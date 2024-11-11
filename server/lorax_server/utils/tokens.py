import re
import warnings
from contextlib import nullcontext
from typing import List, Optional, Set, Tuple, Union

import torch
from transformers import (
    PreTrainedTokenizerBase,
    RepetitionPenaltyLogitsProcessor,
)

from lorax_server.pb import generate_pb2
from lorax_server.pb.generate_pb2 import FinishReason
from lorax_server.utils.logits_process import (
    HeterogeneousFrequencyPenaltyLogitsProcessor,
    HeterogeneousProcessorWrapper,
    HeterogeneousRepetitionPenaltyLogitsProcessor,
    HeterogeneousSchemaLogitsProcessor,
    HeterogeneousTemperatureLogitsWarper,
    HeterogeneousTopKLogitsWarper,
    HeterogeneousTopPLogitsWarper,
    HeterogeneousTypicalLogitsWarper,
    OutlinesLogitsProcessor,
    static_warper,
)
from lorax_server.utils.state import use_ngram
from lorax_server.utils.watermark import WatermarkLogitsProcessor


class NextTokenChooser:
    """
    Class representing a next token chooser.

    Args:
        watermark (bool): Whether to apply watermark processing to logits. Default is False.
        temperature (float): The temperature value for warping logits. Default is 1.0.
        repetition_penalty (float): The penalty value for repetition in logits. Default is 1.0.
        schema (str): A JSON schema string for Outlines logits warping.
        top_k (int): The value for top-k warping of logits. Default is None.
        top_p (float): The value for top-p warping of logits. Default is None.
        typical_p (float): The value for typical-p warping of logits. Default is None.
        do_sample (bool): Whether to perform sampling. Default is False.
        seed (int): The seed value for random number generation. Default is 0.
        device (str): The device to use for computation. Default is "cpu".
        tokenizer (PreTrainedTokenizerBase): A tokenizer to use for processing the tokens.

    Returns:
        next_id (torch.Tensor): The next token ID.
        next_logprob (torch.Tensor): The log probability of the next token.
    """

    def __init__(
        self,
        watermark: bool = False,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        schema: str = None,
        top_k: int = None,
        top_p: float = None,
        typical_p: float = None,
        do_sample: bool = False,
        seed: int = 0,
        device: str = "cpu",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        self.watermark_processor = WatermarkLogitsProcessor(device=device) if watermark else None
        self.repetition_processor = (
            RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty) if repetition_penalty else None
        )

        self.schema_processor = OutlinesLogitsProcessor(schema, tokenizer) if schema and tokenizer else None

        # Temperature = 1 does not change logits; do not use warper
        # Temperature = 0 invokes determinstic token choosing; do not warp
        has_warpers = (
            (temperature is not None and temperature != 1.0 and temperature != 0)
            or (top_k is not None and top_k != 0)
            or (top_p is not None and top_p < 1.0)
            or (typical_p is not None and typical_p < 1.0)
        )
        if has_warpers:
            self.static_warper = static_warper(temperature=temperature, top_k=top_k, top_p=top_p, typical_p=typical_p)
        else:
            self.static_warper = None

        sampling = do_sample or has_warpers

        # do not sample if temperature is 0, even if do_sample flag is set True
        # warn user about deterministic sampling
        if sampling and temperature == 0:
            sampling = False
            warnings.warn("Temperature is set to 0, token sampling will be disabled")

        self.choice = Sampling(seed, device) if sampling else Greedy()

    def __call__(self, input_ids, scores):
        if self.watermark_processor is not None:
            scores = self.watermark_processor(input_ids, scores)
        if self.repetition_processor is not None:
            scores = self.repetition_processor(input_ids, scores)
        if self.schema_processor is not None:
            scores = self.schema_processor(scores)

        if self.static_warper is None:
            next_logprob = torch.log_softmax(scores, -1)
        else:
            scores, next_logprob = self.static_warper(scores)

        next_id = self.choice(scores[-1]).view(1, 1)

        return next_id, next_logprob

    def next_state(self, next_token_id: int):
        if self.schema_processor is not None:
            self.schema_processor.next_state(next_token_id)

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.NextTokenChooserParameters,
        device: torch.device,
        tokenizer: Optional[PreTrainedTokenizerBase],
    ) -> "NextTokenChooser":
        """
        Create a NextTokenChooser instance from a protobuf message.

        Args:
            pb (generate_pb2.NextTokenChooserParameters): The protobuf message containing the parameters.
            device (torch.device): The device to use for computation.
            tokenizer (PreTrainedTokenizerBase): A tokenizer for use in processing the tokens.

        Returns:
            NextTokenChooser: The NextTokenChooser instance.
        """
        return NextTokenChooser(
            watermark=pb.watermark,
            temperature=pb.temperature,
            repetition_penalty=pb.repetition_penalty,
            schema=pb.schema,
            top_k=pb.top_k,
            top_p=pb.top_p,
            typical_p=pb.typical_p,
            do_sample=pb.do_sample,
            seed=pb.seed,
            device=device,
            tokenizer=tokenizer,
        )


class StopSequenceCriteria:
    def __init__(self, stop_sequence: str):
        stop_sequence = re.escape(stop_sequence)
        self.regex = re.compile(f".*{stop_sequence}$")

    def __call__(self, output: str) -> bool:
        if self.regex.findall(output):
            return True
        return False


class StoppingCriteria:
    """
    Class representing the stopping criteria for token generation.

    Args:
        eos_token_id (int): The ID of the end-of-sequence token.
        stop_sequence_criterias (List[StopSequenceCriteria]): A list of stop sequence criteria.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 20.
        ignore_eos_token (bool, optional): Whether to ignore the end-of-sequence token. Defaults to False.
    """

    def __init__(
        self,
        eos_token_id: Optional[Union[int, Set[int]]],
        stop_sequence_criterias: List[StopSequenceCriteria],
        max_new_tokens: int = 20,
        ignore_eos_token: bool = False,
    ):
        self.eos_token_ids = (
            (eos_token_id if isinstance(eos_token_id, set) else set([eos_token_id]))
            if eos_token_id is not None
            else set()
        )
        self.stop_sequence_criterias = stop_sequence_criterias
        self.max_new_tokens = max_new_tokens
        self.current_tokens = 0
        self.current_output = ""
        self.ignore_eos_token = ignore_eos_token

    def __call__(self, last_token: int, last_output: str) -> Tuple[bool, Optional[str]]:
        self.current_tokens += 1
        if self.current_tokens >= self.max_new_tokens:
            return True, FinishReason.FINISH_REASON_LENGTH

        # If the last token is a tensor, convert it to an integer
        # Otherwise the set membership check will fail
        if isinstance(last_token, torch.Tensor):
            last_token = last_token.item()

        if not self.ignore_eos_token and last_token in self.eos_token_ids:
            return True, FinishReason.FINISH_REASON_EOS_TOKEN

        self.current_output += last_output
        for stop_sequence_criteria in self.stop_sequence_criterias:
            if stop_sequence_criteria(self.current_output):
                return True, FinishReason.FINISH_REASON_STOP_SEQUENCE

        return False, None

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.StoppingCriteriaParameters,
        tokenizer: PreTrainedTokenizerBase,
    ) -> "StoppingCriteria":
        stop_sequence_criterias = [StopSequenceCriteria(sequence) for sequence in pb.stop_sequences]
        eos_token_id = getattr(tokenizer, "eos_token_ids", tokenizer.eos_token_id)
        return StoppingCriteria(
            eos_token_id,
            stop_sequence_criterias,
            pb.max_new_tokens,
            pb.ignore_eos_token,
        )


class HeterogeneousNextTokenChooser:
    """
    A class that represents a heterogeneous next token chooser for generating tokens.

    Args:
        dtype (torch.dtype): The data type of the tokens.
        device (torch.device): The device on which the tokens are processed.
        watermark (List[bool]): A list of booleans indicating whether watermark processing should be applied for each token.
        temperature (List[float]): A list of temperature values for temperature-based logits warping.
        repetition_penalty (List[float]): A list of repetition penalty values for repetition penalty-based logits warping.
        frequency_penalty (List[float]): A list of frequency penalty values for frequency penalty-based logits warping.
        presence_penalty (List[float]): A list of presence penalty values for presence penalty-based logits warping.
        schemas (List[str]): A list of JSON schema strings for Outlines logits warping.
        top_k (List[int]): A list of top-k values for top-k-based logits warping.
        top_p (List[float]): A list of top-p values for top-p-based logits warping.
        typical_p (List[float]): A list of typical-p values for typical-p-based logits warping.
        do_sample (List[bool]): A list of booleans indicating whether sampling should be applied for each token.
        seeds (List[int]): A list of seed values for random number generation.
        tokenizers (List[PreTrainedTokenizerBase]): A list of tokenizers to use for processing the tokens.

    Attributes:
        watermark_processor (HeterogeneousProcessorWrapper): The watermark logits processor.
        repetition_processor (HeterogeneousRepetitionPenaltyLogitsProcessor): The repetition penalty logits processor.
        frequency_processor (HeterogeneousFrequencyPenaltyLogitsProcessor): The frequency penalty logits processor.
        schema_processor (HeterogeneousSchemaLogitsProcessor): The JSON schema logits processor.
        warpers (List[HeterogeneousLogitsWarper]): The list of logits warpers.
        choice (HeterogeneousSampling or Greedy): The token choice strategy.
        seeds (List[int]): The list of seed values.
        do_sample (List[bool]): The list of booleans indicating whether sampling should be applied.
        dtype (torch.dtype): The data type of the tokens.
        device (torch.device): The device on which the tokens are processed.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        watermark: List[bool],
        temperature: List[float],
        repetition_penalty: List[float],
        frequency_penalty: List[float],
        presence_penalty: List[float],
        schemas: List[str],
        top_k: List[int],
        top_p: List[float],
        typical_p: List[float],
        do_sample: List[bool],
        seeds: List[int],
        tokenizers: List[PreTrainedTokenizerBase],
        sequence_processors: Optional[List[OutlinesLogitsProcessor]] = None,
    ):
        warpers = []

        self.watermark_processor = (
            HeterogeneousProcessorWrapper(
                {i: WatermarkLogitsProcessor(device=device) for i, do_watermark in enumerate(watermark) if do_watermark}
            )
            if any(watermark)
            else None
        )

        self.repetition_processor = (
            HeterogeneousRepetitionPenaltyLogitsProcessor(repetition_penalty, dtype, device)
            if any([x != 1.0 for x in repetition_penalty])
            else None
        )

        self.frequency_processor = (
            HeterogeneousFrequencyPenaltyLogitsProcessor(frequency_penalty, presence_penalty, dtype, device)
            if any([x != 0.0 for x in frequency_penalty]) or any([x != 0.0 for x in presence_penalty])
            else None
        )

        if sequence_processors is not None:
            # Reuse the state from the previous generation steps
            self.schema_processor = (
                HeterogeneousSchemaLogitsProcessor(sequence_processors) if any(sequence_processors) else None
            )
        else:
            self.schema_processor = (
                HeterogeneousSchemaLogitsProcessor.from_schemas(schemas, tokenizers) if any(schemas) else None
            )

        if any([(x != 1.0 and x != 0) for x in temperature]):
            # set sample flags for each index
            # do not sample this index if temperature is 0 or 1
            do_sample = [sample or (x != 1.0 and x != 0) for x, sample in zip(temperature, do_sample)]
            warpers.append(HeterogeneousTemperatureLogitsWarper(temperature, dtype, device))

        if any([x != 0 for x in top_k]):
            do_sample = [sample or x != 0 for x, sample in zip(top_k, do_sample)]
            warpers.append(HeterogeneousTopKLogitsWarper(top_k, device))

        if any([x < 1.0 for x in top_p]):
            do_sample = [sample or x < 1.0 for x, sample in zip(top_p, do_sample)]
            warpers.append(HeterogeneousTopPLogitsWarper(top_p, dtype, device))

        if any([x < 1.0 for x in typical_p]):
            do_sample = [sample or x < 1.0 for x, sample in zip(typical_p, do_sample)]
            warpers.append(HeterogeneousTypicalLogitsWarper(typical_p, dtype, device))

        self.warpers = warpers

        if any(do_sample):
            # sample tokens from distribution if any sample flags are set True
            self.choice = HeterogeneousSampling(do_sample, seeds, device)
        else:
            # sampling for all requests is set false, do Greedy / deterministic sampling
            self.choice = Greedy()

        self.seeds = seeds
        self.do_sample = do_sample
        self.dtype = dtype
        self.device = device

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        speculate: int,
        speculated_ids: Optional[torch.Tensor] = None,
        speculative_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Chooses the next tokens based on the input IDs and scores.

        Args:
            input_ids (torch.Tensor): The input tensor containing the token IDs.
            scores (torch.Tensor): The tensor containing the scores for each token.

        Returns:
            torch.Tensor: The tensor containing the next token IDs.
            torch.Tensor: The tensor containing the log probabilities of the next tokens.
            torch.Tensor: The tensor containing the accepted next tokens for this step.
            Optional[torch.Tensor]: The tensor containing the speculative token IDs for the next step.
        """
        if speculated_ids is not None:
            B = scores.shape[0] // (speculated_ids.shape[1] + 1)
            S = speculated_ids.shape[1] + 1
        else:
            B = scores.shape[0]
            S = 1
        scores = scores.view(B, S, -1)

        next_ids = torch.zeros((B, S), device=scores.device, dtype=torch.long)
        with self.schema_processor.restore_state() if self.schema_processor is not None else nullcontext():
            for j in range(S):
                scores_j = scores[:, j]

                if self.watermark_processor is not None:
                    scores_j = self.watermark_processor(input_ids, scores_j)
                if self.repetition_processor is not None:
                    scores_j = self.repetition_processor(input_ids, scores_j)
                if self.frequency_processor is not None:
                    scores_j = self.frequency_processor(input_ids, scores_j)
                if self.schema_processor is not None:
                    scores_j = self.schema_processor(input_ids, scores_j)

                for warper in self.warpers:
                    scores_j = warper(input_ids, scores_j)

                next_ids_j = self.choice(scores_j)
                scores[:, j] = scores_j
                next_ids[:, j] = next_ids_j

                # need to update schema processor state for next_ids_j before the next loop iteration
                # can revert this at the end of the loop
                if self.schema_processor is not None:
                    for batch_idx in range(B):
                        self.schema_processor.next_state(batch_idx, next_ids_j[batch_idx].item())

        next_ids = next_ids.view(B * S)
        allscores = scores.view(B * S, -1)
        alllogprobs = torch.log_softmax(allscores, -1)

        if speculated_ids is not None:
            accepted_ids = []
            B = next_ids.shape[0] // (speculated_ids.shape[1] + 1)
            S = speculated_ids.shape[1] + 1
            indices = []
            for i in range(B):
                next_ids_i = next_ids[i * S : (i + 1) * S]
                speculated_ids_i = speculated_ids[i]
                validate_speculative = next_ids_i[:-1] == speculated_ids_i
                index = i * S
                accepted = 1
                # First is always valid
                indices.append(index)
                for valid in validate_speculative.tolist():
                    if valid:
                        index += 1
                        accepted += 1
                        indices.append(index)
                    else:
                        break
                accepted_ids.append(accepted)

            accepted_ids = torch.tensor(accepted_ids, device=input_ids.device, dtype=input_ids.dtype)
            next_ids = next_ids[indices]
            logprobs = alllogprobs[indices]
            indices = torch.arange(B, device=input_ids.device) * S
            if speculative_scores is not None:
                speculative_scores = speculative_scores[indices + accepted_ids - 1]
        else:
            accepted_ids = torch.ones_like(next_ids)
            logprobs = alllogprobs

        next_logprobs = torch.gather(logprobs, 1, next_ids.view(-1, 1)).view(-1)

        speculative_ids = None
        if speculate > 0:
            if speculative_scores is not None:
                # Only use greedy sampling for speculative tokens
                speculative_ids = Greedy()(speculative_scores)
            elif use_ngram():
                speculative_ids = ngram_speculate(input_ids, next_ids, accepted_ids, speculate)

        return next_ids, next_logprobs, accepted_ids, speculative_ids

    def filter(self, indices):
        """
        Filters the chooser based on the given indices.

        Args:
            indices: The indices to filter the chooser with.

        Returns:
            HeterogeneousNextTokenChooser: The filtered chooser.
        """
        if self.watermark_processor is not None:
            self.watermark_processor = self.watermark_processor.filter(indices)

        if self.repetition_processor is not None:
            self.repetition_processor = self.repetition_processor.filter(indices)

        if self.frequency_processor is not None:
            self.frequency_processor = self.frequency_processor.filter(indices)

        if self.schema_processor is not None:
            self.schema_processor = self.schema_processor.filter(indices)

        filtered_warpers = []
        for warper in self.warpers:
            filtered_warper = warper.filter(indices)
            if filtered_warper is not None:
                filtered_warpers.append(filtered_warper)
        self.warpers = filtered_warpers

        self.seeds = [self.seeds[i] for i in indices]
        self.do_sample = [self.do_sample[i] for i in indices]

        if any(self.do_sample):
            self.choice.filter(indices)
        else:
            self.choice = Greedy()

        return self

    def next_state(self, batch_idx: int, next_token_id: int):
        if self.schema_processor is not None:
            self.schema_processor.next_state(batch_idx, next_token_id)

    @classmethod
    def from_pb(
        cls,
        pb: List[generate_pb2.NextTokenChooserParameters],
        tokenizers: List[PreTrainedTokenizerBase],
        dtype: torch.dtype,
        device: torch.device,
        sequence_processors: Optional[List[OutlinesLogitsProcessor]] = None,
    ) -> "HeterogeneousNextTokenChooser":
        """
        Creates a `HeterogeneousNextTokenChooser` instance from the given protocol buffer.

        Args:
            pb (List[generate_pb2.NextTokenChooserParameters]): The protocol buffer containing the parameters.
            tokenizers (List[PreTrainedTokenizerBase]): The tokenizers to use for processing the tokens.
            dtype (torch.dtype): The data type of the tokens.
            device (torch.device): The device on which the tokens are processed.
            sequence_processors (Optional[List[OutlinesLogitsProcessor]]): The sequence processors to use for processing the tokens.

        Returns:
            HeterogeneousNextTokenChooser: The created `HeterogeneousNextTokenChooser` instance.
        """
        return HeterogeneousNextTokenChooser(
            watermark=[pb_.watermark for pb_ in pb],
            temperature=[pb_.temperature for pb_ in pb],
            repetition_penalty=[pb_.repetition_penalty for pb_ in pb],
            frequency_penalty=[pb_.frequency_penalty for pb_ in pb],
            presence_penalty=[pb_.presence_penalty for pb_ in pb],
            schemas=[pb_.schema for pb_ in pb],
            top_k=[pb_.top_k for pb_ in pb],
            top_p=[pb_.top_p for pb_ in pb],
            typical_p=[pb_.typical_p for pb_ in pb],
            do_sample=[pb_.do_sample for pb_ in pb],
            seeds=[pb_.seed for pb_ in pb],
            tokenizers=tokenizers,
            sequence_processors=sequence_processors,
            device=device,
            dtype=dtype,
        )


class Sampling:
    def __init__(self, seed: int, device: str = "cpu"):
        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)
        self.seed = seed

    def __call__(self, logits):
        probs = torch.nn.functional.softmax(logits, -1)
        # Avoid GPU<->CPU sync done by torch multinomial
        # See: https://github.com/pytorch/pytorch/blob/925a3788ec5c06db62ca732a0e9425a26a00916f/aten/src/ATen/native/Distributions.cpp#L631-L637
        q = torch.empty_like(probs).exponential_(1, generator=self.generator)
        return probs.div_(q).argmax()


class Greedy:
    def __call__(self, logits):
        return logits.argmax(dim=-1)


class HeterogeneousSampling:
    r"""
    Mixed greedy and probabilistic sampling. Compute both and pick the right one for each sample.
    """

    def __init__(self, do_sample: List[bool], seeds: List[int], device: torch.device):
        self.seeds = seeds

        self.greedy_indices = []
        self.sampling_mapping = {}
        for i, (sample, seed) in enumerate(zip(do_sample, seeds)):
            if sample:
                self.sampling_mapping[i] = Sampling(seed, device)
            else:
                self.greedy_indices.append(i)

        self.greedy = Greedy()

    def __call__(self, logits):
        out = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
        if self.greedy_indices:
            # Computing for all indices is faster than slicing
            torch.argmax(logits, -1, out=out)

        for i, sampling in self.sampling_mapping.items():
            out[i] = sampling(logits[i])
        return out

    def filter(self, indices):
        new_greedy_indices = []
        new_sampling_mapping = {}
        for i, idx in enumerate(indices):
            if idx in self.sampling_mapping:
                new_sampling_mapping[i] = self.sampling_mapping[idx]
            else:
                new_greedy_indices.append(i)

        self.greedy_indices = new_greedy_indices
        self.sampling_mapping = new_sampling_mapping
        return self


def ngram_speculate(
    input_ids: torch.Tensor,
    next_ids: torch.Tensor,
    accepted_ids: torch.Tensor,
    speculate: int,
) -> torch.Tensor:
    # Inspired by TGI implementation of:
    # https://github.com/apoorvumang/prompt-lookup-decoding
    B = accepted_ids.shape[0]

    # Find the last match of the seed tokens in the input_ids
    seeds = next_ids[accepted_ids.cumsum(dim=-1) - 1]
    indices = (input_ids == seeds.unsqueeze(-1)).max(dim=1).indices + 1

    # Speculate out from the last match by the number of speculative tokens `speculate`
    # Clamp the indices to the maximum length of the input_ids to prevent out-of-bound errors
    all_indices = indices.unsqueeze(-1).expand(B, speculate) + torch.arange(speculate, device=input_ids.device)
    all_indices = torch.clamp(all_indices, max=input_ids.shape[1] - 1)

    # Gather the speculative tokens from the input_ids to form a [B, S] tensor
    speculative_ids = input_ids.gather(dim=-1, index=all_indices)
    return speculative_ids
