import re
import torch

from transformers import (
    RepetitionPenaltyLogitsProcessor,
    PreTrainedTokenizerBase,
)
from typing import List, Tuple, Optional

from lorax_server.pb import generate_pb2
from lorax_server.pb.generate_pb2 import FinishReason
from lorax_server.utils.watermark import WatermarkLogitsProcessor
from lorax_server.utils.logits_process import (
    static_warper,
    HeterogeneousRepetitionPenaltyLogitsProcessor,
    HeterogeneousTemperatureLogitsWarper,
    HeterogeneousTopKLogitsWarper,
    HeterogeneousTopPLogitsWarper,
    HeterogeneousTypicalLogitsWarper,
    HeterogeneousProcessorWrapper,
)


class NextTokenChooser:
    """
    Class representing a next token chooser.

    Args:
        watermark (bool): Whether to apply watermark processing to logits. Default is False.
        temperature (float): The temperature value for warping logits. Default is 1.0.
        repetition_penalty (float): The penalty value for repetition in logits. Default is 1.0.
        top_k (int): The value for top-k warping of logits. Default is None.
        top_p (float): The value for top-p warping of logits. Default is None.
        typical_p (float): The value for typical-p warping of logits. Default is None.
        do_sample (bool): Whether to perform sampling. Default is False.
        seed (int): The seed value for random number generation. Default is 0.
        device (str): The device to use for computation. Default is "cpu".

    Returns:
        next_id (torch.Tensor): The next token ID.
        next_logprob (torch.Tensor): The log probability of the next token.
    """

    def __init__(
        self,
        watermark=False,
        temperature=1.0,
        repetition_penalty=1.0,
        top_k=None,
        top_p=None,
        typical_p=None,
        do_sample=False,
        seed=0,
        device="cpu",
    ):
        self.watermark_processor = (
            WatermarkLogitsProcessor(device=device) if watermark else None
        )
        self.repetition_processor = (
            RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
            if repetition_penalty
            else None
        )

        has_warpers = (
            (temperature is not None and temperature != 1.0)
            or (top_k is not None and top_k != 0)
            or (top_p is not None and top_p < 1.0)
            or (typical_p is not None and typical_p < 1.0)
        )
        if has_warpers:
            self.static_warper = static_warper(
                temperature=temperature, top_k=top_k, top_p=top_p, typical_p=typical_p
            )
        else:
            self.static_warper = None

        sampling = do_sample or has_warpers
        self.choice = Sampling(seed, device) if sampling else Greedy()

    def __call__(self, input_ids, scores):
        if self.watermark_processor is not None:
            scores = self.watermark_processor(input_ids, scores)
        if self.repetition_processor is not None:
            scores = self.repetition_processor(input_ids, scores)

        if self.static_warper is None:
            next_logprob = torch.log_softmax(scores, -1)
        else:
            scores, next_logprob = self.static_warper(scores)

        next_id = self.choice(scores[-1]).view(1, 1)

        return next_id, next_logprob

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.NextTokenChooserParameters,
        device: torch.device,
    ) -> "NextTokenChooser":
        """
        Create a NextTokenChooser instance from a protobuf message.

        Args:
            pb (generate_pb2.NextTokenChooserParameters): The protobuf message containing the parameters.
            device (torch.device): The device to use for computation.

        Returns:
            NextTokenChooser: The NextTokenChooser instance.
        """
        return NextTokenChooser(
            watermark=pb.watermark,
            temperature=pb.temperature,
            repetition_penalty=pb.repetition_penalty,
            top_k=pb.top_k,
            top_p=pb.top_p,
            typical_p=pb.typical_p,
            do_sample=pb.do_sample,
            seed=pb.seed,
            device=device,
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
        eos_token_id: int,
        stop_sequence_criterias: List[StopSequenceCriteria],
        max_new_tokens: int = 20,
        ignore_eos_token: bool = False,
    ):
        self.eos_token_id = eos_token_id
        self.stop_sequence_criterias = stop_sequence_criterias
        self.max_new_tokens = max_new_tokens
        self.current_tokens = 0
        self.current_output = ""
        self.ignore_eos_token = ignore_eos_token

    def __call__(self, last_token: int, last_output: str) -> Tuple[bool, Optional[str]]:
        self.current_tokens += 1
        if self.current_tokens >= self.max_new_tokens:
            return True, FinishReason.FINISH_REASON_LENGTH

        if not self.ignore_eos_token and last_token == self.eos_token_id:
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
        stop_sequence_criterias = [
            StopSequenceCriteria(sequence) for sequence in pb.stop_sequences
        ]
        return StoppingCriteria(
            tokenizer.eos_token_id,
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
        top_k (List[int]): A list of top-k values for top-k-based logits warping.
        top_p (List[float]): A list of top-p values for top-p-based logits warping.
        typical_p (List[float]): A list of typical-p values for typical-p-based logits warping.
        do_sample (List[bool]): A list of booleans indicating whether sampling should be applied for each token.
        seeds (List[int]): A list of seed values for random number generation.

    Attributes:
        watermark_processor (HeterogeneousProcessorWrapper): The watermark logits processor.
        repetition_processor (HeterogeneousRepetitionPenaltyLogitsProcessor): The repetition penalty logits processor.
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
        top_k: List[int],
        top_p: List[float],
        typical_p: List[float],
        do_sample: List[bool],
        seeds: List[int],
    ):
        warpers = []

        self.watermark_processor = (
            HeterogeneousProcessorWrapper(
                {
                    i: WatermarkLogitsProcessor(device=device)
                    for i, do_watermark in enumerate(watermark)
                    if do_watermark
                }
            )
            if any(watermark)
            else None
        )

        self.repetition_processor = (
            HeterogeneousRepetitionPenaltyLogitsProcessor(
                repetition_penalty, dtype, device
            )
            if any([x != 1.0 for x in repetition_penalty])
            else None
        )

        if any([x != 1.0 for x in temperature]):
            do_sample = [
                sample or x != 1.0 for x, sample in zip(temperature, do_sample)
            ]
            warpers.append(
                HeterogeneousTemperatureLogitsWarper(temperature, dtype, device)
            )

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
            self.choice = HeterogeneousSampling(do_sample, seeds, device)
        else:
            self.choice = Greedy()

        self.seeds = seeds
        self.do_sample = do_sample
        self.dtype = dtype
        self.device = device

    def create_n_gram_speculation(self, input_ids, next_ids, accepted_ids, speculate, verbose):
        batch_size = accepted_ids.shape[0]
        device = input_ids.device

        seed_indices = next_ids[accepted_ids.cumsum(dim=-1) - 1]
        match_indices = (input_ids == seed_indices.unsqueeze(-1)).max(dim=1).indices + 1

        spec_range = torch.arange(speculate, device=device)
        all_indices = match_indices.unsqueeze(-1).expand(batch_size, speculate) + spec_range
        all_indices = torch.clamp(all_indices, max=input_ids.shape[1] - 1)

        speculative_ids = input_ids.gather(dim=-1, index=all_indices)
        return speculative_ids

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        speculate: int,
        speculation_ids: Optional[torch.Tensor] = None,
        speculation_scores: Optional[torch.Tensor] = None,
        verbose=False,
    ):
        """
        Perform token processing and selection based on input scores.

        Args:
            input_ids (torch.Tensor): The input tensor of token IDs.
            scores (torch.Tensor): The scores tensor representing the likelihood of each token.
            speculate (int): The number of speculative tokens to generate.
            speculation_ids (Optional[torch.Tensor]): The tensor of speculated token IDs.
            speculation_scores (Optional[torch.Tensor]): The scores tensor for speculated tokens.
            verbose (bool): Whether to enable verbose mode.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: A tuple containing the following:
                - next_ids (torch.Tensor): The selected token IDs for the next step.
                - next_logprobs (torch.Tensor): The log probabilities of the selected token IDs.
                - logprobs (torch.Tensor): The log probabilities of all token IDs.
                - accepted_ids (torch.Tensor): The accepted tokens for each input sequence.
                - speculative_ids (Optional[torch.Tensor]): The selected speculative token IDs.
        """
        if speculation_ids is not None:
            _batches = scores.shape[0] // (speculation_ids.shape[1] + 1) if speculation_ids is not None else scores.shape[0]
            _speculations = speculation_ids.shape[1] + 1 if speculation_ids is not None else 1
            scores = scores.view(_batches, _speculations, -1)

        next_ids = torch.zeros((_batches, _speculations), device=scores.device, dtype=torch.long)
        for j in range(_speculations):
            _scores = scores[:, j]
            if self.watermark_processor is not None:
                _scores = self.watermark_processor(input_ids, _scores)
            if self.repetition_processor is not None:
                _scores = self.repetition_processor(input_ids, _scores)

            for warper in self.warpers:
                _scores = warper(input_ids, _scores)

            _next_ids = self.choice(_scores)
            scores[:, j] = _scores
            next_ids[:, j] = _next_ids
        next_ids = next_ids.view(_batches * _speculations)
        scores = scores.view(_batches * _speculations, -1)

        if speculation_ids is not None:
            accepted_ids = []
            # number of batches
            _batches = next_ids.shape[0] // (speculation_ids.shape[1] + 1)
            # number of speculations
            _speculations = speculation_ids.shape[1] + 1
            indices = []
            for i in range(_batches):
                _next_ids = next_ids[i * _speculations : (i + 1) * _speculations]
                _speculated_ids = speculation_ids[i]
                validate_speculative = _next_ids[:-1] == _speculated_ids
                index = i * _speculations
                accepted = 1
                indices.append(index)
                for valid in validate_speculative.tolist():
                    if valid:
                        index += 1
                        accepted += 1
                        indices.append(index)
                    else:
                        break
                accepted_ids.append(accepted)

            accepted_ids = torch.tensor(
                accepted_ids, device=input_ids.device, dtype=input_ids.dtype
            )
            next_ids = next_ids[indices]
            scores = scores[indices]
            indices = torch.arange(_batches, device=input_ids.device) * _speculations
            if speculation_scores is not None:
                speculation_scores = speculation_scores[indices + accepted_ids - 1]
        else:
            accepted_ids = torch.ones_like(next_ids)

        logprobs = torch.log_softmax(scores, -1)
        next_logprobs = torch.gather(
            torch.log_softmax(scores, -1), 1, next_ids.view(-1, 1)
        ).view(-1)

        if speculate > 0:
            speculative_ids = self.create_n_gram_speculation(
                    input_ids, next_ids, accepted_ids, speculate, verbose
                )
        else:
            speculative_ids = None

        return next_ids, next_logprobs, logprobs, accepted_ids, speculative_ids

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

    @classmethod
    def from_pb(
        cls,
        pb: List[generate_pb2.NextTokenChooserParameters],
        dtype: torch.dtype,
        device: torch.device,
    ) -> "HeterogeneousNextTokenChooser":
        """
        Creates a `HeterogeneousNextTokenChooser` instance from the given protocol buffer.

        Args:
            pb (List[generate_pb2.NextTokenChooserParameters]): The protocol buffer containing the parameters.
            dtype (torch.dtype): The data type of the tokens.
            device (torch.device): The device on which the tokens are processed.

        Returns:
            HeterogeneousNextTokenChooser: The created `HeterogeneousNextTokenChooser` instance.
        """
        return HeterogeneousNextTokenChooser(
            watermark=[pb_.watermark for pb_ in pb],
            temperature=[pb_.temperature for pb_ in pb],
            repetition_penalty=[pb_.repetition_penalty for pb_ in pb],
            top_k=[pb_.top_k for pb_ in pb],
            top_p=[pb_.top_p for pb_ in pb],
            typical_p=[pb_.typical_p for pb_ in pb],
            do_sample=[pb_.do_sample for pb_ in pb],
            seeds=[pb_.seed for pb_ in pb],
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
