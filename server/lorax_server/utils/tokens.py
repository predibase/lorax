import re
import torch
import warnings

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

        # Temperature = 1 does not change logits; do not use warper
        # Temperature = 0 invokes determinstic token choosing; do not warp
        has_warpers = (
            (temperature is not None and temperature != 1.0 and temperature != 0)
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
            # set sample flags for each index
            # do not sample this index if temperature is 0 or 1
            do_sample = [
                sample or (x != 1.0 and x != 0)
                for x, sample in zip(temperature, do_sample)
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

        # sample tokens from distribution if any sample flags are set True
        if any(do_sample):
            self.choice = HeterogeneousSampling(do_sample, seeds, device)
        # all tokens are set false, do Greedy / deterministic sampling
        else:
            self.choice = Greedy()

        self.seeds = seeds
        self.do_sample = do_sample
        self.dtype = dtype
        self.device = device

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):
        """
        Chooses the next tokens based on the input IDs and scores.

        Args:
            input_ids (torch.Tensor): The input tensor containing the token IDs.
            scores (torch.Tensor): The tensor containing the scores for each token.

        Returns:
            torch.Tensor: The tensor containing the next token IDs.
            torch.Tensor: The tensor containing the log probabilities of the next tokens.
        """
        if self.watermark_processor is not None:
            scores = self.watermark_processor(input_ids, scores)
        if self.repetition_processor is not None:
            scores = self.repetition_processor(input_ids, scores)

        for warper in self.warpers:
            scores = warper(input_ids, scores)

        next_ids = self.choice(scores)
        next_logprobs = torch.gather(
            torch.log_softmax(scores, -1), 1, next_ids.view(-1, 1)
        ).view(-1)

        return next_ids, next_logprobs

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
