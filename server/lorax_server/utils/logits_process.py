import math
from contextlib import contextmanager
from functools import lru_cache
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    LogitsProcessor,
    LogitsWarper,
    PreTrainedTokenizerBase,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)

try:
    from outlines.fsm.fsm import RegexFSM
    from outlines.fsm.json_schema import build_regex_from_schema

    HAS_OUTLINES = True
except ImportError:
    HAS_OUTLINES = False

mempool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None


class StaticWarper:
    def __init__(
        self,
        temperature=1.0,
        top_k=None,
        top_p=None,
        typical_p=None,
    ):
        self.warpers = []

        if temperature is not None and temperature != 1.0 and temperature != 0:
            temperature = float(temperature)
            self.warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            self.warpers.append(TopKLogitsWarper(top_k=top_k))
        if top_p is not None and top_p < 1.0:
            self.warpers.append(TopPLogitsWarper(top_p=top_p))
        if typical_p is not None and typical_p < 1.0:
            self.warpers.append(TypicalLogitsWarper(mass=typical_p))

        self.cuda_graph = None
        self.static_scores = None
        self.static_warped_scores = None
        self.static_next_logprob = None

    def __call__(self, scores):
        if torch.cuda.is_available():
            if self.cuda_graph is None:
                self.static_scores = scores
                self.cuda_graph = torch.cuda.CUDAGraph()

                with torch.cuda.graph(self.cuda_graph, pool=mempool):
                    local_scores = self.static_scores
                    for warper in self.warpers:
                        local_scores = warper(None, local_scores)

                    self.static_warped_scores = local_scores
                    # Compute logprobs
                    self.static_next_logprob = torch.log_softmax(self.static_warped_scores, -1)

            self.static_scores.copy_(scores)
            self.cuda_graph.replay()

            return self.static_warped_scores, self.static_next_logprob

        # CPU branch
        for warper in self.warpers:
            scores = warper(None, scores)
        return scores, torch.log_softmax(scores, -1)


@lru_cache(10)
def static_warper(
    temperature: Optional[float],
    top_k: Optional[int],
    top_p: Optional[float],
    typical_p: Optional[float],
) -> StaticWarper:
    return StaticWarper(temperature=temperature, top_k=top_k, top_p=top_p, typical_p=typical_p)


class FrequencyPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    Frequency penalty as defined by OpenAI

    Args:
        penalty (`float`):
            The parameter for frequency penalty. 0.0 means no penalty.
    """

    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)
        # if score < 0 then penalty has to be multiplied to reduce the previous token probability
        score = -torch.where(score < 0, score * self.penalty, score / self.penalty)
        # set score to 0 where input_ids is a padding token
        score *= input_ids.ne(0)

        return scores.scatter_add_(1, input_ids, score)


class HeterogeneousRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing an exponential penalty on repeated sequences.
    This version allows for a separate value for each sample and runs inplace when possible.
    It doesn't validate inputs.

    Args:
        repetition_penalty (`List[float]`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """

    def __init__(self, penalty: List[float], dtype: torch.dtype, device: torch.device):
        self.penalty = penalty
        self.penalty_tensor = torch.tensor(penalty, dtype=dtype, device=device).unsqueeze(1)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * self.penalty_tensor, score / self.penalty_tensor)

        scores.scatter_(1, input_ids, score)
        return scores

    def filter(self, indices):
        self.penalty = [self.penalty[i] for i in indices]
        if any([x != 1.0 for x in self.penalty]):
            self.penalty_tensor = self.penalty_tensor[indices]
            return self
        return None


class HeterogeneousFrequencyPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    Frequency penalty as defined by OpenAI in
    https://platform.openai.com/docs/guides/text-generation/parameter-details

    Args:
        frequency_penalty (`List[float]`):
            The parameter for frequency penalty. 0.0 means no penalty.
        presence_penalty (`List[float]`):
            The parameter for presence penalty. 0.0 means no penalty.
    """

    def __init__(
        self, frequency_penalty: List[float], presence_penalty: List[float], dtype: torch.dtype, device: torch.device
    ):
        self.frequency_penalty = frequency_penalty
        self.frequency_penalty_tensor = torch.tensor(frequency_penalty, dtype=dtype, device=device).unsqueeze(1)

        self.presence_penalty = presence_penalty
        self.presence_penalty_tensor = torch.tensor(presence_penalty, dtype=dtype, device=device).unsqueeze(1)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        batch_size, input_size = input_ids.size()
        vocab_size = scores.size(1)

        # Calculate the frequency for each token so far
        token_freq = torch.zeros(batch_size, vocab_size, device=input_ids.device)
        token_freq.scatter_add_(1, input_ids, torch.ones_like(input_ids, dtype=torch.float))
        mask = token_freq > 0
        token_freq /= input_size

        # Apply the frequency and presence penalties to logits
        scores -= token_freq * self.frequency_penalty_tensor
        scores -= mask * self.presence_penalty_tensor

        return scores

    def filter(self, indices):
        self.frequency_penalty = [self.frequency_penalty[i] for i in indices]
        self.presence_penalty = [self.presence_penalty[i] for i in indices]
        if any([x != 0.0 for x in self.frequency_penalty]) or any([x != 0.0 for x in self.presence_penalty]):
            self.frequency_penalty_tensor = self.frequency_penalty_tensor[indices]
            self.presence_penalty_tensor = self.presence_penalty_tensor[indices]
            return self
        return None


class HeterogeneousTemperatureLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] for temperature (exponential scaling output probability distribution).
    This version allows for a separate value for each sample and runs inplace when possible.
    It doesn't validate inputs.

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: List[float], dtype: torch.dtype, device: torch.device):
        self.temperature = temperature
        self.temperature_tensor = torch.tensor(temperature, dtype=dtype, device=device).unsqueeze(1)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores.div_(self.temperature_tensor)
        return scores

    def filter(self, indices):
        self.temperature = [self.temperature[i] for i in indices]
        if any([x != 1.0 for x in self.temperature]):
            self.temperature_tensor = self.temperature_tensor[indices]
            return self
        return None


class HeterogeneousTopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
    This version allows for a separate value for each sample and runs inplace when possible.
    It doesn't validate inputs.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(
        self,
        top_p: List[float],
        dtype: torch.dtype,
        device: torch.device,
        filter_value: float = -math.inf,
        min_tokens_to_keep: int = 1,
    ):
        self.top_p = top_p
        self.top_p_opposite = 1 - torch.tensor(top_p, dtype=dtype, device=device).unsqueeze(1)
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        probs = sorted_logits.softmax(dim=-1)
        # This is way faster for some reason
        for i in range(probs.shape[0]):
            probs[i] = probs[i].cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = probs <= self.top_p_opposite
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        warped_scores = scores.masked_fill_(indices_to_remove, self.filter_value)

        return warped_scores

    def filter(self, indices):
        self.top_p = [self.top_p[i] for i in indices]
        if any([x < 1.0 for x in self.top_p]):
            self.top_p_opposite = self.top_p_opposite[indices]
            return self
        return None


class HeterogeneousTopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.
    This version allows for a separate value for each sample and runs inplace when possible.
    It doesn't validate inputs.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(
        self,
        top_k: List[int],
        device: torch.device,
        filter_value: float = -math.inf,
        min_tokens_to_keep: int = 1,
    ):
        self.top_k = top_k
        self.max_top_k = max(top_k)
        # value - 1 as we will use top_k to index and python uses 0 based numbering
        self.top_k_tensor = torch.tensor(
            [max(x - 1, min_tokens_to_keep - 1) for x in top_k],
            dtype=torch.int64,
            device=device,
        ).unsqueeze(1)

        # 0 is a special value that disables top_k warping for this member of the batch
        disabled = [x == 0 for x in top_k]

        if any(disabled):
            self.top_k_disabled_mask = torch.tensor(disabled, dtype=torch.bool, device=device).view(-1, 1)
        else:
            self.top_k_disabled_mask = None

        self.filter_value = filter_value

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # If max_top_k is superior to the vocab, we need to clamp or the warper will fail
        if scores.size(-1) < self.max_top_k:
            max_top_k = scores.size(-1)
            top_k = torch.clamp_max(self.top_k_tensor, max_top_k)
        else:
            max_top_k = self.max_top_k
            top_k = self.top_k_tensor

        # Get the kth score for each member of the batch
        kth_scores = torch.gather(torch.topk(scores, max_top_k)[0], 1, top_k)

        # Mask member of kth_scores that do not want to use top_k warping
        if self.top_k_disabled_mask is not None:
            kth_scores.masked_fill_(self.top_k_disabled_mask, self.filter_value)

        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < kth_scores
        scores.masked_fill_(indices_to_remove, self.filter_value)
        return scores

    def filter(self, indices):
        self.top_k = [self.top_k[i] for i in indices]
        disabled = [x == 0 for x in self.top_k]

        if not all(disabled):
            self.top_k_tensor = self.top_k_tensor[indices]
            self.max_top_k = max(self.top_k)

            if self.top_k_disabled_mask is not None:
                self.top_k_disabled_mask = self.top_k_disabled_mask[indices] if any(disabled) else None

            return self
        return None


class HeterogeneousTypicalLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs typical decoding. See [Typical Decoding for Natural Language
    Generation](https://arxiv.org/abs/2202.00666) for more information.
    This version allows for a separate value for each sample and runs inplace when possible.
    It doesn't validate inputs.

    Args:
        mass (`float`):
            Value of typical_p between 0 and 1 inclusive, defaults to 0.9.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(
        self,
        mass: List[float],
        dtype: torch.dtype,
        device: torch.device,
        filter_value: float = -math.inf,
        min_tokens_to_keep: int = 1,
    ):
        self.mass = mass
        self.mass_tensor = torch.tensor(mass, dtype=dtype, device=device).unsqueeze(1)

        # 1 is a special value that disables typical_p warping for this member of the batch
        disabled = [x == 1.0 for x in mass]

        if any(disabled):
            self.disabled_mask = torch.tensor(disabled, dtype=torch.bool, device=device)
        else:
            self.disabled_mask = None

        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # shift and sort
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        probs = sorted_logits.softmax(dim=-1)
        # This is way faster for some reason
        for i in range(probs.shape[0]):
            probs[i] = probs[i].cumsum(dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (probs < self.mass_tensor).sum(dim=1)
        last_ind[last_ind < 0] = 0

        if self.disabled_mask is not None:
            last_ind.masked_fill_(self.disabled_mask, scores.shape[-1] - 1)

        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        warped_scores = scores.masked_fill_(indices_to_remove, self.filter_value)

        return warped_scores

    def filter(self, indices):
        self.mass = [self.mass[i] for i in indices]
        disabled = [x == 1.0 for x in self.mass]

        if not all(disabled):
            self.mass_tensor = self.mass_tensor[indices]

            if self.disabled_mask is not None:
                self.disabled_mask = self.disabled_mask[indices] if any(disabled) else None

            return self
        return None


class HeterogeneousProcessorWrapper(LogitsProcessor):
    r"""
    A wrapper for logit warpers or processors without heterogeneous parameter support.
    Args:
        processors (`Dict[int, Union[LogitsProcessor, LogitsWarper]]`):
            A mapping of sample indices to logit warpers or processors, to be run sequentially.
    """

    def __init__(
        self,
        processors: Dict[int, Union[LogitsProcessor, LogitsWarper]],
    ):
        self.processors = processors

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for i, processor in self.processors.items():
            scores[i : i + 1] = processor(input_ids[i : i + 1], scores[i : i + 1])
        return scores

    def filter(self, indices):
        new_processors = {}
        for i, idx in enumerate(indices):
            if idx in self.processors:
                new_processors[i] = self.processors[idx]

        if new_processors:
            self.processors = new_processors
            return self
        return None


class HeterogeneousSchemaLogitsProcessor(LogitsProcessor):
    """
    [`LogitsWarper`] for JSON schema enforcement.
    This version uses Outlines to perform the constrained decoding.

    Args:
        sequence_processors (`List[Optional[OutlinesLogitsProcessor]]`):
            The Outlines processors to use for each request.
    """

    def __init__(
        self,
        sequence_processors: List[Optional["OutlinesLogitsProcessor"]],
    ):
        self.sequence_processors = sequence_processors

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for i, processor in enumerate(self.sequence_processors):
            if processor is not None:
                scores[i : i + 1] = processor(scores[i : i + 1])
        return scores

    def filter(self, indices):
        self.sequence_processors = [self.sequence_processors[i] for i in indices]
        if any([x is not None for x in self.sequence_processors]):
            return self
        return None

    def next_state(self, batch_idx: int, next_token_id: int):
        if self.sequence_processors[batch_idx] is not None:
            self.sequence_processors[batch_idx].next_state(next_token_id)

    @contextmanager
    def restore_state(self):
        states = [processor.fsm_state if processor is not None else None for processor in self.sequence_processors]
        try:
            yield
        finally:
            for i, state in enumerate(states):
                if state is not None:
                    self.sequence_processors[i].fsm_state = state

    @classmethod
    def from_schemas(
        cls,
        schemas: List[Optional[str]],
        tokenizers: List[Optional[PreTrainedTokenizerBase]],
    ) -> "HeterogeneousSchemaLogitsProcessor":
        """
        Args:
            schemas (`List[Optional[str]]`):
                The JSON encoded schemas to enforce. `None` means no enforcement.
            tokenizers (`List[Optional[PreTrainedTokenizerBase]]`):
                The tokenizers to use for each request.
        """
        if schemas is None:
            schemas = []
        if tokenizers is None:
            tokenizers = []

        sequence_processors = [
            OutlinesLogitsProcessor(schema, tokenizer) if schema and tokenizer else None
            for schema, tokenizer in zip(schemas, tokenizers)
        ]
        return cls(sequence_processors)


# Source: https://github.com/outlines-dev/outlines/blob/main/outlines/serve/vllm.py
class OutlinesLogitsProcessor(LogitsProcessor):
    def __init__(self, schema: str, tokenizer: PreTrainedTokenizerBase):
        """Compile the FSM that drives the regex-guided generation.

        Args:
            schema (str):
                JSON schema to enforce.
            tokenizer (PreTrainedTokenizerBase):
                The tokenizer to use for the FSM.
        """
        if not HAS_OUTLINES:
            raise ImportError("Unable to enforce JSON schema: `outlines` is not installed.")

        self.tokenizer = OutlinesLogitsProcessor.adapt_tokenizer(tokenizer)
        self.fsm = OutlinesLogitsProcessor.compile_fsm(schema, self.tokenizer)
        self.fsm_state = 0

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        if self.fsm_state == -1 or self.fsm is None:
            return scores

        allowed_tokens = self.fsm.allowed_token_ids(self.fsm_state)

        # Filter out tokens that are out of bounds
        # TODO(travis): figure out why this happens
        allowed_tokens = [t for t in allowed_tokens if t < scores.shape[-1]]

        mask = torch.full_like(scores, -math.inf, device=scores.device)
        mask[:, allowed_tokens] = 0
        biased_scores = scores + mask
        return biased_scores

    def next_state(self, next_token_id: int):
        if self.fsm_state != -1:
            self.fsm_state = self.fsm.next_state(self.fsm_state, next_token_id)

    @staticmethod
    @lru_cache(maxsize=32, typed=True)
    def compile_fsm(schema, tokenizer):
        regex_string = build_regex_from_schema(schema)
        return RegexFSM(regex_string, tokenizer)

    @staticmethod
    @lru_cache(maxsize=32, typed=True)
    def adapt_tokenizer(tokenizer: PreTrainedTokenizerBase):
        """Adapt the HF tokenizer to use to compile the FSM.

        The API of Outlines tokenizers is slightly different to that of
        `transformers`. In addition, we need to handle the missing spaces to
        Llama's tokenizer to be able to compile FSMs for this model.
        """
        if hasattr(tokenizer, "vocabulary"):
            # We've already adapted the tokenizer from a previous request
            return tokenizer

        tokenizer.vocabulary = tokenizer.get_vocab()
        tokenizer.special_tokens = set(tokenizer.all_special_tokens)

        def convert_token_to_string(token: str) -> str:
            from transformers.file_utils import SPIECE_UNDERLINE

            string = tokenizer.convert_tokens_to_string([token])

            # A hack to handle missing spaces to HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

            return string

        tokenizer.convert_token_to_string = convert_token_to_string

        return tokenizer
