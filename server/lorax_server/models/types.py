from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import PreTrainedTokenizerBase

from lorax_server.pb import generate_pb2
from lorax_server.pb.generate_pb2 import FinishReason
from lorax_server.utils.tokenizer import TokenizerManager


class Batch(ABC):
    @abstractmethod
    def to_pb(self) -> generate_pb2.CachedBatch:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        tokenizers: TokenizerManager,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "Batch":
        raise NotImplementedError

    @abstractmethod
    def filter(self, request_ids: List[int]) -> "Batch":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concatenate(cls, batches: List["Batch"]) -> "Batch":
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


@dataclass
class GeneratedText:
    text: str
    generated_tokens: int
    finish_reason: FinishReason
    seed: Optional[int]

    def to_pb(self) -> generate_pb2.GeneratedText:
        return generate_pb2.GeneratedText(
            text=self.text,
            generated_tokens=self.generated_tokens,
            finish_reason=self.finish_reason,
            seed=self.seed,
        )


@dataclass
class PrefillTokens:
    token_ids: List[int]
    logprobs: List[float]
    texts: List[str]

    def to_pb(self) -> generate_pb2.PrefillTokens:
        return generate_pb2.PrefillTokens(ids=self.token_ids, logprobs=self.logprobs, texts=self.texts)

    def __len__(self):
        return len(self.token_ids)


@dataclass
class AlternativeTokens:
    token_ids: List[int]
    logprobs: List[float]
    texts: List[str]

    def to_pb(self) -> generate_pb2.AlternativeTokens:
        return generate_pb2.AlternativeTokens(ids=self.token_ids, logprobs=self.logprobs, texts=self.texts)

    def __len__(self):
        return len(self.token_ids)


@dataclass
class NextTokens:
    token_ids: List[int]
    logprobs: List[float]
    texts: List[str]
    is_special: List[bool]
    alternative_tokens: Optional[List[AlternativeTokens]]

    def to_pb(self) -> generate_pb2.PrefillTokens:
        return generate_pb2.NextTokens(
            ids=self.token_ids,
            logprobs=self.logprobs,
            texts=self.texts,
            is_special=self.is_special,
            alternative_tokens=(
                [alt_tokens.to_pb() for alt_tokens in self.alternative_tokens]
                if self.alternative_tokens is not None
                else None
            ),
        )

    def __len__(self):
        return len(self.token_ids)


@dataclass
class Generation:
    request_id: int
    prefill_tokens: Optional[PrefillTokens]
    prefill_tokens_length: int
    next_tokens: NextTokens
    generated_text: Optional[GeneratedText]

    def to_pb(self) -> generate_pb2.Generation:
        return generate_pb2.Generation(
            request_id=self.request_id,
            prefill_tokens=self.prefill_tokens.to_pb() if self.prefill_tokens is not None else None,
            prefill_tokens_length=self.prefill_tokens_length,
            next_tokens=self.next_tokens.to_pb(),
            generated_text=self.generated_text.to_pb() if self.generated_text is not None else None,
        )


@dataclass
class FlashEmbeddingBatch(ABC):
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    position_ids: torch.Tensor

    cu_seqlens: torch.Tensor
    max_s: int
    size: int

    def __len__(self):
        return self.size

    def from_pb(self, *args, **kwargs):
        return None
