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

    def __len__(self) -> int:
        return self.size

    @classmethod
    def from_pb(
        self, 
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        tokenizers: TokenizerManager,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "FlashEmbeddingBatch":
        batch_inputs = []
        max_truncation = 0
        for r in pb.requests:
            inputs = tokenizers.get_inputs(r, tokenizer)
            batch_inputs.append(inputs)
            max_truncation = max(max_truncation, r.truncate)

        batch_tokenized_inputs = tokenizer(
            batch_inputs, 
            return_token_type_ids=True, 
            truncation=True, 
            max_length=max_truncation,
        )

        max_s = 0
        position_ids = []

        cumulative_length = 0
        cu_seqlens = [0]

        for i, (r, tokenized_input) in enumerate(zip(pb.requests, batch_tokenized_inputs)):
            tokenized_input = tokenized_input[-r.truncate :]

            input_length = len(tokenized_input)
            max_s = max(max_s, input_length)
            cu_seqlens.append(cumulative_length + input_length)

            # Position ids
            request_position_ids = torch.arange(0, input_length, dtype=torch.int32)
            position_ids.append(request_position_ids)

            cumulative_length += input_length

        return FlashEmbeddingBatch(
            input_ids=torch.tensor(batch_tokenized_inputs["input_ids"], dtype=torch.int32, device=device),
            token_type_ids=torch.tensor(batch_tokenized_inputs["token_type_ids"], dtype=torch.int32, device=device),
            position_ids=torch.tensor(position_ids, dtype=torch.int32, device=device),
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
            max_s=max_s,
            size=len(batch_inputs),
        )
