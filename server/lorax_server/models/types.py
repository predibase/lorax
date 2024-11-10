from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from lorax_server.adapters import AdapterBatchMetadata
from lorax_server.pb import generate_pb2
from lorax_server.pb.generate_pb2 import FinishReason
from lorax_server.utils.segments import find_segments
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
        processor,
        config,
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
class FlashEmbeddingClassificationBatch(ABC):
    request_ids: List[int]
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    position_ids: torch.Tensor

    cu_seqlens: torch.Tensor
    max_s: int
    size: int

    adapter_meta: AdapterBatchMetadata

    def __len__(self) -> int:
        return self.size

    @classmethod
    def from_pb(
        self,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        tokenizers: TokenizerManager,
        processor,
        config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "FlashEmbeddingClassificationBatch":
        batch_inputs = []
        max_truncation = 0
        for r in pb.requests:
            inputs = tokenizers.get_inputs(r, tokenizer)
            batch_inputs.append(inputs)
            max_truncation = max(max_truncation, r.truncate)

        if all(r.HasField("tokenized_inputs") and len(r.tokenized_inputs.ids) > 0 for r in pb.requests):
            batch_tokenized_inputs = [r.tokenized_inputs.ids[-max_truncation:] for r in pb.requests]
        else:
            batch_tokenized_inputs = tokenizer(batch_inputs, padding=True, truncation=True, max_length=max_truncation)[
                "input_ids"
            ]

        pad_to = max(len(x) for x in batch_tokenized_inputs)
        all_input_ids = []
        position_ids = []
        all_token_type_ids = []
        cu_seqlens = [0]
        adapter_indices_list = []

        max_s = 0
        cumulative_length = 0
        adapter_list = []

        for i, (r, tokenized_input) in enumerate(zip(pb.requests, batch_tokenized_inputs)):
            tokenized_input = tokenized_input[-r.truncate :]
            if len(tokenized_input) < pad_to:
                tokenized_input += [tokenizer.pad_token_id] * (pad_to - len(tokenized_input))
            all_input_ids.append(tokenized_input)
            all_token_type_ids.append([0] * len(tokenized_input))

            input_length = len(tokenized_input)
            max_s = max(max_s, input_length)
            cu_seqlens.append(cumulative_length + input_length)

            # Position ids
            request_position_ids = torch.arange(0, input_length, dtype=torch.int32)
            position_ids.append(request_position_ids)

            adapter_indices_list.append(torch.full((input_length,), r.adapter_index))
            adapter_list.append(r.adapter_index)

            cumulative_length += input_length

        if len(pb.requests) > 1:
            input_ids = np.concatenate(all_input_ids, dtype=np.int64)
            final_token_type_ids = np.concatenate(all_token_type_ids, dtype=np.int64)
            position_ids = torch.cat(position_ids)
            adapter_indices = torch.cat(adapter_indices_list)
        else:
            input_ids = all_input_ids[0]
            final_token_type_ids = all_token_type_ids[0]
            position_ids = position_ids[0]
            adapter_indices = adapter_indices_list[0]

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        final_token_type_ids = torch.tensor(final_token_type_ids, dtype=torch.int64, device=device)
        position_ids = position_ids.to(device)
        adapter_indices = adapter_indices.to(dtype=torch.int64, device=device)

        adapter_segments, adapter_segment_indices = find_segments(adapter_indices)
        adapter_segments = torch.tensor(adapter_segments, dtype=torch.int32, device=device)

        return FlashEmbeddingClassificationBatch(
            request_ids=[r.id for r in pb.requests],
            input_ids=input_ids,
            token_type_ids=final_token_type_ids,
            position_ids=position_ids,
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
            max_s=max_s,
            size=len(batch_tokenized_inputs),
            adapter_meta=AdapterBatchMetadata(
                adapter_indices=adapter_indices,
                adapter_list=adapter_list,
                adapter_set=set(adapter_list),
                adapter_segments=adapter_segments,
                segment_indices=adapter_segment_indices,
            ),
        )

    @classmethod
    def from_pb_embed(
        self,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        tokenizers: TokenizerManager,
        processor,
        config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "FlashEmbeddingClassificationBatch":
        return self.from_pb(pb, tokenizer, tokenizers, processor, config, dtype, device)

    @classmethod
    def to_pb_classify(self, batch, predicted_token_classes, confidence_scores) -> generate_pb2.ClassifyResponse:
        results = []
        for i, (pred, con) in enumerate(zip(predicted_token_classes, confidence_scores)):
            results.append(
                generate_pb2.ClassifyPredictionList(
                    request_id=batch.request_ids[i],
                    predictions=pred,
                    scores=con,
                )
            )

        pb_resp = generate_pb2.ClassifyResponse(classify_prediction_lists=results)
        return pb_resp

    @classmethod
    def to_pb_embed(self, batch, embeddings) -> generate_pb2.EmbedResponse:
        embeddings_proto = []
        for i, embedding in enumerate(embeddings):
            embeddings_proto.append(generate_pb2.Embedding(request_id=batch.request_ids[i], values=embedding))
        return generate_pb2.EmbedResponse(embeddings=embeddings_proto)
