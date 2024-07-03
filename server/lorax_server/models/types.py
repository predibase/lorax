from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
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
    request_ids: List[int]
    strings: List[str]
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
        batch_strings = []
        max_truncation = 0
        for r in pb.requests:
            inputs = tokenizers.get_inputs(r, tokenizer)
            batch_strings.append(inputs)
            max_truncation = max(max_truncation, r.truncate)

        batch_inputs = tokenizer(
            batch_strings, 
            return_token_type_ids=True, 
            truncation=True, 
            max_length=max_truncation,
        )

        batch_tokenized_inputs = batch_inputs["input_ids"]
        batch_token_type_ids = batch_inputs["token_type_ids"]

        all_input_ids = []
        position_ids = []
        all_token_type_ids = []
        cu_seqlens = [0]

        max_s = 0
        cumulative_length = 0
        
        for (r, tokenized_input, token_type_ids) in zip(pb.requests, batch_tokenized_inputs, batch_token_type_ids):
            tokenized_input = tokenized_input[-r.truncate :]
            token_type_ids = token_type_ids[-r.truncate :]
            all_input_ids.append(tokenized_input)
            all_token_type_ids.append(token_type_ids)

            input_length = len(tokenized_input)
            max_s = max(max_s, input_length)
            cu_seqlens.append(cumulative_length + input_length)

            # Position ids
            request_position_ids = torch.arange(0, input_length, dtype=torch.int32)
            position_ids.append(request_position_ids)

            cumulative_length += input_length
        
        if len(pb.requests) > 1:
            input_ids = np.concatenate(all_input_ids, dtype=np.int64)
            final_token_type_ids = np.concatenate(all_token_type_ids, dtype=np.int64)
            position_ids = torch.cat(position_ids)
        else:
            input_ids = all_input_ids[0]
            final_token_type_ids = all_token_type_ids[0]
            position_ids = position_ids[0]
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        final_token_type_ids = torch.tensor(final_token_type_ids, dtype=torch.int64, device=device)
        position_ids = position_ids.to(device)

        return FlashEmbeddingBatch(
            request_ids=[r.id for r in pb.requests],
            strings=batch_strings,
            input_ids=input_ids,
            token_type_ids=final_token_type_ids,
            position_ids=position_ids,
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
            max_s=max_s,
            size=len(batch_inputs),
        )
    
    def to_pb(self, predicted_token_class, confidence_scores, tokenizer):
        res =  _format_ner_output(predicted_token_class, confidence_scores, self.input_ids, tokenizer)
        return res


def _format_ner_output(predicted_token_class, scores, input_ids, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    
    ner_results = []
    current_entity = None
    
    for i, (token, token_class, score) in enumerate(zip(tokens, predicted_token_class, scores)):  # Skip [CLS] and [SEP]
        if token_class != 'O':
            if token_class.startswith('B-') or (current_entity and token_class != current_entity['entity']):
                if current_entity:
                    ner_results.append(current_entity)
                current_entity = {
                    'entity': token_class,
                    'score': score,
                    'index': i,
                    'word': token,
                    'start': len(tokenizer.decode(input_ids[:i+1])),
                    'end': len(tokenizer.decode(input_ids[:i+2]))
                }
            elif token_class.startswith('I-') and current_entity:
                current_entity['word'] += token.replace('##', '')
                current_entity['end'] = len(tokenizer.decode(input_ids[:i+2]))
        else:
            if current_entity:
                ner_results.append(current_entity)
                current_entity = None
    
    if current_entity:
        ner_results.append(current_entity)
    
    return ner_results