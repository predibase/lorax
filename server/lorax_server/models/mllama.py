from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from opentelemetry import trace
from PIL import Image
from transformers import (
    PreTrainedTokenizerBase,
)

from lorax_server.models.vlm_causal_lm import VlmCausalLM, VlmCausalLMBatch
from lorax_server.pb import generate_pb2
from lorax_server.utils.attention.common import Seqlen
from lorax_server.utils.attention.utils import block_tables_to_ragged
from lorax_server.utils.state import PREFIX_CACHING
from lorax_server.utils.tokenizer import TokenizerManager

tracer = trace.get_tracer(__name__)


@dataclass
class MllamaCausalLMBatch(VlmCausalLMBatch):
    image_indices: List[int] = 42
    aspect_ratio_ids: Optional[torch.Tensor] = None
    aspect_ratio_mask: Optional[torch.Tensor] = None
    cross_attention_states: Optional[torch.Tensor] = None

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches):
        batch = super().concatenate(batches)
        batch.pixel_values = None
        batch.pixel_attention_mask = None

        offset = 0
        image_indices = []
        attention_states = []
        for b in batches:
            if b.cross_attention_states is not None:
                attention_states.append(b.cross_attention_states)
            image_indices.extend([i + offset for i in b.image_indices])
            offset += len(b.image_indices)
        if len(attention_states) > 0:
            assert len(image_indices) > 0
            batch.cross_attention_states = torch.cat(attention_states, dim=0)
            batch.image_indices = image_indices
        else:
            batch.cross_attention_states = None
            batch.image_indices = []
        return batch

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]):
        assert self.image_indices is not None
        batch = super().filter(request_ids)
        assert self.image_indices is not None
        indices = []
        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            indices.append(idx)

        offset = 0
        new_image_indices = []
        prev_i = None
        for i in self.image_indices:
            if i in indices:
                new_image_indices.append(offset)
                if i != prev_i:
                    offset += 1
                prev_i = i

        batch.image_indices = new_image_indices
        if len(new_image_indices) > 0:
            assert max(new_image_indices) < self.cross_attention_states.shape[0]
            assert offset <= self.cross_attention_states.shape[0]
            batch.cross_attention_states = self.cross_attention_states[new_image_indices]
        else:
            batch.cross_attention_states = None
        return batch

    @classmethod
    def batch_tokenized_inputs(cls, requests: Iterable[generate_pb2.Request], tokenizer, processor, config):
        image_inputs = []
        texts = []
        image_indices = []
        batch_tokenized_inputs = []
        for i, r in enumerate(requests):
            # Each input is encoded into a list, where each element of this input list is either a string or a URL
            curr_text = ""
            for chunk in r.tokenized_inputs.input_chunks:
                chunk_type = chunk.WhichOneof("chunk")
                if chunk_type == "text":
                    curr_text += chunk.text
                elif chunk_type == "image":
                    image = Image.open(BytesIO(chunk.image.data))
                    # TODO unsure about BOS
                    curr_text += "<|image|>"
                    image_input = processor.image_processor(image, return_tensors="pt")
                    image_inputs.append(image_input)
                    image_indices.append(i)
                else:
                    raise RuntimeError(f"Invalid chunk type {chunk_type}")
            texts.append(curr_text)

            input_ids = tokenizer(
                curr_text,
                truncation=True,
                max_length=r.truncate,
                add_special_tokens=True,
            )["input_ids"]
            batch_tokenized_inputs.append(input_ids)
        if image_inputs:
            image_input = image_inputs[0]
            new_image_inputs = {
                "pixel_values": torch.cat([img["pixel_values"] for img in image_inputs], dim=0),
            }
            if "aspect_ratio_ids" in image_input:
                new_image_inputs["aspect_ratio_ids"] = torch.cat(
                    [img["aspect_ratio_ids"] for img in image_inputs], dim=0
                )
            if "aspect_ratio_mask" in image_input:
                new_image_inputs["aspect_ratio_mask"] = torch.cat(
                    [img["aspect_ratio_mask"] for img in image_inputs], dim=0
                )
            image_inputs = new_image_inputs
            image_inputs["image_indices"] = image_indices
        else:
            image_inputs = None

        if image_inputs is not None:
            assert len(image_indices) == image_inputs["pixel_values"].shape[0]

        return batch_tokenized_inputs, image_inputs

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        tokenizers: TokenizerManager,
        processor,
        config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "VlmCausalLMBatch":
        batch_tokenized_inputs, image_inputs = cls.batch_tokenized_inputs(pb.requests, tokenizer, processor, config)
        batch = super(VlmCausalLMBatch, cls).from_pb(
            pb, tokenizer, tokenizers, processor, config, dtype, device, batch_tokenized_inputs=batch_tokenized_inputs
        )

        # XXX: <|image|> token is actually out of bounds and bugs out the logit processors.
        batch.all_input_ids_tensor = batch.all_input_ids_tensor.clamp(max=config.text_config.vocab_size - 1)
        if isinstance(batch.input_ids, list):
            if len(batch) > 1:
                input_ids = np.concatenate(batch.input_ids, dtype=np.int64)
            else:
                input_ids = batch.input_ids[0]
            batch.input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        batch.input_ids = batch.input_ids.clamp(max=config.text_config.vocab_size - 1)

        if image_inputs is not None:
            batch.pixel_values = image_inputs["pixel_values"].to(device=device, dtype=dtype)
            batch.aspect_ratio_ids = image_inputs["aspect_ratio_ids"].to(device=device)
            batch.aspect_ratio_mask = image_inputs["aspect_ratio_mask"].to(device=device)
            batch.image_indices = image_inputs["image_indices"]
        else:
            batch.pixel_values = None
            batch.aspect_ratio_ids = None
            batch.aspect_ratio_mask = None
            batch.image_indices = []
        assert batch.image_indices is not None
        return batch


class MllamaCausalLM(VlmCausalLM):
    def forward(
        self,
        batch: VlmCausalLMBatch,
        adapter_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Model Forward
        if batch.speculative_ids is not None:
            input_ids = batch.input_ids
            position_ids = batch.position_ids
            cu_seqlen_prefill = batch.cu_seqlen_prefill
            kv_cache = self.kv_cache
            block_tables = batch.block_tables_tensor
            slots = batch.slots[batch.slot_indices]
            input_lengths = batch.input_lengths_tensor
            max_s = batch.max_current_length
            lm_head_indices = batch.prefill_head_indices

            speculative_ids = batch.speculative_ids

            B, speculative_length = speculative_ids.shape
            new_length = speculative_length + 1
            new_input_ids = torch.cat([input_ids.unsqueeze(-1), speculative_ids], dim=1).reshape(-1)
            arange = torch.arange(new_length, device=position_ids.device).unsqueeze(0)
            arange_int = arange.to(dtype=torch.int32)
            new_position_ids = (position_ids.unsqueeze(-1).expand(B, new_length) + arange).view(-1)
            slots = (slots.unsqueeze(-1).expand(B, new_length) + arange_int).view(-1)
            input_lengths = (input_lengths.unsqueeze(-1).expand(B, new_length) + arange_int).view(-1)
            cache_lengths_tensor = (batch.cache_lengths_tensor.unsqueeze(-1).expand(B, new_length)).reshape(-1)

            # Add Copy the block tables for all members
            block_tables = block_tables.unsqueeze(1).expand(B, new_length, -1).reshape(B * new_length, -1).contiguous()
            max_s = max_s + speculative_length

            input_ids = new_input_ids
            position_ids = new_position_ids
        else:
            input_ids = batch.input_ids
            position_ids = batch.position_ids
            cu_seqlen_prefill = batch.cu_seqlen_prefill
            kv_cache = self.kv_cache
            block_tables = batch.block_tables_tensor
            slots = batch.slots[batch.slot_indices]
            input_lengths = batch.input_lengths_tensor
            cache_lengths_tensor = batch.cache_lengths_tensor
            max_s = batch.max_current_length
            lm_head_indices = batch.prefill_head_indices

        if cu_seqlen_prefill is None and self.max_past() is not None:
            # In decode, not prefill, we're actually overwriting the KV-cache
            # in a circular buffer mode.
            # This makes sure the max_s for the decode pass is correct.
            max_s = min(self.max_past(), max_s)

        seqlen = Seqlen(
            input_lengths=input_lengths,
            cache_lengths=cache_lengths_tensor,
            cu_seqlen_q=None,
            max_q=batch.max_input_length,
            max_k=batch.max_current_length,
        )

        # TODO: cuda graph
        input_lengths = input_lengths + cache_lengths_tensor
        if PREFIX_CACHING:
            block_tables = block_tables_to_ragged(
                block_tables=block_tables,
                input_lengths=batch.input_lengths,
                cache_lengths=batch.cache_lengths,
            )
        with self._forward_context(
            block_tables=block_tables,
            cu_seqlen_prefill=cu_seqlen_prefill,
            input_lengths=batch.input_lengths,
            input_lengths_tensor=input_lengths,
            cache_lengths=batch.cache_lengths,
            cache_lengths_tensor=cache_lengths_tensor,
        ):
            # TODO(travis): is this needed?
            # max_k = (input_lengths + cache_lengths_tensor).max().item()
            if batch.pixel_values is not None:
                cross_attention_states = self.model.vision_forward(
                    pixel_values=batch.pixel_values,
                    aspect_ratio_ids=batch.aspect_ratio_ids,
                    aspect_ratio_mask=batch.aspect_ratio_mask,
                )
                batch.cross_attention_states = cross_attention_states

            cross_attention_states = batch.cross_attention_states

            logits, speculative_logits = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seqlen_prefill=cu_seqlen_prefill,
                kv_cache=kv_cache,
                block_tables=block_tables,
                slots=slots,
                seqlen=seqlen,
                max_s=max_s,
                prefill_cache_indices=batch.prefill_cache_indices,
                lm_head_indices=lm_head_indices,
                cross_attention_states=cross_attention_states,
                adapter_data=adapter_data,
                image_indices=batch.image_indices[:],
            )
            if batch.prefill_cache_indices is not None:
                batch.prefill_cache_indices = None
            if batch.pixel_values is not None:
                batch.pixel_values = None
            return logits, speculative_logits
