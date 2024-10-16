import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.distributed
from loguru import logger
from opentelemetry import trace
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, PretrainedConfig, PreTrainedTokenizerBase

from lorax_server.adapters import AdapterBatchData, AdapterBatchMetadata
from lorax_server.models.model import Model
from lorax_server.models.types import (
    AlternativeTokens,
    Batch,
    GeneratedText,
    Generation,
    NextTokens,
    PrefillTokens,
)
from lorax_server.pb import generate_pb2
from lorax_server.utils import HeterogeneousNextTokenChooser, StoppingCriteria
from lorax_server.utils.adapter import BASE_MODEL_ADAPTER_ID, create_merged_weight_files
from lorax_server.utils.attention.utils import block_tables_to_ragged
from lorax_server.utils.dist import MEMORY_FRACTION, MEMORY_WIGGLE_ROOM, initialize_torch_distributed
from lorax_server.utils.graph import GraphCache
from lorax_server.utils.import_utils import get_cuda_free_memory
from lorax_server.utils.segments import SegmentConcatBuilder, find_segments
from lorax_server.utils.sources import HUB
from lorax_server.utils.sources.hub import weight_files
from lorax_server.utils.state import BLOCK_SIZE, FLASH_INFER, PREFIX_CACHING, get_speculative_tokens, warmup_mode
from lorax_server.utils.tokenizer import TokenizerManager
from lorax_server.utils.weights import Weights

ADAPTER_MEMORY_FRACTION = float(os.getenv("ADAPTER_MEMORY_FRACTION", "0.1"))

# Will be set in init
SLIDING_WINDOW: Optional[int] = None
SLIDING_WINDOW_BLOCKS: Optional[int] = None


tracer = trace.get_tracer(__name__)


@dataclass
class FlashCausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    # request id -> idx in list mapping
    requests_idx_mapping: Dict[int, int]

    # Decoder values
    input_ids: torch.Tensor
    position_ids: torch.Tensor

    # Spculative decoding values
    speculative_ids: Optional[torch.Tensor]

    # Flash Attention values

    # tensor of length b containing the cumulative sequence lengths of the sequences in the batch, only used in prefill
    cu_seqlen_prefill: Optional[torch.Tensor]

    # Paged Attention values

    # Set when creating the batch
    # CPU tensor of length b indicating the start of each sequence in slots
    start_slots: torch.Tensor
    # tensor of indices of the currently used slots, length = \sum_{i=0}^{b} s_i in prefill, length = b in decode
    slot_indices: torch.Tensor

    # list of length b of list of length s_i // block_size
    block_tables: List[List[int]]
    # tensor of size [b, max_seqlen // block_size] holding the paged attention block tables for all sequences
    block_tables_tensor: torch.Tensor
    # tensor of length \sum_{i=0}^{b} max_s_i  holding the paged attention slots for all sequences
    slots: torch.Tensor

    # size [b], containing the number of blocks that can be retrieved from the cache
    prefix_lens: List[int]
    prefix_lens_tensor: torch.Tensor

    max_seqlen: int

    # Prefill metadata tensors to efficiently compute logprobs
    prefill_head_indices: Optional[torch.Tensor]
    prefill_next_token_indices: Optional[torch.tensor]
    prefill_cu_outlens: Optional[List[int]]

    # Prefixes
    prefix_ids: List[List[int]]

    # All tokens
    all_input_ids: List[List[int]]
    all_input_ids_tensor: torch.Tensor

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    input_lengths_tensor: torch.Tensor
    prefix_offsets: List[Optional[int]]
    read_offsets: List[Optional[int]]

    # Generation helpers
    next_token_chooser: HeterogeneousNextTokenChooser
    stopping_criterias: List[StoppingCriteria]

    # Adapter metadata for each request
    adapter_meta: AdapterBatchMetadata

    # Number of blocks in this batch
    num_blocks: int
    # Maximum number of blocks
    max_blocks: int

    # Prefill cache indices is used to slice into the kv tensor before caching it into the paged attention buffers
    # as we only keep SLIDING_WINDOW values instead of the whole tensor
    prefill_cache_indices: Optional[torch.Tensor] = None

    def to_pb(self) -> generate_pb2.CachedBatch:
        return generate_pb2.CachedBatch(
            id=self.batch_id,
            request_ids=[r.id for r in self.requests],
            size=len(self),
            max_tokens=self.num_blocks * BLOCK_SIZE,
        )

    @classmethod
    def to_pb_embed(self, batch, embeddings) -> generate_pb2.EmbedResponse:
        embeddings_proto = []
        for i, embedding in enumerate(embeddings):
            embeddings_proto.append(generate_pb2.Embedding(request_id=batch.requests[i].id, values=embedding))
        return generate_pb2.EmbedResponse(embeddings=embeddings_proto)

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
        batch_tokenized_inputs=None,
    ) -> "FlashCausalLMBatch":
        global SLIDING_WINDOW
        global SLIDING_WINDOW_BLOCKS

        if batch_tokenized_inputs is None:
            batch_inputs = []
            max_truncation = 0
            for r in pb.requests:
                inputs = tokenizers.get_inputs(r, tokenizer)
                batch_inputs.append(inputs)
                max_truncation = max(max_truncation, r.truncate)

            if all(r.HasField("tokenized_inputs") and len(r.tokenized_inputs.ids) > 0 for r in pb.requests):
                batch_tokenized_inputs = [r.tokenized_inputs.ids[-max_truncation:] for r in pb.requests]
            else:
                batch_tokenized_inputs = tokenizer(batch_inputs, truncation=True, max_length=max_truncation)[
                    "input_ids"
                ]

        position_ids = []
        cu_seqlen_prefill = [0]
        start_slots = []
        slot_indices = []
        prefill_cache_indices = []

        input_lengths = []
        prefix_offsets = []
        read_offsets = []
        all_input_ids = []
        prefix_ids = []
        requests_idx_mapping = {}

        all_prefill_logprobs = True
        no_prefill_logprobs = True
        prefill_head_indices = []
        prefill_next_token_indices = []
        prefill_cu_outlens = [0]

        next_token_chooser_parameters = []
        stopping_criterias = []

        adapter_indices_list = []
        adapter_set = set()

        # Cumulative length
        cumulative_length = 0
        cumulative_slot_tokens = 0
        prefill_out_cumulative_length = 0

        num_blocks = 0
        max_seqlen = 0
        max_length = 0
        max_blocks = 0

        block_tables = []
        slots = []
        prefix_lens = []

        # Parse batch
        for i, (r, tokenized_input) in enumerate(zip(pb.requests, batch_tokenized_inputs)):
            # request id -> idx in list mapping
            requests_idx_mapping[r.id] = i

            tokenized_input = tokenized_input[-r.truncate :]

            orig_input_length = len(tokenized_input)
            if PREFIX_CACHING:
                prefix_len = r.prefix_len
                if prefix_len == orig_input_length:
                    assert prefix_len > 0
                    prefix_len -= 1
            else:
                prefix_len = 0

            prefix_ids.append(tokenized_input[:prefix_len])
            tokenized_input = tokenized_input[prefix_len:]

            input_length = len(tokenized_input)
            input_lengths.append(input_length)

            prefix_offsets.append(input_length - 5)
            read_offsets.append(input_length)

            all_input_ids.append(tokenized_input)

            # Position ids
            request_position_ids = torch.arange(prefix_len, orig_input_length, dtype=torch.int32)
            position_ids.append(request_position_ids)

            # Add cumulative lengths of all previous inputs
            cu_seqlen_prefill.append(cumulative_length + input_length)

            next_token_chooser_parameters.append(r.parameters)

            stopping_criteria = StoppingCriteria.from_pb(r.stopping_parameters, tokenizer)
            max_new_tokens = stopping_criteria.max_new_tokens
            stopping_criterias.append(stopping_criteria)

            adapter_indices_list.append(torch.full((input_length,), r.adapter_index))
            adapter_set.add(r.adapter_index)

            speculative_tokens = get_speculative_tokens()

            # Tokens that need to be mapped to blocks.
            # Remove one as the first token des not have a past
            block_tokens = orig_input_length + max_new_tokens - 1 + speculative_tokens

            # Tokens that need to be mapped to slots. We don't need slots for the
            # cached prefix (if present).
            slot_tokens = input_length + max_new_tokens - 1 + speculative_tokens

            # blocks and slots can be empty (for example in warmup)
            if not r.blocks:
                needed_blocks = math.ceil(block_tokens / BLOCK_SIZE)
                request_blocks = [b for b in range(num_blocks, num_blocks + needed_blocks)]
                request_slots = [s for b in request_blocks for s in range(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE)]
            else:
                request_blocks = r.blocks
                request_slots = r.slots[
                    prefix_len:  #: orig_input_length + max_new_tokens + speculative_length
                ]

            block_tables.append(request_blocks)
            slots.extend(request_slots)
            prefix_lens.append(prefix_len)
            num_blocks += len(request_blocks)
            start_slots.append(cumulative_slot_tokens)

            request_slot_indices = torch.arange(
                cumulative_slot_tokens,
                cumulative_slot_tokens + input_length,
                dtype=torch.int64,
            )
            slot_indices.append(request_slot_indices)

            # Create tensor to slice into the kv tensor in prefill
            if SLIDING_WINDOW is not None:
                request_prefill_cache_indices = torch.arange(
                    cumulative_length + max(0, input_length - SLIDING_WINDOW),
                    cumulative_length + input_length,
                    dtype=torch.int64,
                )
                prefill_cache_indices.append(request_prefill_cache_indices)

            all_prefill_logprobs = all_prefill_logprobs and r.prefill_logprobs
            no_prefill_logprobs = no_prefill_logprobs and not r.prefill_logprobs

            if r.prefill_logprobs:
                prefill_head_indices.append(request_position_ids + cumulative_length)
                prefill_next_token_indices.append(prefill_out_cumulative_length + input_length - 1)
                prefill_cu_outlens.append(prefill_out_cumulative_length + input_length)
                prefill_out_cumulative_length += input_length
            else:
                prefill_head_indices.append(torch.tensor([cumulative_length + input_length - 1], dtype=torch.int32))
                prefill_next_token_indices.append(prefill_out_cumulative_length)
                prefill_cu_outlens.append(prefill_out_cumulative_length + 1)
                prefill_out_cumulative_length += 1

            # Update
            cumulative_length += input_length
            cumulative_slot_tokens += slot_tokens
            max_seqlen = max(max_seqlen, input_length)
            max_blocks = max(max_blocks, len(request_blocks))
            max_length = max(max_length, input_length + max_new_tokens + speculative_tokens)

        adapter_indices = torch.cat(adapter_indices_list).to(dtype=torch.int64, device=device)

        # always use the base model tokenizer for the next token chooser until we revisit adding back support
        # for per-request tokenizers
        request_tokenizers = [tokenizer for _ in pb.requests]
        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters, request_tokenizers, dtype, device
        )
        start_slots = torch.tensor(start_slots, dtype=torch.int64)

        # Padded all_input_ids_tensor
        all_input_ids_tensor = np.zeros((len(all_input_ids), max_length), dtype=np.int64)
        for i, input_ids in enumerate(all_input_ids):
            all_input_ids_tensor[i, : len(input_ids)] = input_ids

        # Create tensors on device
        all_input_ids_tensor = torch.tensor(all_input_ids_tensor, dtype=torch.int64, device=device)

        if len(pb.requests) > 1:
            input_ids = np.concatenate(all_input_ids, dtype=np.int64)
            position_ids = torch.cat(position_ids)
            slot_indices = torch.cat(slot_indices)
            if SLIDING_WINDOW is not None:
                prefill_cache_indices = torch.cat(prefill_cache_indices)
        else:
            input_ids = all_input_ids[0]
            position_ids = position_ids[0]
            slot_indices = slot_indices[0]
            if SLIDING_WINDOW is not None:
                prefill_cache_indices = prefill_cache_indices[0]

        cu_seqlen_prefill = torch.tensor(cu_seqlen_prefill, device=device, dtype=torch.int32)

        position_ids = position_ids.to(device)
        slot_indices = slot_indices.to(device)
        if SLIDING_WINDOW is not None:
            prefill_cache_indices = prefill_cache_indices.to(device)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.int32, device=device)

        adapter_segments, adapter_segment_indices = find_segments(adapter_indices)
        adapter_segments = torch.tensor(adapter_segments, dtype=torch.int32, device=device)

        if all_prefill_logprobs:
            prefill_head_indices = None
            prefill_next_token_indices = cu_seqlen_prefill[1:] - 1
        elif no_prefill_logprobs:
            prefill_head_indices = cu_seqlen_prefill[1:] - 1
            prefill_next_token_indices = None
        else:
            prefill_head_indices = torch.tensor(torch.cat(prefill_head_indices), dtype=torch.int64, device=device)
            prefill_next_token_indices = torch.tensor(prefill_next_token_indices, dtype=torch.int64, device=device)

        slots = torch.tensor(slots, dtype=torch.int64, device=device)
        block_tables_tensor = torch.zeros((len(block_tables), max_blocks), dtype=torch.int32, device="cpu")
        for i, request_blocks in enumerate(block_tables):
            block_tables_tensor[i, : len(request_blocks)] = torch.tensor(request_blocks)
        block_tables_tensor = block_tables_tensor.to(device)
        prefix_lens_tensor = torch.tensor(prefix_lens, dtype=torch.int32, device=device)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            speculative_ids=None,
            cu_seqlen_prefill=cu_seqlen_prefill,
            start_slots=start_slots,
            slot_indices=slot_indices,
            block_tables=block_tables,
            block_tables_tensor=block_tables_tensor,
            slots=slots,
            prefix_lens=prefix_lens,
            prefix_lens_tensor=prefix_lens_tensor,
            max_seqlen=max_seqlen,
            prefill_head_indices=prefill_head_indices,
            prefill_next_token_indices=prefill_next_token_indices,
            prefill_cu_outlens=prefill_cu_outlens,
            input_lengths=input_lengths,
            input_lengths_tensor=input_lengths_tensor,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            prefix_ids=prefix_ids,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            num_blocks=num_blocks,
            max_blocks=max_blocks,
            adapter_meta=AdapterBatchMetadata(
                adapter_indices=adapter_indices,
                adapter_set=adapter_set,
                adapter_segments=adapter_segments,
                segment_indices=adapter_segment_indices,
            ),
            prefill_cache_indices=prefill_cache_indices if SLIDING_WINDOW is not None else None,
        )

    @classmethod
    def from_pb_embed(
        self,
        pb: generate_pb2.EmbedRequest,
        tokenizer: PreTrainedTokenizerBase,
        tokenizers: TokenizerManager,
        processor,
        config,
        dtype,
        device,
    ) -> "FlashCausalLMBatch":
        return self.from_pb(pb, tokenizer, tokenizers, None, None, dtype, device)

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]) -> "FlashCausalLMBatch":
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")
        # We assume that if len(requests) == len(self) then the requests are the same
        if len(request_ids) == len(self):
            return self

        device = self.input_ids.device

        # New values after filtering
        requests_idx_mapping = {}

        # Used to index into tensors
        indices = []

        # slots to keep after filtering
        slot_filtering_indices = torch.zeros(self.slots.shape[0], dtype=torch.bool, device=device)

        # Create on CPU to only move to GPU once instead of at every copy
        slot_indices = torch.empty(len(request_ids), dtype=torch.int64)
        max_seqlen = 0

        requests = []
        start_slots = []
        block_tables = []
        all_input_ids = []
        prefix_ids = []

        input_lengths = []
        prefix_lens = []
        prefix_offsets = []
        read_offsets = []

        stopping_criterias = []
        adapter_set = set()

        num_blocks = 0
        max_blocks = 0
        # Cumulative length
        cumulative_max_length = 0

        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            indices.append(idx)
            requests_idx_mapping[request_id] = i

            requests.append(self.requests[idx])

            # Get length
            request_input_length = self.input_lengths[idx]
            prefix_len = self.prefix_lens[idx]
            max_seqlen = max(max_seqlen, request_input_length)

            all_input_ids.append(self.all_input_ids[idx])
            prefix_ids.append(self.prefix_ids[idx])

            input_lengths.append(request_input_length)
            prefix_lens.append(prefix_len)
            prefix_offsets.append(self.prefix_offsets[idx])
            read_offsets.append(self.read_offsets[idx])

            stopping_criteria = self.stopping_criterias[idx]
            stopping_criterias.append(stopping_criteria)

            adapter_set.add(self.requests[idx].adapter_index)

            remaining_tokens = stopping_criteria.max_new_tokens - stopping_criteria.current_tokens

            request_block_table = self.block_tables[idx]
            num_blocks += len(request_block_table)
            block_tables.append(request_block_table)
            start_slots.append(cumulative_max_length)

            # Copy to tensor (CPU)
            slot_indices[i] = cumulative_max_length + request_input_length - 1

            # Set slice
            slot_filtering_indices[
                self.start_slots[idx] : self.start_slots[idx] + request_input_length + remaining_tokens - 1
            ] = True

            cumulative_max_length += request_input_length + remaining_tokens - 1

            max_blocks = max(max_blocks, len(request_block_table))

        # Index into tensors
        input_ids = self.input_ids[indices]
        position_ids = self.position_ids[indices]
        adapter_indices = self.adapter_meta.adapter_indices[indices]
        all_input_ids_tensor = self.all_input_ids_tensor[indices]
        block_tables_tensor = self.block_tables_tensor[indices]
        input_lengths_tensor = self.input_lengths_tensor[indices]
        slots = self.slots[slot_filtering_indices]
        prefix_lens_tensor = self.prefix_lens_tensor[indices]
        next_token_chooser = self.next_token_chooser.filter(indices)
        speculative_ids = self.speculative_ids[indices] if self.speculative_ids is not None else None

        start_slots = torch.tensor(start_slots, dtype=torch.int64)

        # Move to GPU now that we have the whole tensor
        slot_indices = slot_indices.to(device)

        adapter_segments, adapter_segment_indices = find_segments(adapter_indices)
        adapter_segments = torch.tensor(adapter_segments, dtype=torch.int32, device=device)

        return type(self)(
            batch_id=self.batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            speculative_ids=speculative_ids,
            cu_seqlen_prefill=None,
            start_slots=start_slots,
            slot_indices=slot_indices,
            block_tables=block_tables,
            block_tables_tensor=block_tables_tensor,
            slots=slots,
            max_seqlen=max_seqlen,
            prefill_head_indices=None,
            prefill_next_token_indices=None,
            prefill_cu_outlens=None,
            input_lengths=input_lengths,
            input_lengths_tensor=input_lengths_tensor,
            prefix_lens=prefix_lens,
            prefix_lens_tensor=prefix_lens_tensor,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            prefix_ids=prefix_ids,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            num_blocks=num_blocks,
            max_blocks=max_blocks,
            adapter_meta=AdapterBatchMetadata(
                adapter_indices=adapter_indices,
                adapter_set=adapter_set,
                adapter_segments=adapter_segments,
                segment_indices=adapter_segment_indices,
            ),
        )

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["FlashCausalLMBatch"]) -> "FlashCausalLMBatch":
        # Batch attributes
        requests = []
        requests_idx_mapping = {}

        num_blocks = 0
        total_batch_size = 0
        total_slots = 0
        max_blocks = 0
        max_length = 0
        max_seqlen = 0
        for b in batches:
            total_batch_size += len(b)
            total_slots += len(b.slots)
            num_blocks += b.num_blocks
            max_blocks = max(max_blocks, b.max_blocks)
            max_seqlen = max(max_seqlen, b.max_seqlen)

            speculative_length = b.speculative_ids.shape[1] if b.speculative_ids is not None else 0
            max_length = max(
                max_length,
                max(
                    input_length
                    + stopping_criteria.max_new_tokens
                    + speculative_length
                    - stopping_criteria.current_tokens
                    for input_length, stopping_criteria in zip(b.input_lengths, b.stopping_criterias)
                ),
            )

        input_ids = batches[0].input_ids.new_empty(total_batch_size)
        position_ids = batches[0].position_ids.new_empty(total_batch_size)
        slots = batches[0].slots.new_empty(total_slots)
        slot_indices = batches[0].slot_indices.new_empty(total_batch_size)
        input_lengths_tensor = batches[0].input_lengths_tensor.new_empty(total_batch_size)
        block_tables_tensor = batches[0].block_tables_tensor.new_zeros((total_batch_size, max_blocks))
        prefix_lens_tensor = batches[0].prefix_lens_tensor.new_empty(total_batch_size)
        all_input_ids_tensor = batches[0].all_input_ids_tensor.new_zeros((total_batch_size, max_length))

        total_indices_size = sum(b.adapter_meta.adapter_indices.shape[0] for b in batches)

        adapter_indices = batches[0].adapter_meta.adapter_indices.new_empty(total_indices_size)
        adapter_set = set()
        adapter_segment_builder = SegmentConcatBuilder()

        start_slots = []
        block_tables = []
        prefix_lens = []
        all_input_ids = []
        prefix_ids = []

        input_lengths = []
        prefix_offsets = []
        read_offsets = []

        next_token_chooser_parameters = []
        sequence_processors = []
        stopping_criterias = []

        # Cumulative length
        cumulative_batch_size = 0
        cumulative_slots = 0
        cumulative_adapter_indices_size = 0

        for i, batch in enumerate(batches):
            requests.extend(batch.requests)

            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                # We need to offset the mapping for each batch by the cumulative batch size
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + cumulative_batch_size

            start_index = cumulative_batch_size
            end_index = cumulative_batch_size + len(batch)
            slots_start_index = cumulative_slots
            slots_end_index = cumulative_slots + len(batch.slots)

            # Copy tensors (GPU)
            input_ids[start_index:end_index] = batch.input_ids
            position_ids[start_index:end_index] = batch.position_ids
            slot_indices[start_index:end_index] = batch.slot_indices + cumulative_slots
            input_lengths_tensor[start_index:end_index] = batch.input_lengths_tensor
            slots[slots_start_index:slots_end_index] = batch.slots

            # Copy over adapter indices
            adapter_start_index = cumulative_adapter_indices_size
            adapter_end_index = cumulative_adapter_indices_size + batch.adapter_meta.adapter_indices.shape[0]
            adapter_indices[adapter_start_index:adapter_end_index] = batch.adapter_meta.adapter_indices
            cumulative_adapter_indices_size = adapter_end_index
            adapter_set.update(batch.adapter_meta.adapter_set)

            # Update adapter segments
            adapter_segment_builder.concat(batch.adapter_meta.adapter_segments, batch.adapter_meta.segment_indices)

            all_input_ids_tensor[start_index:end_index, : batch.all_input_ids_tensor.shape[1]] = (
                batch.all_input_ids_tensor[:, :max_length]
            )

            block_tables_tensor[start_index:end_index, : batch.block_tables_tensor.shape[1]] = (
                batch.block_tables_tensor[:, :max_blocks]
            )

            prefix_lens_tensor[start_index:end_index] = batch.prefix_lens_tensor

            start_slots.append(batch.start_slots + cumulative_slots)

            block_tables.extend(batch.block_tables)
            prefix_lens.extend(batch.prefix_lens)
            all_input_ids.extend(batch.all_input_ids)
            prefix_ids.extend(batch.prefix_ids)

            input_lengths.extend(batch.input_lengths)
            prefix_offsets.extend(batch.prefix_offsets)
            read_offsets.extend(batch.read_offsets)

            next_token_chooser_parameters.extend([r.parameters for r in batch.requests])
            if batch.next_token_chooser.schema_processor is not None:
                sequence_processors.extend(batch.next_token_chooser.schema_processor.sequence_processors)
            else:
                # No sequence processors, so pad with Nones
                sequence_processors.extend([None for _ in batch.requests])
            stopping_criterias.extend(batch.stopping_criterias)

            # Update
            cumulative_batch_size += len(batch)
            cumulative_slots += len(batch.slots)

        start_slots = torch.concat(start_slots)

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters,
            tokenizers=[],
            dtype=batches[0].next_token_chooser.dtype,
            device=batches[0].next_token_chooser.device,
            sequence_processors=sequence_processors,
        )

        speculative_ids = (
            torch.cat([b.speculative_ids for b in batches], dim=0) if batches[0].speculative_ids is not None else None
        )

        adapter_segments, adapter_segment_indices = adapter_segment_builder.build()

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            speculative_ids=speculative_ids,
            cu_seqlen_prefill=None,
            start_slots=start_slots,
            slot_indices=slot_indices,
            block_tables=block_tables,
            block_tables_tensor=block_tables_tensor,
            prefix_lens=prefix_lens,
            prefix_lens_tensor=prefix_lens_tensor,
            slots=slots,
            max_seqlen=max_seqlen,
            prefill_head_indices=None,
            prefill_next_token_indices=None,
            prefill_cu_outlens=None,
            input_lengths=input_lengths,
            input_lengths_tensor=input_lengths_tensor,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            prefix_ids=prefix_ids,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            num_blocks=num_blocks,
            max_blocks=max_blocks,
            adapter_meta=AdapterBatchMetadata(
                adapter_indices=adapter_indices,
                adapter_set=adapter_set,
                adapter_segments=adapter_segments,
                segment_indices=adapter_segment_indices,
            ),
        )

    def __len__(self):
        return len(self.requests)


class FlashCausalLM(Model):
    def __init__(
        self,
        model_id: str,
        model_cls: Type[torch.nn.Module],
        dtype: torch.dtype,
        revision: Optional[str] = None,
        adapter_id: str = BASE_MODEL_ADAPTER_ID,
        adapter_source: str = HUB,
        tokenizer_cls: Type[PreTrainedTokenizerBase] = AutoTokenizer,
        config_cls: Type[PretrainedConfig] = AutoConfig,
        # Used for Santacoder override of config
        num_kv_heads: Optional[int] = None,
        # Deepseek V2 uses different QK and V dims.
        head_size: Optional[int] = None,
        quantize: Optional[str] = None,
        compile: bool = False,
        merge_adapter_weights: bool = False,
        embedding_dim: Optional[int] = None,
        trust_remote_code: bool = False,
        processor=None,
    ):
        global SLIDING_WINDOW
        global SLIDING_WINDOW_BLOCKS

        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashLlama is only available on GPU")

        tokenizer = tokenizer_cls.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        try:
            # Override the tokenizer's eos_token_id with the one from the generation_config
            # if it is a list or set. We need to do this by adding a new property as the tokenizer
            # does not officially support multiple eos_token_ids.
            generation_config = GenerationConfig.from_pretrained(
                model_id, revision=revision, trust_remote_code=trust_remote_code
            )

            if isinstance(generation_config.eos_token_id, (list, set)):
                tokenizer.eos_token_ids = set(generation_config.eos_token_id)
        except Exception:
            pass

        config = config_cls.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors", embedding_dim=embedding_dim)
        merged_weight_filenames = None
        if merge_adapter_weights:
            if len(adapter_id) > 0:
                logger.info(f"Merging adapter weights from adapter_id {adapter_id} into model weights.")
                # Need to pass the adapter source here
                merged_weight_filenames = create_merged_weight_files(
                    adapter_id, model_id, model_weight_filenames=filenames, adapter_source=adapter_source
                )
                self.dynamic_adapter_loading_enabled = False
                self.adapter_id = adapter_id
            else:
                raise ValueError("Cannot merge adapter weights without an adapter_id")

        weights = Weights(
            filenames,
            device,
            dtype,
            process_group=self.process_group,
            merged_weight_filenames=merged_weight_filenames,
        )
        weights._set_config(model_id, config)

        self._supports_embeddings = embedding_dim is not None
        if not (weights.has_tensor("lm_head.weight") or weights.has_tensor("language_model.lm_head.weight")) and not self._supports_embeddings:
            raise ValueError(
                "Model does not have lm head so it is presumed to be for embeddings."
                "No embedding_dim was provided so we cannot load the model."
                "Please pass in an embedding_dim to the model."
            )

        prefix = ""
        model = model_cls(prefix, config, weights)

        torch.distributed.barrier(group=self.process_group)

        # VLM models define the config we care about in their text_config
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            config = text_config

        sliding_window = None
        if getattr(config, "sliding_window", None) is not None:
            sliding_window = config.sliding_window
        else:
            config.sliding_window = None

        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads // self.process_group.size()
        # Validation is done in the model itself
        if num_kv_heads is None:
            num_kv_heads = getattr(config, "num_key_value_heads", None)
            # GPT-2 workaround
            if num_kv_heads is None:
                num_kv_heads = getattr(config, "n_head", None)
        if num_kv_heads is None:
            raise ValueError("Cannot get the number of key/value heads")
        self.num_kv_heads = num_kv_heads // self.process_group.size() if num_kv_heads > 1 else num_kv_heads
        assert self.num_kv_heads > 0

        if head_size is None:
            # Some models use GQA and different sizes for o_proj
            # and q_proj, that allows for that.
            if hasattr(config, "head_dim"):
                self.head_size = config.head_dim
            else:
                self.head_size = config.hidden_size // config.num_attention_heads
        else:
            self.head_size = head_size

        super(FlashCausalLM, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            sliding_window=sliding_window,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            dynamic_adapter_loading_enabled=not merge_adapter_weights,
            trust_remote_code=trust_remote_code,
            processor=processor,
        )

        if sliding_window is not None:
            # Set context windows
            SLIDING_WINDOW = sliding_window
            SLIDING_WINDOW_BLOCKS = math.ceil(sliding_window / BLOCK_SIZE)

        self.compile = compile
        self.model_graph_wrapper: GraphCache = None
        self.kv_cache = []

        self.prefill_state = None
        self.prefill_with_paged_kv_state = None
        self.decode_state = None
        if FLASH_INFER:
            from lorax_server.utils.flashinfer_attention import (
                create_decode_state,
                create_prefill_state,
                create_prefill_with_paged_kv_state,
            )

            self.prefill_state = create_prefill_state(device=device)
            self.prefill_with_paged_kv_state = create_prefill_with_paged_kv_state(device=device)
            self.decode_state = create_decode_state(
                device=device,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
            )

    @property
    def block_size(self) -> int:
        return BLOCK_SIZE

    @property
    def sliding_window_blocks(self) -> Optional[int]:
        return SLIDING_WINDOW_BLOCKS

    @property
    def batch_type(self) -> Type[FlashCausalLMBatch]:
        return FlashCausalLMBatch

    def max_past(self) -> int:
        return getattr(self.model, "max_past", None)

    def init_kv_cache(
        self,
        num_blocks: int,
        num_layers: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.kv_cache = []
        torch.cuda.empty_cache()

        element_size = torch.tensor([], dtype=dtype).element_size()
        x = BLOCK_SIZE // element_size

        if FLASH_INFER:
            self.kv_cache = [
                (
                    torch.empty(
                        (num_blocks, BLOCK_SIZE, num_heads, head_size),
                        dtype=dtype,
                        device=device,
                    ),
                    torch.empty(
                        (num_blocks, BLOCK_SIZE, num_heads, head_size),
                        dtype=dtype,
                        device=device,
                    ),
                )
                for _ in range(num_layers)
            ]
        else:
            self.kv_cache = [
                (
                    torch.empty(
                        (num_blocks, num_heads, head_size // x, BLOCK_SIZE, x),
                        dtype=dtype,
                        device=device,
                    ),
                    torch.empty(
                        (num_blocks, num_heads, head_size, BLOCK_SIZE),
                        dtype=dtype,
                        device=device,
                    ),
                )
                for _ in range(num_layers)
            ]

    def adapter_memory_size(self) -> int:
        total_gpu_memory = torch.cuda.get_device_properties(self.device).total_memory
        return ADAPTER_MEMORY_FRACTION * total_gpu_memory

    def warmup(self, batch: FlashCausalLMBatch, max_new_tokens: int, embedding_model: bool = False):
        # The warmup batch is the biggest batch we could ever receive
        max_total_tokens = batch.max_seqlen + max_new_tokens + get_speculative_tokens()

        torch.cuda.empty_cache()
        try:
            self.init_kv_cache(
                batch.num_blocks,
                self.num_layers,
                self.num_kv_heads,
                self.head_size,
                self.dtype,
                self.device,
            )

            if not embedding_model:
                with warmup_mode():
                    logger.info("Warming up to max_new_tokens: {}", max_new_tokens)
                    with tqdm(total=max_new_tokens, desc="Warmup to max_total_tokens") as pbar:
                        for _ in range(max_new_tokens):
                            cur_seqlen = batch.max_seqlen
                            _, batch = self.generate_token(batch, is_warmup=True)
                            new_seqlen = batch.max_seqlen
                            pbar.update(new_seqlen - cur_seqlen)
                            if new_seqlen >= max_total_tokens - get_speculative_tokens():
                                break
                    logger.info("Finished generating warmup tokens")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
                raise RuntimeError(
                    f"Not enough memory to handle {len(batch.input_ids)} prefill tokens. "
                    f"You need to decrease `--max-batch-prefill-tokens`"
                ) from e
            else:
                raise

        torch.cuda.synchronize(self.device)

        graph_cache_memory = 0
        if self.compile:
            if self.world_size > 1:
                raise ValueError("Cannot enable `--compile` when sharding across multiple GPUs")

            # This will be recalculated in the graph step
            self.decode_state = None

            # Estimate the memory overhead from CUDA graphs so we can subtract it from the kv cache.
            # Needs to be estimated here rather than fully initialized as the graph cache relies on the
            # cache manager being set.
            self.model_graph_wrapper = GraphCache(
                self.model,
                self.device,
                self.kv_cache,
                self.adapter_layers,
                self.default_traced_adapter_layers,
                self._forward_context,
                max_total_tokens,
                self.num_heads,
                self.num_kv_heads,
                self.sliding_window_blocks,
            )
            graph_cache_memory = self.model_graph_wrapper.get_estimated_cache_memory()
            logger.info("Estimated graph cache memory: {} MB", graph_cache_memory / 1024 / 1024)
            torch.cuda.synchronize(self.device)

        # Inspired by the original implementation in [vllm](https://github.com/vllm-project/vllm)
        # Calculate the number of blocks that can be allocated with the free memory
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        cache_block_size = BLOCK_SIZE * self.num_kv_heads * self.head_size
        total_cache_size = self.num_layers * cache_block_size * 2 * dtype_size

        free_memory = get_cuda_free_memory(self.device, MEMORY_FRACTION - ADAPTER_MEMORY_FRACTION)
        free_memory -= graph_cache_memory
        logger.info("Memory remaining for kv cache: {} MB", free_memory / 1024 / 1024)

        batch_num_blocks = batch.num_blocks if batch is not None else 0
        num_blocks = (
            # Leave 5% for some wiggle room
            int((free_memory * MEMORY_WIGGLE_ROOM) // total_cache_size)
            # Add batch.num_blocks as we allocated it above, so it is included in the peak memory.
            + batch_num_blocks
        )

        del batch

        self.init_kv_cache(
            num_blocks,
            self.num_layers,
            self.num_kv_heads,
            self.head_size,
            self.dtype,
            self.device,
        )

        torch.cuda.synchronize(self.device)

        if self.model_graph_wrapper is not None:
            # Warmup the graph cache. Needs to be done after setting cache manager as
            # tracing will use the static kv cache tensors
            self.model_graph_wrapper.kv_cache = self.kv_cache
            self.model_graph_wrapper.warmup()
            torch.cuda.synchronize(self.device)

        return int(num_blocks * BLOCK_SIZE)

    def decode(self, generated_ids: Union[torch.Tensor, List[int]]) -> str:
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _forward_context(
        self,
        *,
        block_tables: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        input_lengths: List[int],
        input_lengths_tensor: torch.Tensor,
        prefix_lens: List[int],
        prefix_lens_tensor: torch.Tensor,
        state: Optional[Any] = None,
    ) -> ContextManager:
        if not FLASH_INFER:
            return nullcontext()

        from lorax_server.utils.flashinfer_attention import (
            use_decode_state,
            use_prefill_with_paged_kv_state,
        )

        # has_prefix_lens = any(prefix_len > 0 for prefix_len in prefix_lens)
        if cu_seqlen_prefill is not None:
            return use_prefill_with_paged_kv_state(
                state=(state if state is not None else self.prefill_with_paged_kv_state),
                # block_tables=block_tables_to_ragged(
                #     block_tables=block_tables,
                #     input_lengths=input_lengths,
                #     prefix_lens=prefix_lens,
                # ),
                block_tables=block_tables,
                cu_seqlens=cu_seqlen_prefill,
                input_lengths=input_lengths_tensor,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                page_size=BLOCK_SIZE,
            )
        else:
            assert input_lengths_tensor is not None
            return use_decode_state(
                state=state if state is not None else self.decode_state,
                input_lengths=input_lengths_tensor,
                block_tables=block_tables,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                page_size=BLOCK_SIZE,
            )

    def forward(self, batch: FlashCausalLMBatch, adapter_data: AdapterBatchData) -> Tuple[torch.Tensor, torch.Tensor]:
        prefill = batch.cu_seqlen_prefill is not None
        model = self.model
        use_graph = False
        if (
            self.model_graph_wrapper is not None
            and not prefill
        ):
            if self.model_graph_wrapper.can_use_graph(batch, adapter_data):
                use_graph = True
                model = self.model_graph_wrapper
            else:
                logger.info("CUDA graphs enabled but batch is incompatible, falling back to eager mode.")

        input_ids = batch.input_ids
        position_ids = batch.position_ids
        block_tables = batch.block_tables_tensor
        slots = batch.slots[batch.slot_indices]
        input_lengths = batch.input_lengths_tensor
        prefix_lens_tensor = batch.prefix_lens_tensor
        max_s = batch.max_seqlen

        if batch.speculative_ids is not None:
            speculative_ids = batch.speculative_ids

            B, speculative_length = speculative_ids.shape
            new_length = speculative_length + 1
            new_input_ids = torch.cat([input_ids.unsqueeze(-1), speculative_ids], dim=1).reshape(-1)
            arange = torch.arange(new_length, device=position_ids.device).unsqueeze(0)
            arange_int = arange.to(dtype=torch.int32)
            new_position_ids = (position_ids.unsqueeze(-1).expand(B, new_length) + arange).view(-1)
            slots = (slots.unsqueeze(-1).expand(B, new_length) + arange_int).view(-1)
            input_lengths = (input_lengths.unsqueeze(-1).expand(B, new_length) + arange_int).view(-1)
            prefix_lens_tensor = (batch.prefix_lens_tensor.unsqueeze(-1).expand(B, new_length)).reshape(-1)

            block_tables = block_tables.unsqueeze(1).expand(B, new_length, -1).reshape(B * new_length, -1).contiguous()
            max_s = max_s + speculative_length

            input_ids = new_input_ids
            position_ids = new_position_ids

        # Model Forward
        if not use_graph:
            # eager mode
            input_lengths = input_lengths + prefix_lens_tensor
            if FLASH_INFER:
                block_tables = block_tables_to_ragged(
                    block_tables=block_tables,
                    input_lengths=batch.input_lengths,
                    prefix_lens=batch.prefix_lens,
                )

            with self._forward_context(
                block_tables=block_tables,
                cu_seqlen_prefill=batch.cu_seqlen_prefill,
                input_lengths=batch.input_lengths,
                input_lengths_tensor=input_lengths,
                prefix_lens=batch.prefix_lens,
                prefix_lens_tensor=prefix_lens_tensor,
            ):
                out = model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cu_seqlen_prefill=batch.cu_seqlen_prefill,
                    kv_cache=self.kv_cache,
                    block_tables=block_tables,
                    slots=slots,
                    input_lengths=input_lengths,
                    max_s=max_s,
                    adapter_data=adapter_data,
                    prefill_cache_indices=batch.prefill_cache_indices,
                    lm_head_indices=batch.prefill_head_indices,
                )
        else:
            # CUDA graph mode
            out = model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seqlen_prefill=batch.cu_seqlen_prefill,
                kv_cache=self.kv_cache,
                block_tables=block_tables,
                slots=slots,
                input_lengths=input_lengths,
                prefix_lens=batch.prefix_lens,
                prefix_lens_tensor=prefix_lens_tensor,
                max_s=max_s,
                adapter_data=adapter_data,
                prefill_cache_indices=batch.prefill_cache_indices,
                lm_head_indices=batch.prefill_head_indices,
            )

        if batch.prefill_cache_indices is not None:
            batch.prefill_cache_indices = None

        return out

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: FlashCausalLMBatch, is_warmup: bool = False
    ) -> Tuple[List[Generation], Optional[FlashCausalLMBatch]]:
        prefill = batch.cu_seqlen_prefill is not None
        prefill_logprobs = batch.prefill_next_token_indices is not None
        return_alternatives = any(req.parameters.return_k_alternatives > 0 for req in batch.requests)

        # Update adapter indices for speculative tokens (if present)
        adapter_meta = batch.adapter_meta
        if batch.speculative_ids is not None:
            B, speculative_length = batch.speculative_ids.shape
            new_length = speculative_length + 1
            adapter_indices = adapter_meta.adapter_indices.unsqueeze(-1).expand(B, new_length).reshape(-1)
            adapter_segments = adapter_meta.adapter_segments * new_length
            adapter_meta = AdapterBatchMetadata(
                adapter_indices=adapter_indices,
                adapter_set=adapter_meta.adapter_set,
                adapter_segments=adapter_segments,
                segment_indices=adapter_meta.segment_indices,
            )

        # Assign pointers to adapter weights
        # TODO(travis): don't update this if indices haven't changed
        adapter_data = AdapterBatchData.from_meta(
            adapter_meta, self.layer_to_adapter_weights, prefill, batch.prefill_head_indices
        )

        out, speculative_logits = self.forward(batch, adapter_data)

        if prefill:
            next_token_logits = out[batch.prefill_next_token_indices] if prefill_logprobs else out
            if speculative_logits is not None:
                speculative_logits = (
                    speculative_logits[batch.prefill_next_token_indices] if prefill_logprobs else speculative_logits
                )
        else:
            next_token_logits = out

        speculative_tokens = get_speculative_tokens()
        (
            next_input_ids,
            next_token_logprobs,
            accepted_ids,
            speculative_ids,
        ) = batch.next_token_chooser(
            batch.all_input_ids_tensor[:, : batch.max_seqlen],
            next_token_logits,
            speculative_tokens,
            batch.speculative_ids,
            speculative_logits,
        )

        if return_alternatives:
            alternative_token_logprobs, alternative_token_ids = torch.sort(
                torch.log_softmax(next_token_logits, -1), dim=-1, stable=True, descending=True
            )

        if prefill:
            if len(batch) > 1 and prefill_logprobs:
                # We create the prefill_tokens_indices tensor that will be used to gather prefill logprobs
                # When batch == 1, we will just use the batch.input_ids values directly
                prefill_tokens_indices = batch.input_ids.new_zeros(len(out))

            next_position_ids = batch.position_ids.new_empty(len(batch))
            batch.slot_indices = batch.slot_indices[batch.cu_seqlen_prefill[1:] - 1]
            # We do not need cu_seqlen_prefill anymore
            batch.cu_seqlen_prefill = None

            next_adapter_indices = batch.adapter_meta.adapter_indices.new_empty(len(batch))
        else:
            prefill_logprobs = None
            next_position_ids = batch.position_ids
            next_adapter_indices = batch.adapter_meta.adapter_indices

        # Cumulative length
        cumulative_length = 0

        # Results
        generations: List[Generation] = []

        # During warmup, do not allow early stopping
        stopped = not is_warmup

        # Zipped iterator
        iterator = zip(
            batch.input_lengths,
            batch.all_input_ids,
            accepted_ids,
        )

        # We do two for loops as the first one can run completely asynchronously from the GPU while for the second
        # one, we need to first do a GPU <-> CPU sync
        # It is faster if we delay this sync for the maximum amount of time

        # For each member of the batch
        idx = 0
        for i, (
            input_length,
            all_input_ids,
            num_accepted_ids,
        ) in enumerate(iterator):
            # Indexing metadata
            start_index = cumulative_length
            end_index = cumulative_length + input_length

            if prefill:
                # Indexing metadata
                out_start_index = batch.prefill_cu_outlens[i]
                out_end_index = batch.prefill_cu_outlens[i + 1]
                out_length = out_end_index - out_start_index

                # Initialize position_ids
                # In decode, we do not need this as we can just increment position ids
                next_position_ids[i] = batch.position_ids[end_index - 1]

                # Initialize adapter indices
                # In decode, we only have one token per row in the batch, so grab last index
                next_adapter_indices[i] = batch.adapter_meta.adapter_indices[end_index - 1]

                # Used to gather prefill logprobs
                # Copy batch.input_ids to prefill_token_indices
                if prefill_logprobs:
                    if len(batch) > 1:
                        prefill_tokens_indices[out_start_index : out_end_index - 1] = batch.input_ids[
                            start_index + 1 : start_index + out_length
                        ]
                    else:
                        # Set prefill_tokens_indices to the correct slice
                        prefill_tokens_indices = batch.input_ids[start_index + 1 : start_index + out_length]

            batch.all_input_ids_tensor[i, input_length] = next_input_ids[i]

            for j in range(num_accepted_ids):
                batch.all_input_ids_tensor[i, input_length + j] = next_input_ids[idx]
                idx += 1

            cumulative_length += input_length

        # Set values in batch
        batch.input_ids = next_input_ids[accepted_ids.cumsum(dim=-1) - 1]
        batch.position_ids = next_position_ids + accepted_ids
        batch.adapter_meta.adapter_indices = next_adapter_indices
        batch.speculative_ids = speculative_ids
        batch.input_lengths_tensor += accepted_ids
        batch.slot_indices += accepted_ids

        if prefill:
            # adjust segment lengths to account for all request lengths being 1 during decoding
            adapter_segments, _ = find_segments(batch.adapter_meta.adapter_indices)
            batch.adapter_meta.adapter_segments = torch.tensor(
                adapter_segments,
                dtype=torch.int32,
                device=batch.adapter_meta.adapter_segments.device,
            )

        if prefill and prefill_logprobs:
            # Get prefill logprobs
            prefill_logprobs_tensor = torch.log_softmax(out, -1)
            prefill_logprobs = torch.gather(prefill_logprobs_tensor, 1, prefill_tokens_indices.view(-1, 1))
            # GPU <-> CPU sync
            prefill_logprobs = prefill_logprobs.view(-1).tolist()

        # GPU <-> CPU sync
        next_token_logprobs = next_token_logprobs.tolist()
        next_token_ids = next_input_ids.tolist()

        if return_alternatives:
            alternative_token_logprobs = alternative_token_logprobs.tolist()
            alternative_token_ids = alternative_token_ids.tolist()

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.prefix_offsets,
            batch.read_offsets,
            batch.stopping_criterias,
            batch.all_input_ids,
            batch.prefix_ids,
            batch.next_token_chooser.do_sample,
            batch.next_token_chooser.seeds,
            accepted_ids,
        )

        # For each member of the batch
        idx = 0
        for i, (
            request,
            input_length,
            prefix_offset,
            read_offset,
            stopping_criteria,
            all_input_ids,
            prefix_ids,
            do_sample,
            seed,
            num_accepted_ids,
        ) in enumerate(iterator):
            all_alternative_tokens = [] if request.parameters.return_k_alternatives > 0 else None
            next_token_texts = []
            left = 0
            current_stopped = False
            for j in range(num_accepted_ids):
                token_idx = idx + j

                # Generated token
                next_token_id = next_token_ids[token_idx]
                all_input_ids.append(next_token_id)
                next_token_text, prefix_offset, read_offset = self.decode_token(
                    all_input_ids,
                    prefix_offset,
                    read_offset,
                )
                next_token_texts.append(next_token_text)

                if request.parameters.return_k_alternatives > 0:
                    # Limit the number of alternatives to the vocabulary size
                    num_alternatives = min(
                        request.parameters.return_k_alternatives,
                        len(alternative_token_ids[token_idx]),
                    )

                    # Select top-k logprobs
                    request_alternative_token_ids = alternative_token_ids[token_idx][:num_alternatives]
                    request_alternative_token_logprobs = alternative_token_logprobs[token_idx][:num_alternatives]

                    # Decode tokens
                    request_alternative_token_texts = []
                    for alternative_token_id in request_alternative_token_ids:
                        all_input_ids.append(alternative_token_id)
                        alternative_token_text, _, _ = self.decode_token(
                            all_input_ids,
                            prefix_offset,
                            read_offset,
                        )
                        request_alternative_token_texts.append(alternative_token_text)
                        all_input_ids.pop()
                    alternative_tokens = AlternativeTokens(
                        request_alternative_token_ids,
                        request_alternative_token_logprobs,
                        request_alternative_token_texts,
                    )
                    all_alternative_tokens.append(alternative_tokens)

                stop, reason = stopping_criteria(
                    next_token_id,
                    next_token_text,
                )

                if stop:
                    left = num_accepted_ids - j - 1
                    current_stopped = True
                    break
                else:
                    current_stopped = False
            stopped = stopped and current_stopped

            accepted_token_ids = next_token_ids[idx : idx + num_accepted_ids - left]
            accepted_token_logprobs = next_token_logprobs[idx : idx + num_accepted_ids - left]
            idx += num_accepted_ids

            # Shard generations
            # All generations will be appended in the rust sharded client
            if i % self.world_size == self.rank:
                if stop:
                    # Decode generated tokens
                    output_text = self.decode(all_input_ids[-stopping_criteria.current_tokens :])
                    generated_text = GeneratedText(
                        output_text,
                        stopping_criteria.current_tokens,
                        reason,
                        seed if do_sample else None,
                    )
                else:
                    generated_text = None

                # Prefill
                if prefill and request.prefill_logprobs:
                    out_start_index = batch.prefill_cu_outlens[i]
                    out_end_index = batch.prefill_cu_outlens[i + 1]

                    # Remove generated token to only have prefill and add nan for first prompt token
                    request_prefill_logprobs = ([float("nan")] * (len(prefix_ids) + 1)) + prefill_logprobs[
                        out_start_index : out_end_index - 1
                    ]
                    prefill_token_ids = all_input_ids[:-1]
                    prefill_texts = self.tokenizer.batch_decode(
                        prefix_ids + prefill_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )
                    prefill_tokens = PrefillTokens(prefill_token_ids, request_prefill_logprobs, prefill_texts)
                else:
                    prefill_tokens = None

                generation = Generation(
                    request.id,
                    prefill_tokens,
                    len(all_input_ids[:-1]) if prefill else 0,
                    NextTokens(
                        accepted_token_ids,
                        accepted_token_logprobs,
                        next_token_texts,
                        [tid in self.all_special_ids for tid in accepted_token_ids],
                        all_alternative_tokens,
                    ),
                    generated_text,
                )

                generations.append(generation)

            # advance the FSM for each accepted token (as we may have more than one from speculative decoding)
            for next_token_id in accepted_token_ids:
                batch.next_token_chooser.next_state(i, next_token_id)

            # Update values
            batch.input_lengths[i] = input_length + num_accepted_ids.item()
            if batch.input_lengths[i] > batch.max_seqlen:
                batch.max_seqlen = batch.input_lengths[i]
            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.all_input_ids[i] = all_input_ids

        if stopped:
            # No need to return a batch if we know that all requests stopped
            return generations, None

        batch.prefill_cu_outlens = None
        batch.prefill_head_indices = None
        batch.prefill_next_token_indices = None
        batch.max_seqlen = batch.max_seqlen + 1

        return generations, batch
