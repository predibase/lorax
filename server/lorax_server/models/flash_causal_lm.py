import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.distributed
import torch.profiler
from loguru import logger
from opentelemetry import trace
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, PretrainedConfig, PreTrainedTokenizerBase

from lorax_server.adapters import AdapterBatchData, AdapterBatchMetadata
from lorax_server.models.metadata_kernels import (
    block_tables_to_padded,
    block_tables_to_ragged,
    copy_next_input_ids_inplace,
    has_triton,
    prepare_position_slot_ids,
    slots_filtering,
)
from lorax_server.models.model import Model
from lorax_server.models.types import (
    Batch,
    GeneratedText,
    Generation,
    NextTokens,
)
from lorax_server.pb import generate_pb2
from lorax_server.utils import HeterogeneousNextTokenChooser, StoppingCriteria
from lorax_server.utils.adapter import BASE_MODEL_ADAPTER_ID, create_merged_weight_files
from lorax_server.utils.attention.common import Seqlen
from lorax_server.utils.dist import MEMORY_FRACTION, MEMORY_WIGGLE_ROOM, initialize_torch_distributed
from lorax_server.utils.graph import GraphCache
from lorax_server.utils.import_utils import get_cuda_free_memory
from lorax_server.utils.punica import LORAX_PUNICA_TRITON_DISABLED, PunicaWrapper
from lorax_server.utils.segments import SegmentConcatBuilder, find_segments
from lorax_server.utils.sources import HUB
from lorax_server.utils.sources.hub import weight_files
from lorax_server.utils.state import (
    BLOCK_SIZE,
    FLASH_INFER,
    get_max_prefill_tokens,
    get_speculative_tokens,
    get_supports_chunking,
    warmup_mode,
)
from lorax_server.utils.tokenizer import TokenizerManager
from lorax_server.utils.torch_utils import is_fp8, is_fp8_kv, is_fp8_supported
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
    # Can be a list for easy filtering
    # If `input_ids` is a list, it needs to be materialized to a tensor first
    input_ids: Union[torch.Tensor, List[List[int]]]
    # Will be set by `generate_token` and reset after each prefill forward before staying set in decode
    position_ids: Optional[torch.Tensor]

    # Spculative decoding values
    speculative_ids: Optional[torch.Tensor]

    # tensor of indices of the currently used slots, length = \sum_{i=0}^{b} s_i in prefill, length = b in decode
    # Will be set by `generate_token` and reset after each prefill forward before staying set in decode
    slot_indices: Optional[torch.Tensor]

    # list of length b of list of length s_i // block_size
    block_tables: List[List[int]]
    # tensor of size [b, max_seqlen // block_size] holding the paged attention block tables for all sequences
    block_tables_tensor: torch.Tensor
    # tensor of length \sum_{i=0}^{b} max_s_i  holding the paged attention slots for all sequences
    # Will be set by `generate_token` and reset after each prefill forward before staying set in decode
    slots: Optional[torch.Tensor]

    # list of length b + 1  containing the cumulative sequence slot lengths of the sequences in the batch
    # used for filtering
    cu_slots: torch.Tensor

    max_input_length: int
    max_current_length: int

    # Whether this batch contains at least one request that is prefilling
    prefilling: bool
    # Whether each request is prefilling
    prefilling_mask: List[bool]
    prefilling_mask_tensor: Optional[torch.Tensor]

    # Prefill metadata tensors to efficiently compute logprobs
    # tensor of length b+1 containing the cumulative sequence lengths of the sequences in the batch, only used in prefill
    cu_seqlen_prefill: Optional[torch.Tensor]
    # Prefill cache indices is used to slice into the kv tensor before caching it into the paged attention buffers
    # as we only keep SLIDING_WINDOW values instead of the whole tensor
    prefill_cache_indices: Optional[torch.Tensor]
    # Will be set by `generate_token` and reset after each prefill forward
    prefill_head_indices: Optional[torch.Tensor]
    # Will be set by `generate_token` and reset after each prefill forward
    prefill_next_token_indices: Optional[torch.tensor]
    # Will be set by `generate_token` and reset after each prefill forward
    prefill_cu_outlens: Optional[List[int]]
    # Will be set by `generate_token` and reset after each prefill forward
    prefill_logprob_tokens: List[Optional[NextTokens]]

    # All tokens
    all_input_ids: List[List[int]]
    all_input_ids_tensor: torch.Tensor

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    # size [b], containing the number of blocks that can be retrieved from the cache
    cache_lengths: List[int]
    prompt_lengths: List[int]
    # Will be set by `generate_token` and reset after each prefill forward before staying set in decode
    input_lengths_tensor: Optional[torch.Tensor]
    cache_lengths_tensor: Optional[torch.Tensor]
    prompt_lengths_tensor: torch.Tensor

    prefix_offsets: List[Optional[int]]
    read_offsets: List[Optional[int]]

    # Generation helpers
    next_token_chooser: HeterogeneousNextTokenChooser
    stopping_criterias: List[StoppingCriteria]

    # Adapter metadata for each request
    # Will be set by `generate_token` and reset after each prefill forward before staying set in decode
    adapter_meta: AdapterBatchMetadata

    # Number of blocks in this batch
    num_blocks: int
    # Maximum number of blocks
    max_blocks: int

    def to_pb(self) -> generate_pb2.CachedBatch:
        return generate_pb2.CachedBatch(
            id=self.batch_id,
            request_ids=[r.id for r in self.requests],
            size=len(self),
            max_tokens=self.num_blocks * BLOCK_SIZE,
            current_tokens=(
                sum([len(i) for i in self.input_ids]) if isinstance(self.input_ids, list) else len(self.input_ids)
            ),
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

        speculative_tokens = get_speculative_tokens()

        cache_lengths = []
        input_lengths = []
        prompt_lengths = []
        prefix_offsets = []
        read_offsets = []
        all_input_ids = []
        all_postfix_ids = []
        requests_idx_mapping = {}
        slots = []
        cu_slots = [0]

        next_token_chooser_parameters = []
        stopping_criterias = []

        num_blocks = 0
        max_input_length = 0
        max_current_length = 0
        max_length = 0
        max_blocks = 0

        cu_blocks = [0]
        block_tables = []
        block_tables_ragged = []

        # Parse batch
        for i, (r, tokenized_input) in enumerate(zip(pb.requests, batch_tokenized_inputs)):
            # request id -> idx in list mapping
            requests_idx_mapping[r.id] = i

            tokenized_input = tokenized_input[-r.truncate :]

            prompt_length = len(tokenized_input)
            prompt_lengths.append(prompt_length)

            cache_length = r.cache_len
            assert cache_length <= prompt_length, f"Prefix {cache_length} vs input {prompt_length}"
            if cache_length == prompt_length:
                assert False, "unreachable"

            # `chunk_len` is an optional field in the protobuf
            # It is only set if the model support chunking
            if r.HasField("chunk_len"):
                input_length = r.chunk_len

                if cache_length + input_length < prompt_length:
                    # FIXME: speculate is not supported for context chunking at the moment
                    assert speculative_tokens == 0
                    assert get_supports_chunking()
                    assert input_length > 0

                postfix_ids = tokenized_input[cache_length : cache_length + input_length]
                assert len(postfix_ids) == input_length, "Rust and Python tokenizers are not aligned"
            else:
                # Use all the remaining ids
                postfix_ids = tokenized_input[cache_length:]
                input_length = len(postfix_ids)

            input_lengths.append(input_length)

            prefix_offsets.append(prompt_length - 5)
            read_offsets.append(prompt_length)

            all_postfix_ids.append(postfix_ids)
            all_input_ids.append(tokenized_input)

            next_token_chooser_parameters.append(r.parameters)

            stopping_criteria = StoppingCriteria.from_pb(r.stopping_parameters, tokenizer)
            max_new_tokens = stopping_criteria.max_new_tokens
            stopping_criterias.append(stopping_criteria)

            # adapter_indices_list.append(torch.full((input_length,), r.adapter_index))
            # adapter_set.add(r.adapter_index)

            # Tokens that need to be mapped to blocks.
            # Remove one as the first token des not have a past
            block_tokens = prompt_length + max_new_tokens - 1 + speculative_tokens

            # blocks and slots can be empty (for example in warmup)
            if not r.blocks:
                needed_blocks = math.ceil(block_tokens / BLOCK_SIZE)
                request_blocks = [b for b in range(num_blocks, num_blocks + needed_blocks)]
                request_slots = [s for b in request_blocks for s in range(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE)]
            else:
                request_blocks = r.blocks
                request_slots = r.slots

            block_tables.append(request_blocks)
            block_tables_ragged.extend(request_blocks)
            cu_blocks.append(len(block_tables_ragged))

            slots.extend(request_slots)
            cu_slots.append(len(slots))

            cache_lengths.append(cache_length)
            num_blocks += len(request_blocks)

            # Update
            max_blocks = max(max_blocks, len(request_blocks))
            max_input_length = max(max_input_length, input_length)
            max_current_length = max(max_current_length, cache_length + input_length)
            max_length = max(
                max_length,
                prompt_length + max_new_tokens + speculative_tokens,
            )

        # always use the base model tokenizer for the next token chooser until we revisit adding back support
        # for per-request tokenizers
        request_tokenizers = [tokenizer for _ in pb.requests]
        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters, request_tokenizers, dtype, device
        )

        # Padded all_input_ids_tensor
        all_input_ids_tensor = np.zeros((len(all_input_ids), max_length), dtype=np.int64)
        for i, input_ids in enumerate(all_input_ids):
            all_input_ids_tensor[i, : len(input_ids)] = input_ids

        # Create tensors on device
        all_input_ids_tensor = torch.tensor(all_input_ids_tensor, dtype=torch.int64, device=device)

        block_tables_ragged = torch.tensor(block_tables_ragged, device=device, dtype=torch.int32)
        cu_blocks = torch.tensor(cu_blocks, device=device, dtype=torch.int64)
        block_tables_tensor = torch.empty(
            (len(block_tables), max_blocks),
            device=device,
            dtype=torch.int32,
        )

        # If the device supports Triton, we can use a fused kernel
        if has_triton():
            block_tables_to_padded(max_blocks, cu_blocks, block_tables_tensor, block_tables_ragged)
        else:
            for i, request_blocks in enumerate(block_tables):
                block_tables_tensor[i, : len(request_blocks)] = torch.tensor(request_blocks)

        prompt_lengths_tensor = torch.tensor(prompt_lengths, dtype=torch.int32, device=device)

        slots = torch.tensor(slots, dtype=torch.int64, device=device)
        cu_slots = torch.tensor(cu_slots, dtype=torch.int64)

        prefilling_mask = [True] * len(pb.requests)
        prefilling_mask_tensor = torch.tensor(prefilling_mask, dtype=torch.bool, device=device)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=all_postfix_ids,
            block_tables=block_tables,
            block_tables_tensor=block_tables_tensor,
            cache_lengths=cache_lengths,
            max_input_length=max_input_length,
            max_current_length=max_current_length,
            prefilling=True,
            prefilling_mask=prefilling_mask,
            prefilling_mask_tensor=prefilling_mask_tensor,
            prefill_logprob_tokens=[None] * len(pb.requests),
            input_lengths=input_lengths,
            prompt_lengths=prompt_lengths,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            num_blocks=num_blocks,
            max_blocks=max_blocks,
            speculative_ids=None,
            prompt_lengths_tensor=prompt_lengths_tensor,
            # These values will be set by `FlashCausalLMBatch.prepare_for_prefill`
            position_ids=None,
            cu_seqlen_prefill=None,
            prefill_cache_indices=None,
            slot_indices=None,
            slots=slots,
            cu_slots=cu_slots,
            prefill_head_indices=None,
            prefill_next_token_indices=None,
            prefill_cu_outlens=None,
            cache_lengths_tensor=None,
            input_lengths_tensor=None,
            adapter_meta=None,
        )

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]) -> "FlashCausalLMBatch":
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")

        device = self.block_tables_tensor.device

        # New values after filtering
        requests_idx_mapping = {}

        # Used to index into tensors
        indices = []

        # slots to keep after filtering
        if not has_triton():
            # slots to keep after filtering
            slot_filtering_indices = torch.zeros(self.slots.shape[0], dtype=torch.bool, device=device)

        # Create on CPU to only move to GPU once instead of at every copy
        slot_indices = torch.empty(len(request_ids), dtype=torch.int64)
        max_input_length = 0
        max_current_length = 0

        requests = []
        block_tables = []
        all_input_ids = []
        input_ids = []

        prompt_lengths = []
        input_lengths = []
        cache_lengths = []
        prefix_offsets = []
        read_offsets = []
        cu_slots = [0]

        prefilling_mask = []
        prefill_logprob_tokens = []

        stopping_criterias = []
        adapter_list = []

        num_blocks = 0
        max_blocks = 0
        max_slots = 0
        cumulative_slot_tokens = 0

        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            indices.append(idx)
            requests_idx_mapping[request_id] = i

            requests.append(self.requests[idx])

            # Prefilling
            request_prefilling = self.prefilling_mask[idx]
            prefilling_mask.append(request_prefilling)

            # Get length
            request_input_length = self.input_lengths[idx]
            request_cache_length = self.cache_lengths[idx]
            max_input_length = max(max_input_length, request_input_length)
            max_current_length = max(max_current_length, request_cache_length + request_input_length)

            all_input_ids.append(self.all_input_ids[idx])

            prompt_lengths.append(self.prompt_lengths[idx])
            input_lengths.append(request_input_length)
            cache_lengths.append(request_cache_length)
            prefix_offsets.append(self.prefix_offsets[idx])
            read_offsets.append(self.read_offsets[idx])

            stopping_criteria = self.stopping_criterias[idx]
            stopping_criterias.append(stopping_criteria)

            prefill_logprob_tokens.append(self.prefill_logprob_tokens[idx])

            adapter_list.append(self.requests[idx].adapter_index)

            request_block_table = self.block_tables[idx]
            num_blocks += len(request_block_table)
            block_tables.append(request_block_table)

            start_slot = self.cu_slots[idx]
            end_slot = self.cu_slots[idx + 1]
            slot_length = end_slot - start_slot

            if not has_triton():
                # Set slice
                slot_filtering_indices[start_slot:end_slot] = True

            cu_slots.append(cumulative_slot_tokens + slot_length)

            # Input ids if the request was part of a prefilling batch
            # If the batch was decoding we can index into the tensor directly later
            if self.prefilling:
                input_ids.append(self.input_ids[idx])
            else:
                # Copy to tensor (CPU)
                slot_indices[i] = cumulative_slot_tokens + request_cache_length

            cumulative_slot_tokens += slot_length
            max_blocks = max(max_blocks, len(request_block_table))
            max_slots = max(max_slots, slot_length)

        all_input_ids_tensor = self.all_input_ids_tensor[indices]

        block_tables_tensor = self.block_tables_tensor[indices]
        next_token_chooser = self.next_token_chooser.filter(indices)
        speculative_ids = self.speculative_ids[indices] if self.speculative_ids is not None else None
        prompt_lengths_tensor = self.prompt_lengths_tensor[indices]
        cu_slots = torch.tensor(cu_slots, dtype=torch.int64)

        if not has_triton():
            slots = self.slots[slot_filtering_indices]
        else:
            slots = self.slots.new_empty(cumulative_slot_tokens)
            gpu_cu_slots = cu_slots.to(device)
            slots_indexing_start = self.cu_slots.to(device)[indices]
            slots_filtering(max_slots, self.slots, slots, gpu_cu_slots, slots_indexing_start)

        if self.prefilling:
            # These values will be set by `FlashCausalLMBatch.prepare_for_prefill`
            position_ids = None
            slot_indices = None
            cache_lengths_tensor = None
            input_lengths_tensor = None
            adapter_meta = None
            prefilling_mask_tensor = self.prefilling_mask_tensor[indices]
        else:
            # Index into tensors
            input_ids = self.input_ids[indices]
            position_ids = self.position_ids[indices]
            adapter_indices = self.adapter_meta.adapter_indices[indices]
            input_lengths_tensor = self.input_lengths_tensor[indices]
            cache_lengths_tensor = self.cache_lengths_tensor[indices]
            prefilling_mask_tensor = None

            # Move to GPU now that we have the whole tensor
            slot_indices = slot_indices.to(device)

            adapter_segments, adapter_segment_indices = find_segments(adapter_indices)
            adapter_segments = torch.tensor(adapter_segments, dtype=torch.int32, device=device)
            adapter_meta = AdapterBatchMetadata(
                adapter_indices=adapter_indices,
                adapter_list=adapter_list,
                adapter_set=set(adapter_list),
                adapter_segments=adapter_segments,
                segment_indices=adapter_segment_indices,
            )

        return type(self)(
            batch_id=self.batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            speculative_ids=speculative_ids,
            cu_seqlen_prefill=None,
            prefill_cache_indices=None,
            slot_indices=slot_indices,
            block_tables=block_tables,
            block_tables_tensor=block_tables_tensor,
            slots=slots,
            cu_slots=cu_slots,
            max_input_length=max_input_length,
            max_current_length=max_current_length,
            prefilling=self.prefilling,
            prefilling_mask=prefilling_mask,
            prefilling_mask_tensor=prefilling_mask_tensor,
            prefill_head_indices=None,
            prefill_next_token_indices=None,
            prefill_cu_outlens=None,
            prefill_logprob_tokens=prefill_logprob_tokens,
            prompt_lengths=prompt_lengths,
            prompt_lengths_tensor=prompt_lengths_tensor,
            input_lengths=input_lengths,
            input_lengths_tensor=input_lengths_tensor,
            cache_lengths=cache_lengths,
            cache_lengths_tensor=cache_lengths_tensor,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            num_blocks=num_blocks,
            max_blocks=max_blocks,
            adapter_meta=adapter_meta,
        )

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["FlashCausalLMBatch"]) -> "FlashCausalLMBatch":
        # Batch attributes
        requests = []
        requests_idx_mapping = {}

        prefilling = False
        num_blocks = 0
        total_batch_size = 0
        total_slots = 0
        max_blocks = 0
        max_length = 0
        max_input_length = 0
        max_current_length = 0
        for b in batches:
            total_batch_size += len(b)
            max_blocks = max(max_blocks, b.max_blocks)
            # If `b` is prefilling and was just filtered, `b.slots` is None
            # `total_slots` is not used if any of the batches is prefilling
            total_slots += len(b.slots) if not b.prefilling else 0
            num_blocks += b.num_blocks
            speculative_length = b.speculative_ids.shape[1] if b.speculative_ids is not None else 0
            max_input_length = max(max_input_length, b.max_input_length)
            max_current_length = max(max_current_length, b.max_current_length)
            max_length = max(
                max_length,
                max(
                    prompt_length + stopping_criteria.max_new_tokens + speculative_length
                    for prompt_length, stopping_criteria in zip(b.prompt_lengths, b.stopping_criterias)
                ),
            )
            prefilling = prefilling or b.prefilling

        slots = batches[0].slots.new_empty(total_slots)
        cu_slots = torch.zeros(total_batch_size + 1, dtype=torch.int64)
        if prefilling:
            input_ids = []
            # These values will be set by `FlashCausalLMBatch.prepare_for_prefill`
            position_ids = None
            slot_indices = None
            cache_lengths_tensor = None
            input_lengths_tensor = None
            prefilling_mask_tensor = batches[0].prefilling_mask_tensor.new_empty(total_batch_size)
            adapter_meta = None
            adapter_segment_builder = None
        else:
            input_ids = batches[0].input_ids.new_empty(total_batch_size)
            position_ids = batches[0].position_ids.new_empty(total_batch_size)
            slot_indices = batches[0].slot_indices.new_empty(total_batch_size)
            input_lengths_tensor = batches[0].input_lengths_tensor.new_empty(total_batch_size)
            cache_lengths_tensor = batches[0].cache_lengths_tensor.new_empty(total_batch_size)
            prefilling_mask_tensor = None
            total_indices_size = sum(b.adapter_meta.adapter_indices.shape[0] for b in batches)
            adapter_indices = batches[0].adapter_meta.adapter_indices.new_empty(total_indices_size)
            adapter_segment_builder = SegmentConcatBuilder()
            adapter_list = []
            adapter_set = set()

        prompt_lengths_tensor = batches[0].prompt_lengths_tensor.new_empty(total_batch_size)
        block_tables_tensor = batches[0].block_tables_tensor.new_zeros((total_batch_size, max_blocks))
        all_input_ids_tensor = batches[0].all_input_ids_tensor.new_zeros((total_batch_size, max_length))

        block_tables = []
        cache_lengths = []
        all_input_ids = []

        prompt_lengths = []
        input_lengths = []
        prefix_offsets = []
        read_offsets = []

        prefill_logprob_tokens = []

        next_token_chooser_parameters = []
        sequence_processors = []
        stopping_criterias = []
        prefilling_mask = []

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

            # Copy tensors (GPU)
            all_input_ids_tensor[start_index:end_index, : batch.all_input_ids_tensor.shape[1]] = (
                batch.all_input_ids_tensor[:, :max_length]
            )

            block_tables_tensor[start_index:end_index, : batch.block_tables_tensor.shape[1]] = (
                batch.block_tables_tensor[:, :max_blocks]
            )

            prompt_lengths_tensor[start_index:end_index] = batch.prompt_lengths_tensor

            slots_start_index = cumulative_slots
            slots_end_index = cumulative_slots + len(batch.slots)
            slots[slots_start_index:slots_end_index] = batch.slots
            cu_slots[start_index + 1 : end_index + 1] = batch.cu_slots[1:] + cumulative_slots

            if not prefilling:
                input_ids[start_index:end_index] = batch.input_ids
                position_ids[start_index:end_index] = batch.position_ids
                slot_indices[start_index:end_index] = batch.slot_indices + cumulative_slots
                input_lengths_tensor[start_index:end_index] = batch.input_lengths_tensor
                cache_lengths_tensor[start_index:end_index] = batch.cache_lengths_tensor

                # Copy over adapter indices
                adapter_start_index = cumulative_adapter_indices_size
                adapter_end_index = cumulative_adapter_indices_size + batch.adapter_meta.adapter_indices.shape[0]
                adapter_indices[adapter_start_index:adapter_end_index] = batch.adapter_meta.adapter_indices
                cumulative_adapter_indices_size = adapter_end_index
                adapter_list.extend(batch.adapter_meta.adapter_list)
                adapter_set.update(batch.adapter_meta.adapter_set)
                adapter_segment_builder.concat(
                    batch.adapter_meta.adapter_segments,
                    batch.adapter_meta.segment_indices,
                )
            else:
                if isinstance(batch.input_ids, torch.Tensor):
                    batch.input_ids = batch.input_ids.view(-1, 1).tolist()
                input_ids.extend(batch.input_ids)
                prefilling_mask_tensor[start_index:end_index] = batch.prefilling_mask_tensor

            prefilling_mask.extend(batch.prefilling_mask)
            block_tables.extend(batch.block_tables)
            cache_lengths.extend(batch.cache_lengths)
            all_input_ids.extend(batch.all_input_ids)

            prompt_lengths.extend(batch.prompt_lengths)
            input_lengths.extend(batch.input_lengths)
            prefix_offsets.extend(batch.prefix_offsets)
            read_offsets.extend(batch.read_offsets)

            prefill_logprob_tokens.extend(batch.prefill_logprob_tokens)

            next_token_chooser_parameters.extend([r.parameters for r in batch.requests])
            if batch.next_token_chooser.schema_processor is not None:
                sequence_processors.extend(batch.next_token_chooser.schema_processor.sequence_processors)
            else:
                # No sequence processors, so pad with Nones
                sequence_processors.extend([None for _ in batch.requests])
            stopping_criterias.extend(batch.stopping_criterias)

            # Update
            cumulative_slots += len(batch.slots)
            cumulative_batch_size += len(batch)

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters,
            tokenizers=[],
            dtype=batches[0].next_token_chooser.dtype,
            device=batches[0].next_token_chooser.device,
            sequence_processors=sequence_processors,
        )

        # We skip computing the speculative_ids when the batch size is too large, so
        # we must check that all batches have them, otherwise they must be discarded
        speculative_ids = None
        if get_speculative_tokens() > 0:
            if all(b.speculative_ids is not None for b in batches):
                speculative_ids = torch.cat([b.speculative_ids for b in batches], dim=0)
            else:
                logger.info("Discarding speculative IDs, not every batch has them")

        if adapter_segment_builder is not None:
            adapter_segments, adapter_segment_indices = adapter_segment_builder.build()
            adapter_meta = AdapterBatchMetadata(
                adapter_indices=adapter_indices,
                adapter_list=adapter_list,
                adapter_set=adapter_set,
                adapter_segments=adapter_segments,
                segment_indices=adapter_segment_indices,
            )

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            speculative_ids=speculative_ids,
            cu_seqlen_prefill=None,
            prefill_cache_indices=None,
            slot_indices=slot_indices,
            block_tables=block_tables,
            block_tables_tensor=block_tables_tensor,
            cache_lengths=cache_lengths,
            cache_lengths_tensor=cache_lengths_tensor,
            slots=slots,
            cu_slots=cu_slots,
            max_input_length=max_input_length,
            max_current_length=max_current_length,
            prefilling=prefilling,
            prefilling_mask=prefilling_mask,
            prefilling_mask_tensor=prefilling_mask_tensor,
            prefill_head_indices=None,
            prefill_next_token_indices=None,
            prefill_cu_outlens=None,
            prefill_logprob_tokens=prefill_logprob_tokens,
            prompt_lengths=prompt_lengths,
            prompt_lengths_tensor=prompt_lengths_tensor,
            input_lengths=input_lengths,
            input_lengths_tensor=input_lengths_tensor,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            num_blocks=num_blocks,
            max_blocks=max_blocks,
            adapter_meta=adapter_meta,
        )

    def prepare_for_prefill(self):
        global SLIDING_WINDOW
        global SLIDING_WINDOW_BLOCKS

        # Prepare values if we need to continue prefilling
        # Speculation must be ignored while we prefill even with chunking
        # it simplifies everything
        assert self.speculative_ids is None

        device = self.block_tables_tensor.device

        if isinstance(self.input_ids, list):
            if len(self) > 1:
                input_ids = np.concatenate(self.input_ids, dtype=np.int64)
            else:
                input_ids = self.input_ids[0]
            self.input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)

        self.input_lengths_tensor = torch.tensor(self.input_lengths, dtype=torch.int32, device=device)
        self.cu_seqlen_prefill = torch.nn.functional.pad(torch.cumsum(self.input_lengths_tensor, dim=0), (1, 0)).to(
            torch.int32
        )
        self.cache_lengths_tensor = torch.tensor(self.cache_lengths, dtype=torch.int32, device=device)

        # If the device supports Triton, we can use a fused kernel
        if has_triton():
            self.position_ids = torch.empty(len(self.input_ids), dtype=torch.int32, device=device)
            self.slot_indices = torch.empty(len(self.input_ids), dtype=torch.int64, device=device)
            cu_slots_gpu = self.cu_slots.to(device)

            prepare_position_slot_ids(
                self.max_input_length,
                self.cache_lengths_tensor,
                self.cu_seqlen_prefill,
                cu_slots_gpu,
                self.position_ids,
                self.slot_indices,
            )

        position_ids = []
        slot_indices = []
        prefill_cache_indices = []
        all_prefill_logprobs = True
        no_prefill_logprobs = True
        prefill_cu_outlens = [0]

        # Cumulative length
        cumulative_length = 0
        cumulative_slot_tokens = 0
        prefill_out_cumulative_length = 0

        adapter_indices_list = []
        adapter_list = []

        for i, (
            r,
            cache_length,
            input_length,
            prompt_length,
            request_prefilling,
            blocks,
        ) in enumerate(
            zip(
                self.requests,
                self.cache_lengths,
                self.input_lengths,
                self.prompt_lengths,
                self.prefilling_mask,
                self.block_tables,
            )
        ):
            next_chunk_length = input_length

            if not has_triton():
                # Position ids
                request_position_ids = torch.arange(cache_length, cache_length + input_length, dtype=torch.int32)
                position_ids.append(request_position_ids)

                if not r.slots:
                    request_slots = [s for b in blocks for s in range(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE)]
                else:
                    request_slots = r.slots

                request_slot_indices = torch.arange(
                    cache_length + cumulative_slot_tokens,
                    cache_length + cumulative_slot_tokens + input_length,
                    dtype=torch.int64,
                )

                slot_indices.append(request_slot_indices)

                # Update
                cumulative_slot_tokens += len(request_slots)

            # Create tensor to slice into the kv tensor in prefill
            if SLIDING_WINDOW is not None:
                request_prefill_cache_indices = torch.arange(
                    cumulative_length + max(0, input_length - SLIDING_WINDOW),
                    cumulative_length + input_length,
                    dtype=torch.int64,
                )

            # Prefill logprobs is ignored if the request is done prefilling
            prefill_logprobs = r.prefill_logprobs and request_prefilling

            all_prefill_logprobs = all_prefill_logprobs and prefill_logprobs
            no_prefill_logprobs = no_prefill_logprobs and not prefill_logprobs

            if prefill_logprobs:
                prefill_cu_outlens.append(prefill_out_cumulative_length + input_length)
                prefill_out_cumulative_length += input_length
            else:
                prefill_cu_outlens.append(prefill_out_cumulative_length + 1)
                prefill_out_cumulative_length += 1

            if SLIDING_WINDOW is not None:
                prefill_cache_indices.append(request_prefill_cache_indices)

            adapter_indices_list.append(torch.full((next_chunk_length,), r.adapter_index))
            adapter_list.append(r.adapter_index)

            # Update
            cumulative_length += next_chunk_length

        if not all_prefill_logprobs and not no_prefill_logprobs:
            prefill_head_indices = []
            prefill_next_token_indices = []

            # Cumulative length
            cumulative_length = 0
            prefill_out_cumulative_length = 0

            for i, (
                r,
                input_length,
                request_prefilling,
            ) in enumerate(
                zip(
                    self.requests,
                    self.input_lengths,
                    self.prefilling_mask,
                )
            ):
                # Prefill logprobs is ignored if the request is done prefilling
                prefill_logprobs = r.prefill_logprobs and request_prefilling

                if prefill_logprobs:
                    prefill_head_indices.append(
                        torch.arange(
                            cumulative_length,
                            cumulative_length + input_length,
                            dtype=torch.int64,
                        )
                    )
                    prefill_next_token_indices.append(prefill_out_cumulative_length + input_length - 1)
                    prefill_out_cumulative_length += input_length
                else:
                    prefill_head_indices.append(
                        torch.tensor(
                            [cumulative_length + input_length - 1],
                            dtype=torch.int64,
                        )
                    )
                    prefill_next_token_indices.append(prefill_out_cumulative_length)
                    prefill_out_cumulative_length += 1

                # Update
                cumulative_length += input_length

        if len(self) > 1:
            if position_ids:
                position_ids = torch.cat(position_ids)
            if slot_indices:
                slot_indices = torch.cat(slot_indices)
            if SLIDING_WINDOW is not None:
                prefill_cache_indices = torch.cat(prefill_cache_indices)
        else:
            if position_ids:
                position_ids = position_ids[0]
            if slot_indices:
                slot_indices = slot_indices[0]
            if SLIDING_WINDOW is not None:
                prefill_cache_indices = prefill_cache_indices[0]

        if not has_triton():
            self.position_ids = position_ids.to(device)
            self.slot_indices = slot_indices.to(device)

        self.prefill_cu_outlens = prefill_cu_outlens
        self.prefill_cache_indices = prefill_cache_indices.to(device) if SLIDING_WINDOW is not None else None

        if all_prefill_logprobs:
            prefill_head_indices = None
            prefill_next_token_indices = self.cu_seqlen_prefill[1:] - 1
        elif no_prefill_logprobs:
            prefill_head_indices = self.cu_seqlen_prefill[1:] - 1
            prefill_next_token_indices = None
        else:
            prefill_head_indices = torch.cat(prefill_head_indices).to(device)
            prefill_next_token_indices = torch.tensor(prefill_next_token_indices, dtype=torch.int64, device=device)

        self.prefill_head_indices = prefill_head_indices
        self.prefill_next_token_indices = prefill_next_token_indices
        adapter_indices = torch.cat(adapter_indices_list).to(dtype=torch.int64, device=device)
        adapter_segments, adapter_segment_indices = find_segments(adapter_indices)
        adapter_segments = torch.tensor(adapter_segments, dtype=torch.int32, device=device)
        self.adapter_meta = AdapterBatchMetadata(
            adapter_indices=adapter_indices,
            adapter_list=adapter_list,
            adapter_set=set(adapter_list),
            adapter_segments=adapter_segments,
            segment_indices=adapter_segment_indices,
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
        supports_chunking: bool = True,
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

        if is_fp8(config.quantize) and not is_fp8_supported():
            raise ValueError("FP8 quantization is only supported on hardware that supports FP8")

        if is_fp8_kv(config.quantize):
            if not FLASH_INFER:
                raise ValueError("FP8 KV cache requires FLASH_INFER backend")
            self.kv_dtype = torch.float8_e4m3fn
            logger.info("Enabling FP8 KV cache. Prefix caching will not work.")
        else:
            self.kv_dtype = dtype

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
        if (
            not (weights.has_tensor("lm_head.weight") or weights.has_tensor("language_model.lm_head.weight"))
            and not self._supports_embeddings
        ):
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
            supports_chunking=supports_chunking,
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

        self.punica_wrapper = None

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
        max_total_tokens = batch.max_input_length + max_new_tokens + get_speculative_tokens()

        self.punica_wrapper = PunicaWrapper(
            max_num_batched_tokens=get_max_prefill_tokens(),
            max_batches=256,  # TODO(travis): find a better way to set this programmatically
            device=self.device,
            enabled=(
                not self.dynamic_adapter_loading_enabled  # only supported for now with statically loaded adapters
                and not LORAX_PUNICA_TRITON_DISABLED
            ),
        )

        torch.cuda.empty_cache()
        try:
            self.init_kv_cache(
                batch.num_blocks,
                self.num_layers,
                self.num_kv_heads,
                self.head_size,
                self.kv_dtype,
                self.device,
            )

            if not embedding_model:
                with warmup_mode():
                    logger.info("Warming up to max_new_tokens: {}", max_new_tokens)
                    with tqdm(total=max_new_tokens, desc="Warmup to max_total_tokens") as pbar:
                        for _ in range(max_new_tokens):
                            cur_seqlen = batch.max_current_length
                            _, batch = self.generate_token(batch, is_warmup=True)
                            new_seqlen = batch.max_current_length
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

            # Estimate the memory overhead from CUDA graphs so we can subtract it from the kv cache.
            # Needs to be estimated here rather than fully initialized as the graph cache relies on the
            # cache manager being set.
            self.model_graph_wrapper = GraphCache(
                self.model,
                self.device,
                self.kv_cache,
                self.adapter_layers,
                self.traced_adapter_layers,
                self._forward_context,
                max_total_tokens,
                self.num_heads,
                self.num_kv_heads,
                self.sliding_window_blocks,
                self.layer_to_lora_weights,
                self.punica_wrapper,
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
            self.kv_dtype,
            self.device,
        )

        torch.cuda.synchronize(self.device)

        if self.model_graph_wrapper is not None:
            # Warmup the graph cache. Needs to be done after setting cache manager as
            # tracing will use the static kv cache tensors
            self.model_graph_wrapper.kv_cache = self.kv_cache
            self.model_graph_wrapper.warmup()
            torch.cuda.synchronize(self.device)

        if self.profiler is not None:
            self.profiler.start()

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
        cache_lengths: List[int],
        cache_lengths_tensor: torch.Tensor,
        state: Optional[Any] = None,
    ) -> ContextManager:
        if not FLASH_INFER:
            return nullcontext()

        from lorax_server.utils.flashinfer_attention import (
            use_decode_state,
            use_prefill_state,
            use_prefill_with_paged_kv_state,
        )

        if cu_seqlen_prefill is not None:
            if self.kv_dtype == torch.float8_e4m3fn:
                return use_prefill_state(
                    state=(state if state is not None else self.prefill_state),
                    cu_seqlens=cu_seqlen_prefill,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_size=self.head_size,
                    query_dtype=self.dtype,
                    window_left=self.sliding_window,
                )
            return use_prefill_with_paged_kv_state(
                state=(state if state is not None else self.prefill_with_paged_kv_state),
                block_tables=block_tables,
                cu_seqlens=cu_seqlen_prefill,
                input_lengths=input_lengths_tensor + cache_lengths_tensor,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                page_size=BLOCK_SIZE,
                dtype=self.dtype,
                window_left=self.sliding_window,
            )
        else:
            assert input_lengths_tensor is not None
            return use_decode_state(
                state=state if state is not None else self.decode_state,
                input_lengths=input_lengths_tensor + cache_lengths_tensor,
                block_tables=block_tables,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                page_size=BLOCK_SIZE,
                dtype=self.dtype,
                window_left=self.sliding_window,
            )

    def forward(self, batch: FlashCausalLMBatch, adapter_data: AdapterBatchData) -> Tuple[torch.Tensor, torch.Tensor]:
        prefill = batch.cu_seqlen_prefill is not None
        model = self.model
        use_graph = False
        if self.model_graph_wrapper is not None and not prefill:
            if self.model_graph_wrapper.can_use_graph(batch, adapter_data):
                use_graph = True
                model = self.model_graph_wrapper
            else:
                logger.info("CUDA graphs enabled but batch is incompatible, falling back to eager mode.")

        input_ids = batch.input_ids
        position_ids = batch.position_ids
        cu_seqlen_prefill = batch.cu_seqlen_prefill
        block_tables = batch.block_tables_tensor
        slots = batch.slots[batch.slot_indices]
        input_lengths = batch.input_lengths_tensor
        cache_lengths_tensor = batch.cache_lengths_tensor
        max_s = batch.max_current_length

        if batch.speculative_ids is not None:
            speculative_ids = batch.speculative_ids

            B, speculative_length = speculative_ids.shape
            new_length = speculative_length + 1
            new_input_ids = torch.cat([input_ids.unsqueeze(-1), speculative_ids], dim=1).reshape(-1)
            arange = torch.arange(new_length, device=position_ids.device).unsqueeze(0)
            arange_int = arange.to(dtype=torch.int32)
            new_position_ids = (position_ids.unsqueeze(-1).expand(B, new_length) + arange).view(-1)

            input_lengths = (input_lengths.unsqueeze(-1).expand(B, new_length) + arange_int).view(-1)
            cache_lengths_tensor = (batch.cache_lengths_tensor.unsqueeze(-1).expand(B, new_length)).reshape(-1)

            # Slots can be discontiguous when prefix caching is enabled, so we need to expand the slot_indices,
            # then update the slots with the additional indices to ensure we're grabbing the ones that have been
            # allocated
            slot_indices = (batch.slot_indices.unsqueeze(-1).expand(B, new_length) + arange_int).view(-1)
            slots = batch.slots[slot_indices]

            block_tables = block_tables.unsqueeze(1).expand(B, new_length, -1).reshape(B * new_length, -1).contiguous()
            max_s = max_s + speculative_length

            input_ids = new_input_ids
            position_ids = new_position_ids

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

        # Model Forward
        if not use_graph:
            # eager mode
            if FLASH_INFER:
                block_tables = block_tables_to_ragged(
                    block_tables=block_tables,
                    input_lengths=batch.input_lengths,
                    cache_lengths=batch.cache_lengths,
                    input_lengths_tensor=batch.input_lengths_tensor,
                    cache_lengths_tensor=batch.cache_lengths_tensor,
                    max_current_length=max_s,
                )

            with self._forward_context(
                block_tables=block_tables,
                cu_seqlen_prefill=batch.cu_seqlen_prefill,
                input_lengths=batch.input_lengths,
                input_lengths_tensor=input_lengths,
                cache_lengths=batch.cache_lengths,
                cache_lengths_tensor=cache_lengths_tensor,
            ):
                out = model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cu_seqlen_prefill=batch.cu_seqlen_prefill,
                    kv_cache=self.kv_cache,
                    block_tables=block_tables,
                    slots=slots,
                    seqlen=seqlen,
                    max_s=max_s,
                    adapter_data=adapter_data,
                    prefill_cache_indices=batch.prefill_cache_indices,
                    lm_head_indices=batch.prefill_head_indices,
                )
        else:
            skip_lm_head = get_speculative_tokens() > 0

            # CUDA graph mode
            out = model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seqlen_prefill=batch.cu_seqlen_prefill,
                kv_cache=self.kv_cache,
                block_tables=block_tables,
                slots=slots,
                seqlen=seqlen,
                cache_lengths=batch.cache_lengths,
                cache_lengths_tensor=cache_lengths_tensor,
                max_s=max_s,
                adapter_data=adapter_data,
                prefill_cache_indices=batch.prefill_cache_indices,
                lm_head_indices=batch.prefill_head_indices,
            )

            if skip_lm_head and hasattr(self.model, "lm_head"):
                # re-run through the LM head as the graph did not capture it
                out = self.model.lm_head(out[0], adapter_data)

        if batch.prefill_cache_indices is not None:
            batch.prefill_cache_indices = None

        return out

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: FlashCausalLMBatch, is_warmup: bool = False
    ) -> Tuple[List[Generation], Optional[FlashCausalLMBatch]]:
        prefill = batch.prefilling
        if prefill:
            batch.prepare_for_prefill()
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
                adapter_list=adapter_meta.adapter_list,
                adapter_set=adapter_meta.adapter_set,
                adapter_segments=adapter_segments,
                segment_indices=adapter_meta.segment_indices,
            )

        # Assign pointers to adapter weights
        # TODO(travis): don't update this if indices haven't changed
        self.punica_wrapper.update_metadata(adapter_meta, prefill)
        adapter_data = AdapterBatchData.from_meta(
            adapter_meta,
            self.layer_to_adapter_weights,
            self.layer_to_lora_weights,
            self.punica_wrapper,
            prefill,
            batch.prefill_head_indices,
        )

        out, speculative_logits = self.forward(batch, adapter_data)

        if prefill:
            next_token_logits = out[batch.prefill_next_token_indices] if prefill_logprobs else out
            if speculative_logits is not None:
                speculative_logits = (
                    speculative_logits[batch.prefill_next_token_indices] if prefill_logprobs else speculative_logits
                )
            if len(batch) > 1 and prefill_logprobs:
                # We create the prefill_tokens_indices tensor that will be used to gather prefill logprobs
                # When batch == 1, we will just use the batch.input_ids values directly
                prefill_tokens_indices = batch.input_ids.new_zeros(len(out))
        else:
            prefill_logprobs = None
            next_token_logits = out

        finished_prefilling = True
        next_chunk_lengths = []
        current_prefilling_mask = batch.prefilling_mask
        if prefill:
            if get_supports_chunking():
                next_prefilling_mask = []
                # Budget in tokens for the next batch
                # We remove (len(batch) - 1) to always have enough space for at least a single decode
                # for the remaining requests -1 because the first request does not need to be removed from the budget
                # (ex: you have one request in the batch, you want it to take the full budget not budget -1)
                batch_budget = get_max_prefill_tokens() - (len(batch) - 1)
                # We reverse to prioritize older requests
                # zip() is not reversible so reverse the underlying lists instead
                for cache_length, input_length, prompt_length in zip(
                    reversed(batch.cache_lengths),
                    reversed(batch.input_lengths),
                    reversed(batch.prompt_lengths),
                ):
                    remaining_prefill_tokens = max(prompt_length - cache_length - input_length, 0)
                    if remaining_prefill_tokens > 0:
                        next_chunk_length = max(min(remaining_prefill_tokens, batch_budget), 1)
                        batch_budget -= next_chunk_length
                        finished_prefilling = False
                        next_prefilling_mask.append(True)
                    else:
                        # FIXME: use true number of accepted tokens instead of 1
                        # Since speculation will be turned off, this is always true
                        next_chunk_length = 1
                        next_prefilling_mask.append(False)
                    next_chunk_lengths.append(next_chunk_length)

                # Reverse back the obtained values²
                next_chunk_lengths.reverse()
                next_prefilling_mask.reverse()
            else:
                # The model does not support chunking
                # We know we only do a single prefill
                finished_prefilling = True
                next_prefilling_mask = [False] * len(batch)

            batch.prefilling = not finished_prefilling
            batch.prefilling_mask = next_prefilling_mask

        speculative_tokens = get_speculative_tokens()
        (
            next_input_ids,
            next_token_logprobs,
            accepted_ids,
            speculative_ids,
        ) = batch.next_token_chooser(
            batch.all_input_ids_tensor[:, : batch.max_current_length],
            next_token_logits,
            speculative_tokens,
            batch.speculative_ids,
            speculative_logits,
        )

        if return_alternatives:
            alternative_token_logprobs, alternative_token_ids = torch.sort(
                torch.log_softmax(next_token_logits, -1), dim=-1, stable=True, descending=True
            )

        # Since we are done prefilling, all the tensors that were concatenating values for all the requests
        # instantly become of shape [BATCH_SIZE]
        if prefill and finished_prefilling:
            indices = batch.cu_seqlen_prefill[1:] - 1
            batch.position_ids = batch.position_ids[indices]
            batch.slot_indices = batch.slot_indices[indices]
            batch.adapter_meta.adapter_indices = batch.adapter_meta.adapter_indices[indices]

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.prompt_lengths,
            batch.cache_lengths,
            batch.input_lengths,
            batch.all_input_ids,
            accepted_ids,
            current_prefilling_mask,
            batch.prefilling_mask,
        )

        # We do two for loops as the first one can run completely asynchronously from the GPU while for the second
        # one, we need to first do a GPU <-> CPU sync
        # It is faster if we delay this sync for the maximum amount of time

        # For each member of the batch
        # Cumulative length
        cu_accepted_ids = torch.nn.functional.pad(torch.cumsum(accepted_ids, dim=0), (1, 0))
        cumulative_length = 0
        for i, (
            request,
            prompt_length,
            cache_length,
            input_length,
            all_input_ids,
            n_accepted_ids,
            request_was_prefilling,
            request_is_prefilling,
        ) in enumerate(iterator):
            # Used to gather prefill logprobs
            # Copy batch.all_input_ids_tensor to prefill_token_indices
            if request.prefill_logprobs and request_was_prefilling:
                # Indexing metadata
                out_start_index = batch.prefill_cu_outlens[i]
                out_end_index = batch.prefill_cu_outlens[i + 1]

                # Logprobs generated by the model are for the next token
                # So we need to translate the id tensor by 1
                ids = batch.all_input_ids_tensor[i, cache_length + 1 : cache_length + input_length + 1]
                if len(batch) > 1:
                    prefill_tokens_indices[out_start_index:out_end_index] = ids
                else:
                    # Set prefill_tokens_indices to the correct slice
                    prefill_tokens_indices = ids

            # If the device does not support triton, we copy one by one
            if not request_is_prefilling and not has_triton():
                # Only save tokens if we are done prefilling for this request
                batch.all_input_ids_tensor[
                    i,
                    batch.cache_lengths_tensor[i] + batch.input_lengths[i] : batch.cache_lengths_tensor[i]
                    + batch.input_lengths[i]
                    + accepted_ids[i],
                ] = next_input_ids[cu_accepted_ids[i] : cu_accepted_ids[i + 1]]

            cumulative_length += input_length

        # If the device support triton, we can use a fused kernel
        if has_triton():
            copy_next_input_ids_inplace(
                speculative_tokens + 1,
                batch.all_input_ids_tensor,
                batch.cache_lengths_tensor,
                batch.input_lengths_tensor,
                batch.prompt_lengths_tensor,
                next_input_ids,
                cu_accepted_ids,
            )

        # Update values
        # These values can be updated without a GPU -> CPU sync
        if not prefill or (prefill and finished_prefilling):
            batch.input_ids = next_input_ids[cu_accepted_ids[1:] - 1]
            batch.speculative_ids = speculative_ids
            batch.position_ids += accepted_ids
            batch.cache_lengths_tensor += batch.input_lengths_tensor + accepted_ids - 1
            batch.input_lengths_tensor = torch.ones_like(batch.input_lengths_tensor)
            batch.slot_indices += accepted_ids

        if prefill and prefill_logprobs:
            # Get prefill logprobs with inplace softmax (avoid copying the `out` tensor (max_batch_prefill_tokens * vocab_size))
            torch.log_softmax(out, -1, out=out)
            prefill_logprobs_tensor = out
            prefill_logprobs = torch.gather(prefill_logprobs_tensor, 1, prefill_tokens_indices.view(-1, 1))
            # GPU <-> CPU sync
            prefill_logprobs = prefill_logprobs.view(-1).tolist()

        # Does a GPU <-> CPU sync internally
        if prefill and finished_prefilling:
            # adjust segment lengths to account for all request lengths being 1 during decoding
            adapter_segments, _ = find_segments(batch.adapter_meta.adapter_indices)
            batch.adapter_meta.adapter_segments = torch.tensor(
                adapter_segments,
                dtype=torch.int32,
                device=batch.adapter_meta.adapter_segments.device,
            )

        # GPU <-> CPU sync
        next_token_logprobs = next_token_logprobs.tolist()
        next_token_ids = next_input_ids.tolist()
        accepted_ids = accepted_ids.tolist()

        if return_alternatives:
            alternative_token_logprobs = alternative_token_logprobs.tolist()
            alternative_token_ids = alternative_token_ids.tolist()

        # Update values if we need to continue prefilling
        # This represents the `else` case of the `Update values` if above
        # but since this require the `next_token_ids` to be on CPU, it is better to do it here
        if prefill and not finished_prefilling:
            # Speculation must be ignored while we prefill even with chunking
            # it simplifies everything
            assert batch.speculative_ids is None

            all_postfix_ids = []
            for i, (
                request_prefilling,
                next_token_id,
                all_input_ids,
                cache_length,
                input_length,
                next_chunk_length,
            ) in enumerate(
                zip(
                    batch.prefilling_mask,
                    next_token_ids,
                    batch.all_input_ids,
                    batch.cache_lengths,
                    batch.input_lengths,
                    next_chunk_lengths,
                )
            ):
                if request_prefilling:
                    next_cache_length = cache_length + input_length
                    # Get new prompt IDs to prefill
                    postfix_ids = all_input_ids[next_cache_length : next_cache_length + next_chunk_length]
                else:
                    # This request is done prefilling, the new id is the one selected the sampling method
                    postfix_ids = [next_token_id]

                all_postfix_ids.append(postfix_ids)

            batch.input_ids = all_postfix_ids

        # Results
        generations: List[Generation] = []
        stopped = not is_warmup

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.prompt_lengths,
            batch.cache_lengths,
            batch.input_lengths,
            batch.prefix_offsets,
            batch.read_offsets,
            batch.stopping_criterias,
            batch.all_input_ids,
            batch.next_token_chooser.do_sample,
            batch.next_token_chooser.seeds,
            current_prefilling_mask,
            batch.prefilling_mask,
            accepted_ids,
        )

        # Reset max_input_length
        batch.max_input_length = 0
        # For each member of the batch
        index = 0
        for i, (
            request,
            prompt_length,
            cache_length,
            input_length,
            prefix_offset,
            read_offset,
            stopping_criteria,
            all_input_ids,
            do_sample,
            seed,
            request_was_prefilling,
            request_is_prefilling,
            n_accepted_ids,
        ) in enumerate(iterator):
            all_alternative_tokens = [] if request.parameters.return_k_alternatives > 0 else None

            # TODO(travis): return_k_alternatives
            # if request.parameters.return_k_alternatives > 0:
            #         # Limit the number of alternatives to the vocabulary size
            #         num_alternatives = min(
            #             request.parameters.return_k_alternatives,
            #             len(alternative_token_ids[token_idx]),
            #         )

            #         # Select top-k logprobs
            #         request_alternative_token_ids = alternative_token_ids[token_idx][:num_alternatives]
            #         request_alternative_token_logprobs = alternative_token_logprobs[token_idx][:num_alternatives]

            #         # Decode tokens
            #         request_alternative_token_texts = []
            #         for alternative_token_id in request_alternative_token_ids:
            #             all_input_ids.append(alternative_token_id)
            #             alternative_token_text, _, _ = self.decode_token(
            #                 all_input_ids,
            #                 prefix_offset,
            #                 read_offset,
            #             )
            #             request_alternative_token_texts.append(alternative_token_text)
            #             all_input_ids.pop()
            #         alternative_tokens = AlternativeTokens(
            #             request_alternative_token_ids,
            #             request_alternative_token_logprobs,
            #             request_alternative_token_texts,
            #         )
            #         all_alternative_tokens.append(alternative_tokens)

            # Compute logprobs first as, even though we might skip the token,
            # it can still be required to compute the logprobs
            # modulo on request.id as it is robust to batch.filter whereas the index in the batch is not and we need
            # this state to be stable
            if request.id % self.world_size == self.rank:
                # Prefill
                if request_was_prefilling and request.prefill_logprobs:
                    out_start_index = batch.prefill_cu_outlens[i]
                    out_end_index = batch.prefill_cu_outlens[i + 1]
                    if not request_is_prefilling:
                        # The request is dones prefilling, meaning that we started generating new tokens
                        # The last logprob is a logprob for a generated token that was not part of the prompt
                        # We need to remove it
                        out_end_index -= 1

                    request_prefill_logprobs = prefill_logprobs[out_start_index:out_end_index]
                    # Logprobs generated by the model are for the next token
                    # So we need to translate the id tensor by 1
                    prefill_token_ids = all_input_ids[cache_length + 1 : cache_length + input_length + 1]

                    past_prefill_logprob_tokens = batch.prefill_logprob_tokens[i]

                    if past_prefill_logprob_tokens is None:
                        # add nan for cached prompt tokens/first token
                        request_prefill_logprobs = [float("nan")] * (cache_length + 1) + request_prefill_logprobs
                        prefill_token_ids = all_input_ids[: cache_length + 1] + prefill_token_ids

                    prefill_texts = self.tokenizer.batch_decode(
                        prefill_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )

                    prefill_logprob_tokens = NextTokens(
                        prefill_token_ids,
                        request_prefill_logprobs,
                        prefill_texts,
                        [],
                        all_alternative_tokens,
                    )
                    if past_prefill_logprob_tokens is not None:
                        prefill_logprob_tokens = past_prefill_logprob_tokens + prefill_logprob_tokens

                    batch.prefill_logprob_tokens[i] = prefill_logprob_tokens
                else:
                    batch.prefill_logprob_tokens[i] = None

            # If it is, the tokens we decoded should be ignored
            if request_is_prefilling:
                # Make sure that we do not stop as even though this request did not create a token, it is still
                # processing
                stopped = False
                new_input_length = next_chunk_lengths[i]
                new_cache_length = cache_length + input_length
            else:
                new_input_length = 1
                new_cache_length = cache_length + input_length + n_accepted_ids - 1
                # Append next token to all tokens
                next_token_texts = []
                left = 0

                if n_accepted_ids > 1:
                    logger.debug(f"speculated ids {n_accepted_ids - 1}")

                current_stopped = False
                for j in range(index, index + n_accepted_ids):
                    # Generated token
                    next_token_id = next_token_ids[j]
                    all_input_ids.append(next_token_id)
                    next_token_text, prefix_offset, read_offset = self.decode_token(
                        all_input_ids,
                        prefix_offset,
                        read_offset,
                    )
                    next_token_texts.append(next_token_text)

                    stop, reason = stopping_criteria(
                        next_token_id,
                        next_token_text,
                    )

                    if stop:
                        left = index + n_accepted_ids - j - 1
                        current_stopped = True
                        break
                    else:
                        current_stopped = False
                stopped = stopped and current_stopped

                _next_token_ids = next_token_ids[index : index + n_accepted_ids - left]
                _next_token_logprobs = next_token_logprobs[index : index + n_accepted_ids - left]

                # Shard generations
                # All generations will be appended in the rust sharded client
                if request.id % self.world_size == self.rank:
                    if stop:
                        # Decode generated tokens
                        output_text, _, _ = self.decode_token(
                            all_input_ids,
                            prefix_offset=len(all_input_ids) - stopping_criteria.current_tokens - 1,
                            read_offset=len(all_input_ids) - stopping_criteria.current_tokens,
                            skip_special_tokens=True,
                        )
                        generated_text = GeneratedText(
                            output_text,
                            stopping_criteria.current_tokens,
                            reason,
                            seed if do_sample else None,
                        )
                    else:
                        generated_text = None

                    # TODO(travis): top tokens
                    # if top_n_tokens > 0:
                    #     all_top_tokens = []
                    #     for top_token_ids, top_token_logprobs in zip(
                    #         top_token_ids, top_token_logprobs
                    #     ):
                    #         toptoken_texts = self.tokenizer.batch_decode(
                    #             top_token_ids,
                    #             clean_up_tokenization_spaces=False,
                    #             skip_special_tokens=False,
                    #         )
                    #         special_toptokens = [
                    #             token_id in self.all_special_ids
                    #             for token_id in top_token_ids
                    #         ]
                    #         top_tokens = Tokens(
                    #             top_token_ids,
                    #             top_token_logprobs,
                    #             toptoken_texts,
                    #             special_toptokens,
                    #         )
                    #         all_top_tokens.append(top_tokens)
                    #     top_tokens = all_top_tokens
                    # else:
                    #     top_tokens = None

                    generation = Generation(
                        request.id,
                        batch.prefill_logprob_tokens[i],
                        len(all_input_ids[:-1]) if prefill else 0,
                        NextTokens(
                            _next_token_ids,
                            _next_token_logprobs,
                            next_token_texts,
                            [nid in self.all_special_ids for nid in _next_token_ids],
                            all_alternative_tokens,
                        ),
                        generated_text,
                    )

                    generations.append(generation)

                # advance the FSM for each accepted token (as we may have more than one from speculative decoding)
                for next_token_id in _next_token_ids:
                    batch.next_token_chooser.next_state(i, next_token_id)

            # Update values
            index += n_accepted_ids
            batch.cache_lengths[i] = new_cache_length
            batch.max_input_length = max(batch.max_input_length, new_input_length)
            batch.input_lengths[i] = new_input_length
            current_length = new_cache_length + new_input_length
            batch.max_current_length = max(batch.max_current_length, current_length)

            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.all_input_ids[i] = all_input_ids

        if stopped:
            # No need to return a batch if we know that all requests stopped
            return generations, None

        if prefill and finished_prefilling:
            # We do not need prefill tensors anymore
            batch.cu_seqlen_prefill = None
            batch.prefill_cache_indices = None
            batch.prefill_cu_outlens = None
            batch.prefill_head_indices = None
            batch.prefill_next_token_indices = None

        return generations, batch
