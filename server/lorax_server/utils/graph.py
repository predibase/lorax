# CUDA Graph implementation modified from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py

from dataclasses import dataclass
from functools import lru_cache
from statistics import median
from typing import TYPE_CHECKING, List, Optional, Tuple
import numpy as np

import torch
from torch import nn
from tqdm import tqdm

from lorax_server.utils.lora import AdapterBatchData, AdapterBatchMetadata, AdapterWeightData, RankSegments
from lorax_server.models.cache_manager import get_cache_manager, BLOCK_SIZE
from lorax_server.utils.sgmv import get_tmp_expand_size, get_tmp_tensors, use_cutlass_shrink

if TYPE_CHECKING:
    from lorax_server.models.flash_causal_lm import FlashCausalLMBatch


# TODO(travis): make this configurable by model / user
MAX_BATCH_SIZE = 256
MAX_CONTEXT_LENGTH = 8192
MAX_RANK = 64

SLOT_PAD_VALUE = -1
SEGMENT_PAD_VALUE = -1

# Cached batch sizes used in vLLM. This and the helper function `get_cached_batch_size` below
# must be kept in sync.
BATCH_SIZE_INCREMENT = 32
CACHED_BATCH_SIZES = [1, 2, 4, 8, 16] + [BATCH_SIZE_INCREMENT * (i + 1) for i in range(8)]

# Include 0 to ensure we can use cuda graphs without adapters
# TODO(travis): use padding to allow for more ranks without increasing memory usage
CACHED_MAX_RANKS = [0, 8, 16, 32, 64]
_allowed_ranks = set(CACHED_MAX_RANKS)

MAX_SAMPLES = 3


def get_cached_batch_size(batch_size: int) -> int:
    if batch_size == 1:
        return 1
    if batch_size == 2:
        return 2
    if batch_size <= 4:
        return 4
    if batch_size <= 8:
        return 8
    if batch_size <= 16:
        return 16
    return (batch_size + BATCH_SIZE_INCREMENT - 1) // BATCH_SIZE_INCREMENT * BATCH_SIZE_INCREMENT


def pad_and_fill(dest: torch.Tensor, src: torch.Tensor, pad_value: int):
    dest[:src.shape[0]] = src
    dest[src.shape[0]:].fill_(pad_value)


def next_pow_2(x: int) -> int:
    assert x > 0
    return 1 << (x-1).bit_length()


@dataclass
class GraphState:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    block_tables: torch.Tensor
    slots: torch.Tensor
    input_lengths: torch.Tensor
    adapter_data: AdapterBatchData


@lru_cache(maxsize=1)
def get_max_graph_state(device: torch.device, adapter_layers: Tuple[str]) -> GraphState:
    max_num_blocks = (MAX_CONTEXT_LENGTH + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables_arr = np.zeros((MAX_BATCH_SIZE, max_num_blocks), dtype=np.int32)
    block_tables = torch.from_numpy(block_tables_arr).to(device=device)

    input_ids = torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int64, device=device)
    position_ids = torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int32, device=device)
    slots = torch.full((MAX_BATCH_SIZE,), SLOT_PAD_VALUE, dtype=torch.int64, device=device)
    input_lengths = torch.ones((MAX_BATCH_SIZE,), dtype=torch.int32, device=device)

    tmp_shrink, tmp_expand = get_tmp_tensors(MAX_BATCH_SIZE, MAX_RANK, device)

    adapter_weight_data = {}
    for layer_name in adapter_layers:
        adapter_weight_data[layer_name] = AdapterWeightData(
            lora_a={},
            lora_b={},
            adapter_index_configs={},
            rank_data={
                MAX_RANK: RankSegments(
                    rank=MAX_RANK,
                    tmp_shrink=tmp_shrink,
                    tmp_expand=tmp_expand,
                    lora_a_ptr=torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int64, device=device),
                    lora_b_ptr=torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int64, device=device),
                    segment_starts=torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int32, device=device),
                    segment_ends=torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int32, device=device),
                ),
            },
        )

    return GraphState(
        input_ids=input_ids,
        position_ids=position_ids,
        block_tables=block_tables,
        slots=slots,
        input_lengths=input_lengths,
        adapter_data=AdapterBatchData(
            meta=AdapterBatchMetadata(
                adapter_indices=torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int64, device=device),
                adapter_set=set(),
                adapter_segments=torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int64, device=device),
                segment_indices=[],
            ),
            data=adapter_weight_data,
        ),
    )


class GraphWrapper:
    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        memory_pool: Tuple[int, int],
        input_state: GraphState,
        output_states: torch.Tensor,
        model,
    ):
        self.graph = graph
        self.memory_pool = memory_pool
        self.input_state = input_state
        self.output_states = output_states
        self.model = model
        
    @staticmethod
    def trace(
        model: nn.Module,
        device: torch.device,
        adapter_layers: Tuple[str],
        batch_size: int,
        max_rank: int,
        memory_pool: Tuple[int, int],
    ) -> "GraphWrapper":
        max_input_state = get_max_graph_state(device, adapter_layers)

        # WARNING: for some reason the SGMV kernel can hang if we don't use a power of 2
        # as the segment size. This is a workaround until we can figure out why.
        # Specifically, this issue has been observed with batch_size=96.
        # I suspect it is related to synchronization and the chunk size (256) used in the kernel.
        # But we need to investigate further.
        segment_size = next_pow_2(batch_size)

        adapter_weight_data = {}
        for layer_name, weight_data in max_input_state.adapter_data.data.items():
            tmp_expand_size = get_tmp_expand_size(segment_size)

            tmp_shrink = weight_data.rank_data[MAX_RANK].tmp_shrink
            if use_cutlass_shrink(max_rank):
                # cutlass shrink uses a custom temp buffer per rank
                tmp_shrink = tmp_shrink[:tmp_expand_size]

            adapter_weight_data[layer_name] = AdapterWeightData(
                lora_a={},
                lora_b={},
                adapter_index_configs={},
                rank_data={
                    max_rank: RankSegments(
                        rank=max_rank,
                        tmp_shrink=tmp_shrink,
                        tmp_expand=weight_data.rank_data[MAX_RANK].tmp_expand[:tmp_expand_size],
                        lora_a_ptr=weight_data.rank_data[MAX_RANK].lora_a_ptr[:segment_size],
                        lora_b_ptr=weight_data.rank_data[MAX_RANK].lora_b_ptr[:segment_size],
                        segment_starts=weight_data.rank_data[MAX_RANK].segment_starts[:segment_size],
                        segment_ends=weight_data.rank_data[MAX_RANK].segment_ends[:segment_size],
                    ),
                } if max_rank > 0 else {},
            )

        input_state = GraphState(
            input_ids=max_input_state.input_ids[:batch_size],
            position_ids=max_input_state.position_ids[:batch_size],
            block_tables=max_input_state.block_tables[:batch_size],
            slots=max_input_state.slots[:batch_size],
            input_lengths=max_input_state.input_lengths[:batch_size],
            adapter_data=AdapterBatchData(
                meta=AdapterBatchMetadata(
                    adapter_indices=max_input_state.adapter_data.meta.adapter_indices[:batch_size],
                    adapter_set=max_input_state.adapter_data.meta.adapter_set,
                    adapter_segments=max_input_state.adapter_data.meta.adapter_segments[:batch_size],
                    segment_indices=max_input_state.adapter_data.meta.segment_indices,
                ),
                data=adapter_weight_data,
            ),
        )

        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=memory_pool):  # noqa: SIM117
            output_states = model.forward(
                input_ids=input_state.input_ids,
                position_ids=input_state.position_ids,
                cu_seqlen_prefill=None,
                kv_cache=get_cache_manager().kv_cache,
                block_tables=input_state.block_tables,
                slots=input_state.slots,
                input_lengths=input_state.input_lengths,
                max_s=MAX_CONTEXT_LENGTH,
                adapter_data=input_state.adapter_data,
                lm_head_indices=None,
            )

        torch.cuda.synchronize(device)

        return GraphWrapper(
            graph, graph.pool(), input_state, output_states, model
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        adapter_data: AdapterBatchData,
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> None:
        pad_and_fill(self.input_state.input_ids, input_ids, 0)
        pad_and_fill(self.input_state.position_ids, position_ids, 0)
        pad_and_fill(self.input_state.slots, slots, SLOT_PAD_VALUE)
        pad_and_fill(self.input_state.input_lengths, input_lengths, 0)

        self.input_state.block_tables.zero_()
        self.input_state.block_tables[:block_tables.shape[0], :block_tables.shape[1]] = block_tables

        for layer_name, weight_data in self.input_state.adapter_data.data.items():
            if layer_name not in adapter_data.data:
                # zero out all the segments
                for rank_data in weight_data.rank_data.values():
                    rank_data.segment_starts.fill_(SEGMENT_PAD_VALUE)
                    rank_data.segment_ends.fill_(SEGMENT_PAD_VALUE)
                continue
            
            source_data = adapter_data.data[layer_name]
            dest_data = weight_data
            for rank, source_rank_data in source_data.rank_data.items():
                dest_rank_data = dest_data.rank_data[rank]

                pad_and_fill(dest_rank_data.lora_a_ptr, source_rank_data.lora_a_ptr, 0)
                pad_and_fill(dest_rank_data.lora_b_ptr, source_rank_data.lora_b_ptr, 0)

                pad_and_fill(dest_rank_data.segment_starts, source_rank_data.segment_starts, SEGMENT_PAD_VALUE)
                pad_and_fill(dest_rank_data.segment_ends, source_rank_data.segment_ends, SEGMENT_PAD_VALUE)
        
        self.graph.replay()

        return self.output_states[:input_ids.shape[0]]
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class GraphCache:
    def __init__(self, model: nn.Module, device: torch.device, adapter_layers: List[str]):
        self.model = model
        self.device = device
        self.adapter_layers = tuple(adapter_layers)
        self.memory_pool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.cache = {}

    def can_use_graph(
        self,
        batch: "FlashCausalLMBatch",
        adapter_data: AdapterBatchData,
    ) -> bool:
        ranks = adapter_data.ranks()
        nranks = len(ranks)
        max_rank = max(ranks) if len(ranks) > 0 else 0

        batch_size = batch.input_ids.shape[0]
        max_s = batch.max_seqlen

        # TODO(travis): allow using CUDA graphs with multi-rank batches
        return (
            torch.cuda.is_available()
            and batch_size <= MAX_BATCH_SIZE
            and max_s <= MAX_CONTEXT_LENGTH
            and max_rank <= MAX_RANK
            and nranks <= 1
            and max_rank in _allowed_ranks
        )
    
    def get_estimated_cache_memory(self) -> int:
        # Store off graphs into temporary cache to discard after estimation
        tmp_cache = {}
        pool = None

        # Use the largest batch size to overestimate memory overhead
        batch_size = CACHED_BATCH_SIZES[-1]

        samples = []
        for i, max_rank in enumerate(reversed(CACHED_MAX_RANKS)):
            torch.cuda.synchronize(self.device)
            free_memory_before, _ = torch.cuda.mem_get_info(self.device)
            
            key = (batch_size, max_rank)
            graph = GraphWrapper.trace(
                self.model,
                self.device,
                self.adapter_layers,
                batch_size,
                max_rank,
                pool,
            )
            tmp_cache[key] = graph
            pool = graph.memory_pool

            torch.cuda.synchronize(self.device)
            free_memory_after, _ = torch.cuda.mem_get_info(self.device)

            # Measure memory difference after tracing the graph,
            # discard first sample to account for global state initialization
            delta_memory = free_memory_before - free_memory_after
            if i > 0:
                samples.append(delta_memory)
            
            # Tracing all graphs can take a while, so limit the number of samples
            if len(samples) == MAX_SAMPLES:
                break
            
        # Estimate memory usage for all batch sizes and ranks
        ngraphs = len(CACHED_BATCH_SIZES) * len(CACHED_MAX_RANKS)
        per_graph_memory = median(samples)
        return ngraphs * per_graph_memory

    def warmup(self):
        ngraphs = len(CACHED_BATCH_SIZES) * len(CACHED_MAX_RANKS)
        pool = None
        with tqdm(total=ngraphs, desc="Trace CUDA graphs") as pbar:
            for batch_size in reversed(CACHED_BATCH_SIZES):
                pbar.set_postfix({'batch_size': batch_size})
                for max_rank in reversed(CACHED_MAX_RANKS):
                    key = (batch_size, max_rank)
                    graph = GraphWrapper.trace(
                        self.model,
                        self.device,
                        self.adapter_layers,
                        batch_size,
                        max_rank,
                        pool,
                    )
                    self.cache[key] = graph
                    pool = graph.memory_pool
                    pbar.update(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        adapter_data: AdapterBatchData,
        lm_head_indices: Optional[torch.Tensor] = None,
        **kwargs
    ) -> None:
        batch_size = get_cached_batch_size(input_ids.shape[0])
        max_rank = adapter_data.max_rank

        key = (batch_size, max_rank)
        if key not in self.cache:
            self.cache[key] = GraphWrapper.trace(
                self.model,
                self.device,
                self.adapter_layers,
                batch_size,
                max_rank,
                self.memory_pool,
            )
            
        output_states = self.cache[key].forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            input_lengths=input_lengths,
            max_s=max_s,
            adapter_data=adapter_data,
            lm_head_indices=lm_head_indices,
        )

        return output_states
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
