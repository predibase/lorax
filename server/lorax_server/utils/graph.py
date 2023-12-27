from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple
import numpy as np

import torch
from torch import nn

from lorax_server.utils.lora import AdapterBatchData, AdapterBatchMetadata, AdapterWeightData, RankSegments
from lorax_server.models.cache_manager import get_cache_manager, BLOCK_SIZE


# TODO(travis): make this configurable by model / user
MAX_BATCH_SIZE = 256
MAX_CONTEXT_LENGTH = 8192
MAX_RANK = 128

SLOT_PAD_VALUE = -1


def get_cached_batch_size(batch_size: int) -> int:
    if batch_size == 1:
        return 1
    if batch_size == 2:
        return 2
    if batch_size <= 4:
        return 4
    return (batch_size + 7) // 8 * 8


@dataclass
class GraphState:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    block_tables: torch.Tensor
    slots: torch.Tensor
    input_lengths: torch.Tensor
    adapter_data: AdapterBatchData


@lru_cache(maxsize=1)
def get_max_graph_state(model: nn.Module) -> GraphState:
    device = model.device
    
    # TODO(travis): cite vllm
    max_num_blocks = (MAX_CONTEXT_LENGTH + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables_arr = np.zeros((MAX_BATCH_SIZE, max_num_blocks), dtype=np.int32)
    block_tables = torch.from_numpy(block_tables_arr).to(device=device)

    input_ids = torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int64, device=device)
    position_ids = torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int32, device=device)
    slots = torch.full((MAX_BATCH_SIZE,), SLOT_PAD_VALUE, dtype=torch.int64, device=device)
    input_lengths = torch.ones((MAX_BATCH_SIZE,), dtype=torch.int32, device=device)

    adapter_weight_data = {}
    for layer_name in model.adapter_layers:
        adapter_weight_data[layer_name] = AdapterWeightData(
            lora_a={},
            lora_b={},
            adapter_index_configs={},
            rank_data={
                MAX_RANK: RankSegments(
                    rank=MAX_RANK,
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
    ):
        self.graph = graph
        self.memory_pool = memory_pool
        self.input_state = input_state
        self.output_states = output_states
    
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
        self.input_state.input_ids[:input_ids.shape[0]] = input_ids
        self.input_state.position_ids[:position_ids.shape[0]] = position_ids
        self.input_state.block_tables[:block_tables.shape[0], :block_tables.shape[1]] = block_tables
        self.input_state.slots[:slots.shape[0]] = slots
        self.input_state.input_lengths[:input_lengths.shape[0]] = input_lengths

        for layer_name, weight_data in self.input_state.adapter_data.data.items():
            if layer_name not in adapter_data.data:
                continue
            
            source_data = adapter_data.data[layer_name]
            dest_data = weight_data
            for rank, source_rank_data in source_data.rank_data.items():
                dest_rank_data = dest_data.rank_data[rank]
                dest_rank_data.lora_a_ptr[:source_rank_data.lora_a_ptr.shape[0]] = source_rank_data.lora_a_ptr
                dest_rank_data.lora_b_ptr[:source_rank_data.lora_b_ptr.shape[0]] = source_rank_data.lora_b_ptr
                dest_rank_data.segment_starts[:source_rank_data.segment_starts.shape[0]] = source_rank_data.segment_starts
                dest_rank_data.segment_ends[:source_rank_data.segment_ends.shape[0]] = source_rank_data.segment_ends

                # pad remainder of segments with zeros
                dest_rank_data.lora_a_ptr[source_rank_data.lora_a_ptr.shape[0]:] = 0
                dest_rank_data.lora_b_ptr[source_rank_data.lora_b_ptr.shape[0]:] = 0
        
        self.graph.replay()

        return self.output_states.clone()
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    @staticmethod
    def trace(
        model: nn.Module,
        batch_size: int,
        max_rank: int,
        memory_pool: Tuple[int, int],
    ) -> Tuple["GraphWrapper", torch.Tensor]:
        torch.cuda.synchronize(model.device)

        max_input_state = get_max_graph_state(model)

        adapter_weight_data = {}
        for layer_name, weight_data in max_input_state.adapter_data.data.items():
            adapter_weight_data[layer_name] = AdapterWeightData(
                lora_a={},
                lora_b={},
                adapter_index_configs={},
                rank_data={
                    max_rank: RankSegments(
                        rank=max_rank,
                        lora_a_ptr=weight_data.rank_data[max_rank].lora_a_ptr[:batch_size],
                        lora_b_ptr=weight_data.rank_data[max_rank].lora_b_ptr[:batch_size],
                        segment_starts=weight_data.rank_data[max_rank].segment_starts[:batch_size],
                        segment_ends=weight_data.rank_data[max_rank].segment_ends[:batch_size],
                    ),
                },
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

        torch.cuda.synchronize(model.device)

        return GraphWrapper(
            graph, memory_pool, input_state, output_states
        )


class GraphCache:
    def __init__(self, model: nn.Module):
        self.model = model
        self.memory_pool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.cache = {}

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
        batch_size = get_cached_batch_size(input_ids.shape[0])

        ranks = adapter_data.ranks()
        max_rank = max(ranks) if len(ranks) > 0 else 0

        key = (batch_size, max_rank)
        if key not in self.cache:
            print("cache miss")
            self.cache[key] = GraphWrapper.trace(
                self.model,
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
            print(output_states)
        else:
            print("cache hit")
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
            print(output_states)

        return output_states
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
