from typing import Tuple

import torch
from torch import nn

from lorax_server.models.types import Batch
from lorax_server.utils.lora import AdapterBatchData
from lorax_server.models.cache_manager import get_cache_manager


# TODO(travis): make this confgiurable by model / user
MAX_CONTEXT_LENGTH = 8192

SLOT_PAD_VALUE = -1


def get_cached_batch_size(batch_size: int) -> int:
    if batch_size == 1:
        return 1
    if batch_size == 2:
        return 2
    if batch_size <= 4:
        return 4
    return (batch_size + 7) // 8 * 8


class GraphWrapper:
    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        batch: Batch,
        adapter_data: AdapterBatchData,
        output_states: torch.Tensor,
        memory_pool: Tuple[int, int],
    ):
        self.graph = graph
        self.batch = batch
        self.adapter_data = adapter_data
        self.output_states = output_states
        self.memory_pool = memory_pool
        self.kv_cache = get_cache_manager().kv_cache
    
    def forward(self, batch: Batch, adapter_data: torch.Tensor) -> None:
        self.batch.copy_(batch)
        # self.adapter_data.copy_(adapter_data)
        # iterate over every list in kv_cache and every tuple in the list and copy the tensor data
        # into the tensor in the graph
        # batch_kv_cache = get_cache_manager().kv_cache
        # for i, kv in enumerate(self.kv_cache):
        #     for j, v in enumerate(kv):
        #         v.copy_(batch_kv_cache[i][j])

        # self.kv_cache.copy_(get_cache_manager().kv_cache)
        self.graph.replay()

        # batch.copy_(self.batch)

        return self.output_states.clone()
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    @staticmethod
    def trace(
        model: nn.Module,
        batch: Batch,
        adapter_data: torch.Tensor,
        memory_pool: Tuple[int, int],
        slots_buffer: torch.Tensor,
    ) -> Tuple["GraphWrapper", torch.Tensor]:
        torch.cuda.synchronize(model.device)

        batch = batch.clone()
        batch.slots = slots_buffer

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=memory_pool):  # noqa: SIM117
            output_states = model.forward(batch, adapter_data)

        torch.cuda.synchronize(model.device)

        return GraphWrapper(graph, batch, adapter_data, output_states, memory_pool)


class GraphCache:
    def __init__(self, model: nn.Module):
        self.model = model
        self.memory_pool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.cache = {}
        self.slots_buffer = torch.full((MAX_CONTEXT_LENGTH,), SLOT_PAD_VALUE, dtype=torch.int64, device=model.device)

    def forward(self, batch: Batch, adapter_data: AdapterBatchData) -> None:
        batch_size = get_cached_batch_size(len(batch))
        key = (batch_size, adapter_data.key())
        if key not in self.cache:
            print("cache miss")
            print(batch.input_ids)
            print(batch.position_ids)
            print(batch.slots)
            print(batch.block_tables_tensor)
            print(batch.input_lengths_tensor)
            print(batch.max_seqlen)
            self.cache[key] = GraphWrapper.trace(
                self.model,
                batch,
                adapter_data,
                self.memory_pool,
                self.slots_buffer
            )

            output_states = self.cache[key].forward(batch, adapter_data)

            print()
            print(output_states)

            # print("!!! REPLAY !!!")
            # print(batch.input_ids)
            # print(batch.position_ids)
            # print(batch.slots)
            # print(batch.block_tables_tensor)
            # print(batch.input_lengths_tensor)
            # print(batch.max_seqlen)
            # output_states, hidden_states = self.cache[key].forward(batch, adapter_data)
            # print()
            # print(output_states)
            # print(hidden_states, hidden_states.shape, hidden_states.float().norm())
        else:
            print("cache hit")
            # print(batch.input_ids)
            # print(batch.position_ids)
            # print(batch.slots)
            # print(batch.block_tables_tensor)
            # print(batch.input_lengths_tensor)
            # print(batch.max_seqlen)
            # output_states, hidden_states = self.model.forward(batch, adapter_data)
            # print(output_states)
            # print(hidden_states, hidden_states.shape, hidden_states.float().norm())

            print("!!! REPLAY !!!")
            print(batch.input_ids)
            print(batch.position_ids)
            print(batch.slots)
            print(batch.block_tables_tensor)
            print(batch.input_lengths_tensor)
            print(batch.max_seqlen)
            output_states = self.cache[key].forward(batch, adapter_data)
            print(output_states)
            # print(hidden_states, hidden_states.shape, hidden_states.float().norm())

        return output_states
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
