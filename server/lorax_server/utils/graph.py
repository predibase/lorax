from typing import Tuple

import torch
from torch import nn

from lorax_server.models.types import Batch
from lorax_server.utils.lora import AdapterBatchData


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
    
    def forward(self, batch: Batch, adapter_data: torch.Tensor) -> None:
        self.batch.copy_(batch)
        self.adapter_data.copy_(adapter_data)
        self.graph.replay()
        return self.output_states
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    @staticmethod
    def trace(
        model: nn.Module,
        batch: Batch,
        adapter_data: torch.Tensor,
        memory_pool: Tuple[int, int],
    ) -> Tuple["GraphWrapper", torch.Tensor]:
        torch.cuda.synchronize(model.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=memory_pool):  # noqa: SIM117
            output_states = model.forward(batch, adapter_data)
        
        torch.cuda.synchronize(model.device)

        return GraphWrapper(graph, batch, adapter_data, output_states, memory_pool), output_states


class GraphCache:
    def __init__(self, model: nn.Module):
        self.model = model
        self.memory_pool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.cache = {}

    def forward(self, batch: Batch, adapter_data: AdapterBatchData) -> None:
        key = (len(batch), adapter_data.key())
        if key not in self.cache:
            print("cache miss")
            self.cache[key], output_states = GraphWrapper.trace(
                self.model,
                batch,
                adapter_data,
                self.memory_pool,
            )
        else:
            print("cache hit")
            output_states = self.cache[key].forward(batch, adapter_data)
        return output_states
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
