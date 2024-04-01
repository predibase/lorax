from abc import ABC, abstractclassmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Type

import torch

from lorax_server.adapters.types import LORA


@dataclass
class AdapterBatchMetadata:
    # [batch_size]
    adapter_indices: torch.Tensor

    # [num_adapters]
    adapter_set: Set[int]

    # [num_segments + 1]
    adapter_segments: torch.Tensor

    # [num_segments]
    # maps from segment index to adapter index, i.e.:
    # segment_indices[s] == adapter_indices[i]
    segment_indices: List[int]


class AdapterWeights(ABC):
    @abstractclassmethod
    def get_batch_type(cls) -> "BatchAdapterWeights":
        pass

    @property
    def speculative_tokens(self) -> int:
        return 0


class BatchAdapterWeights(ABC):
    @abstractclassmethod
    def has_adapter(self, adapter_index: int) -> bool:
        pass

    @abstractclassmethod
    def key(cls) -> str:
        pass

    @abstractclassmethod
    def load(cls, adapter_weights: Dict[int, AdapterWeights], meta: "AdapterBatchMetadata") -> "BatchAdapterWeights":
        pass


class LayerAdapterWeights:
    """Adapter weights that apply to a particular layer."""

    def __init__(self):
        self.adapter_weights: Dict[int, AdapterWeights] = {}

    def add_adapter(self, adapter_idx: int, weights: AdapterWeights):
        self.adapter_weights[adapter_idx] = weights

    def remove_adapter(self, adapter_idx: int):
        if adapter_idx not in self.adapter_weights:
            return
        del self.adapter_weights[adapter_idx]
    
    @property
    def max_speculative_tokens(self) -> int:
        return max(
            adapter_weights.speculative_tokens
            for adapter_weights in self.adapter_weights.values()
        )

    def is_empty(self) -> bool:
        return len(self.adapter_weights) == 0

    def get_data(self, meta: AdapterBatchMetadata) -> Dict[str, BatchAdapterWeights]:
        # bucket adapters by batch class
        adapter_batch_types: Dict[Type[BatchAdapterWeights], Dict[int, AdapterWeights]] = defaultdict(dict)
        for adapter_index, adapter_weights in self.adapter_weights.items():
            adapter_batch_types[adapter_weights.get_batch_type()][adapter_index] = adapter_weights
        
        batch_data = {}
        for batch_type, adapter_weights in adapter_batch_types.items():
            batch_data[batch_type.key()] = batch_type.load(adapter_weights, meta)
        return batch_data


@dataclass
class AdapterBatchData:
    meta: AdapterBatchMetadata

    # layer type -> adapter type -> batch weight data
    data: Dict[str, Dict[str, BatchAdapterWeights]]

    @staticmethod
    def from_meta(meta: AdapterBatchMetadata, weights: Dict[str, LayerAdapterWeights]) -> "AdapterBatchData":
        data = {}
        for k, v in weights.items():
            if v.is_empty():
                continue
            data[k] = v.get_data(meta)
        return AdapterBatchData(meta=meta, data=data)
    
    def ranks(self) -> Set[int]:
        # TODO(travis): refactor to be less coupled to lora implementation
        return set(
            rank_data.rank
            for layer_data in self.data.values()
            for rank_data in layer_data.get(LORA, []).rank_data.values()
        )
    
    @property
    def max_rank(self) -> int:
        ranks = self.ranks()
        return max(ranks) if len(ranks) > 0 else 0
