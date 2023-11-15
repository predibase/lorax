from dataclasses import dataclass
from typing import Dict, List, Set

import torch
from peft import LoraConfig
from torch.distributed import ProcessGroup

from text_generation_server.utils.weights import shard_on_dim


# Constants
Q_PROJ = "q_proj"
K_PROJ = "k_proj"
V_PROJ = "v_proj"
O_PROJ = "o_proj"


EMPTY_TENSOR = torch.tensor([])


@dataclass
class AdapterWeightData:
    lora_a_ptr: torch.Tensor
    lora_b_ptr: torch.Tensor
    lora_a: List[torch.Tensor]
    lora_b: List[torch.Tensor]

    r: Set[int]
    alpha: Set[int]
    adapter_index_configs: Dict[int, LoraConfig]

    @property
    def can_vectorize(self) -> bool:
        return len(self.r) == 1 and len(self.alpha) == 1
    
    def has_adapter(self, adapter_index: int) -> bool:
        return adapter_index in self.adapter_index_configs
    
    @property
    def rank(self) -> int:
        return next(iter(self.r))
    
    @property
    def scaling(self) -> float:
        alpha = next(iter(self.alpha))
        return alpha / self.rank
    
    def scaling_for_adapter(self, adapter_idx: int) -> float:
        return self.adapter_index_configs[adapter_idx].alpha / self.adapter_index_configs[adapter_idx].r


@dataclass
class AdapterBatchMetadata:
    adapter_indices: torch.Tensor
    adapter_set: Set[int]
    adapter_segments: torch.Tensor
    segment_indices: List[int]


@dataclass
class AdapterBatchData:
    meta: AdapterBatchMetadata
    data: Dict[str, AdapterWeightData]

    @staticmethod
    def from_meta(meta: AdapterBatchMetadata, weights: Dict[str, "BatchedLoraWeights"]) -> "AdapterBatchData":
        data = {}
        for k, v in weights.items():
            data[k] = v.get_data(meta)
        return AdapterBatchData(meta=meta, data=data)


class MergedLoraWeights:
    """LoRA weights for a single adapter merged across all layers."""

    def __init__(
        self,
        weights_a: List[torch.Tensor],
        weights_b: List[torch.Tensor],
        adapter_config: LoraConfig,
        process_group: ProcessGroup,
    ):
        # [num_layers, hidden_size, r]
        self.weights_a = shard_on_dim(torch.stack(weights_a), dim=0, process_group=process_group)

        # [num_layers, r, hidden_size]
        self.weights_b = shard_on_dim(torch.stack(weights_b), dim=0, process_group=process_group)

        self.adapter_config = adapter_config


class BatchedLoraWeights:
    """LoRA weights for multiple adapters."""

    def __init__(self):
        self.lora_weights: Dict[int, MergedLoraWeights] = {}

    def add_adapter(self, adapter_idx: int, weights: MergedLoraWeights):
        self.lora_weights[adapter_idx] = weights

    def remove_adapter(self, adapter_idx: int):
        del self.lora_weights[adapter_idx]

    def get_data(self, segment_indices: List[int]) -> AdapterWeightData:
        device = list(self.lora_weights.values())[0].weights_a.device
        lora_a = [
            self.lora_weights[idx].weights_a 
            if idx in self.lora_weights else None 
            for idx in segment_indices
        ]
        lora_a_ptr = torch.tensor(
            [w.data_ptr() if w is not None else EMPTY_TENSOR.data_ptr() for w in lora_a],
            dtype=torch.int64,
            device=device,
        )
        lora_b = [
            self.lora_weights[idx].weights_b 
            if idx in self.lora_weights else None 
            for idx in segment_indices
        ]
        lora_b_ptr = torch.tensor(
            [w.data_ptr() if w is not None else EMPTY_TENSOR.data_ptr() for w in lora_b],
            dtype=torch.int64,
            device=device,
        )

        r = set([
            self.lora_weights[idx].adapter_config.r 
            if idx in self.lora_weights else None 
            for idx in segment_indices
        ])
        alpha = set([
            self.lora_weights[idx].adapter_config.alpha 
            if idx in self.lora_weights else None 
            for idx in segment_indices
        ])
        adapter_index_configs = {
            idx: self.lora_weights[idx].adapter_config 
            if idx in self.lora_weights else None 
            for idx in segment_indices
        }

        return AdapterWeightData(
            lora_a_ptr=lora_a_ptr, 
            lora_b_ptr=lora_b_ptr, 
            lora_a=lora_a, 
            lora_b=lora_b,
            r=r,
            alpha=alpha,
            adapter_index_configs=adapter_index_configs,
        )
