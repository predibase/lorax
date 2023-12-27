from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

import torch
from peft import LoraConfig
from torch.distributed import ProcessGroup

from lorax_server.utils.sgmv import MIN_SGMV_RANK, orient_for_rank
from lorax_server.utils.weights import shard_on_dim


# Constants
Q_PROJ = "q_proj"
K_PROJ = "k_proj"
V_PROJ = "v_proj"
O_PROJ = "o_proj"

GATE_PROJ = "gate_proj"
UP_PROJ = "up_proj"
DOWN_PROJ = "down_proj"

LM_HEAD = "lm_head"

EMPTY_TENSOR = torch.tensor([])


@dataclass
class RankSegments:
    rank: int
    lora_a_ptr: torch.Tensor
    lora_b_ptr: torch.Tensor
    segment_starts: torch.Tensor
    segment_ends: torch.Tensor

    def copy_(self, data: "RankSegments") -> None:
        self.rank = data.rank
        self.lora_a_ptr.copy_(data.lora_a_ptr)
        self.lora_b_ptr.copy_(data.lora_b_ptr)
        self.segment_starts.copy_(data.segment_starts)
        self.segment_ends.copy_(data.segment_ends)


@dataclass
class AdapterWeightData:
    lora_a: Dict[int, torch.Tensor]
    lora_b: Dict[int, torch.Tensor]
    adapter_index_configs: Dict[int, LoraConfig]
    rank_data: Dict[int, RankSegments]
    
    def has_adapter(self, adapter_index: int) -> bool:
        return adapter_index in self.adapter_index_configs
    
    def can_vectorize(self, pg: ProcessGroup) -> bool:
        return all(
            rank_data.rank // pg.size() >= MIN_SGMV_RANK
            for rank_data in self.rank_data.values()
        )
    

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
            if v.is_empty():
                continue
            data[k] = v.get_data(meta)
        return AdapterBatchData(meta=meta, data=data)
    
    def copy_(self, data: "AdapterBatchData") -> None:
        self.meta = data.meta

        for layer_name, source_data in data.data.items():
            dest_data = self.data[layer_name]
            for r, source_rank_data in source_data.rank_data.items():
                dest_rank_data = dest_data.rank_data[r]
                dest_rank_data.copy_(source_rank_data)
    
    def key(self) -> str:
        if len(self.data) == 0:
            return ""
        
        nsegments = self.meta.adapter_segments.shape[0]
        layers_str = "-".join(sorted(self.data.keys()))
        layer = next(iter(self.data.values()))
        rank_str = "-".join(sorted([str(r) for r in layer.rank_data.keys()]))
        return f"{nsegments}-{layers_str}-{rank_str}"
    
    def ranks(self) -> Set[int]:
        return set(
            rank_data.rank
            for layer in self.data.values()
            for rank_data in layer.rank_data.values()
        )
    
    @property
    def max_r(self) -> int:
        if len(self.data) == 0:
            return 0
        
        return max(
            max(rank_data.rank for rank_data in layer.rank_data.values())
            for layer in self.data.values()
        )


class MergedLoraWeights:
    """LoRA weights for a single adapter merged across all layers."""

    def __init__(
        self,
        weights_a: List[torch.Tensor],
        weights_b: List[torch.Tensor],
        adapter_config: LoraConfig,
    ):
        # [num_layers, hidden_size, r]
        weights_a = [
            orient_for_rank(w, adapter_config.r).contiguous()
            for w in weights_a
        ]
        self.weights_a = torch.stack(weights_a)

        # [num_layers, r, hidden_size]
        self.weights_b = torch.stack(weights_b)

        self.adapter_config = adapter_config


class BatchedLoraWeights:
    """LoRA weights for multiple adapters."""

    def __init__(self):
        self.lora_weights: Dict[int, MergedLoraWeights] = {}

    def add_adapter(self, adapter_idx: int, weights: MergedLoraWeights):
        self.lora_weights[adapter_idx] = weights

    def remove_adapter(self, adapter_idx: int):
        if adapter_idx not in self.lora_weights:
            return
        del self.lora_weights[adapter_idx]

    def is_empty(self) -> bool:
        return len(self.lora_weights) == 0

    def get_data(self, meta: AdapterBatchMetadata) -> AdapterWeightData:
        """
        Get the adapter weight data for a given metadata.

        Args:
            meta (AdapterBatchMetadata): The metadata for the adapter batch.

        Returns:
            AdapterWeightData: The adapter weight data.

        """
        device = list(self.lora_weights.values())[0].weights_a.device
        segment_indices = meta.segment_indices

        lora_a = {
            idx: self.lora_weights[idx].weights_a
            for idx in segment_indices
            if idx in self.lora_weights
        }
        lora_a_ptr = torch.tensor(
            [
                (
                    self.lora_weights[idx].weights_a.data_ptr()
                    if idx in self.lora_weights 
                    else EMPTY_TENSOR.data_ptr()
                ) for idx in segment_indices
            ],
            dtype=torch.int64,
            device=device,
        )
        lora_b = {
            idx: self.lora_weights[idx].weights_b
            for idx in segment_indices
            if idx in self.lora_weights
        }
        lora_b_ptr = torch.tensor(
            [
                (
                    self.lora_weights[idx].weights_b.data_ptr() 
                    if idx in self.lora_weights 
                    else EMPTY_TENSOR.data_ptr()
                ) for idx in segment_indices
            ],
            dtype=torch.int64,
            device=device,
        )

        adapter_index_configs = {
            idx: self.lora_weights[idx].adapter_config
            for idx in segment_indices
            if idx in self.lora_weights
        }

        rank_indices = defaultdict(list)
        for segment_idx, adapter_idx in enumerate(segment_indices):
            if adapter_idx not in self.lora_weights:
                continue
            rank_indices[self.lora_weights[adapter_idx].adapter_config.r].append(segment_idx)

        rank_data = {}
        for rank, indices in rank_indices.items():
            rank_data[rank] = RankSegments(
                rank=rank,
                lora_a_ptr=lora_a_ptr[indices],
                lora_b_ptr=lora_b_ptr[indices],
                segment_starts=meta.adapter_segments[indices],
                segment_ends=meta.adapter_segments[[i+1 for i in indices]],
            )

        return AdapterWeightData(
            lora_a=lora_a, 
            lora_b=lora_b,
            adapter_index_configs=adapter_index_configs,
            rank_data=rank_data,
        )
