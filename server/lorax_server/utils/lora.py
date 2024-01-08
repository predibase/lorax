from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

import torch
from peft import LoraConfig
from torch.distributed import ProcessGroup

from lorax_server.utils.sgmv import MIN_SGMV_RANK, get_tmp_tensors, orient_for_rank


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
    tmp_shrink: torch.Tensor
    tmp_expand: torch.Tensor
    lora_a_ptr: torch.Tensor
    lora_b_ptr: torch.Tensor
    segment_starts: torch.Tensor
    segment_ends: torch.Tensor


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
    
    def ranks(self) -> Set[int]:
        return set(
            rank_data.rank
            for layer in self.data.values()
            for rank_data in layer.rank_data.values()
        )
    
    @property
    def max_rank(self) -> int:
        ranks = self.ranks()
        return max(ranks) if len(ranks) > 0 else 0


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
        first_weights = list(self.lora_weights.values())[0]
        device = first_weights.weights_a.device
        dtype = first_weights.weights_a.dtype
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
            lora_a_ptr_indices = lora_a_ptr[indices]
            tmp_shrink, tmp_expand = get_tmp_tensors(lora_a_ptr_indices.size(0), rank, device)

            rank_data[rank] = RankSegments(
                rank=rank,
                tmp_shrink=tmp_shrink,
                tmp_expand=tmp_expand,
                lora_a_ptr=lora_a_ptr_indices,
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
