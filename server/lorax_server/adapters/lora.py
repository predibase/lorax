from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import torch
from peft import LoraConfig as _LoraConfig
from torch.distributed import ProcessGroup

from lorax_server.adapters.config import AdapterConfig, ModuleMap
from lorax_server.adapters.weights import AdapterWeights, BatchAdapterWeights
from lorax_server.utils.adapter import get_scaling_factor
from lorax_server.adapters.weights import AdapterBatchMetadata
from lorax_server.utils.sgmv import MAX_RANK_CUSTOM, get_tmp_tensors, orient_for_rank, pad_rank

if TYPE_CHECKING:
    from lorax_server.models.model import Model

EMPTY_TENSOR = torch.tensor([])


@dataclass
class LoraConfig(AdapterConfig):
    r: int
    target_modules: Optional[Union[List[str], str]]
    fan_in_fan_out: bool
    lora_alpha: int
    use_rslora: bool

    def map_weights_for_model(
        self, adapter_weights: Dict, weight_names: Tuple[str],
    ) -> Tuple[ModuleMap, Set[str]]:
        adapter_weight_names = set()
        module_map = {}
        for weight_name in weight_names:
            lora_a_name = f"base_model.model.{weight_name}.lora_A.weight"
            lora_b_name = f"base_model.model.{weight_name}.lora_B.weight"
            if lora_a_name not in adapter_weights or lora_b_name not in adapter_weights:
                continue
            
            module_map[weight_name] = {
                "lora_A": (adapter_weights[lora_a_name], lora_a_name),
                "lora_B": (adapter_weights[lora_b_name], lora_b_name),
            }
            adapter_weight_names.add(lora_a_name)
            adapter_weight_names.add(lora_b_name)
        return module_map, adapter_weight_names
    
    def load_batched_adapter_weights(
        self, 
        model: "Model",
        module_map: Dict[str, Dict], 
        layer_type: str,
        unused_weight_names: Set[str],
    ) -> AdapterWeights:
        return LoraWeights.load(
            self,
            model,
            module_map,
            layer_type,
            unused_weight_names,
        )
    
    @classmethod
    def load(cls, adapter_id: str, api_token: str) -> "LoraConfig":
        hf_config = _LoraConfig.from_pretrained(adapter_id, token=api_token)
        return cls(
            base_model_name_or_path=hf_config.base_model_name_or_path,
            r=hf_config.r,
            target_modules=hf_config.target_modules,
            fan_in_fan_out=hf_config.fan_in_fan_out,
            lora_alpha=hf_config.lora_alpha,
            use_rslora=hf_config.use_rslora if hasattr(hf_config, "use_rslora") else False,
        )


class LoraWeights(AdapterWeights):
    """LoRA weights for a single adapter merged across all layers."""

    def __init__(
        self,
        weights_a: List[torch.Tensor],
        weights_b: List[torch.Tensor],
        adapter_config: LoraConfig,
    ):
        self.lora_a_r = weights_a[0].size(1) if len(weights_a) > 0 else 1
        self.lora_b_r = weights_b[0].size(0) if len(weights_a) > 0 else 1

        # [num_layers, hidden_size, r]
        weights_a = [
            orient_for_rank(w, w.size(1)).contiguous()
            for w in weights_a
        ]
        self.weights_a = torch.stack(weights_a)

        # [num_layers, r, hidden_size]
        self.weights_b = torch.stack(weights_b)

        self.adapter_config = adapter_config
    
    @classmethod
    def get_batch_type(cls) -> BatchAdapterWeights:
        return BatchLoraWeights

    @classmethod
    def load(
        cls, 
        config: LoraConfig,
        model: "Model",
        module_map: Dict[str, Dict], 
        layer_type: str,
        unused_weight_names: Set[str],
    ) -> AdapterWeights:
        nlayers = model.get_num_layers_for_type(layer_type)
        lora_a_list = [None] * nlayers
        lora_b_list = [None] * nlayers

        for layer_id in range(nlayers):
            key = (layer_id, layer_type)
            weight_name, layer = model.target_to_layer[key]

            base_weight = layer.base_layer.linear.weight
            base_device = base_weight.device

            if weight_name not in module_map:
                # There is no LoRA weight for this layer type in the adapter
                return

            lora_a, lora_a_name = module_map[weight_name]["lora_A"]
            lora_a = lora_a.to(base_device, model.dtype)

            lora_b, lora_b_name = module_map[weight_name]["lora_B"]
            lora_b = lora_b.to(base_device, model.dtype)

            scale = get_scaling_factor(
                config.lora_alpha,
                config.r,
                uses_rslora=config.use_rslora,
            )

            unused_weight_names.discard(lora_a_name)
            unused_weight_names.discard(lora_b_name)

            # Merge scaling factor into lora_b due to associativity of matrix multiplication:
            # (A * B) * C = A * (B * C)
            lora_a_list[layer_id] = lora_a.transpose(0, 1)
            lora_b_list[layer_id] = lora_b.transpose(0, 1) * scale

        # pad lora ranks to be compatible with sgmv
        lora_a_list = [pad_rank(w, dim=1, world_size=model.world_size) for w in lora_a_list]
        lora_b_list = [pad_rank(w, dim=0, world_size=model.world_size) for w in lora_b_list]

        if lora_a_list:
            # update rank if it was padded
            padded_rank = lora_a_list[0].size(1)
            config.r = padded_rank

        return LoraWeights(
            *model.shard_lora_weights(lora_a_list, lora_b_list, layer_type), config,
        )


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
class BatchLoraWeights:
    lora_a: Dict[int, torch.Tensor]
    lora_b: Dict[int, torch.Tensor]
    adapter_index_configs: Dict[int, LoraConfig]
    rank_data: Dict[int, RankSegments]
    
    def has_adapter(self, adapter_index: int) -> bool:
        return adapter_index in self.adapter_index_configs
    
    def can_vectorize(self, pg: ProcessGroup) -> bool:
        return all(
            rank_data.rank // pg.size() <= MAX_RANK_CUSTOM
            for rank_data in self.rank_data.values()
        )

    @classmethod
    def load(self, adapter_weights: Dict[int, LoraWeights], meta: AdapterBatchMetadata) -> "BatchLoraWeights":
        first_weights = list(adapter_weights.values())[0]
        device = first_weights.weights_a.device
        segment_indices = meta.segment_indices

        lora_a = {
            idx: adapter_weights[idx].weights_a
            for idx in segment_indices
            if idx in adapter_weights
        }
        lora_a_ptr = torch.tensor(
            [
                (
                    adapter_weights[idx].weights_a.data_ptr()
                    if idx in adapter_weights 
                    else EMPTY_TENSOR.data_ptr()
                ) for idx in segment_indices
            ],
            dtype=torch.int64,
            device=device,
        )
        lora_b = {
            idx: adapter_weights[idx].weights_b
            for idx in segment_indices
            if idx in adapter_weights
        }
        lora_b_ptr = torch.tensor(
            [
                (
                    adapter_weights[idx].weights_b.data_ptr() 
                    if idx in adapter_weights 
                    else EMPTY_TENSOR.data_ptr()
                ) for idx in segment_indices
            ],
            dtype=torch.int64,
            device=device,
        )

        adapter_index_configs = {
            idx: adapter_weights[idx].adapter_config
            for idx in segment_indices
            if idx in adapter_weights
        }

        rank_indices = defaultdict(list)
        for segment_idx, adapter_idx in enumerate(segment_indices):
            if adapter_idx not in adapter_weights:
                continue
            rank_indices[adapter_weights[adapter_idx].lora_a_r].append(segment_idx)

        rank_data = {}
        for rank, indices in rank_indices.items():
            lora_a_ptr_indices = lora_a_ptr[indices]
            tmp_shrink, tmp_expand = get_tmp_tensors(
                lora_a_ptr_indices.size(0),
                rank,
                device
            )

            rank_data[rank] = RankSegments(
                rank=rank,
                tmp_shrink=tmp_shrink,
                tmp_expand=tmp_expand,
                lora_a_ptr=lora_a_ptr_indices,
                lora_b_ptr=lora_b_ptr[indices],
                segment_starts=meta.adapter_segments[indices],
                segment_ends=meta.adapter_segments[[i+1 for i in indices]],
            )

        return BatchLoraWeights(
            lora_a=lora_a, 
            lora_b=lora_b,
            adapter_index_configs=adapter_index_configs,
            rank_data=rank_data,
        )
