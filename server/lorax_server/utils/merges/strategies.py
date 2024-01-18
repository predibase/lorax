from abc import ABC
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from peft import LoraConfig

from lorax_server.utils.merges.utils import calculate_majority_sign_mask, disjoint_merge, prune
from lorax_server.utils.adapter import ModuleMap


def _apply_weights(tensors: List[torch.Tensor], w: torch.Tensor) -> torch.Tensor:
    t = torch.stack(tensors, dim=0)

    # element-wise weighting of each task tensor
    # need to unsqueeze weights to match task tensor dimensions
    # for multiplication to apply element-wise
    while len(t.shape) > len(w.shape):
        w = w.unsqueeze(-1)
    return t * w


class MergeStrategy(ABC):
    def merge(self, task_tensors: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class LinearMerge(MergeStrategy):
    def merge(self, task_tensors: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        weighted_task_tensors = _apply_weights(task_tensors, weights)
        return weighted_task_tensors.sum(dim=0)


class TiesMerge(MergeStrategy):
    def __init__(self, density: float, majority_sign_method: str = "total"):
        self.density = density
        self.majority_sign_method = majority_sign_method
    
    def merge(self, task_tensors: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        # sparsify
        task_tensors = [prune(tensor, self.density, method="magnitude") for tensor in task_tensors]
        weighted_task_tensors = _apply_weights(task_tensors, weights)
        
        # elect sign
        majority_sign_mask = calculate_majority_sign_mask(weighted_task_tensors, method=self.majority_sign_method)
        
        # disjoint merge
        return disjoint_merge(weighted_task_tensors, majority_sign_mask)


class DareLinearMerge(MergeStrategy):
    def __init__(self, density: float):
        self.density = density
    
    def merge(self, task_tensors: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        # sparsify
        task_tensors = [prune(tensor, self.density, method="random", rescale=True) for tensor in task_tensors]
        weighted_task_tensors = _apply_weights(task_tensors, weights)
        return weighted_task_tensors.sum(dim=0)


class DareTiesMerge(MergeStrategy):
    def __init__(self, density: float, majority_sign_method: str = "total"):
        self.density = density
        self.majority_sign_method = majority_sign_method

    def merge(self, task_tensors: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        # sparsify
        task_tensors = [prune(tensor, self.density, method="random", rescale=True) for tensor in task_tensors]
        weighted_task_tensors = _apply_weights(task_tensors, weights)
        
        # elect sign
        majority_sign_mask = calculate_majority_sign_mask(weighted_task_tensors, method=self.majority_sign_method)
        
        # disjoint merge
        mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
        return mixed_task_tensors


strategy_registry = {
    "linear": LinearMerge,
    "ties": TiesMerge,
    "dare_linear": DareLinearMerge,
    "dare_ties": DareTiesMerge,
}


def merge_adapters(
    adapters: List[Tuple[ModuleMap, LoraConfig]],
    merge_config: Dict,
) -> Tuple[ModuleMap, LoraConfig]:
    merge_config = merge_config.copy()
    strategy_name = merge_config.pop("strategy")
    merge_strategy = strategy_registry[strategy_name](**merge_config)

    module_maps = defaultdict(dict)
    lora_configs = []

    for module_map, lora_config in adapters:
        
        for weight_name, weights in module_map.items():
            for k, (param, param_name) in weights.items():
                module_maps[weight_name][k] = (param, param_name)
        
        lora_configs.append(lora_config)


