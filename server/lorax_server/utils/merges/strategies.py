from abc import ABC
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Tuple, Type

import torch
from peft import LoraConfig

from lorax_server.pb.generate_pb2 import (
    AdapterParameters, 
    MajoritySignMethod as MajoritySignMethodEnum,
    MergeStrategy as MergeStrategyEnum,
)
from lorax_server.utils.merges.utils import calculate_majority_sign_mask, disjoint_merge, prune

if TYPE_CHECKING:
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
    def merge(self, task_tensors: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()


class LinearMerge(MergeStrategy):
    def __init__(self, weights: torch.Tensor, **kwargs):
        self.weights = weights

    def merge(self, task_tensors: List[torch.Tensor]) -> torch.Tensor:
        weighted_task_tensors = _apply_weights(task_tensors, self.weights)
        return weighted_task_tensors.sum(dim=0)


class TiesMerge(MergeStrategy):
    def __init__(self, weights: torch.Tensor, density: float, majority_sign_method: str = "total", **kwargs):
        self.weights = weights
        self.density = density
        self.majority_sign_method = majority_sign_method
    
    def merge(self, task_tensors: List[torch.Tensor]) -> torch.Tensor:
        # sparsify
        task_tensors = [prune(tensor, self.density, method="magnitude") for tensor in task_tensors]
        weighted_task_tensors = _apply_weights(task_tensors, self.weights)
        
        # elect sign
        majority_sign_mask = calculate_majority_sign_mask(weighted_task_tensors, method=self.majority_sign_method)
        
        # disjoint merge
        return disjoint_merge(weighted_task_tensors, majority_sign_mask)


class DareLinearMerge(MergeStrategy):
    def __init__(self, weights: torch.Tensor, density: float, **kwargs):
        self.weights = weights
        self.density = density
    
    def merge(self, task_tensors: List[torch.Tensor]) -> torch.Tensor:
        # sparsify
        task_tensors = [prune(tensor, self.density, method="random", rescale=True) for tensor in task_tensors]
        weighted_task_tensors = _apply_weights(task_tensors, self.weights)
        return weighted_task_tensors.sum(dim=0)


class DareTiesMerge(MergeStrategy):
    def __init__(self, weights: torch.Tensor, density: float, majority_sign_method: str = "total", **kwargs):
        self.weights = weights
        self.density = density
        self.majority_sign_method = majority_sign_method

    def merge(self, task_tensors: List[torch.Tensor]) -> torch.Tensor:
        # sparsify
        task_tensors = [prune(tensor, self.density, method="random", rescale=True) for tensor in task_tensors]
        weighted_task_tensors = _apply_weights(task_tensors, self.weights)
        
        # elect sign
        majority_sign_mask = calculate_majority_sign_mask(weighted_task_tensors, method=self.majority_sign_method)
        
        # disjoint merge
        mixed_task_tensors = disjoint_merge(weighted_task_tensors, majority_sign_mask)
        return mixed_task_tensors


strategy_registry: Dict[str, Type[MergeStrategy]] = {
    "linear": LinearMerge,
    "ties": TiesMerge,
    "dare_linear": DareLinearMerge,
    "dare_ties": DareTiesMerge,
}


def merge_adapters(
    adapters: List[Tuple["ModuleMap", LoraConfig]],
    merge_params: AdapterParameters,
) -> Tuple["ModuleMap", LoraConfig]:
    strategy_name = MergeStrategyEnum.Name(merge_params.merge_strategy).lower()

    weights = merge_params.weights
    if not weights:
        weights = torch.ones(len(adapters))
    else:
        weights = torch.tensor(weights)

    merge_config = {
        "density": merge_params.density,
        "majority_sign_method": MajoritySignMethodEnum.Name(merge_params.majority_sign_method).lower(),
    }
    merge_strategy = strategy_registry[strategy_name](weights=weights, **merge_config)

    module_maps: Dict[str, Dict[str, Dict[str, List[torch.Tensor]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    lora_configs = []

    # input is list of (module_map, lora_config) tuples
    # convert into dict[k][param_name] -> list of tensors
    for module_map, lora_config in adapters:
        for weight_name, data in module_map.items():
            for k, (param_data, param_name) in data.items():
                module_maps[weight_name][k][param_name].append(param_data)
        lora_configs.append(lora_config)

    # validate lora configs are compatible
    _validate_lora_configs(lora_configs)

    # merge tensors for each module such that we have a single ModuleMap:
    # dict[k] -> merged tensor
    merged_module_map: "ModuleMap" = defaultdict(dict)
    for weight_name, data in module_maps.items():
        for k, param_data in data.items():
            for param_name, tensors in param_data.items():
                merged_tensor = merge_strategy.merge(tensors)
                merged_module_map[weight_name][k] = (merged_tensor, param_name)

    # merge lora configs
    merged_lora_config = _merge_lora_configs(lora_configs)

    return merged_module_map, merged_lora_config


def _validate_lora_configs(lora_configs: List[LoraConfig]):
    # check that all configs have the same rank
    ranks = set(lora_config.r for lora_config in lora_configs)
    if len(ranks) > 1:
        raise ValueError(f"unable to merge adapters, lora configs have different ranks: {ranks}")

    # check that all configs have the same target modules
    target_modules = set([" | ".join(sorted(lora_config.target_modules)) for lora_config in lora_configs])
    if len(target_modules) > 1:
        raise ValueError(f"unable to merge adapters, lora configs have different target modules: {target_modules}")


def _merge_lora_configs(lora_configs: List[LoraConfig]) -> LoraConfig:
    # for now, due to rank and target constraints, we can just return one config
    # may revisit this in the future if we loosen these constraints
    return lora_configs[0]
