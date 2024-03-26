import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

import torch

from lorax_server.adapters.weights import AdapterWeights
from lorax_server.adapters.lora import LoraConfig

if TYPE_CHECKING:
    from server.lorax_server.models.model import Model


ModuleMap = Dict[str, Dict[str, Tuple[torch.Tensor, str]]]


def load_adapter_config(
    config_path: Optional[Path],
    adapter_config_path: Optional[Path],
    api_token: str,
) -> "AdapterConfig":
    if adapter_config_path is not None and adapter_config_path.exists():
        return LoraConfig.load(str(adapter_config_path), api_token)
    
    if config_path is not None and config_path.exists():
        config = json.load(config_path.open())
        if "medusa_num_heads" in config:
            return MedusaConfig.load(config)
    
    raise ValueError(f"No valid adapter config file found: "
                     f"tried {adapter_config_path} and {config_path}")


@dataclass
class AdapterConfig(ABC):
    base_model_name_or_path: str

    @abstractmethod
    def map_weights_for_model(
        self, adapter_weights: Dict, weight_names: Tuple[str],
    ) -> Tuple[ModuleMap, Set[str]]:
        pass

    @abstractmethod
    def load_batched_adapter_weights(
        self, 
        model: "Model",
        module_map: Dict[str, Dict], 
        layer_type: str,
        unused_weight_names: Set[str],
    ) -> AdapterWeights:
        pass


@dataclass
class MedusaConfig(AdapterConfig):
    medusa_num_heads: int
    medusa_num_layers: int

    def map_weights_for_model(
        self, adapter_weights: Dict, weight_names: Tuple[str],
    ) -> Tuple[ModuleMap, Set[str]]:
        return adapter_weights, set(weight_names)

    @classmethod
    def load(cls, config: dict) -> "MedusaConfig":
        return cls(
            base_model_name_or_path=config["base_model_name_or_path"],
            medusa_num_heads=config["medusa_num_heads"],
            medusa_num_layers=config["medusa_num_layers"],
        )
