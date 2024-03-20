import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from peft import LoraConfig as _LoraConfig


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


@dataclass
class MedusaConfig(AdapterConfig):
    medusa_num_heads: int
    medusa_num_layers: int

    def map_weights_for_model(
        self, adapter_weights: Dict, weight_names: Tuple[str],
    ) -> Tuple[ModuleMap, Set[str]]:
        adapter_weight_names = set()
        module_map = {}
        for weight_name in weight_names:
            medusa_name = f"base_model.model.{weight_name}.medusa.weight"
            if medusa_name not in adapter_weights:
                continue
            
            module_map[weight_name] = {"medusa": (adapter_weights[medusa_name], medusa_name)}
            adapter_weight_names.add(medusa_name)
        return module_map, adapter_weight_names

    @classmethod
    def load(cls, config: dict) -> "MedusaConfig":
        return cls(
            base_model_name_or_path=config["base_model_name_or_path"],
            medusa_num_heads=config["medusa_num_heads"],
            medusa_num_layers=config["medusa_num_layers"],
        )
