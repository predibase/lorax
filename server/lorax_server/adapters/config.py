import json
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from peft import LoraConfig as _LoraConfig


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


@dataclass
class LoraConfig(AdapterConfig):
    r: int
    target_modules: Optional[Union[List[str], str]]
    fan_in_fan_out: bool
    lora_alpha: int
    use_rslora: bool
    
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

    @classmethod
    def load(cls, config: dict) -> "MedusaConfig":
        return cls(
            base_model_name_or_path=config["base_model_name_or_path"],
            medusa_num_heads=config["medusa_num_heads"],
            medusa_num_layers=config["medusa_num_layers"],
        )
