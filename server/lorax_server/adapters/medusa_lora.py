from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type

import torch

from lorax_server.adapters.config import AdapterConfig, ModuleMap
from lorax_server.adapters.lora import BatchLoraWeights, LoraConfig, LoraWeights
from lorax_server.adapters.medusa import BatchMedusaWeights, MedusaConfig, MedusaWeights
from lorax_server.adapters.weights import AdapterWeights, BatchAdapterWeights

if TYPE_CHECKING:
    from lorax_server.models.model import Model

EMPTY_TENSOR = torch.tensor([])


@dataclass
class MedusaLoraModuleMap:
    lora_module_map: ModuleMap
    medusa_module_map: ModuleMap


@dataclass
class MedusaLoraConfig(AdapterConfig):
    lora_config: LoraConfig
    medusa_config: MedusaConfig

    def map_weights_for_model(
        self,
        adapter_weights: Dict,
        weight_names: Tuple[str],
    ) -> Tuple[MedusaLoraModuleMap, Set[str]]:
        lora_module_map, weight_names = self.lora_config.map_weights_for_model(adapter_weights, weight_names)
        medusa_module_map, _ = self.medusa_config.map_weights_for_model(adapter_weights, weight_names)
        return MedusaLoraModuleMap(lora_module_map, medusa_module_map), weight_names

    def load_batched_adapter_weights(
        self,
        model: "Model",
        module_map: MedusaLoraModuleMap,
        layer_type: str,
        unused_weight_names: Set[str],
        dynamic: bool,
    ) -> Optional[AdapterWeights]:
        lora_weights = self.lora_config.load_batched_adapter_weights(
            model, module_map.lora_module_map, layer_type, unused_weight_names, dynamic
        )
        medusa_weights = self.medusa_config.load_batched_adapter_weights(
            model, module_map.medusa_module_map, layer_type, unused_weight_names, dynamic
        )
        return MedusaLoraWeights.load(
            lora_weights,
            medusa_weights,
        )

    @classmethod
    def load(cls, adapter_id: str, config: dict, api_token: str) -> "MedusaLoraConfig":
        lora_config = LoraConfig.load(adapter_id, api_token)
        medusa_config = MedusaConfig.load(config)
        return cls(
            base_model_name_or_path=lora_config.base_model_name_or_path,
            lora_config=lora_config,
            medusa_config=medusa_config,
        )


class MedusaLoraWeights(AdapterWeights):
    def __init__(
        self,
        lora_weights: LoraWeights,
        medusa_weights: MedusaWeights,
    ):
        self.lora_weights = lora_weights
        self.medusa_weights = medusa_weights

    @classmethod
    def get_batch_types(cls) -> List[Type[BatchAdapterWeights]]:
        return [BatchLoraWeights, BatchMedusaWeights]

    @property
    def speculative_tokens(self) -> int:
        return self.medusa_weights.speculative_tokens

    @classmethod
    def load(
        cls,
        lora_weights: LoraWeights,
        medusa_weights: MedusaWeights,
    ) -> Optional[AdapterWeights]:
        return MedusaLoraWeights(
            lora_weights,
            medusa_weights,
        )
