from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

import torch

from lorax_server.adapters.config import AdapterConfig, ModuleMap
from lorax_server.adapters.types import MEDUSA
from lorax_server.adapters.weights import AdapterBatchMetadata, AdapterWeights, BatchAdapterWeights
from lorax_server.utils.layers import FastLinear
from lorax_server.utils.weights import AbstractWeights, InMemoryWeights

if TYPE_CHECKING:
    from lorax_server.models.model import Model


@dataclass
class MedusaConfig(AdapterConfig):
    medusa_num_heads: int
    medusa_num_layers: int

    def map_weights_for_model(
        self, adapter_weights: Dict, weight_names: Tuple[str],
    ) -> Tuple[ModuleMap, Set[str]]:
        # TODO(travis): this isn't technically the ModuleMap structure, make this more generic
        return adapter_weights, set(weight_names)
    
    def load_batched_adapter_weights(
        self, 
        model: "Model",
        module_map: Dict[str, Dict], 
        layer_type: str,
        unused_weight_names: Set[str],
        dynamic: bool,
    ) -> Optional[AdapterWeights]:
        if dynamic:
            raise ValueError(
                "Dynamic adapter loading is not supported for Medusa at this time. "
                "Instead, initialize the LoRAX server with the Medusa adapter and it will be applied to every request."
            )
        
        return MedusaWeights.load(
            self,
            model,
            module_map,
            layer_type,
            unused_weight_names,
        )

    @classmethod
    def load(cls, config: dict) -> "MedusaConfig":
        return cls(
            base_model_name_or_path=config["base_model_name_or_path"],
            medusa_num_heads=config["medusa_num_heads"],
            medusa_num_layers=config["medusa_num_layers"],
        )


class ResBlock(torch.nn.Module):
    def __init__(self, config: MedusaConfig, prefix: str, weights: AbstractWeights):
        super().__init__()
        self.linear = FastLinear.load(
            config, prefix=f"{prefix}.linear", weights=weights, bias=True
        )
        self.act = torch.nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class MedusaHead(torch.nn.Module):
    def __init__(self, config: MedusaConfig, prefix: str, weights: AbstractWeights):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                ResBlock(config, prefix=f"{prefix}.{i}", weights=weights)
                for i in range(config.medusa_num_layers)
            ]
        )
        n = len(self.blocks)
        self.out = FastLinear.load(
            config, prefix=f"{prefix}.{n}", weights=weights, bias=False
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.out(x)
        return x


class MedusaModel(torch.nn.Module):
    def __init__(self, config: MedusaConfig, weights: AbstractWeights):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [
                MedusaHead(config, prefix=f"{i}", weights=weights)
                for i in range(config.medusa_num_heads)
            ]
        )

    def forward(self, x):
        speculative_logits = torch.stack([head(x) for head in self.heads], dim=1)
        return speculative_logits


class MedusaWeights(AdapterWeights):
    def __init__(self, config: MedusaConfig, module_map: ModuleMap, model: "Model"):
        self.config = config
        self.model = MedusaModel(config, InMemoryWeights(module_map, model.device, model.dtype))
    
    @classmethod
    def get_batch_type(cls) -> BatchAdapterWeights:
        return BatchMedusaWeights
    
    @property
    def speculative_tokens(self) -> int:
        return self.config.medusa_num_heads

    @classmethod
    def load(
        cls, 
        config: MedusaConfig,
        model: "Model",
        module_map: Dict[str, Dict], 
        layer_type: str,
        unused_weight_names: Set[str],
    ) -> Optional[AdapterWeights]:
        # Unused weights not needed for Medusa
        unused_weight_names.clear()
        return MedusaWeights(config, module_map, model)


@dataclass
class BatchMedusaWeights(BatchAdapterWeights):
    adapter_to_medusa: Dict[int, MedusaWeights]
    default_medusa: Optional[MedusaWeights] = None

    def has_adapter(self, adapter_index: int) -> bool:
        return adapter_index in self.adapter_to_medusa

    @classmethod
    def key(cls) -> str:
        return MEDUSA

    @classmethod
    def load(
        cls, adapter_weights: Dict[int, AdapterWeights], meta: "AdapterBatchMetadata"
    ) -> "BatchMedusaWeights":
        adapter_weights = {
            k: v
            for k, v in adapter_weights.items()
            if isinstance(v, MedusaWeights)
        }

        adapter_to_medusa = {
            idx: adapter_weights[idx]
            for idx in meta.segment_indices
            if idx in adapter_weights
        }

        return BatchMedusaWeights(
            adapter_to_medusa=adapter_to_medusa,
            default_medusa=adapter_weights.get(0),
        )
