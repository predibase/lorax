from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

import torch

from lorax_server.adapters.config import AdapterConfig, ModuleMap
from lorax_server.adapters.types import MEDUSA
from lorax_server.adapters.weights import AdapterBatchData, AdapterBatchMetadata, AdapterWeights, BatchAdapterWeights
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
        print("Mapping weights for Medusa", adapter_weights.keys(), weight_names)
        return adapter_weights, set(weight_names)
    
    def load_batched_adapter_weights(
        self, 
        model: "Model",
        module_map: Dict[str, Dict], 
        layer_type: str,
        unused_weight_names: Set[str],
    ) -> Optional[AdapterWeights]:
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
        self.model = MedusaModel(config, InMemoryWeights(module_map, model.device, model.dtype))
    
    @classmethod
    def get_batch_type(cls) -> BatchAdapterWeights:
        return BatchMedusaWeights

    @classmethod
    def load(
        cls, 
        config: MedusaConfig,
        model: "Model",
        module_map: Dict[str, Dict], 
        layer_type: str,
        unused_weight_names: Set[str],
    ) -> Optional[AdapterWeights]:
        return MedusaWeights(config, module_map, model)


@dataclass
class BatchMedusaWeights(BatchAdapterWeights):
    adapter_to_medusa: Dict[int, MedusaWeights]

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
            adapter_to_medusa=adapter_to_medusa
        )


class SpeculativeHead(torch.nn.Module):
    def __init__(self, lm_head, medusa):
        super().__init__()
        self.lm_head = lm_head
        self.medusa = medusa

    @staticmethod
    def load(lm_head: torch.nn.Module, weights, adapter_id):
        from huggingface_hub import hf_hub_download
        from pathlib import Path

        is_local = Path(adapter_id).exists()
        if not is_local:
            medusa_config = hf_hub_download(
                adapter_id, revision=None, filename="config.json"
            )
            hf_hub_download(
                adapter_id,
                revision=None,
                filename="medusa_lm_head.safetensors",
            )
            medusa_path = Path(medusa_config).parent
        else:
            medusa_path = Path(adapter_id)

        if medusa_path:
            from pathlib import Path
            from safetensors import safe_open
            import json

            medusa_config = str(medusa_path / "config.json")
            filename = str(medusa_path / "medusa_lm_head.safetensors")

            with open(medusa_config, "r") as f:
                config = json.load(f)
            routing = weights.routing
            with safe_open(filename, framework="pytorch") as f:
                for k in f.keys():
                    if k in routing:
                        raise RuntimeError(
                            f"Key {k} was found in multiple files: {filename} and {routing[k]}"
                        )
                    weights.routing[k] = filename

            medusa = MedusaModel(config, weights)
        else:
            medusa = None
        return SpeculativeHead(lm_head, medusa)

    def forward(
        self, input: torch.Tensor, adapter_data: AdapterBatchData,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = self.lm_head(input, adapter_data)
        speculative_logits = self.medusa(input) if self.medusa is not None else None
        return logits, speculative_logits
