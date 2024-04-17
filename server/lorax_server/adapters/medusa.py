from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

import torch
import torch.distributed

from lorax_server.adapters.config import AdapterConfig, ModuleMap
from lorax_server.adapters.types import MEDUSA
from lorax_server.adapters.weights import AdapterBatchMetadata, AdapterWeights, BatchAdapterWeights
from lorax_server.utils.layers import FastLinear, TensorParallelColumnLinear
from lorax_server.utils.weights import AbstractWeights, InMemoryWeights

if TYPE_CHECKING:
    from lorax_server.models.model import Model


@dataclass
class MedusaConfig(AdapterConfig):
    medusa_num_heads: int
    medusa_num_layers: int

    @property
    def quantize(self) -> Optional[str]:
        return None

    def map_weights_for_model(
        self,
        adapter_weights: Dict,
        weight_names: Tuple[str],
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
        self.linear = FastLinear.load(config, prefix=f"{prefix}.linear", weights=weights, bias=True)
        # self.lora_A = FastLinear.load(config, prefix=f"{prefix}.lora_A", weights=weights, bias=False)
        # self.lora_B = FastLinear.load(config, prefix=f"{prefix}.lora_B", weights=weights, bias=False)
        self.act = torch.nn.SiLU()
        self.scaling = 1

    def forward(self, x):
        return x + self.act(self.linear(x))
        # return x + self.act(self.lora_B(self.lora_A(x)) * self.scaling)


class MedusaHead(torch.nn.Module):
    def __init__(self, config: MedusaConfig, prefix: str, weights: AbstractWeights):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [ResBlock(config, prefix=f"{prefix}.{i}", weights=weights) for i in range(config.medusa_num_layers)]
        )
        # n = len(self.blocks)
        # self.out = FastLinear.load(config, prefix=f"{prefix}.{n}", weights=weights, bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        # x = self.out(x)
        return x


class MedusaModel(torch.nn.Module):
    def __init__(self, config: MedusaConfig, weights: AbstractWeights):
        super().__init__()
        self.n_medusa_heads = config.medusa_num_heads

        assert config.medusa_num_layers == 1
        self.linear = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{i}.0.linear" for i in range(self.n_medusa_heads)],
            dim=0,
            weights=weights,
            bias=True,
        )
        self.process_group = weights.process_group
        self.world_size = self.process_group.size()
        self.rank = self.process_group.rank()

        self.act = torch.nn.SiLU()

    # def forward(self, x):
    #     speculative_logits = torch.stack([head(x) for head in self.heads], dim=1)
    #     return speculative_logits

    def forward(self, x, lm_head):
        # If we have too many tokens, we skip speculative logits
        if x.shape[0] > 128:
            logits = lm_head(x)
            return logits, None

        size = x.shape[-1]
        block_size = (size + self.world_size - 1) // self.world_size
        start = self.rank * block_size
        stop = (self.rank + 1) * block_size

        x_block = x[:, start:stop]

        # Compute all medusa heads at the same time, then reshape and move the n_medusa_heads dim to dim 1
        medusa_res = self.act(self.linear(x)).reshape(*x_block.shape[:-1], self.n_medusa_heads, x_block.shape[-1])

        # Apply all residual medusa heads
        output = x[:, start:stop].unsqueeze(-2) + medusa_res

        # Gather medusa heads
        world_output = [torch.empty_like(output) for _ in range(self.process_group.size())]
        torch.distributed.all_gather(world_output, output, group=self.process_group)
        world_output = torch.cat(world_output, dim=-1)

        # Stack x and medusa residual x
        stacked_x = torch.cat([x.unsqueeze(-2), world_output], dim=-2)

        # Compute lm head on x + medusa residual x
        logits = lm_head(stacked_x)

        # Finally, split logits from speculative logits
        logits, speculative_logits = torch.split(logits, [1, self.n_medusa_heads], dim=-2)
        # Squeeze added dimension
        logits = logits.squeeze(-2)

        return logits, speculative_logits


class MedusaWeights(AdapterWeights):
    def __init__(self, config: MedusaConfig, module_map: ModuleMap, model: "Model"):
        self.config = config
        self.model = MedusaModel(config, InMemoryWeights(module_map, model.device, model.dtype, model.process_group))

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
    def load(cls, adapter_weights: Dict[int, AdapterWeights], meta: "AdapterBatchMetadata") -> "BatchMedusaWeights":
        adapter_weights = {k: v for k, v in adapter_weights.items() if isinstance(v, MedusaWeights)}

        adapter_to_medusa = {idx: adapter_weights[idx] for idx in meta.segment_indices if idx in adapter_weights}

        return BatchMedusaWeights(
            adapter_to_medusa=adapter_to_medusa,
            default_medusa=adapter_weights.get(0),
        )
