from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type

import torch
import torch.distributed
from loguru import logger

from lorax_server.adapters.config import AdapterConfig, ModuleMap
from lorax_server.adapters.types import MEDUSA
from lorax_server.adapters.weights import AdapterBatchMetadata, AdapterWeights, BatchAdapterWeights
from lorax_server.layers import FastLinear, TensorParallelColumnLinear
from lorax_server.utils.punica import segmented_matmul
from lorax_server.utils.segments import find_segments
from lorax_server.utils.state import LORAX_SPECULATION_MAX_BATCH_SIZE, get_speculative_tokens
from lorax_server.utils.weights import AbstractWeights, InMemoryWeights

if TYPE_CHECKING:
    from lorax_server.models.model import Model


EMPTY_TENSOR = torch.tensor([])

_MEDUSA_ENABLED = False


@dataclass
class MedusaConfig(AdapterConfig):
    medusa_num_heads: int
    medusa_num_layers: int
    version: int

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
        global _MEDUSA_ENABLED
        if dynamic:
            if not _MEDUSA_ENABLED:
                raise ValueError(
                    "Medusa adapters can only be loaded at request time when LoRAX was initialized with a default "
                    "Medusa adapter."
                )

            if self.version < 2:
                raise ValueError(
                    f"Dynamic adapter loading is not supported for Medusa version {self.version} at this time. "
                    f"Instead, initialize the LoRAX server with the Medusa adapter and it will be applied to every "
                    f"request, or migrate to a v2 adapter."
                )

            if get_speculative_tokens() != self.medusa_num_heads:
                raise ValueError(
                    f"Cannot load a Medusa adapter dynamically with a different number of heads "
                    f"({self.medusa_num_heads}) from the default speculative tokens ({get_speculative_tokens()})."
                )
        else:
            _MEDUSA_ENABLED = True

        # TODO(travis): load to GPU and offload to CPU in accordance with lorax scheduler
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
            version=float(config.get("version", 1)),
        )


@dataclass
class MedusaSegments:
    w: List[torch.Tensor]
    b: List[torch.Tensor]
    s_start: torch.Tensor
    s_end: torch.Tensor


class ResBlock(torch.nn.Module):
    def __init__(self, config: MedusaConfig, prefix: str, weights: AbstractWeights):
        super().__init__()
        self.linear = FastLinear.load(config, prefix=f"{prefix}.linear", weights=weights, bias=True)
        self.act = torch.nn.SiLU()
        self.scaling = 1

    def forward(self, x):
        return x + self.act(self.linear(x))


class MedusaHead(torch.nn.Module):
    def __init__(self, config: MedusaConfig, prefix: str, weights: AbstractWeights):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [ResBlock(config, prefix=f"{prefix}.{i}", weights=weights) for i in range(config.medusa_num_layers)]
        )
        n = len(self.blocks)
        self.out = FastLinear.load(config, prefix=f"{prefix}.{n}", weights=weights, bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.out(x)
        return x


class MedusaV1(torch.nn.Module):
    def __init__(self, config: MedusaConfig, weights: AbstractWeights):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [MedusaHead(config, prefix=f"{i}", weights=weights) for i in range(config.medusa_num_heads)]
        )

    def forward(self, x, lm_head, segments: Optional[MedusaSegments] = None):
        logits = lm_head(x)
        speculative_logits = torch.stack([head(x) for head in self.heads], dim=1)
        return logits, speculative_logits


class MedusaV2(torch.nn.Module):
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

    def forward(self, x, lm_head, segments: Optional[MedusaSegments] = None):
        # If we have too many tokens, we skip speculative logits
        if x.shape[0] > LORAX_SPECULATION_MAX_BATCH_SIZE:
            logger.info(f"Skipping speculation at batch size = {x.shape[0]}")
            logits = lm_head(x)
            return logits, None

        size = x.shape[-1]
        block_size = (size + self.world_size - 1) // self.world_size
        start = self.rank * block_size
        stop = (self.rank + 1) * block_size

        x_block = x[:, start:stop]

        if segments is not None:
            # Multi-Medusa
            # TODO(travis): custom kernel similar to SGMV
            y = torch.empty((x.shape[0], self.n_medusa_heads * x_block.shape[-1]), device=x.device, dtype=x.dtype)
            segmented_matmul(
                y,
                x,
                segments.w,
                segments.b,
                segments.s_start,
                segments.s_end,
            )
        else:
            y = self.linear(x)

        # Compute all medusa heads at the same time, then reshape and move the n_medusa_heads dim to dim 1
        medusa_res = self.act(y).reshape(*x_block.shape[:-1], self.n_medusa_heads, x_block.shape[-1])

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


class MedusaModel(torch.nn.Module):
    def __init__(self, config: MedusaConfig, weights: AbstractWeights):
        super().__init__()
        if config.medusa_num_layers > 1 or weights.has_tensor(f"0.{config.medusa_num_layers}.weight"):
            self.medusa = MedusaV1(config, weights)
        else:
            self.medusa = MedusaV2(config, weights)

    def forward(self, x, lm_head, segments: Optional[MedusaSegments] = None):
        return self.medusa(x, lm_head, segments)


class MedusaWeights(AdapterWeights):
    def __init__(self, config: MedusaConfig, module_map: ModuleMap, model: "Model"):
        self.config = config
        self.model = MedusaModel(config, InMemoryWeights(module_map, model.device, model.dtype, model.process_group))
        self.process_group = model.process_group

    @classmethod
    def get_batch_types(cls) -> List[Type[BatchAdapterWeights]]:
        return [BatchMedusaWeights]

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
    segments: Optional[MedusaSegments] = None

    def has_adapter(self, adapter_index: int) -> bool:
        # If we have a default Medusa, we always have an adapter
        return self.default_medusa is not None or adapter_index in self.adapter_to_medusa

    @classmethod
    def key(cls) -> str:
        return MEDUSA

    def __call__(self, x, lm_head):
        if self.default_medusa:
            return self.default_medusa.model(x, lm_head, self.segments)
        return lm_head(x)

    @classmethod
    def load(
        cls,
        adapter_weights: Dict[int, AdapterWeights],
        meta: "AdapterBatchMetadata",
        layer_name: str,
        prefill: bool,
        prefill_head_indices: Optional[torch.Tensor],
    ) -> Optional["BatchMedusaWeights"]:
        adapter_weights = {k: _convert_medusa(v) for k, v in adapter_weights.items()}
        adapter_weights = {k: v for k, v in adapter_weights.items() if isinstance(v, MedusaWeights)}
        if not adapter_weights:
            return None

        default_medusa = adapter_weights.get(0)

        segments = meta.adapter_segments
        segment_indices = meta.segment_indices
        if default_medusa is not None:
            # Replace all non-existent segment indices with 0 (default medusa)
            # This happens when the segment corresponds to a different adapter type (like LoRA) but we still wish
            # to apply the default Medusa adapter
            if len(segment_indices) > 1:
                # merge segments
                adapter_indices = [idx if idx in adapter_weights else 0 for idx in meta.adapter_indices.cpu().tolist()]
                segments, segment_indices = find_segments(adapter_indices)
                segments = torch.tensor(
                    segments,
                    dtype=torch.int32,
                    device=meta.adapter_segments.device,
                )
            else:
                # update segment in place
                segment_indices = [idx if idx in adapter_weights else 0 for idx in meta.segment_indices]

        indices = [idx for idx, s in enumerate(segment_indices) if s in adapter_weights]

        return BatchMedusaWeights(
            adapter_to_medusa=adapter_weights,
            default_medusa=default_medusa,
            segments=MedusaSegments(
                w=[
                    (
                        adapter_weights[idx].model.medusa.linear.linear.weight.data
                        if idx in adapter_weights
                        else EMPTY_TENSOR
                    )
                    for idx in segment_indices
                ],
                b=[
                    (
                        adapter_weights[idx].model.medusa.linear.linear.bias.data
                        if idx in adapter_weights
                        else EMPTY_TENSOR
                    )
                    for idx in segment_indices
                ],
                s_start=segments[indices],
                s_end=segments[[i + 1 for i in indices]],
            ),
        )


def _convert_medusa(v: AdapterWeights) -> AdapterWeights:
    if hasattr(v, "medusa_weights"):
        return v.medusa_weights
    return v
