from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, Union

import torch
from peft import LoraConfig as _LoraConfig
from torch.distributed import ProcessGroup

from lorax_server.adapters.config import AdapterConfig, ModuleMap
from lorax_server.adapters.types import LORA
from lorax_server.adapters.weights import AdapterBatchMetadata, AdapterWeights, BatchAdapterWeights
from lorax_server.utils.lora import EMBED_TOKENS
from lorax_server.utils.sgmv import (
    BGMV_MAX_RANK,
    MAX_RANK_CUSTOM,
    get_tmp_tensors,
    orient_for_rank,
    pad_rank,
    use_cutlass_shrink,
)

if TYPE_CHECKING:
    from lorax_server.models.model import Model

EMPTY_TENSOR = torch.tensor([])


@dataclass
class LoraConfig(AdapterConfig):
    r: int
    target_modules: Optional[Union[List[str], str]]
    fan_in_fan_out: bool
    lora_alpha: int
    use_rslora: bool

    def map_weights_for_model(
        self,
        adapter_weights: Dict,
        weight_names: Tuple[str],
    ) -> Tuple[ModuleMap, Set[str]]:
        adapter_weight_names = set()
        module_map = {}
        for weight_name in weight_names:
            if EMBED_TOKENS in weight_name:
                lora_a_name = f"base_model.model.{weight_name}.lora_embedding_A"
                lora_b_name = f"base_model.model.{weight_name}.lora_embedding_B"
            else:
                lora_a_name = f"base_model.model.{weight_name}.lora_A.weight"
                lora_b_name = f"base_model.model.{weight_name}.lora_B.weight"
            if lora_a_name not in adapter_weights or lora_b_name not in adapter_weights:
                continue

            # note(ajinkya): popping the weights so that we know which weights are
            # can be used as lora weights (supported) and which cannot
            module_map[weight_name] = {
                "lora_A": (adapter_weights.pop(lora_a_name), lora_a_name),
                "lora_B": (adapter_weights.pop(lora_b_name), lora_b_name),
            }
            adapter_weight_names.add(lora_a_name)
            adapter_weight_names.add(lora_b_name)
        return module_map, adapter_weight_names

    def load_batched_adapter_weights(
        self,
        model: "Model",
        module_map: Dict[str, Dict],
        layer_type: str,
        unused_weight_names: Set[str],
        dynamic: bool,
    ) -> Optional[AdapterWeights]:
        return LoraWeights.load(
            self,
            model,
            module_map,
            layer_type,
            unused_weight_names,
        )

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


class LoraWeights(AdapterWeights):
    """LoRA weights for a single adapter merged across all layers."""

    def __init__(
        self,
        weights_a: List[torch.Tensor],
        weights_b: List[torch.Tensor],
        adapter_config: LoraConfig,
        is_embed: bool = False,
    ):
        self.lora_a_r = weights_a[0].size(1) if len(weights_a) > 0 else 1
        self.lora_b_r = weights_b[0].size(0) if len(weights_a) > 0 else 1

        self._use_cutlass_shrink = use_cutlass_shrink(self.lora_a_r)
        self._is_transposed = False

        # [num_layers, hidden_size, r]
        if not is_embed:
            weights_a = [orient_for_rank(w, w.size(1)).contiguous() for w in weights_a]
        self._weights_a = torch.stack(weights_a)

        # [num_layers, r, hidden_size]
        self._weights_b = torch.stack(weights_b)

        self.adapter_config = adapter_config

    @property
    def weights_a(self) -> torch.Tensor:
        if self._is_transposed:
            self._transpose_weights()
        return self._weights_a

    @property
    def weights_b(self) -> torch.Tensor:
        if self._is_transposed:
            self._transpose_weights()
        return self._weights_b

    @property
    def weights_a_t(self) -> torch.Tensor:
        if not self._is_transposed:
            self._transpose_weights()
        return self._weights_a

    @property
    def weights_b_t(self) -> torch.Tensor:
        if not self._is_transposed:
            self._transpose_weights()
        return self._weights_b

    def _transpose_weights(self):
        if self._use_cutlass_shrink:
            # If we're not using the cutlass shrink, then both SGMV and BGMV use the same orientation
            self._weights_a = self._weights_a.transpose(1, 2).contiguous()
        self._weights_b = self._weights_b.transpose(1, 2).contiguous()
        self._is_transposed = not self._is_transposed

    @classmethod
    def get_batch_types(cls) -> List[Type[BatchAdapterWeights]]:
        return [BatchLoraWeights]

    @classmethod
    def load(
        cls,
        config: LoraConfig,
        model: "Model",
        module_map: Dict[str, Dict],
        layer_type: str,
        unused_weight_names: Set[str],
    ) -> Optional[AdapterWeights]:
        nlayers = model.get_num_layers_for_type(layer_type)
        lora_a_list = [None] * nlayers
        lora_b_list = [None] * nlayers

        for layer_id in range(nlayers):
            key = (layer_id, layer_type)
            weight_name, layer = model.target_to_layer[key]

            if EMBED_TOKENS in layer_type:
                base_device = layer.base_layer.weight.device
            else:
                base_device = layer.base_layer.linear.weight.device

            if weight_name not in module_map:
                # There is no LoRA weight for this layer type in the adapter
                return None

            lora_a, lora_a_name = module_map[weight_name]["lora_A"]
            lora_a = lora_a.to(base_device, model.dtype)

            lora_b, lora_b_name = module_map[weight_name]["lora_B"]
            lora_b = lora_b.to(base_device, model.dtype)

            scale = get_scaling_factor(
                config.lora_alpha,
                config.r,
                uses_rslora=config.use_rslora,
            )

            unused_weight_names.discard(lora_a_name)
            unused_weight_names.discard(lora_b_name)

            # Merge scaling factor into lora_b due to associativity of matrix multiplication:
            # (A * B) * C = A * (B * C)
            lora_a_list[layer_id] = lora_a.transpose(0, 1)
            lora_b_list[layer_id] = lora_b.transpose(0, 1) * scale

        # pad lora ranks to be compatible with sgmv
        lora_a_list = [pad_rank(w, dim=1, world_size=model.world_size) for w in lora_a_list]
        lora_b_list = [pad_rank(w, dim=0, world_size=model.world_size) for w in lora_b_list]

        if lora_a_list:
            # update rank if it was padded
            padded_rank = lora_a_list[0].size(1)
            config.r = padded_rank

        return LoraWeights(
            *model.shard_lora_weights(lora_a_list, lora_b_list, layer_type),
            adapter_config=config,
            is_embed=(layer_type == EMBED_TOKENS),
        )


@dataclass
class RankSegments:
    rank: int
    adapter_index_map: int

    lora_a_ptr: torch.Tensor
    lora_b_ptr: torch.Tensor

    # prefill (sgmv)
    tmp_shrink: torch.Tensor
    tmp_expand: torch.Tensor
    segment_starts: torch.Tensor
    segment_ends: torch.Tensor

    # decode (bgmv)
    indices: torch.Tensor


@dataclass
class BatchLoraWeights(BatchAdapterWeights):
    lora_a: Dict[int, torch.Tensor]
    lora_b: Dict[int, torch.Tensor]
    adapter_index_configs: Dict[int, LoraConfig]
    rank_data: Dict[int, RankSegments]
    use_sgmv: bool

    def has_adapter(self, adapter_index: int) -> bool:
        return adapter_index in self.adapter_index_configs

    def can_vectorize(self, pg: ProcessGroup) -> bool:
        return all(rank_data.rank // pg.size() <= MAX_RANK_CUSTOM for rank_data in self.rank_data.values())

    @classmethod
    def key(cls) -> str:
        return LORA

    @classmethod
    def load(
        self,
        adapter_weights: Dict[int, AdapterWeights],
        meta: AdapterBatchMetadata,
        prefill: bool,
        prefill_head_indices: Optional[torch.Tensor],
        is_embed: bool,
    ) -> Optional["BatchLoraWeights"]:
        adapter_weights = {k: _convert_lora(v) for k, v in adapter_weights.items()}
        adapter_weights = {k: v for k, v in adapter_weights.items() if isinstance(v, LoraWeights)}
        if not adapter_weights:
            return None

        first_weights = list(adapter_weights.values())[0]
        device = first_weights.weights_a.device
        segment_indices = meta.segment_indices

        lora_a, lora_b, adapter_index_configs = {}, {}, {}
        max_rank, rank_indices = 0, defaultdict(list)
        for segment_idx, adapter_idx in enumerate(segment_indices):
            if adapter_idx in adapter_weights:
                adapter_weight = adapter_weights[adapter_idx]
                adapter_index_configs[adapter_idx] = adapter_weight.adapter_config
                max_rank = max(max_rank, adapter_weight.lora_a_r)
                rank_indices[adapter_weight.lora_a_r].append(segment_idx)
                lora_a[adapter_idx] = adapter_weight.weights_a
                lora_b[adapter_idx] = adapter_weight.weights_b

        if not max_rank:
            return None

        use_sgmv = prefill or max_rank > BGMV_MAX_RANK

        if prefill_head_indices is not None:
            j, prefill_head_segment_starts, prefill_head_segment_ends = 1, [0], [0]
            # prefill_head_indices is used to slice the tokens in the batch such
            # that we only forward the last token for each request through lm_head
            # there can be multiple head_index associated with each adapter segment
            for head_index in prefill_head_indices:
                # j cannot go out of bounds as that would mean there are tokens without segments
                if head_index < meta.adapter_segments[j]:
                    # head_index is part of the current adapter
                    # so increment the current segment end
                    prefill_head_segment_ends[-1] += 1
                else:
                    # head_index in not part of the current adapter
                    # close the previous segment and start a new one
                    prefill_head_segment_starts.append(prefill_head_segment_ends[-1])
                    prefill_head_segment_ends.append(prefill_head_segment_ends[-1] + 1)
                    j += 1

        rank_data = {}
        for rank, indices in rank_indices.items():
            tmp_shrink = None
            tmp_expand = None
            segment_starts = None
            segment_ends = None
            batch_indices = None
            lora_a_ptr_indices = []
            lora_b_ptr_indices = []

            if use_sgmv:
                for segment_idx in indices:
                    adapter_weight = adapter_weights[segment_indices[segment_idx]]
                    lora_a_ptr_indices.append(adapter_weight.weights_a.data_ptr())
                    lora_b_ptr_indices.append(adapter_weight.weights_b.data_ptr())
                tmp_shrink, tmp_expand = get_tmp_tensors(len(lora_a_ptr_indices), rank, device)
                segment_starts = meta.adapter_segments[indices]
                segment_ends = meta.adapter_segments[[i + 1 for i in indices]]
                if prefill_head_indices is not None:
                    # since prefill_head_indices is present the segment starts and ends
                    # need to be adjusted according to the number of head tokens in each
                    for i, segment_idx in enumerate(indices):
                        segment_starts[i] = prefill_head_segment_starts[segment_idx]
                        segment_ends[i] = prefill_head_segment_ends[segment_idx]
            else:
                adapter_idx_to_pointer_idx = {}
                # find out which adapters are present in the segments for this rank
                # iterate over each segment index and use it to find adapter index and weights
                for segment_idx in indices:
                    adapter_idx = segment_indices[segment_idx]
                    adapter_weight = adapter_weights[adapter_idx]
                    # if the adapter hasn't been seen before, then append its weight pointers
                    # and save the index to the just added pointers for later
                    if adapter_idx not in adapter_idx_to_pointer_idx:
                        lora_a_ptr_indices.append(
                            (adapter_weight.weights_a if is_embed else adapter_weight.weights_a_t).data_ptr()
                        )
                        lora_b_ptr_indices.append(adapter_weight.weights_b_t.data_ptr())
                        adapter_idx_to_pointer_idx[adapter_idx] = len(lora_a_ptr_indices) - 1
                # for each token in the batch, see if its adapter is present in the segments for this rank
                # if present, then store the index of its weight pointers otherwise store -1
                batch_indices = torch.tensor([
                    adapter_idx_to_pointer_idx.get(adapter_idx, -1) for adapter_idx in meta.adapter_indices.tolist()
                ], dtype=torch.int64, device=device)

            lora_a_ptr_indices = torch.tensor(lora_a_ptr_indices, dtype=torch.int64, device=device)
            lora_b_ptr_indices = torch.tensor(lora_b_ptr_indices, dtype=torch.int64, device=device)

            rank_data[rank] = RankSegments(
                rank=rank,
                adapter_index_map=indices,
                tmp_shrink=tmp_shrink,
                tmp_expand=tmp_expand,
                lora_a_ptr=lora_a_ptr_indices,
                lora_b_ptr=lora_b_ptr_indices,
                segment_starts=segment_starts,
                segment_ends=segment_ends,
                indices=batch_indices,
            )

        return BatchLoraWeights(
            lora_a=lora_a,
            lora_b=lora_b,
            adapter_index_configs=adapter_index_configs,
            rank_data=rank_data,
            use_sgmv=use_sgmv,
        )


def get_scaling_factor(
    lora_alpha: int,
    r: int,
    uses_rslora: bool = False,
) -> float:
    """Computes the scaling factor for the lora weights."""
    if uses_rslora:
        return lora_alpha / (r**0.5)
    return lora_alpha / r


def _convert_lora(v: AdapterWeights) -> AdapterWeights:
    if hasattr(v, "lora_weights"):
        return v.lora_weights
    return v
