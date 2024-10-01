from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from opentelemetry import trace

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_qwen_modeling import (
    ATTN_C_ATTN,
    ATTN_C_PROJ,
    MLP_C_PROJ,
    MLP_W1,
    MLP_W2,
    FlashQwenForCausalLM,
    QwenConfig,
)
from lorax_server.utils.lora import LM_HEAD
from lorax_server.utils.weights import shard_on_dim

tracer = trace.get_tracer(__name__)


ADAPTER_LAYERS = [ATTN_C_ATTN, ATTN_C_PROJ, MLP_W1, MLP_W2, MLP_C_PROJ, LM_HEAD]
ROW_PARALLEL = {ATTN_C_PROJ, MLP_C_PROJ, LM_HEAD}


class FlashQwen(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        adapter_id: str,
        adapter_source: str,
        revision: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__(
            model_id=model_id,
            model_cls=FlashQwenForCausalLM,
            dtype=dtype,
            revision=revision,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            config_cls=QwenConfig,
            **kwargs,
        )

    @property
    def supports_adapter_loading(self) -> bool:
        return True

    def adapter_target_to_layer(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        layer_weights = {}

        prefix = "transformer.h"
        for i, layer in enumerate(self.model.transformer.h):
            layer_weights[(i, ATTN_C_ATTN)] = (f"{prefix}.{i}.attn.c_attn", layer.attn.c_attn)
            layer_weights[(i, ATTN_C_PROJ)] = (f"{prefix}.{i}.attn.c_proj", layer.attn.c_proj)

            layer_weights[(i, MLP_W1)] = (f"{prefix}.{i}.mlp.w1", layer.mlp.gate_up_proj)
            layer_weights[(i, MLP_W2)] = (f"{prefix}.{i}.mlp.w2", layer.mlp.gate_up_proj)
            layer_weights[(i, MLP_C_PROJ)] = (f"{prefix}.{i}.mlp.c_proj", layer.mlp.c_proj)

        layer_weights[(0, LM_HEAD)] = ("lm_head", self.model.lm_head)
        return layer_weights

    @property
    def adapter_layers(self) -> List[str]:
        return ADAPTER_LAYERS

    @property
    def default_traced_adapter_layers(self) -> List[str]:
        return [ATTN_C_ATTN]

    def get_num_layers_for_type(self, layer_type: str) -> int:
        return 1 if layer_type == LM_HEAD else len(self.model.transformer.h)

    def is_row_parallel(self, layer_type: str) -> bool:
        return layer_type in ROW_PARALLEL

    def split_lora_b_qkv(self, t: torch.Tensor, projection_size: int) -> torch.Tensor:
        # Because we're splitting on the hidden size dimension, we need to
        # account for the separate q, k, and v matrices.
        chunks = torch.split(t, projection_size, dim=1)
        assert len(chunks) == 3
        chunks = [shard_on_dim(w, dim=1, process_group=self.process_group) for w in chunks]
        return torch.cat(chunks, dim=1)

    def shard_lora_weights(
        self,
        weights_a: List[torch.Tensor],
        weights_b: List[torch.Tensor],
        layer_type: str,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # TODO(travis): genralize this for other layers and architectures
        if layer_type == ATTN_C_ATTN:
            # [hidden_size, r]
            split_dim = 0 if self.is_row_parallel(layer_type) else 1
            weights_a = [shard_on_dim(w, dim=split_dim, process_group=self.process_group) for w in weights_a]

            # [r, hidden_size]
            # Because we're splitting on the hidden size dimension, we need to
            # account for the separate q, k, and v matrices.
            projection_size = (
                self.config.hidden_size // self.config.num_attention_heads
            ) * self.config.num_attention_heads
            weights_b = [self.split_lora_b_qkv(w, projection_size) for w in weights_b]

            return weights_a, weights_b
        else:
            return super().shard_lora_weights(weights_a, weights_b, layer_type)
