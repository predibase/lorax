from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from opentelemetry import trace
from transformers import AutoTokenizer

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_gemma_modeling import (
    GemmaConfig,
    GemmaForCausalLM,
)
from lorax_server.utils.lora import DOWN_PROJ, GATE_PROJ, K_PROJ, O_PROJ, Q_PROJ, UP_PROJ, V_PROJ

tracer = trace.get_tracer(__name__)


# TODO(travis): re-enable LM_HEAD after resolving issues with outputs
ADAPTER_LAYERS = [Q_PROJ, K_PROJ, V_PROJ, O_PROJ, GATE_PROJ, UP_PROJ, DOWN_PROJ]
ROW_PARALLEL = {O_PROJ, DOWN_PROJ}


class FlashGemma(FlashCausalLM):
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
            model_cls=GemmaForCausalLM,
            dtype=dtype,
            revision=revision,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            tokenizer_cls=AutoTokenizer,
            config_cls=GemmaConfig,
            **kwargs,
        )

    @property
    def supports_adapter_loading(self) -> bool:
        return True

    def adapter_target_to_layer(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        layer_weights = {}

        prefix = "model.layers"
        for i, layer in enumerate(self.model.model.layers):
            layer_weights[(i, Q_PROJ)] = (
                f"{prefix}.{i}.self_attn.q_proj",
                layer.self_attn.query_key_value,
            )
            layer_weights[(i, K_PROJ)] = (
                f"{prefix}.{i}.self_attn.k_proj",
                layer.self_attn.query_key_value,
            )
            layer_weights[(i, V_PROJ)] = (
                f"{prefix}.{i}.self_attn.v_proj",
                layer.self_attn.query_key_value,
            )
            layer_weights[(i, O_PROJ)] = (f"{prefix}.{i}.self_attn.o_proj", layer.self_attn.o_proj)

            layer_weights[(i, GATE_PROJ)] = (f"{prefix}.{i}.mlp.gate_proj", layer.mlp.gate_proj)
            layer_weights[(i, UP_PROJ)] = (f"{prefix}.{i}.mlp.up_proj", layer.mlp.up_proj)
            layer_weights[(i, DOWN_PROJ)] = (f"{prefix}.{i}.mlp.down_proj", layer.mlp.down_proj)

        return layer_weights

    @property
    def adapter_layers(self) -> List[str]:
        return ADAPTER_LAYERS

    @property
    def default_traced_adapter_layers(self) -> List[str]:
        return [Q_PROJ, V_PROJ]

    def get_num_layers_for_type(self, layer_type: str) -> int:
        return len(self.model.model.layers)

    def is_row_parallel(self, layer_type: str) -> bool:
        return layer_type in ROW_PARALLEL
