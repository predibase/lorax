from typing import Dict, List, Optional, Tuple, cast

import torch
import torch.distributed
from opentelemetry import trace
from transformers import AutoTokenizer

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_exaone_modeling import (
    ExaoneConfig,
    ExaoneForCausalLM,
)
from lorax_server.utils.lora import DOWN_PROJ, GATE_PROJ, K_PROJ, O_PROJ, Q_PROJ, UP_PROJ, V_PROJ

tracer = trace.get_tracer(__name__)


# TODO(travis): re-enable LM_HEAD after resolving issues with outputs
ADAPTER_LAYERS = [Q_PROJ, K_PROJ, V_PROJ, O_PROJ, GATE_PROJ, UP_PROJ, DOWN_PROJ]

class FlashExaOne(FlashCausalLM):
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
            model_cls=ExaoneForCausalLM,
            dtype=dtype,
            revision=revision,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            tokenizer_cls=AutoTokenizer,
            config_cls=ExaoneConfig,
            **kwargs,
        )

    @property
    def supports_adapter_loading(self) -> bool:
        return True

    def adapter_target_to_layer(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        layer_weights = {}

        prefix = "transformer.h"
        for i, layer in enumerate(cast(ExaoneForCausalLM, self.model).transformer.h):
            layer_weights[(i, Q_PROJ)] = (
                f"{prefix}.{i}.attn.attention.q_proj",
                layer.attn.attention.q_proj,
            )
            layer_weights[(i, K_PROJ)] = (
                f"{prefix}.{i}.attn.attention.k_proj",
                layer.attn.attention.k_proj,
            )
            layer_weights[(i, V_PROJ)] = (
                f"{prefix}.{i}.attn.attention.v_proj",
                layer.attn.attention.v_proj,
            )
            layer_weights[(i, O_PROJ)] = (
                f"{prefix}.{i}.attn.attention.out_proj",
                layer.attn.attention.out_proj,
            )
            layer_weights[(i, GATE_PROJ)] = (f"{prefix}.{i}.mlp.c_fc_0", layer.mlp.c_fc_0)
            layer_weights[(i, UP_PROJ)] = (f"{prefix}.{i}.mlp.c_fc_1", layer.mlp.c_fc_1)
            layer_weights[(i, DOWN_PROJ)] = (f"{prefix}.{i}.mlp.c_proj", layer.mlp.c_proj)

        return layer_weights

    @property
    def adapter_layers(self) -> List[str]:
        return ADAPTER_LAYERS

    @property
    def default_traced_adapter_layers(self) -> List[str]:
        return [Q_PROJ, V_PROJ, K_PROJ]

    def get_num_layers_for_type(self, layer_type: str) -> int:
        return len(self.model.transformer.h)
