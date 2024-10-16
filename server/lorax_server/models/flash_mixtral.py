from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from opentelemetry import trace
from transformers.models.llama import LlamaTokenizerFast

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_mixtral_modeling import (
    ATTN_K_PROJ,
    ATTN_O_PROJ,
    ATTN_Q_PROJ,
    ATTN_V_PROJ,
    FlashMixtralForCausalLM,
    MixtralConfig,
)
from lorax_server.utils.lora import LM_HEAD

tracer = trace.get_tracer(__name__)

ADAPTER_LAYERS = [
    ATTN_Q_PROJ,
    ATTN_K_PROJ,
    ATTN_V_PROJ,
    ATTN_O_PROJ,
]  # TODO(travis): add back LM_HEAD following https://github.com/predibase/lorax/issues/231
ROW_PARALLEL = {ATTN_O_PROJ, LM_HEAD}


class FlashMixtral(FlashCausalLM):
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
            model_cls=FlashMixtralForCausalLM,
            dtype=dtype,
            revision=revision,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            tokenizer_cls=LlamaTokenizerFast,
            config_cls=MixtralConfig,
            **kwargs,
        )

    @property
    def supports_adapter_loading(self) -> bool:
        return True

    def adapter_target_to_layer(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        layer_weights = {}

        prefix = "model.layers"
        for i, layer in enumerate(self.model.model.layers):
            layer_weights[(i, ATTN_Q_PROJ)] = (
                f"{prefix}.{i}.self_attn.q_proj",
                layer.self_attn.query_key_value,
            )
            layer_weights[(i, ATTN_K_PROJ)] = (
                f"{prefix}.{i}.self_attn.k_proj",
                layer.self_attn.query_key_value,
            )
            layer_weights[(i, ATTN_V_PROJ)] = (
                f"{prefix}.{i}.self_attn.v_proj",
                layer.self_attn.query_key_value,
            )
            layer_weights[(i, ATTN_O_PROJ)] = (
                f"{prefix}.{i}.self_attn.o_proj",
                layer.self_attn.o_proj,
            )

            # TODO(travis): requires implementing this for block sparse MoE
            # layer_weights[(i, MOE_W1)] = (f"{prefix}.{i}.moe.w1", layer.moe.w1)
            # layer_weights[(i, MOE_W2)] = (f"{prefix}.{i}.moe.w2", layer.moe.w2)
            # layer_weights[(i, MOE_W3)] = (f"{prefix}.{i}.moe.w3", layer.moe.w3)

        layer_weights[(0, LM_HEAD)] = ("lm_head", self.model.lm_head)
        return layer_weights

    @property
    def adapter_layers(self) -> List[str]:
        return ADAPTER_LAYERS

    @property
    def default_traced_adapter_layers(self) -> List[str]:
        return [ATTN_Q_PROJ, ATTN_V_PROJ]

    def get_num_layers_for_type(self, layer_type: str) -> int:
        return 1 if layer_type == LM_HEAD else len(self.model.model.layers)

    def is_row_parallel(self, layer_type: str) -> bool:
        return layer_type in ROW_PARALLEL
