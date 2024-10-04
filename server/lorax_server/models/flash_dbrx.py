from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from opentelemetry import trace
from transformers import AutoTokenizer

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_dbrx_modeling import (
    ATTN_O_PROJ,
    ATTN_WQKV,
    DbrxConfig,
    FlashDbrxForCausalLM,
)
from lorax_server.utils.lora import LM_HEAD

tracer = trace.get_tracer(__name__)

ADAPTER_LAYERS = [ATTN_WQKV, ATTN_O_PROJ, LM_HEAD]
ROW_PARALLEL = {ATTN_O_PROJ, LM_HEAD}


class FlashDbrx(FlashCausalLM):
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
            model_cls=FlashDbrxForCausalLM,
            dtype=dtype,
            revision=revision,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            tokenizer_cls=AutoTokenizer,
            config_cls=DbrxConfig,
            **kwargs,
        )

    @property
    def supports_adapter_loading(self) -> bool:
        return True

    def adapter_target_to_layer(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        layer_weights = {}

        prefix = "transformer.blocks"
        for i, layer in enumerate(self.model.model.layers):
            layer_weights[(i, ATTN_WQKV)] = (
                f"{prefix}.{i}.norm_attn_norm.attn.q_proj",
                layer.attn.self_attn.query_key_value,
            )
            layer_weights[(i, ATTN_O_PROJ)] = (
                f"{prefix}.{i}.norm_attn_norm.attn.out_proj",
                layer.attn.self_attn.o_proj,
            )

        layer_weights[(0, LM_HEAD)] = ("lm_head", self.model.lm_head)
        return layer_weights

    @property
    def adapter_layers(self) -> List[str]:
        return ADAPTER_LAYERS

    @property
    def default_traced_adapter_layers(self) -> List[str]:
        return [ATTN_WQKV]

    def get_num_layers_for_type(self, layer_type: str) -> int:
        return len(self.model.model.layers)

    def is_row_parallel(self, layer_type: str) -> bool:
        return layer_type in ROW_PARALLEL
