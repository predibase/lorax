from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from opentelemetry import trace
from transformers.models.qwen2 import Qwen2Config

from lorax_server.adapters import AdapterBatchData
from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_qwen2_modeling import (
    ATTN_K_PROJ,
    ATTN_O_PROJ,
    ATTN_Q_PROJ,
    ATTN_V_PROJ,
    MLP_DOWN_PROJ,
    MLP_GATE_PROJ,
    MLP_UP_PROJ,
    FlashQwen2ForCausalLM,
    FlashQwen2ForEmbeddings,
)
from lorax_server.utils.lora import LM_HEAD

tracer = trace.get_tracer(__name__)


ADAPTER_LAYERS = [
    ATTN_Q_PROJ,
    ATTN_K_PROJ,
    ATTN_V_PROJ,
    ATTN_O_PROJ,
    MLP_GATE_PROJ,
    MLP_UP_PROJ,
    MLP_DOWN_PROJ,
    LM_HEAD,
]
ROW_PARALLEL = {ATTN_O_PROJ, MLP_DOWN_PROJ, LM_HEAD}


class FlashQwen2(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        adapter_id: str,
        adapter_source: str,
        revision: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        embedding_dim: Optional[int] = None,
        **kwargs,
    ):
        model_cls = FlashQwen2ForEmbeddings if embedding_dim is not None else FlashQwen2ForCausalLM
        super().__init__(
            model_id=model_id,
            model_cls=model_cls,
            dtype=dtype,
            revision=revision,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            config_cls=Qwen2Config,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    @property
    def supports_adapter_loading(self) -> bool:
        return True

    @property
    def supports_embeddings(self) -> bool:
        return self._supports_embeddings

    @property
    def supports_text_generation(self) -> bool:
        return not self._supports_embeddings

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

            layer_weights[(i, MLP_GATE_PROJ)] = (
                f"{prefix}.{i}.mlp.gate_proj",
                layer.mlp.gate_up_proj,
            )
            layer_weights[(i, MLP_UP_PROJ)] = (f"{prefix}.{i}.mlp.up_proj", layer.mlp.gate_up_proj)
            layer_weights[(i, MLP_DOWN_PROJ)] = (f"{prefix}.{i}.mlp.down_proj", layer.mlp.down_proj)

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

    def embed(self, batch) -> torch.Tensor:
        adapter_meta = batch.adapter_meta
        prefill = False
        adapter_data = AdapterBatchData.from_meta(
            meta=adapter_meta,
            weights=self.layer_to_adapter_weights,
            layer_to_lora_weights={},
            punica_wrapper=None,
            prefill=prefill,
            prefill_head_indices=batch.prefill_head_indices,
        )
        embedding, _ = self.forward(batch, adapter_data=adapter_data)
        return embedding.cpu().tolist()

    def warmup(self, batch, max_new_tokens):
        return super().warmup(batch, max_new_tokens, embedding_model=self._supports_embeddings)
