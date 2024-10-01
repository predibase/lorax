from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from opentelemetry import trace
from transformers import AutoTokenizer
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
from lorax_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
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
        quantize: Optional[str] = None,
        compile: bool = False,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        embedding_dim: Optional[int] = None,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashQwen is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = Qwen2Config.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
        config.quantize = quantize
        if not config.use_sliding_window:
            config.sliding_window = None

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors", embedding_dim=embedding_dim)
        weights = Weights(
            filenames,
            device,
            dtype,
            process_group=self.process_group,
        )
        weights._set_config(model_id, config)

        self._supports_embeddings = embedding_dim is not None

        if not weights.has_tensor("lm_head.weight") and not self._supports_embeddings:
            raise ValueError(
                "Model does not have lm head so it is presumed to be for embeddings."
                "No embedding_dim was provided so we cannot load the model."
                "Please pass in an embedding_dim to the model."
            )

        if self._supports_embeddings:
            model = FlashQwen2ForEmbeddings(config, weights)
        else:
            model = FlashQwen2ForCausalLM(config, weights)

        self.config = config

        torch.distributed.barrier(group=self.process_group)
        super(FlashQwen2, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            num_layers=len(model.model.layers),
            num_kv_heads=model.model.num_key_value_heads,
            head_size=model.model.head_size,
            num_heads=model.model.num_heads,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            sliding_window=config.sliding_window,
            compile=compile,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            trust_remote_code=trust_remote_code,
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
            adapter_meta, self.layer_to_adapter_weights, prefill, batch.prefill_head_indices
        )
        embedding, _ = self.forward(batch, adapter_data=adapter_data)
        return embedding.cpu().tolist()

    def warmup(self, batch, max_new_tokens):
        return super().warmup(batch, max_new_tokens, embedding_model=self._supports_embeddings)
