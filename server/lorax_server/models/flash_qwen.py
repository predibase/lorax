from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from opentelemetry import trace
from transformers import AutoTokenizer

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
from lorax_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
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
        quantize: Optional[str] = None,
        compile: bool = False,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
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

        config = QwenConfig.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames,
            device,
            dtype,
            process_group=self.process_group,
        )
        weights._set_config(model_id, config)

        model = FlashQwenForCausalLM(config, weights)
        self.config = config

        torch.distributed.barrier(group=self.process_group)
        super(FlashQwen, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            num_layers=len(model.transformer.h),
            num_kv_heads=model.transformer.num_key_value_heads,
            head_size=model.transformer.head_size,
            num_heads=model.transformer.num_heads,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            compile=compile,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            trust_remote_code=trust_remote_code,
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
