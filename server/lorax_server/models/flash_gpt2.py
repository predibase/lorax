from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from opentelemetry import trace
from transformers import AutoTokenizer

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_gpt2_modeling import (
    ATTN_C_ATTN,
    ATTN_C_PROJ,
    LM_HEAD,
    MLP_C_FC,
    MLP_C_PROJ,
    FlashGPT2ForCausalLM,
    GPT2Config,
)
from lorax_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
)

tracer = trace.get_tracer(__name__)

ADAPTER_LAYERS = [ATTN_C_ATTN, ATTN_C_PROJ, MLP_C_FC, MLP_C_PROJ]
ROW_PARALLEL = {ATTN_C_PROJ, MLP_C_PROJ}


class FlashGPT2(FlashCausalLM):
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
            raise NotImplementedError("FlashLlama is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = GPT2Config.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
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

        model = FlashGPT2ForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashGPT2, self).__init__(
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
            layer_weights[(i, ATTN_C_ATTN)] = (f"{prefix}.{i}.{ATTN_C_ATTN}", layer.attn.c_attn)
            layer_weights[(i, ATTN_C_PROJ)] = (f"{prefix}.{i}.{ATTN_C_PROJ}", layer.attn.c_proj)

            layer_weights[(i, MLP_C_FC)] = (f"{prefix}.{i}.{MLP_C_FC}", layer.mlp.c_fc)
            layer_weights[(i, MLP_C_PROJ)] = (f"{prefix}.{i}.{MLP_C_PROJ}", layer.mlp.c_proj)

        # TODO: make Embedding layers adapter-compatible
        # layer_weights[(0, LM_HEAD)] = ("transformer.wte", self.model.transformer.wte)
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
