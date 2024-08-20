from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from opentelemetry import trace
from transformers import AutoTokenizer
from transformers.models.phi.modeling_phi import PhiConfig

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_phi_modeling import (
    ATTN_DENSE,
    ATTN_K_PROJ,
    ATTN_Q_PROJ,
    ATTN_V_PROJ,
    MLP_FC1,
    MLP_FC2,
    FlashPhiForCausalLM,
)
from lorax_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
)
from lorax_server.utils.lora import LM_HEAD

tracer = trace.get_tracer(__name__)


ADAPTER_LAYERS = [ATTN_Q_PROJ, ATTN_K_PROJ, ATTN_V_PROJ, ATTN_DENSE, MLP_FC1, MLP_FC2, LM_HEAD]
ROW_PARALLEL = {ATTN_DENSE, MLP_FC2, LM_HEAD}


class FlashPhi(FlashCausalLM):
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
            raise NotImplementedError("FlashPhi is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = PhiConfig.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
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

        model = FlashPhiForCausalLM(config, weights)
        self.config = config

        torch.distributed.barrier(group=self.process_group)
        super(FlashPhi, self).__init__(
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

        prefix = "model.layers"
        for i, layer in enumerate(self.model.model.layers):
            layer_weights[(i, ATTN_Q_PROJ)] = (
                f"{prefix}.{i}.self_attn.q_proj",
                layer.self_attn.qkv_proj,
            )
            layer_weights[(i, ATTN_K_PROJ)] = (
                f"{prefix}.{i}.self_attn.k_proj",
                layer.self_attn.qkv_proj,
            )
            layer_weights[(i, ATTN_V_PROJ)] = (
                f"{prefix}.{i}.self_attn.v_proj",
                layer.self_attn.qkv_proj,
            )
            layer_weights[(i, ATTN_DENSE)] = (
                f"{prefix}.{i}.self_attn.dense",
                layer.self_attn.dense,
            )

            layer_weights[(i, MLP_FC1)] = (f"{prefix}.{i}.mlp.fc1", layer.mlp.fc1)
            layer_weights[(i, MLP_FC2)] = (f"{prefix}.{i}.mlp.fc2", layer.mlp.fc2)

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
