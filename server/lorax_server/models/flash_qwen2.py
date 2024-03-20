import torch
import torch.distributed

from loguru import logger
from opentelemetry import trace
from transformers import AutoTokenizer
from transformers.models.qwen2 import Qwen2Config
from typing import Dict, List, Optional, Tuple

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_qwen2_modeling import (
    ATTN_Q_PROJ,
    ATTN_K_PROJ,
    ATTN_V_PROJ,
    ATTN_O_PROJ,
    MLP_GATE_PROJ,
    MLP_UP_PROJ,
    MLP_DOWN_PROJ,
    FlashQwen2ForCausalLM,
)
from lorax_server.utils import (
    create_merged_weight_files,
    initialize_torch_distributed,
    weight_files,
    Weights,
)
from lorax_server.utils.adapter import BASE_MODEL_ADAPTER_ID
from lorax_server.utils.lora import LM_HEAD

tracer = trace.get_tracer(__name__)


ADAPTER_LAYERS = [ATTN_Q_PROJ, ATTN_K_PROJ, ATTN_V_PROJ, ATTN_O_PROJ, MLP_GATE_PROJ, MLP_UP_PROJ, MLP_DOWN_PROJ, LM_HEAD]
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

        config = Qwen2Config.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")

        # if adapter_id passed in as part of model instantiation, then we merge 
        # the adapter weights with the model weights. This also disables dynamic
        # adapter loading, since the model is now itself initialized with an adapter.
        merged_weight_filenames = None
        dynamic_adapter_loading_enabled = True
        if len(adapter_id) > 0:
            logger.info(f"Merging adapter weights from adapter_id {adapter_id} into model weights.")
            # Need to pass the adapter source here
            merged_weight_filenames = create_merged_weight_files(
                adapter_id, model_id, model_weight_filenames=filenames, adapter_source=adapter_source
            )
            dynamic_adapter_loading_enabled = False
            adapter_id = adapter_id
        else:
            adapter_id = BASE_MODEL_ADAPTER_ID

        weights = Weights(
            filenames, 
            device, 
            dtype, 
            process_group=self.process_group, 
            merged_weight_filenames=merged_weight_filenames
        )

        if config.quantize in ["gptq", "awq", "eetq"]:
            weights._set_gptq_params(model_id)

        model = FlashQwen2ForCausalLM(config, weights)

        if config.sliding_window is None:
            config.sliding_window = config.max_position_embeddings
        self.config = config

        torch.distributed.barrier(group=self.process_group)
        super(FlashQwen2, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            num_layers=len(model.model.layers),
            num_kv_heads=model.model.num_key_value_heads,
            head_size=model.model.head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            sliding_window=config.sliding_window,
            compile=compile,
            adapter_id=adapter_id,
            dynamic_adapter_loading_enabled=dynamic_adapter_loading_enabled,
        )
    
    @property
    def supports_adapter_loading(self) -> bool:
        return True
    
    def adapter_target_to_layer(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        layer_weights = {}

        prefix = "model.layers"
        for i, layer in enumerate(self.model.model.layers):
            layer_weights[(i, ATTN_Q_PROJ)] = (f"{prefix}.{i}.self_attn.q_proj", layer.self_attn.query_key_value)
            layer_weights[(i, ATTN_K_PROJ)] = (f"{prefix}.{i}.self_attn.k_proj", layer.self_attn.query_key_value)
            layer_weights[(i, ATTN_V_PROJ)] = (f"{prefix}.{i}.self_attn.v_proj", layer.self_attn.query_key_value)
            layer_weights[(i, ATTN_O_PROJ)] = (f"{prefix}.{i}.self_attn.o_proj", layer.self_attn.o_proj)

            layer_weights[(i, MLP_GATE_PROJ)] = (f"{prefix}.{i}.mlp.gate_proj", layer.mlp.gate_up_proj)
            layer_weights[(i, MLP_UP_PROJ)] = (f"{prefix}.{i}.mlp.up_proj", layer.mlp.gate_up_proj)
            layer_weights[(i, MLP_DOWN_PROJ)] = (f"{prefix}.{i}.mlp.down_proj", layer.mlp.down_proj)
        
        layer_weights[(0, LM_HEAD)] = ("lm_head", self.model.lm_head)
        return layer_weights
    
    @property
    def adapter_layers(self) -> List[str]:
        return ADAPTER_LAYERS
    
    def get_num_layers_for_type(self, layer_type: str) -> int:
        return 1 if layer_type == LM_HEAD else len(self.model.model.layers)
    
    def is_row_parallel(self, layer_type: str) -> bool:
        return layer_type in ROW_PARALLEL
