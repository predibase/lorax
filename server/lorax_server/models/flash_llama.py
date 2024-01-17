import torch
import torch.distributed

from loguru import logger
from opentelemetry import trace
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_llama_modeling import (
    FlashLlamaForCausalLM,
    LlamaConfig,
)
from lorax_server.utils import (
    create_merged_weight_files,
    initialize_torch_distributed,
    weight_files,
    Weights,
)
from lorax_server.utils.adapter import BASE_MODEL_ADAPTER_ID
from lorax_server.utils.lora import DOWN_PROJ, GATE_PROJ, K_PROJ, LM_HEAD, O_PROJ, Q_PROJ, UP_PROJ, V_PROJ

tracer = trace.get_tracer(__name__)


ADAPTER_LAYERS = [Q_PROJ, K_PROJ, V_PROJ, O_PROJ, GATE_PROJ, UP_PROJ, DOWN_PROJ, LM_HEAD]
ROW_PARALLEL = {O_PROJ, DOWN_PROJ, LM_HEAD}


class FlashLlama(FlashCausalLM):
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

        config = LlamaConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")

        # if adapter_id passed in as part of model instantiation, then we merge 
        # the adapter weights with the model weights. This also disables dynamic
        # adapter loading, since the model is now itself initialized with an adapter.
        merged_weight_filenames = None
        self.dynamic_adapter_loading_enabled = True
        self.adapter_id = BASE_MODEL_ADAPTER_ID
        if len(adapter_id) > 0:
            logger.info(f"Merging adapter weights from adapter_id {adapter_id} into model weights.")
            # Need to pass the adapter source here
            merged_weight_filenames = create_merged_weight_files(
                adapter_id, model_id, model_weight_filenames=filenames, adapter_source=adapter_source
            )
            self.dynamic_adapter_loading_enabled = False
            self.adapter_id = adapter_id

        weights = Weights(
            filenames, 
            device, 
            dtype, 
            process_group=self.process_group, 
            merged_weight_filenames=merged_weight_filenames
        )

        if config.quantize in ["gptq", "awq"]:
            weights._set_gptq_params(model_id)

        self.model_id = model_id
        model = FlashLlamaForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashLlama, self).__init__(
            model=model,
            tokenizer=tokenizer,
            num_layers=len(model.model.layers),
            num_kv_heads=model.model.num_key_value_heads,
            head_size=model.model.head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            compile=compile,
        )
    
    @property
    def supports_adapter_loading(self) -> bool:
        return True
    
    def adapter_target_to_layer(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        layer_weights = {}

        prefix = "model.layers"
        for i, layer in enumerate(self.model.model.layers):
            layer_weights[(i, Q_PROJ)] = (f"{prefix}.{i}.self_attn.q_proj", layer.self_attn.query_key_value)
            layer_weights[(i, K_PROJ)] = (f"{prefix}.{i}.self_attn.k_proj", layer.self_attn.query_key_value)
            layer_weights[(i, V_PROJ)] = (f"{prefix}.{i}.self_attn.v_proj", layer.self_attn.query_key_value)
            layer_weights[(i, O_PROJ)] = (f"{prefix}.{i}.self_attn.o_proj", layer.self_attn.o_proj)

            layer_weights[(i, GATE_PROJ)] = (f"{prefix}.{i}.mlp.gate_proj", layer.mlp.gate_up_proj)
            layer_weights[(i, UP_PROJ)] = (f"{prefix}.{i}.mlp.up_proj", layer.mlp.gate_up_proj)
            layer_weights[(i, DOWN_PROJ)] = (f"{prefix}.{i}.mlp.down_proj", layer.mlp.down_proj)
        
        layer_weights[(0, LM_HEAD)] = ("lm_head", self.model.lm_head)
        return layer_weights
    
    @property
    def adapter_layers(self) -> List[str]:
        return ADAPTER_LAYERS
    
    def get_num_layers_for_type(self, layer_type: str) -> int:
        return 1 if layer_type == LM_HEAD else len(self.model.model.layers)
    
    def is_row_parallel(self, layer_type: str) -> bool:
        return layer_type in ROW_PARALLEL
