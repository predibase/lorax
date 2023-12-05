import torch
import torch.distributed

from loguru import logger
from opentelemetry import trace
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import Dict, Optional, Tuple

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_qwen_modeling import (
    C_ATTN,
    C_PROJ,
    W1,
    W2,
    FlashQwenForCausalLM,
    QwenConfig,
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


class FlashQwen(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        adapter_id: str,
        adapter_source: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
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
            trust_remote_code=True,
        )

        config = QwenConfig.from_pretrained(
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

        if config.quantize == "gptq":
            weights._set_gptq_params(model_id)

        self.model_id = model_id
        model = FlashQwenForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashQwen, self).__init__(
            model=model,
            tokenizer=tokenizer,
            num_layers=len(model.transformer.h),
            num_kv_heads=model.transformer.num_key_value_heads,
            head_size=model.transformer.head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )
    
    @property
    def supports_adapter_loading(self) -> bool:
        return True
    
    def get_adaptable_weights(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        layer_weights = {}

        prefix = "transformer.h"
        for i, layer in enumerate(self.model.transformer.h):
            layer_weights[(i, C_ATTN)] = (f"{prefix}.{i}.attn.c_attn", layer.attn.c_attn)
            layer_weights[(i, C_PROJ)] = (f"{prefix}.{i}.attn.c_proj", layer.attn.c_proj)

            layer_weights[(i, W1)] = (f"{prefix}.{i}.mlp.w1", layer.mlp.gate_up_proj)
            layer_weights[(i, W2)] = (f"{prefix}.{i}.mlp.w2", layer.mlp.gate_up_proj)
            layer_weights[(i, C_PROJ)] = (f"{prefix}.{i}.mlp.c_proj", layer.mlp.c_proj)
        
        layer_weights[(0, LM_HEAD)] = ("lm_head", self.model.lm_head)
        return layer_weights
