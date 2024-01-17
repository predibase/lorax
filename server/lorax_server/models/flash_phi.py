import torch
import torch.distributed

from loguru import logger
from opentelemetry import trace
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_phi_modeling import (
    ATTN_WQKV,
    ATTN_OUT_PROJ,
    MLP_FC1,
    MLP_FC2,
    FlashPhiForCausalLM,
    PhiConfig,
)
from lorax_server.utils import (
    create_merged_weight_files,
    initialize_torch_distributed,
    weight_files,
    Weights,
)
from lorax_server.utils.adapter import BASE_MODEL_ADAPTER_ID
from lorax_server.utils.lora import LM_HEAD
from lorax_server.utils.weights import shard_on_dim

tracer = trace.get_tracer(__name__)


ADAPTER_LAYERS = [ATTN_WQKV, ATTN_OUT_PROJ, MLP_FC1, MLP_FC2, LM_HEAD]
ROW_PARALLEL = {ATTN_OUT_PROJ, MLP_FC2, LM_HEAD}


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

        config = PhiConfig.from_pretrained(
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
        model = FlashPhiForCausalLM(config, weights)
        self.config = config

        torch.distributed.barrier(group=self.process_group)
        super(FlashPhi, self).__init__(
            model=model,
            tokenizer=tokenizer,
            num_layers=len(model.transformer.h),
            num_kv_heads=model.transformer.num_key_value_heads,
            head_size=model.transformer.head_size,
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

        prefix = "transformer.h"
        for i, layer in enumerate(self.model.transformer.h):
            layer_weights[(i, ATTN_WQKV)] = (f"{prefix}.{i}.mixer.Wqkv", layer.mixer.Wqkv)
            layer_weights[(i, ATTN_OUT_PROJ)] = (f"{prefix}.{i}.mixer.out_proj", layer.mixer.out_proj)

            layer_weights[(i, MLP_FC1)] = (f"{prefix}.{i}.mlp.fc1", layer.mlp.fc1)
            layer_weights[(i, MLP_FC2)] = (f"{prefix}.{i}.mlp.fc2", layer.mlp.fc2)
        
        layer_weights[(0, LM_HEAD)] = ("lm_head.linear", self.model.lm_head.linear)
        return layer_weights
    
    @property
    def adapter_layers(self) -> List[str]:
        return ADAPTER_LAYERS
    
    def get_num_layers_for_type(self, layer_type: str) -> int:
        return 1 if layer_type == LM_HEAD else len(self.model.transformer.h)
    
    def is_row_parallel(self, layer_type: str) -> bool:
        return layer_type in ROW_PARALLEL
    
    def split_lora_b_qkv(self, t: torch.Tensor, head_size: int, num_heads: int, num_key_value_heads: int) -> torch.Tensor:
        # Because we're splitting on the hidden size dimension, we need to
        # account for the separate q, k, and v matrices.
        chunks = t.split(
            [
                head_size * num_heads,
                head_size * num_key_value_heads,
                head_size * num_key_value_heads,
            ],
            dim=1,
        )
        assert len(chunks) == 3
        chunks = [
            shard_on_dim(w, dim=1, process_group=self.process_group)
            for w in chunks
        ]
        return torch.cat(chunks, dim=1)
    
    def shard_lora_weights(
        self,
        weights_a: List[torch.Tensor],
        weights_b: List[torch.Tensor],
        layer_type: str,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # TODO(travis): genralize this for other layers and architectures
        if layer_type == ATTN_WQKV:
            # [hidden_size, r]
            split_dim = 0 if self.is_row_parallel(layer_type) else 1
            weights_a = [
                shard_on_dim(w, dim=split_dim, process_group=self.process_group)
                for w in weights_a
            ]

            # [r, hidden_size]
            # Because we're splitting on the hidden size dimension, we need to
            # account for the separate q, k, and v matrices.
            num_heads = self.config.n_head
            hidden_size = self.config.n_embd
            head_size = hidden_size // num_heads
            num_key_value_heads = getattr(self.config, "n_head_kv", None) or num_heads
            weights_b = [
                self.split_lora_b_qkv(w, head_size, num_heads, num_key_value_heads)
                for w in weights_b
            ]

            return weights_a, weights_b
        else:
            return super().shard_lora_weights(weights_a, weights_b, layer_type)
