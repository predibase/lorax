import torch
import torch.distributed

from loguru import logger
from opentelemetry import trace
from transformers.models.llama import LlamaTokenizer, LlamaTokenizerFast
from tqdm import tqdm
from typing import Optional

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_llama_modeling import (
    FlashLlamaForCausalLM,
    LlamaConfig,
)
from text_generation_server.utils import (
    compute_delta_weight,
    create_merged_weight_files,
    get_start_stop_idxs_for_rank,
    initialize_torch_distributed,
    load_module_map,
    weight_files,
    Weights,
)
from text_generation_server.utils.adapter import BASE_MODEL_ADAPTER_ID

tracer = trace.get_tracer(__name__)


class FlashLlama(FlashCausalLM):
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
            raise NotImplementedError("FlashLlama is only available on GPU")

        try:
            tokenizer = LlamaTokenizer.from_pretrained(
                model_id,
                revision=revision,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            tokenizer = LlamaTokenizerFast.from_pretrained(
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

        if config.quantize == "gptq":
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
        )

        # holds the original weights and the devices they were on as a tuple
        # the original weights are stored in CPU memory, but placed into `device`
        # as needed. Only needed when dynamic_adapter_loading_enabled is True.
        self.orig_weights = None
        if self.dynamic_adapter_loading_enabled:
            # TODO(geoffrey): generalize to non-q_proj and non-v_proj layers
            self.orig_weights = {}
            prefix = "model.layers"
            for i, layer in enumerate(self.model.model.layers):
                q_proj, _, v_proj = layer.self_attn.get_query_key_value_weights(clone=True)

                orig_q_proj_device = q_proj.device
                weight_name = f"{prefix}.{i}.self_attn.q_proj"
                self.orig_weights[weight_name] = (q_proj.cpu(), orig_q_proj_device)
                
                orig_v_proj_device = v_proj.device
                weight_name = f"{prefix}.{i}.self_attn.v_proj"
                self.orig_weights[weight_name] = (v_proj.cpu(), orig_v_proj_device)
    
    def load_adapter(self, adapter_id, adapter_source, adapter_index):
        if not self.dynamic_adapter_loading_enabled:
            if adapter_id == BASE_MODEL_ADAPTER_ID:
                return
            else:
                raise ValueError(f"This model was initialized with the adapter {self.adapter_id} "
                                f"and therefore does not support dynamic adapter loading. "
                                f"Please initialize a new model instance from the base model in "
                                f"order to use the dynamic adapter loading feature.")

        # If we are doing dynamic adapter loading, then we need to reset the weights
        if adapter_id == self.adapter_id:
            return
        elif adapter_id == BASE_MODEL_ADAPTER_ID:
            # if the adapter_id is the base model, then just reset the weights
            prefix = "model.layers"
            for i, layer in enumerate(self.model.model.layers):
                qkv_d, _ = layer.self_attn.query_key_value.base_layer.linear.weight.shape
                q_d = qkv_d // 3  # break up qkv_d into 3 parts
                
                # place the original weights (on their original device) back into the model
                q_proj, q_proj_device = self.orig_weights[f"{prefix}.{i}.self_attn.q_proj"]
                layer.self_attn.query_key_value.base_layer.linear.weight[:q_d] = q_proj.to(q_proj_device)
                v_proj, v_proj_device = self.orig_weights[f"{prefix}.{i}.self_attn.v_proj"]
                layer.self_attn.query_key_value.base_layer.linear.weight[2*q_d:] = v_proj.to(v_proj_device)
                self.adapter_id = adapter_id
        else:
            weight_names = tuple(self.orig_weights.keys())
            module_map, adapter_config = load_module_map(self.model_id, adapter_id, adapter_source, weight_names)
            
            prefix = "model.layers"
            for i, layer in tqdm(
                enumerate(self.model.model.layers), 
                desc=f"Merging weights for adapter {adapter_id}",
                total=len(self.model.model.layers)
            ):
                layer = layer.self_attn.query_key_value
                base_weight = layer.base_layer.linear.weight
                base_device = base_weight.device

                weight_name = f"{prefix}.{i}.self_attn.q_proj"
                q_lora_a = module_map[weight_name]["lora_A"].to(base_device, base_weight.dtype)
                q_lora_b = module_map[weight_name]["lora_B"].to(base_device, base_weight.dtype)

                weight_name = f"{prefix}.{i}.self_attn.v_proj"
                v_lora_a = module_map[weight_name]["lora_A"].to(base_device, base_weight.dtype)
                v_lora_b = module_map[weight_name]["lora_B"].to(base_device, base_weight.dtype)

                # TODO(travis): remove adapter
                layer.add_adapter(
                    (q_lora_a, q_lora_b),
                    (v_lora_a, v_lora_b),
                    adapter_config,
                    self.process_group,
                    adapter_index,
                )

            self.adapter_id = adapter_id

    def offload_adapter(self, adapter_id, adapter_source, adapter_index):
        if not self.dynamic_adapter_loading_enabled:
            if adapter_id == BASE_MODEL_ADAPTER_ID:
                return
            else:
                raise ValueError(f"This model was initialized with the adapter {self.adapter_id} "
                                f"and therefore does not support dynamic adapter loading. "
                                f"Please initialize a new model instance from the base model in "
                                f"order to use the dynamic adapter loading feature.")

        if adapter_id == BASE_MODEL_ADAPTER_ID:
            return
        else:
            for layer in self.model.model.layers:
                layer = layer.self_attn.query_key_value
                layer.remove_adapter(adapter_index)

            self.adapter_id = BASE_MODEL_ADAPTER_ID
