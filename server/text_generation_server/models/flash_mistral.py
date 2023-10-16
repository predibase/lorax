import math
import torch
import torch.distributed

import numpy as np

from dataclasses import dataclass
from loguru import logger
from opentelemetry import trace
from transformers import PreTrainedTokenizerBase
from transformers.models.llama import LlamaTokenizerFast
from tqdm import tqdm
from typing import Optional, Tuple, Type

from text_generation_server.pb import generate_pb2
from text_generation_server.models import FlashCausalLM
from text_generation_server.models.flash_causal_lm import FlashCausalLMBatch, BLOCK_SIZE
from text_generation_server.models.cache_manager import (
    get_cache_manager,
)
from text_generation_server.models.custom_modeling.flash_mistral_modeling import (
    FlashMistralForCausalLM,
    MistralConfig,
)
from text_generation_server.utils import (
    compute_delta_weight,
    create_merged_weight_files,
    get_start_stop_idxs_for_rank,
    initialize_torch_distributed,
    load_module_map,
    weight_files,
    Weights,
    HeterogeneousNextTokenChooser,
    StoppingCriteria,
)
from text_generation_server.utils.adapter import BASE_MODEL_ADAPTER_ID

tracer = trace.get_tracer(__name__)

# Will be set in init
SLIDING_WINDOW: Optional[int] = None
SLIDING_WINDOW_BLOCKS: Optional[int] = None


# Adds windowing logic to FlashCausalLMBatch
@dataclass
class FlashMistralBatch(FlashCausalLMBatch):
    # Prefill cache indices is used to slice into the kv tensor before caching it into the paged attention buffers
    # as we only keep SLIDING_WINDOW values instead of the whole tensor
    prefill_cache_indices: Optional[torch.Tensor] = None

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "FlashCausalLMBatch":
        global SLIDING_WINDOW
        global SLIDING_WINDOW_BLOCKS

        batch_inputs = []
        max_truncation = 0
        for r in pb.requests:
            batch_inputs.append(r.inputs)
            max_truncation = max(max_truncation, r.truncate)

        batch_tokenized_inputs = tokenizer(
            batch_inputs, truncation=True, max_length=max_truncation
        )["input_ids"]

        position_ids = []
        cu_seqlen_prefill = [0]
        needed_blocks_slots = []
        start_slots = []
        slot_indices = []
        prefill_cache_indices = []

        input_lengths = []
        prefix_offsets = []
        read_offsets = []
        all_input_ids = []
        requests_idx_mapping = {}

        all_prefill_logprobs = True
        no_prefill_logprobs = True
        prefill_head_indices = []
        prefill_next_token_indices = []
        prefill_cu_outlens = [0]

        next_token_chooser_parameters = []
        stopping_criterias = []
        # TODO(geoffrey): re-add top_n_tokens functionality in a separate PR
        # top_n_tokens = []

        # Cumulative length
        cumulative_length = 0
        cumulative_max_length = 0
        prefill_out_cumulative_length = 0

        blocks = 0
        max_seqlen = 0
        max_length = 0
        max_blocks = 0

        # Parse batch
        for i, (r, tokenized_input) in enumerate(
            zip(pb.requests, batch_tokenized_inputs)
        ):
            # request id -> idx in list mapping
            requests_idx_mapping[r.id] = i

            tokenized_input = tokenized_input[-r.truncate :]

            input_length = len(tokenized_input)
            input_lengths.append(input_length)

            prefix_offsets.append(input_length - 5)
            read_offsets.append(input_length)

            all_input_ids.append(tokenized_input)

            # Position ids
            request_position_ids = torch.arange(0, input_length, dtype=torch.int32)
            position_ids.append(request_position_ids)

            # Add cumulative lengths of all previous inputs
            cu_seqlen_prefill.append(cumulative_length + input_length)

            next_token_chooser_parameters.append(r.parameters)

            stopping_criteria = StoppingCriteria.from_pb(
                r.stopping_parameters, tokenizer
            )
            max_new_tokens = stopping_criteria.max_new_tokens
            stopping_criterias.append(stopping_criteria)
            # top_n_tokens.append(r.top_n_tokens)

            # Paged attention
            # Remove one as the first token des not have a past
            total_tokens = input_length + max_new_tokens - 1

            # Needed blocks can not go over SLIDING_WINDOW_BLOCKS
            needed_blocks = min(
                math.ceil(total_tokens / BLOCK_SIZE), SLIDING_WINDOW_BLOCKS
            )
            blocks += needed_blocks

            needed_blocks_slots.append((needed_blocks, total_tokens))
            start_slots.append(cumulative_max_length)

            request_slot_indices = torch.arange(
                cumulative_max_length,
                cumulative_max_length + input_length,
                dtype=torch.int64,
            )
            slot_indices.append(request_slot_indices)

            # Create tensor to slice into the kv tensor in prefill
            request_prefill_cache_indices = torch.arange(
                cumulative_length + max(0, input_length - SLIDING_WINDOW),
                cumulative_length + input_length,
                dtype=torch.int64,
            )
            prefill_cache_indices.append(request_prefill_cache_indices)

            all_prefill_logprobs = all_prefill_logprobs and r.prefill_logprobs
            no_prefill_logprobs = no_prefill_logprobs and not r.prefill_logprobs

            if r.prefill_logprobs:
                prefill_head_indices.append(request_position_ids + cumulative_length)
                prefill_next_token_indices.append(
                    prefill_out_cumulative_length + input_length - 1
                )
                prefill_cu_outlens.append(prefill_out_cumulative_length + input_length)
                prefill_out_cumulative_length += input_length
            else:
                prefill_head_indices.append(
                    torch.tensor(
                        [cumulative_length + input_length - 1], dtype=torch.int32
                    )
                )
                prefill_next_token_indices.append(prefill_out_cumulative_length)
                prefill_cu_outlens.append(prefill_out_cumulative_length + 1)
                prefill_out_cumulative_length += 1

            # Update
            cumulative_length += input_length
            cumulative_max_length += total_tokens
            max_seqlen = max(max_seqlen, input_length)
            max_blocks = max(max_blocks, needed_blocks)
            max_length = max(max_length, input_length + max_new_tokens)

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters, dtype, device
        )
        start_slots = torch.tensor(start_slots, dtype=torch.int64)

        # Padded all_input_ids_tensor
        all_input_ids_tensor = np.zeros(
            (len(all_input_ids), max_length), dtype=np.int64
        )
        for i, input_ids in enumerate(all_input_ids):
            all_input_ids_tensor[i, : len(input_ids)] = input_ids

        # Create tensors on device
        all_input_ids_tensor = torch.tensor(
            all_input_ids_tensor, dtype=torch.int64, device=device
        )

        if len(pb.requests) > 1:
            input_ids = np.concatenate(all_input_ids, dtype=np.int64)
            position_ids = torch.cat(position_ids)
            slot_indices = torch.cat(slot_indices)
            prefill_cache_indices = torch.cat(prefill_cache_indices)
        else:
            input_ids = all_input_ids[0]
            position_ids = position_ids[0]
            slot_indices = slot_indices[0]
            prefill_cache_indices = prefill_cache_indices[0]

        cu_seqlen_prefill = torch.tensor(
            cu_seqlen_prefill, device=device, dtype=torch.int32
        )

        position_ids = position_ids.to(device)
        slot_indices = slot_indices.to(device)
        prefill_cache_indices = prefill_cache_indices.to(device)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        input_lengths_tensor = torch.tensor(
            input_lengths, dtype=torch.int32, device=device
        )

        if all_prefill_logprobs:
            prefill_head_indices = None
            prefill_next_token_indices = cu_seqlen_prefill[1:] - 1
        elif no_prefill_logprobs:
            prefill_head_indices = cu_seqlen_prefill[1:] - 1
            prefill_next_token_indices = None
        else:
            prefill_head_indices = torch.tensor(
                torch.cat(prefill_head_indices), dtype=torch.int64, device=device
            )
            prefill_next_token_indices = torch.tensor(
                prefill_next_token_indices, dtype=torch.int64, device=device
            )
        # top_n_tokens_tensor = torch.tensor(
        #     top_n_tokens, device=device, dtype=torch.int64
        # )

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            start_slots=start_slots,
            slot_indices=slot_indices,
            needed_blocks_slots=needed_blocks_slots,
            block_tables=None,
            block_tables_tensor=None,
            slots=None,
            max_seqlen=max_seqlen,
            prefill_head_indices=prefill_head_indices,
            prefill_next_token_indices=prefill_next_token_indices,
            prefill_cu_outlens=prefill_cu_outlens,
            input_lengths=input_lengths,
            input_lengths_tensor=input_lengths_tensor,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            # top_n_tokens=top_n_tokens,
            # top_n_tokens_tensor=top_n_tokens_tensor,
            blocks=blocks,
            max_blocks=max_blocks,
            prefill_cache_indices=prefill_cache_indices,
        )


class FlashMistral(FlashCausalLM):
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
        global SLIDING_WINDOW
        global SLIDING_WINDOW_BLOCKS

        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashLlama is only available on GPU")

        tokenizer = LlamaTokenizerFast.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = MistralConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize

        # Set context windows
        SLIDING_WINDOW = config.sliding_window
        SLIDING_WINDOW_BLOCKS = math.ceil(config.sliding_window / BLOCK_SIZE)

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

        if config.quantize in ["gptq"]:
            weights._set_gptq_params(model_id)

        self.model_id = model_id
        model = FlashMistralForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashMistral, self).__init__(
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

    def load_adapter(self, adapter_id, adapter_source):
        # NOTE: this implementation of `load_adapter` looks VERY similar to the 
        # one in FlashLlama, but we duplicate it here because they are 
        # fundamentally different models, and we want to make it easy to fix
        # model-specific bugs and adding capabilities without breaking others.
        # This philosophy is reflected throughout this repository and in the 
        # broader HuggingFace Transformers ecosystem. More here:
        # https://huggingface.co/blog/transformers-design-philosophy
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
                # replace the target matrices in place
                q_proj, _, v_proj = layer.self_attn.get_query_key_value_weights(clone=False)

                # place original weights (on their original device) by setting in place
                orig_q_proj, orig_q_proj_device = self.orig_weights[f"{prefix}.{i}.self_attn.q_proj"]
                q_proj[:] = orig_q_proj.to(orig_q_proj_device)
                orig_v_proj, orig_v_proj_device = self.orig_weights[f"{prefix}.{i}.self_attn.v_proj"]
                v_proj[:] = orig_v_proj.to(orig_v_proj_device)

                self.adapter_id = adapter_id
        else:
            weight_names = tuple(self.orig_weights.keys())
            module_map, adapter_config = load_module_map(self.model_id, adapter_id, adapter_source, weight_names)
            
            # TODO(geoffrey): merge this with function
            # text_generation_server/utils/adapter.py::merge_adapter_weights
            def compute_merged_weight(weight_name):
                # load the original weights from CPU back onto the device they were on
                orig_weight, orig_device = self.orig_weights[weight_name]
                orig_weight = orig_weight.to(orig_device)
                # ensure the delta has the same dtype and device as the original weight
                lora_A = module_map[weight_name]["lora_A"].to(orig_device, orig_weight.dtype)
                lora_B = module_map[weight_name]["lora_B"].to(orig_device, orig_weight.dtype)
                delta_weight = compute_delta_weight(
                    lora_A, 
                    lora_B, 
                    adapter_config.fan_in_fan_out, 
                    adapter_config.lora_alpha, 
                    adapter_config.r
                )
                start, stop = get_start_stop_idxs_for_rank(delta_weight.shape[0], self.process_group.rank(), self.process_group.size())
                return orig_weight + delta_weight[start:stop]
            
            prefix = "model.layers"
            for i, layer in tqdm(
                enumerate(self.model.model.layers), 
                desc=f"Merging weights for adapter {adapter_id}",
                total=len(self.model.model.layers)
            ):
                # replace the target matrices in place
                q_proj, _, v_proj = layer.self_attn.get_query_key_value_weights(clone=False)
                q_proj[:] = compute_merged_weight(f"{prefix}.{i}.self_attn.q_proj")
                v_proj[:] = compute_merged_weight(f"{prefix}.{i}.self_attn.v_proj")
            self.adapter_id = adapter_id

    @property
    def batch_type(self) -> Type[FlashMistralBatch]:
        return FlashMistralBatch

    def forward(self, batch: FlashMistralBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        # Model Forward
        logits = self.model.forward(
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            cu_seqlen_prefill=batch.cu_seqlen_prefill,
            kv_cache=get_cache_manager().kv_cache,
            block_tables=batch.block_tables_tensor,
            slots=batch.slots[batch.slot_indices],
            input_lengths=batch.input_lengths_tensor,
            max_s=batch.max_seqlen,
            prefill_cache_indices=batch.prefill_cache_indices,
            lm_head_indices=batch.prefill_head_indices,
        )
        if batch.prefill_cache_indices is not None:
            batch.prefill_cache_indices = None
        return logits