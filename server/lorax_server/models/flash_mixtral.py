import math
import torch
import torch.distributed

import numpy as np

from dataclasses import dataclass
from loguru import logger
from opentelemetry import trace
from transformers import PreTrainedTokenizerBase
from transformers.models.llama import LlamaTokenizerFast
from typing import Dict, List, Optional, Tuple, Type

from lorax_server.pb import generate_pb2
from lorax_server.models import FlashCausalLM
from lorax_server.models.flash_causal_lm import FlashCausalLMBatch, BLOCK_SIZE
from lorax_server.models.cache_manager import (
    get_cache_manager,
)
from lorax_server.models.custom_modeling.flash_mixtral_modeling import (
    ATTN_K_PROJ,
    ATTN_O_PROJ,
    ATTN_Q_PROJ,
    ATTN_V_PROJ,
    MOE_W1,
    MOE_W2,
    MOE_W3,
    FlashMixtralForCausalLM,
    MixtralConfig,
)
from lorax_server.utils import (
    create_merged_weight_files,
    initialize_torch_distributed,
    weight_files,
    Weights,
    HeterogeneousNextTokenChooser,
    StoppingCriteria,
)
from lorax_server.utils.adapter import BASE_MODEL_ADAPTER_ID
from lorax_server.utils.lora import LM_HEAD, AdapterBatchData, AdapterBatchMetadata
from lorax_server.utils.segments import find_segments

tracer = trace.get_tracer(__name__)

# Will be set in init
SLIDING_WINDOW: Optional[int] = None
SLIDING_WINDOW_BLOCKS: Optional[int] = None

ADAPTER_LAYERS = [ATTN_Q_PROJ, ATTN_K_PROJ, ATTN_V_PROJ, ATTN_O_PROJ, LM_HEAD]
ROW_PARALLEL = {ATTN_O_PROJ, LM_HEAD}


# Adds windowing logic to FlashCausalLMBatch
@dataclass
class FlashMixtralBatch(FlashCausalLMBatch):
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

        adapter_indices_list = []
        adapter_set = set()

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

            adapter_indices_list.append(torch.full((input_length,), r.adapter_index))
            adapter_set.add(r.adapter_index)

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

        adapter_indices = torch.cat(adapter_indices_list).to(dtype=torch.int64, device=device)
        
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

        adapter_segments, adapter_segment_indices = find_segments(adapter_indices)
        adapter_segments = torch.tensor(adapter_segments, dtype=torch.int32, device=device)

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
            adapter_meta=AdapterBatchMetadata(
                adapter_indices=adapter_indices,
                adapter_set=adapter_set,
                adapter_segments=adapter_segments,
                segment_indices=adapter_segment_indices,
            ),
            prefill_cache_indices=prefill_cache_indices,
        )


class FlashMixtral(FlashCausalLM):
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

        config = MixtralConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize

        if config.sliding_window is None:
            config.sliding_window = config.max_position_embeddings

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

        if config.quantize in ["gptq", "awq"]:
            weights._set_gptq_params(model_id)

        self.model_id = model_id
        model = FlashMixtralForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashMixtral, self).__init__(
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
        )

    @property
    def supports_adapter_loading(self) -> bool:
        return True

    @property
    def batch_type(self) -> Type[FlashMixtralBatch]:
        return FlashMixtralBatch

    def forward(self, batch: FlashMixtralBatch, adapter_data: AdapterBatchData) -> Tuple[torch.Tensor, torch.Tensor]:
        prefill = batch.cu_seqlen_prefill is not None
        model = self.model
        if (
            self.model_graph_wrapper is not None and
            not prefill and
            self.model_graph_wrapper.can_use_graph(batch, adapter_data)
        ):
            model = self.model_graph_wrapper
        
        # Model Forward
        logits = model.forward(
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            cu_seqlen_prefill=batch.cu_seqlen_prefill,
            kv_cache=get_cache_manager().kv_cache,
            block_tables=batch.block_tables_tensor,
            slots=batch.slots[batch.slot_indices],
            input_lengths=batch.input_lengths_tensor,
            max_s=batch.max_seqlen,
            adapter_data=adapter_data,
            prefill_cache_indices=batch.prefill_cache_indices,
            lm_head_indices=batch.prefill_head_indices,
        )
        if batch.prefill_cache_indices is not None:
            batch.prefill_cache_indices = None
        return logits
    
    def adapter_target_to_layer(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        layer_weights = {}

        prefix = "model.layers"
        for i, layer in enumerate(self.model.model.layers):
            layer_weights[(i, ATTN_Q_PROJ)] = (f"{prefix}.{i}.self_attn.q_proj", layer.self_attn.query_key_value)
            layer_weights[(i, ATTN_K_PROJ)] = (f"{prefix}.{i}.self_attn.k_proj", layer.self_attn.query_key_value)
            layer_weights[(i, ATTN_V_PROJ)] = (f"{prefix}.{i}.self_attn.v_proj", layer.self_attn.query_key_value)
            layer_weights[(i, ATTN_O_PROJ)] = (f"{prefix}.{i}.self_attn.o_proj", layer.self_attn.o_proj)

            # TODO(travis): requires implementing this for block sparse MoE
            # layer_weights[(i, MOE_W1)] = (f"{prefix}.{i}.moe.w1", layer.moe.w1)
            # layer_weights[(i, MOE_W2)] = (f"{prefix}.{i}.moe.w2", layer.moe.w2)
            # layer_weights[(i, MOE_W3)] = (f"{prefix}.{i}.moe.w3", layer.moe.w3)
        
        layer_weights[(0, LM_HEAD)] = ("lm_head", self.model.lm_head)
        return layer_weights
    
    @property
    def adapter_layers(self) -> List[str]:
        return ADAPTER_LAYERS
    
    def get_num_layers_for_type(self, layer_type: str) -> int:
        return 1 if layer_type == LM_HEAD else len(self.model.model.layers)
    
    def is_row_parallel(self, layer_type: str) -> bool:
        return layer_type in ROW_PARALLEL
