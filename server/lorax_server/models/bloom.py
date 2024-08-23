from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.distributed
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from lorax_server.adapters import AdapterBatchData
from lorax_server.models.causal_lm import CausalLM, CausalLMBatch
from lorax_server.models.custom_modeling.bloom_modeling import (
    ATTN_DENSE,
    ATTN_QKV,
    LM_HEAD,
    MLP_DENSE_4H_TO_H,
    MLP_DENSE_H_TO_4H,
    BloomForCausalLM,
)
from lorax_server.pb import generate_pb2
from lorax_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
)
from lorax_server.utils.tokenizer import TokenizerManager

ADAPTER_LAYERS = [ATTN_QKV, ATTN_DENSE, MLP_DENSE_H_TO_4H, MLP_DENSE_4H_TO_H]
ROW_PARALLEL = {ATTN_DENSE, MLP_DENSE_4H_TO_H}


class BloomCausalLMBatch(CausalLMBatch):
    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        tokenizers: TokenizerManager,
        processor,
        config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "CausalLMBatch":
        batch = super().from_pb(
            pb=pb,
            tokenizer=tokenizer,
            tokenizers=tokenizers,
            processor=processor,
            config=config,
            dtype=dtype,
            device=device,
        )
        batch.keys_head_dim_last = False
        return batch


class BLOOMSharded(CausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        compile: bool = False,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        if compile:
            raise ValueError("`--compile` is not supported with Bloom")

        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            slow_but_exact=False,
            tp_parallel=True,
            trust_remote_code=trust_remote_code,
        )
        config.pad_token_id = 3
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device=device, dtype=dtype, process_group=self.process_group)
        weights._set_config(model_id, config)

        model = BloomForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(CausalLM, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            trust_remote_code=trust_remote_code,
        )

        self.dynamic_adapter_loading_enabled = True

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return BloomCausalLMBatch

    @property
    def has_adapter_data(self) -> bool:
        return True

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values: Optional = None,
        adapter_data: Optional[AdapterBatchData] = None,
    ):
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            adapter_data=adapter_data,
        )

        logits = outputs.logits
        return logits, outputs.past_key_values

    @property
    def supports_adapter_loading(self) -> bool:
        return True

    def adapter_target_to_layer(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        layer_weights = {}

        prefix = "transformer.h"
        for i, layer in enumerate(self.model.transformer.h):
            layer_weights[(i, ATTN_QKV)] = (
                f"{prefix}.{i}.self_attention.query_key_value",
                layer.self_attention.query_key_value,
            )
            layer_weights[(i, ATTN_DENSE)] = (
                f"{prefix}.{i}.self_attention.dense",
                layer.self_attention.dense,
            )

            layer_weights[(i, MLP_DENSE_H_TO_4H)] = (
                f"{prefix}.{i}.mlp.dense_h_to_4h",
                layer.mlp.dense_h_to_4h,
            )
            layer_weights[(i, MLP_DENSE_4H_TO_H)] = (
                f"{prefix}.{i}.mlp.dense_4h_to_h",
                layer.mlp.dense_4h_to_h,
            )

        # TODO: make Embedding layers adapter-compatible
        # layer_weights[(0, LM_HEAD)] = ("transformer.wte", self.model.transformer.wte)
        return layer_weights

    @property
    def adapter_layers(self) -> List[str]:
        return ADAPTER_LAYERS

    def get_num_layers_for_type(self, layer_type: str) -> int:
        return 1 if layer_type == LM_HEAD else len(self.model.transformer.h)

    def is_row_parallel(self, layer_type: str) -> bool:
        return layer_type in ROW_PARALLEL
