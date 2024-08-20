from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from opentelemetry import trace
from transformers import AutoTokenizer
from transformers.models.gpt2 import GPT2TokenizerFast

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_dbrx_modeling import (
    ATTN_O_PROJ,
    ATTN_WQKV,
    DbrxConfig,
    FlashDbrxForCausalLM,
)
from lorax_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
)
from lorax_server.utils.lora import LM_HEAD

tracer = trace.get_tracer(__name__)

ADAPTER_LAYERS = [ATTN_WQKV, ATTN_O_PROJ, LM_HEAD]
ROW_PARALLEL = {ATTN_O_PROJ, LM_HEAD}


class FlashDbrx(FlashCausalLM):
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
            dtype = torch.bfloat16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashDBRX is only available on GPU")

        try:
            tokenizer = GPT2TokenizerFast.from_pretrained(
                model_id,
                revision=revision,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=trust_remote_code,
                use_fast=True,
                from_slow=False,
            )
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    revision=revision,
                    padding_side="left",
                    truncation_side="left",
                    trust_remote_code=trust_remote_code,
                    use_fast=True,
                    from_slow=False,
                )
            except Exception:
                # FIXME: change back to model id once the tokenizer.json is merged
                tokenizer = GPT2TokenizerFast.from_pretrained(
                    "Xenova/dbrx-instruct-tokenizer",
                    revision=revision,
                    padding_side="left",
                    truncation_side="left",
                    trust_remote_code=trust_remote_code,
                    use_fast=True,
                    from_slow=False,
                )

        config = DbrxConfig.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
        config.quantize = quantize
        config.max_position_embeddings = config.max_seq_len

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames,
            device,
            dtype,
            process_group=self.process_group,
        )
        weights._set_config(model_id, config)

        model = FlashDbrxForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashDbrx, self).__init__(
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

        prefix = "transformer.blocks"
        for i, layer in enumerate(self.model.model.layers):
            layer_weights[(i, ATTN_WQKV)] = (
                f"{prefix}.{i}.norm_attn_norm.attn.q_proj",
                layer.attn.self_attn.query_key_value,
            )
            layer_weights[(i, ATTN_O_PROJ)] = (
                f"{prefix}.{i}.norm_attn_norm.attn.out_proj",
                layer.attn.self_attn.o_proj,
            )

        layer_weights[(0, LM_HEAD)] = ("lm_head", self.model.lm_head)
        return layer_weights

    @property
    def adapter_layers(self) -> List[str]:
        return ADAPTER_LAYERS

    @property
    def default_traced_adapter_layers(self) -> List[str]:
        return [ATTN_WQKV]

    def get_num_layers_for_type(self, layer_type: str) -> int:
        return len(self.model.model.layers)

    def is_row_parallel(self, layer_type: str) -> bool:
        return layer_type in ROW_PARALLEL
