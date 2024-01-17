import torch
import torch.distributed

from typing import Optional, Type

from transformers import (
    AutoTokenizer,
    AutoConfig,
    PreTrainedTokenizerBase,
)

from lorax_server.models.custom_modeling.bloom_modeling import (
    BloomForCausalLM,
)
from lorax_server.models import CausalLM
from lorax_server.models.causal_lm import CausalLMBatch
from lorax_server.pb import generate_pb2
from lorax_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)
from lorax_server.utils.tokenizer import TokenizerManager
from server.lorax_server.utils.lora import AdapterBatchData


class BloomCausalLMBatch(CausalLMBatch):
    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        tokenizers: TokenizerManager,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "CausalLMBatch":
        batch = super().from_pb(pb=pb, tokenizer=tokenizer, tokenizers=tokenizers, dtype=dtype, device=device)
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
        weights = Weights(
            filenames, device=device, dtype=dtype, process_group=self.process_group
        )
        if config.quantize == "gptq":
            weights._set_gptq_params(model_id)

        model = BloomForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(CausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )

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
        adapter_data: Optional[AdapterBatchData] = None
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
