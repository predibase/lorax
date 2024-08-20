from typing import Optional

import torch
import torch.distributed
from opentelemetry import trace
from transformers import AutoConfig, AutoTokenizer

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_neox_modeling import (
    FlashGPTNeoXForCausalLM,
)
from lorax_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
)

tracer = trace.get_tracer(__name__)


class FlashNeoXSharded(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
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
            raise NotImplementedError("FlashNeoX is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device=device, dtype=dtype, process_group=self.process_group)
        weights._set_config(model_id, config)

        model = FlashGPTNeoXForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashNeoXSharded, self).__init__(
            model_id=model_id,
            model=model.to(device),
            tokenizer=tokenizer,
            num_layers=len(model.gpt_neox.layers),
            num_kv_heads=model.gpt_neox.num_heads,
            head_size=model.gpt_neox.head_size,
            num_heads=model.gpt_neox.num_heads,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            compile=compile,
            trust_remote_code=trust_remote_code,
        )
