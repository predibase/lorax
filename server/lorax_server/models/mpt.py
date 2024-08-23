import json
from pathlib import Path
from typing import Optional, Type

import torch
import torch.distributed
from huggingface_hub import hf_hub_download
from opentelemetry import trace
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

from lorax_server.models.causal_lm import CausalLM, CausalLMBatch
from lorax_server.models.custom_modeling.mpt_modeling import (
    MPTForCausalLM,
)
from lorax_server.pb import generate_pb2
from lorax_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
)
from lorax_server.utils.tokenizer import TokenizerManager

tracer = trace.get_tracer(__name__)


class MPTCausalLMBatch(CausalLMBatch):
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


class MPTSharded(CausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        compile: bool = False,
        trust_remote_code: bool = False,
    ):
        if compile:
            raise ValueError("`--compile` is not supported with MPT")

        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16
        else:
            raise NotImplementedError("MPTSharded is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        tokenizer.pad_token = tokenizer.eos_token

        # If model_id is a local path, load the file directly
        local_path = Path(model_id, "config.json")
        if local_path.exists():
            filename = str(local_path.resolve())
        else:
            filename = hf_hub_download(model_id, revision=revision, filename="config.json")
        with open(filename, "r") as f:
            config = json.load(f)
        config = PretrainedConfig(**config)
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)
        weights._set_config(model_id, config)

        config.quantize = quantize
        model = MPTForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(CausalLM, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            trust_remote_code=trust_remote_code,
        )

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return MPTCausalLMBatch
