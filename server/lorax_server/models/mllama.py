from typing import Optional

import torch
import torch.distributed
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from lorax_server.models.custom_modeling.mllama import (
    MllamaForConditionalGeneration,
)
from lorax_server.models.multimodal_causal_lm import MultimodalCausalLM
from lorax_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
)
from lorax_server.utils.state import PREFIX_CACHING


class Mllama(MultimodalCausalLM):
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
            raise NotImplementedError("Mllama is only available on GPU")

        if PREFIX_CACHING:
            raise NotImplementedError("Mllama does not support prefix caching yet")

        if compile:
            raise NotImplementedError("Mllama does not support CUDA graph compilation yet")

        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        config.quantize = quantize
        config.vision_config.quantize = quantize

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames,
            device,
            dtype,
            process_group=self.process_group,
        )
        weights._set_config(model_id, config)

        model = MllamaForConditionalGeneration(prefix="", config=config, weights=weights)
        self.config = config

        torch.distributed.barrier(group=self.process_group)
        super().__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            processor=self.processor,
        )
