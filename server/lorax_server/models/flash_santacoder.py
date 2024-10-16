from typing import List, Optional

import torch
import torch.distributed
from opentelemetry import trace
from transformers import AutoConfig

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_santacoder_modeling import (
    FlashSantacoderForCausalLM,
)

tracer = trace.get_tracer(__name__)


class FlashSantacoderSharded(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__(
            model_id=model_id,
            model_cls=FlashSantacoderForCausalLM,
            dtype=dtype,
            revision=revision,
            config_cls=AutoConfig,
            **kwargs,
        )

    def decode(self, generated_ids: List[int]) -> str:
        # Do not skip special tokens as they are used for custom parsing rules of the generated text
        return self.tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
