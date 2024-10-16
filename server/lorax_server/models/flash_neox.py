from typing import Optional

import torch
import torch.distributed
from opentelemetry import trace

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_neox_modeling import (
    FlashGPTNeoXForCausalLM,
)

tracer = trace.get_tracer(__name__)


class FlashNeoXSharded(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__(
            model_id=model_id,
            model_cls=FlashGPTNeoXForCausalLM,
            dtype=dtype,
            revision=revision,
            **kwargs,
        )
