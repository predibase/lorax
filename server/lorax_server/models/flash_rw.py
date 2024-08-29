from typing import Optional

import torch
import torch.distributed
from opentelemetry import trace
from transformers import AutoTokenizer

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_rw_modeling import (
    FlashRWForCausalLM,
    RWConfig,
)
from lorax_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
)

tracer = trace.get_tracer(__name__)


class FlashRWSharded(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__(
            model_id=model_id,
            model_cls=FlashRWForCausalLM,
            dtype=dtype,
            revision=revision,
            config_cls=RWConfig,
            **kwargs,
        )
