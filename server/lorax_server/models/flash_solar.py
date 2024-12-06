from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from opentelemetry import trace

from lorax_server.models import FlashCausalLM
from lorax_server.models.custom_modeling.flash_solar_modeling import (
    FlashSolarForCausalLM,
    SolarConfig,
)
from lorax_server.utils.lora import (
    DOWN_PROJ,
    GATE_PROJ,
    K_PROJ,
    O_PROJ,
    Q_PROJ,
    UP_PROJ,
    V_PROJ,
)

tracer = trace.get_tracer(__name__)


ADAPTER_LAYERS = [Q_PROJ, K_PROJ, V_PROJ, O_PROJ, GATE_PROJ, UP_PROJ, DOWN_PROJ]
ROW_PARALLEL = {O_PROJ, DOWN_PROJ}


class FlashSolar(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        adapter_id: str,
        adapter_source: str,
        revision: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__(
            model_id=model_id,
            model_cls=FlashSolarForCausalLM,
            dtype=dtype,
            revision=revision,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            config_cls=SolarConfig,
            **kwargs,
        )

    # TODO: Implement adapter_target_to_layer
    def adapter_target_to_layer(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        pass

    @property
    def supports_adapter_loading(self) -> bool:
        return True

    @property
    def adapter_layers(self) -> List[str]:
        return ADAPTER_LAYERS

    @property
    def default_traced_adapter_layers(self) -> List[str]:
        return [Q_PROJ, V_PROJ]

    def get_num_layers_for_type(self, layer_type: str) -> int:
        return len(self.model.model.layers)

    def is_row_parallel(self, layer_type: str) -> bool:
        return layer_type in ROW_PARALLEL
