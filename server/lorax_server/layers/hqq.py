import torch
from torch import nn

HAS_HQQ = True
try:
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear, HQQBackend
    HQQLinear.set_backend(HQQBackend.ATEN)
    class HQQLinearLayer(HQQLinear):
        @property
        def weight(self) -> torch.Tensor:
            return self.W_q

except ImportError:
    HAS_HQQ = False


def get_hqq_linear(quantize, weight, bias=None) -> HQQLinearLayer:
    if quantize == "hqq-4bit":
        quant_config = BaseQuantizeConfig(nbits=4, group_size=128, quant_zero=True, quant_scale=True, offload_meta=True)
    elif quantize == "hqq-3bit":
        quant_config = BaseQuantizeConfig(nbits=3, group_size=64, quant_zero=True, quant_scale=True, offload_meta=True)
    elif quantize == "hqq-2bit":
        quant_config = BaseQuantizeConfig(nbits=2, group_size=32, quant_zero=True, quant_scale=True, offload_meta=True)

    # init nn.linear from weight and bias
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    with torch.no_grad():
        layer.weight.data = weight
        if bias is not None:
            layer.bias.data = bias

    linear = HQQLinearLayer(layer, quant_config, del_orig=True)

    return linear