import torch
from torch import nn
from torch.nn import functional as F

from lorax_server.utils.import_utils import SYSTEM
from lorax_server.utils.torch_utils import is_fp8

if SYSTEM == "rocm":
    try:
        from vllm import _custom_C
    except Exception as e:
        raise ImportError(f"Could not load `vllm._custom_C`. Full error: {e}")


class FastLinear(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_tensor(f"{prefix}.weight")
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class FastLinearROCm(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        if bias is not None:
            self.bias = torch.nn.Parameter(bias)
        else:
            self.bias = None

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_tensor(f"{prefix}.weight")
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(weight, bias)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if SYSTEM == "rocm" and inp.numel() // inp.shape[-1] == 1:
            batched = False
            inp_shape = inp.shape

            if inp.dim() == 3:
                inp = inp.view(-1, inp_shape[-1])
                batched = True

            m, k = weight.shape[0], inp_shape[1]
            out = torch.empty(inp_shape[0], weight.shape[0], dtype=inp.dtype, device="cuda")
            if (k == 8192 and (m == 1280 or m == 7168)) or (k == 3584 and m == 8192):
                _custom_C.LLMM1(weight, inp, out, 8)
            elif k <= 8192 and k % 8 == 0 and m % 4 == 0:
                _custom_C.LLMM1(weight, inp, out, 4)
            else:
                out = F.linear(inp, weight)

            if batched:
                out.view(*inp_shape[:-1], out.shape[-1])

            if bias is not None:
                out = out + bias
            return out
        return F.linear(inp, self.weight, self.bias)


def get_linear(weight, bias, quantize, fan_in_fan_out=False, weight_scale=None, input_scale=None):
    # https://huggingface.co/docs/peft/package_reference/tuners#peft.LoraConfig.fan_in_fan_out
    # Set to True if replacing a Conv1D layer with a Linear layer
    if fan_in_fan_out:
        weight = weight.T.contiguous()

    if quantize is None:
        linear = FastLinear(weight, bias)

    elif is_fp8(quantize):
        from lorax_server.layers.fp8 import Fp8Linear

        linear = Fp8Linear(weight, bias, weight_scale=weight_scale, input_scale=input_scale)

    elif quantize == "bitsandbytes":
        from lorax_server.layers.bnb import Linear8bitLt

        linear = Linear8bitLt(
            weight,
            bias,
            has_fp16_weights=False,
            threshold=6.0,
        )
        if bias is not None:
            linear.bias = nn.Parameter(bias)
    elif quantize == "bitsandbytes-nf4":
        from lorax_server.layers.bnb import Linear4bit

        linear = Linear4bit(
            weight,
            bias,
            quant_type="nf4",
        )
    elif quantize == "bitsandbytes-fp4":
        from lorax_server.layers.bnb import Linear4bit

        linear = Linear4bit(
            weight,
            bias,
            quant_type="fp4",
        )
    elif quantize == "eetq":
        from lorax_server.layers.eetq import EETQLinear

        linear = EETQLinear(weight, bias)
    elif quantize == "gptq":
        try:
            qweight, qzeros, scales, g_idx, bits, groupsize, use_exllama = weight
        except Exception:
            raise NotImplementedError("The passed weight is not `gptq` compatible, loader needs to be updated.")

        if use_exllama:
            from lorax_server.layers.gptq.exllamav2 import QuantLinear as exllamav2QuantLinear

            linear = exllamav2QuantLinear(qweight, qzeros, scales, g_idx, bias, bits, groupsize)
        else:
            from lorax_server.layers.gptq.quant_linear import QuantLinear

            linear = QuantLinear(
                qweight,
                qzeros,
                scales,
                g_idx,
                bias,
                bits,
                groupsize,
            )
    elif quantize == "awq":
        try:
            qweight, qzeros, scales, _, bits, groupsize, _ = weight
        except Exception:
            raise NotImplementedError("The passed weight is not compatible with `awq`")
        from lorax_server.utils.awq.awq import AWQLinear

        linear = AWQLinear(
            w_bit=bits,
            group_size=groupsize,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            bias=bias,
        )
    elif "hqq-" in quantize:
        from lorax_server.layers.hqq import get_hqq_linear

        linear = get_hqq_linear(quantize, weight, bias)
    else:
        raise NotImplementedError(f"Quantization `{quantize}` is not implemented yet.")
    return linear
