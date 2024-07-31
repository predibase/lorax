from typing import Optional

import torch
from vllm import _custom_ops as ops

####### from vLLM code #######


def apply_fp8_linear(
    input: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    qbias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    qinput, x_scale = ops.scaled_fp8_quant(input, input_scale, scale_ub=input_scale_ub, use_per_token_if_dynamic=False)

    output = ops.cutlass_scaled_mm(
        qinput, qweight, out_dtype=input.dtype, scale_a=x_scale, scale_b=weight_scale, bias=qbias
    )

    return output


class Fp8Linear(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
        weight_scale,
        input_scale,
    ) -> None:
        super().__init__()
        self.dtype = weight.dtype
        self.qweight = weight.t()
        self.weight_scale = weight_scale.view(1, -1).contiguous().float()
        self.qbias = bias if bias is not None else None
        self.input_scale = input_scale.float() if input_scale is not None else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return apply_fp8_linear(
            input=input,
            qweight=self.qweight,
            weight_scale=self.weight_scale,
            input_scale=self.input_scale,
            qbias=self.qbias,
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.qweight
