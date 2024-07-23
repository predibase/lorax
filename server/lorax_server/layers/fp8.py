from typing import Optional
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
import torch


####### from vLLM code #######

def apply_fp8_linear(
    input: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
    qbias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # ops.scaled_fp8_quant supports both dynamic and static quant.
    #   If dynamic, layer.input_scale is None and x_scale computed from x.
    #   If static, layer.input_scale is scalar and x_scale is input_scale.

    qinput, x_scale = ops.scaled_fp8_quant(input,
                                            input_scale,
                                            batch_dim_padding=17)

    # Fused GEMM_DQ -- note we padded the input above because
    # torch._scaled_mm is more performant for matrices with
    # batch dimension > 16. Note that this could change
    # in the future.
    output = torch._scaled_mm(qinput,
                                    qweight,
                                    out_dtype=input.dtype,
                                    scale_a=x_scale,
                                    scale_b=weight_scale,
                                    bias=qbias)

    return torch.narrow(output, 0, 0, input.shape[0])

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
        self.qbias = bias if bias is not None else None
        self.weight_scale = weight_scale
        self.input_scale = input_scale

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
