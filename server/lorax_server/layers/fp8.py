from typing import Optional
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
import torch


def fp8_quantize(weight, qdtype=torch.float8_e4m3fn):
    # weight, scale = quant_weights(weight, torch.int8, False)
    finfo = torch.finfo(qdtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / weight.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (weight * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(qdtype)
    scale = scale.float().reciprocal()
    return qweight, scale

####### from vLLM code #######

def cutlass_fp8_supported() -> bool:
    capability = current_platform.get_device_capability()
    capability = capability[0] * 10 + capability[1]

    return ops.cutlass_scaled_mm_supports_fp8(capability)

def apply_fp8_linear(
    input: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
    qbias: Optional[torch.Tensor] = None,
    cutlass_fp8_supported: bool = True,
) -> torch.Tensor:
    # ops.scaled_fp8_quant supports both dynamic and static quant.
    #   If dynamic, layer.input_scale is None and x_scale computed from x.
    #   If static, layer.input_scale is scalar and x_scale is input_scale.

    if qbias is None and cutlass_fp8_supported:
        qinput, x_scale = ops.scaled_fp8_quant(input, input_scale)

        # Fused GEMM_DQ
        output = ops.cutlass_scaled_mm(qinput,
                                       qweight,
                                       out_dtype=input.dtype,
                                       scale_a=x_scale,
                                       scale_b=weight_scale)

    else:
        qinput, x_scale = ops.scaled_fp8_quant(input,
                                               input_scale,
                                               batch_dim_padding=17)

        # Fused GEMM_DQ -- note we padded the input above because
        # torch._scaled_mm is more performant for matrices with
        # batch dimension > 16. Note that this could change
        # in the future.
        output, _ = torch._scaled_mm(qinput,
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
        self.cutlass_fp8_supported = cutlass_fp8_supported()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return apply_fp8_linear(
            input=input,
            qweight=self.qweight,
            weight_scale=self.weight_scale,
            input_scale=self.input_scale,
            qbias=self.qbias,
            cutlass_fp8_supported=self.cutlass_fp8_supported
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.qweight
