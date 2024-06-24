import torch
import transformer_engine.pytorch as te


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


class Fp8Linear(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        self.dtype = weight.dtype
        self.te_linear = te.Linear(weight.shape[1], weight.shape[0])

        state_dict = self.te_linear.state_dict()
        state_dict["weight"] = weight
        state_dict["bias"] = bias
        self.te_linear.load_state_dict(state_dict)

        # self.qweight, self.scale = fp8_quantize(weight)

        # self.bias = bias if bias is not None else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # qinput, scale = fp8_quantize(input)
        # output, _ = torch._scaled_mm(
        #     qinput,
        #     self.qweight.t(),
        #     out_dtype=self.dtype,
        #     scale_a=scale,
        #     scale_b=self.scale,
        #     bias=self.bias,
        # )
        # return output
        return self.te_linear(input)

    @property
    def weight(self) -> torch.Tensor:
        # return self.qweight
        return self.te_linear.weight
