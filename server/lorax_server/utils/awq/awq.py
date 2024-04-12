# Copied logic from https://github.com/mit-han-lab/llm-awq/blob/f084f40bd996f3cf3a0633c1ad7d9d476c318aaa/awq/quantize/qmodule.py

import awq_inference_engine  # with CUDA kernels
import torch
import torch.nn as nn


class AWQLinear(nn.Module):
    def __init__(self, w_bit, group_size, qweight, qzeros, scales, bias):
        super().__init__()

        if w_bit != 4:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = qweight.shape[0]
        self.out_features = qweight.shape[1] * 32 // w_bit

        self.split_k_iters = 8

        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else self.in_features

        assert self.in_features % self.group_size == 0, "in_features must be divisible by group_size"
        assert self.out_features % (32 // self.w_bit) == 0, "out_features must be divisible by 32 // w_bit"

        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.bias = bias

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)

        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()

        out = awq_inference_engine.gemm_forward_cuda(
            x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, 8
        )

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)

    @property
    def weight(self) -> torch.Tensor:
        return self.qweight
