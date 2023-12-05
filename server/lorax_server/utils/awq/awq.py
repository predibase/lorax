# Copied logic from https://github.com/mit-han-lab/llm-awq/blob/19a5a2c9db47f69a2851c83fea90f81ed49269ab/awq/quantize/qmodule.py

import math
import torch
import torch.nn as nn
import awq_inference_engine  # with CUDA kernels

def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError
    
    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width


class AWQLinear(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()
        
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        pack_num = (32 // self.w_bit)
        # TODO (Haotian): a function for buffer shape calculation
        self.register_buffer('qweight', torch.zeros((out_features, in_features // pack_num), dtype=torch.int32, device=dev))
        self.register_buffer('qzeros', torch.zeros((out_features, calculate_zeros_width(in_features, self.group_size)), dtype=torch.int32, device=dev))
        self.register_buffer('scales', torch.zeros((out_features, calculate_zeros_width(in_features, self.group_size) * pack_num), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        inputs = x.reshape(-1, x.shape[-1])
        if inputs.shape[0] > 8:
            out = awq_inference_engine.gemm_forward_cuda(inputs, self.qweight, self.scales, self.qzeros, self.group_size, self.split_k_iters)
        else:
            out = awq_inference_engine.gemv_forward_cuda(inputs, self.qweight, self.scales, self.qzeros, self.group_size)
        out = out + self.bias if self.bias is not None else out
        #print(out)
        #assert 0
        return out.reshape(out_shape)
    
    @property
    def weight(self) -> torch.Tensor:
        return self.qweight
