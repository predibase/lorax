from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from aqlm.utils import _dequantize_weight, unpack_int_data


def forward_pass_quantized_linear(
    input: torch.Tensor,
    codes: torch.IntTensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    match (input.is_cuda, num_codebooks, codebook_size, out_group_size, in_group_size):
        case (True, 1, 65536, 1, 8):
            from .cuda_kernel import CUDA_KERNEL

            assert (
                input.dtype == torch.float16
            ), f"please load the model with `torch_dtype=torch.float16`, as {input.dtype} is not supported on GPU yet"
            return CUDA_KERNEL.code1x16_matmat(input, codes, codebooks, scales) + (bias if bias is not None else 0)
        case (True, 2, 256, 1, 8):
            from .cuda_kernel import CUDA_KERNEL

            assert (
                input.dtype == torch.float16
            ), f"please load the model with `torch_dtype=torch.float16`, as {input.dtype} is not supported on GPU yet"
            return CUDA_KERNEL.code2x8_matmat(input, codes, codebooks, scales) + (bias if bias is not None else 0)
        case (True, _, _, _, _):
            from .triton_kernel import triton_matmul

            return triton_matmul(input, codes, codebooks, scales, bias)
        case (False, _, 256, 1, _):
            from .numba_kernel import numba_gemm_lut

            return numba_gemm_lut(input, codes, codebooks, scales, bias)
        case _:
            dequantized_weight = _dequantize_weight(
                unpack_int_data(codes, codebooks.shape[0].bit_length() - 1),
                codebooks,
                scales,
            )
            return F.linear(input, dequantized_weight, bias)
