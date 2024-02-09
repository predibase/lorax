from typing import Optional

import numba
import numpy as np
import torch

COMPILED_KERNELS = {}


def numba_gemm_lut(
    input: torch.Tensor,  #  [..., in_features]
    codes: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
    codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    scales: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    input_shape = input.shape
    input = input.reshape(-1, input_shape[-1])

    device, dtype = codebooks.device, codebooks.dtype
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    in_features = input.shape[1]
    num_input_groups = in_features // in_group_size
    out_features = codes.shape[1] * out_group_size
    assert input.ndim == 2
    assert scales.shape == (out_features // out_group_size, 1, 1, 1)
    assert in_features % in_group_size == 0
    assert codebook_size == 2**8
    assert codes.dtype == torch.int8
    assert (
        input.dtype == torch.float32 and codebooks.dtype == torch.float32
    ), f"please load the model with `torch_dtype=torch.float32`, as {input.dtype} is not supported for CPU"

    kernel_key = (in_group_size, out_features, in_features, num_codebooks)
    if kernel_key not in COMPILED_KERNELS:
        print(f"Compiling AQLM numba kernel with parameters: {kernel_key=}")

        @numba.njit(parallel=True)
        def numba_gemv_lut_(x, codebooks, codes_alt, scales):
            lut = x.reshape(-1, in_group_size) @ codebooks.reshape(-1, in_group_size).T
            lut = lut.reshape(-1, num_codebooks, codebook_size)

            output_vec = np.zeros(out_features, dtype=x.dtype)
            for j in numba.prange(num_input_groups):
                for i in range(out_features):
                    for c in range(num_codebooks):
                        output_vec[i] += lut[j, c, codes_alt[j, i, c]]
            output_vec *= scales.flatten()
            return output_vec

        COMPILED_KERNELS[kernel_key] = numba_gemv_lut_
    compiled_kernel = COMPILED_KERNELS[kernel_key]

    output = torch.empty(input.shape[0], out_features, device=device, dtype=dtype)
    for i in range(input.shape[0]):
        output[i] = torch.as_tensor(
            compiled_kernel(
                input[i].numpy(),
                codebooks.numpy(),
                codes.view(torch.uint8).numpy(),
                scales.numpy(),
            )
        )
    if bias is not None:
        output += bias
    return output.reshape(input_shape[:-1] + (-1,))
