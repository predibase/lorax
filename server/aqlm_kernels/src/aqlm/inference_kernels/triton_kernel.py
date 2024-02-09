from typing import Optional

import torch
import triton
import triton.language as tl
from torch.autograd import Function


@triton.autotune(
    configs=[
        triton.Config({"UNUSED": 1}, num_stages=num_stages, num_warps=num_warps)
        for num_stages in (1, 2, 3, 4, 5)
        for num_warps in (1, 2, 4, 8)
    ],
    key=[
        "in_features",
        "out_features",
        "num_codebooks",
        "codebook_size",
        "out_group_size",
        "in_group_size",
        "num_input_groups",
        "num_input_groups_next_power_of_2",
        "compute_in_fp32",
        "has_bias",
    ],
)
@triton.jit
def _aqlm_gemv_simple(
    input_vec_ptr,
    output_vec_ptr,
    codes_ptr,
    codebooks_ptr,
    scales_ptr,
    bias_ptr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    num_codebooks: tl.constexpr,
    codebook_size: tl.constexpr,
    out_group_size: tl.constexpr,
    in_group_size: tl.constexpr,
    num_input_groups: tl.constexpr,
    num_input_groups_next_power_of_2: tl.constexpr,
    compute_in_fp32: tl.constexpr,
    has_bias: tl.constexpr,
    UNUSED: tl.constexpr,
):
    # variables ending with "_i" mean "for i-th output unit"
    pid = tl.program_id(axis=0)  # [0, 1, ... {out_features-1}]

    # Stage 1: load input data
    input_vec = tl.load(
        input_vec_ptr
        + tl.arange(0, num_input_groups_next_power_of_2)[:, None, None, None] * in_group_size
        + tl.arange(0, in_group_size)[None, None, None, :],
        mask=tl.arange(0, num_input_groups_next_power_of_2)[:, None, None, None] < num_input_groups,
    )
    # [in_features//in_group_size, 1, 1, group_size]
    # Note: we could simply load input_vec then reshape
    #     input_vec = tl.load(input_vec_ptr + tl.arange(0, in_features))  # [in_features]
    #     input_vec = tl.view(input_vec, [num_input_groups, 1, in_group_size])
    #     , but this does not work because tl.view may reorder elements arbitrarily; see its docstring

    # Stage 2: load integer codes for the active row
    # [in_features // in_group_size, num_codebooks]
    codes_i_ptrs = (
        codes_ptr
        + pid * num_input_groups * num_codebooks
        + tl.arange(0, num_input_groups_next_power_of_2)[:, None] * num_codebooks
        + tl.arange(0, num_codebooks)[None, :]
    )
    codes_i_mask_1d = tl.arange(0, num_input_groups_next_power_of_2) < num_input_groups

    codes_i = tl.load(codes_i_ptrs, mask=codes_i_mask_1d[:, None])  # [in_features//in_group_size, num_codebooks]
    codes_i = codes_i.to(tl.int32)
    codes_i = (codes_i) + (codes_i < 0) * codebook_size  # aka 2 ** nbits_per_codebook
    # ^-- (because codes are int16 tensors that contain uint data)

    # The following alternative does not work:
    #     codes_i = codes_i.to(tl.int32) % codebook_size # aka 2 ** nbits_per_codeboo

    # shift codes_i so that codebooks after 0th point to correct indices in codebooks_ptr
    codes_i += tl.arange(0, num_codebooks)[None, :] * codebook_size  # aka 2 ** nbits_per_codebook
    # ^-- [in_group_size, num_codebooks]

    # Stage 3: convert codes to pointers to every individual (activated) weight in codebooks
    # [in_features // in_group_size, num_codebooks, out_group_size, in_group_size]
    out_group_ix = tl.arange(0, out_group_size)[None, None, :, None]
    in_group_ix = tl.arange(0, in_group_size)[None, None, None, :]
    weight_i_ptrs = (
        codebooks_ptr
        + codes_i[:, :, None, None] * out_group_size * in_group_size
        + out_group_ix * in_group_size
        + in_group_ix
    )

    # Stage 4: reconstruct weights, multiply by inputs and write out
    weights_i = tl.load(weight_i_ptrs, mask=codes_i_mask_1d[:, None, None, None], other=0)
    if compute_in_fp32:
        weights_i = weights_i.to(tl.float32)
        input_vec = input_vec.to(tl.float32)
    # ^-- [in_features // in_group_size, num_codebooks, out_group_size, in_group_size]

    if out_group_size == 1:
        scale = tl.load(scales_ptr + pid).to(weights_i.dtype)  # scalar
        output_i = tl.sum(weights_i * input_vec) * scale
        if has_bias:
            output_i += tl.load(bias_ptr + pid).to(weights_i.dtype)
        tl.store(output_vec_ptr + pid, output_i.to(input_vec.dtype))
    else:
        output_i = tl.sum(tl.sum(weights_i, axis=2) * input_vec, axis=0)  # [out_group_size]
        output_i *= tl.load(scales_ptr + pid).to(weights_i.dtype)
        if has_bias:
            output_i += tl.load(bias_ptr + pid).to(weights_i.dtype)
        tl.store(output_vec_ptr + pid * out_group_size + tl.arange(0, out_group_size), output_i.to(input_vec.dtype))


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def aqlm_gemv_simple(
    input_vec: torch.Tensor,
    codes_i16: torch.ShortTensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    compute_in_fp32: bool = True,
):
    device, dtype = codebooks.device, codebooks.dtype
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    in_features = input_vec.shape[1]
    out_features = codes_i16.shape[0] * out_group_size
    num_input_groups = codes_i16.shape[1]
    assert input_vec.ndim == 2 and input_vec.shape[0] == 1, "do reshape; now!"
    assert scales.shape == (out_features // out_group_size, 1, 1, 1)
    assert in_features % in_group_size == 0
    assert codebooks.shape[1] < 2**32

    output_vec = torch.empty(1, out_features, device=device, dtype=dtype)
    # 1D launch kernel where each block computes output unit
    grid = lambda META: (out_features // out_group_size,)
    _aqlm_gemv_simple[grid](
        input_vec,
        output_vec,
        codes_i16,
        codebooks,
        scales,
        bias,
        in_features,
        out_features,
        num_codebooks,
        codebook_size,
        out_group_size,
        in_group_size,
        num_input_groups,
        next_power_of_2(num_input_groups),
        compute_in_fp32,
        bias is not None,
    )

    return output_vec


def aqlm_gemm_stupid(
    input: torch.Tensor,
    codes_i16: torch.ShortTensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    compute_in_fp32: bool = True,
):
    device, dtype = codebooks.device, codebooks.dtype
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    in_features = input.shape[1]
    out_features = codes_i16.shape[0] * out_group_size
    num_input_groups = codes_i16.shape[1]
    assert input.ndim == 2
    assert scales.shape == (out_features // out_group_size, 1, 1, 1)
    assert in_features % in_group_size == 0
    assert codebooks.shape[1] < 2**32

    output = torch.empty(input.shape[0], out_features, device=device, dtype=dtype)
    for i in range(input.shape[0]):
        # 1D launch kernel where each block computes output unit
        grid = lambda META: (out_features // out_group_size,)
        _aqlm_gemv_simple[grid](
            input[i],
            output[i],
            codes_i16,
            codebooks,
            scales,
            bias,
            in_features,
            out_features,
            num_codebooks,
            codebook_size,
            out_group_size,
            in_group_size,
            num_input_groups,
            next_power_of_2(num_input_groups),
            compute_in_fp32,
            bias is not None,
        )

    return output


def triton_matmul(
    input: torch.Tensor,
    codes: torch.IntTensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    compute_in_fp32: bool = True,
) -> torch.Tensor:
    input_shape = input.shape
    input = input.reshape(-1, input_shape[-1])

    if input.shape[0] == 1:
        return aqlm_gemv_simple(
            input,
            codes,
            codebooks,
            scales,
            bias,
            compute_in_fp32,
        ).reshape(input_shape[:-1] + (-1,))
    else:
        return aqlm_gemm_stupid(
            input,
            codes,
            codebooks,
            scales,
            bias,
            compute_in_fp32,
        ).reshape(input_shape[:-1] + (-1,))
