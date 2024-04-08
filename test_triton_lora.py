"""
Group GEMM
============================
This group gemm kernel launches a fixed number of CTA to compute a group
of gemms. The scheduling is static and we do it on device.
"""

# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import List
import torch

import triton
import triton.language as tl


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 128,
#                 "BLOCK_SIZE_K": 32,
#                 "NUM_SM": 84,
#             }
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 128,
#                 "BLOCK_SIZE_K": 32,
#                 "NUM_SM": 128,
#             }
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 64,
#                 "BLOCK_SIZE_N": 64,
#                 "BLOCK_SIZE_K": 32,
#                 "NUM_SM": 84,
#             }
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 64,
#                 "BLOCK_SIZE_N": 64,
#                 "BLOCK_SIZE_K": 32,
#                 "NUM_SM": 128,
#             }
#         ),
#     ],
#     key=["group_size"],
# )
# @triton.jit
# def sgmv_kernel(
#     # device tensor of matrices pointers
#     group_a_ptrs,
#     group_b_ptrs,
#     group_c_ptrs,
#     # device tensor of gemm sizes. its shape is [group_size, 3]
#     # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
#     group_gemm_sizes,
#     # device tensor of leading dimension sizes. its shape is [group_size, 3]
#     # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
#     g_lds,
#     # number of gemms
#     group_size,
#     # number of virtual SM
#     NUM_SM: tl.constexpr,
#     # tile sizes
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
# ):
#     tile_idx = tl.program_id(0)
#     last_problem_end = 0
#     for g in range(group_size):
#         # get the gemm size of the current problem
#         gm = tl.load(group_gemm_sizes + g * 3)
#         gn = tl.load(group_gemm_sizes + g * 3 + 1)
#         gk = tl.load(group_gemm_sizes + g * 3 + 2)
#         num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
#         num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
#         num_tiles = num_m_tiles * num_n_tiles
#         # iterate through the tiles in the current gemm problem
#         while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
#             # pick up a tile from the current gemm problem
#             k = gk
#             lda = tl.load(g_lds + g * 3)
#             ldb = tl.load(g_lds + g * 3 + 1)
#             ldc = tl.load(g_lds + g * 3 + 2)
#             a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
#             b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
#             c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
#             # figure out tile coordinates
#             tile_idx_in_gemm = tile_idx - last_problem_end
#             tile_m_idx = tile_idx_in_gemm // num_n_tiles
#             tile_n_idx = tile_idx_in_gemm % num_n_tiles

#             # do regular gemm here
#             offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#             offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#             offs_k = tl.arange(0, BLOCK_SIZE_K)
#             a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
#             b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
#             accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#             for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
#                 # hint to Triton compiler to do proper loop pipelining
#                 tl.multiple_of(a_ptrs, [16, 16])
#                 tl.multiple_of(b_ptrs, [16, 16])
#                 # assume full tile for now
#                 a = tl.load(a_ptrs)
#                 b = tl.load(b_ptrs)
#                 accumulator += tl.dot(a, b)
#                 a_ptrs += BLOCK_SIZE_K
#                 b_ptrs += BLOCK_SIZE_K * ldb
#             c = accumulator.to(tl.float16)

#             offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#             offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#             c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

#             # assumes full tile for now
#             tl.store(c_ptrs, c)

#             # go to the next tile by advancing NUM_SM
#             tile_idx += NUM_SM

#         # get ready to go to the next gemm problem
#         last_problem_end = last_problem_end + num_tiles


# def group_gemm_fn(group_A, group_B):
#     device = torch.device("cuda")
#     assert len(group_A) == len(group_B)
#     group_size = len(group_A)

#     A_addrs = []
#     B_addrs = []
#     C_addrs = []
#     g_sizes = []
#     g_lds = []
#     group_C = []
#     for i in range(group_size):
#         A = group_A[i]
#         B = group_B[i]
#         assert A.shape[1] == B.shape[0]
#         M, K = A.shape
#         K, N = B.shape
#         C = torch.empty((M, N), device=device, dtype=A.dtype)
#         group_C.append(C)
#         A_addrs.append(A.data_ptr())
#         B_addrs.append(B.data_ptr())
#         C_addrs.append(C.data_ptr())
#         g_sizes += [M, N, K]
#         g_lds += [A.stride(0), B.stride(0), C.stride(0)]

#     # note these are device tensors
#     d_a_ptrs = torch.tensor(A_addrs, device=device)
#     d_b_ptrs = torch.tensor(B_addrs, device=device)
#     d_c_ptrs = torch.tensor(C_addrs, device=device)
#     d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
#     d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)
#     # we use a fixed number of CTA, and it's auto-tunable
#     grid = lambda META: (META["NUM_SM"],)
#     grouped_matmul_kernel[grid](
#         d_a_ptrs,
#         d_b_ptrs,
#         d_c_ptrs,
#         d_g_sizes,
#         d_g_lds,
#         group_size,
#     )

#     return group_C


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {
#                 "BLOCK_SIZE_S": 16,
#                 "BLOCK_SIZE_H1": 128,
#                 "BLOCK_SIZE_R": 16,
#                 "NUM_SM": 84,
#             }
#         ),
#         # triton.Config(
#         #     {
#         #         "BLOCK_SIZE_M": 128,
#         #         "BLOCK_SIZE_N": 128,
#         #         "BLOCK_SIZE_K": 32,
#         #         "NUM_SM": 128,
#         #     }
#         # ),
#         # triton.Config(
#         #     {
#         #         "BLOCK_SIZE_M": 64,
#         #         "BLOCK_SIZE_N": 64,
#         #         "BLOCK_SIZE_K": 32,
#         #         "NUM_SM": 84,
#         #     }
#         # ),
#         # triton.Config(
#         #     {
#         #         "BLOCK_SIZE_M": 64,
#         #         "BLOCK_SIZE_N": 64,
#         #         "BLOCK_SIZE_K": 32,
#         #         "NUM_SM": 128,
#         #     }
#         # ),
#     ],
#     key=["nsegments"],
# )
@triton.jit
def sgmv_kernel(
    y,
    x,
    wa_ptrs,
    wb_ptrs,
    s_start,
    s_end,
    layer_idx,
    nsegments,
    R,
    H1,
    H2,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_H1: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for s in range(nsegments):
        s1 = tl.load(s_start + s)
        s2 = tl.load(s_end + s)
        if s2 - s1 > 0:
            S = s2 - s1

            wa_ptr = tl.load(wa_ptrs + s).to(tl.pointer_type(tl.float16))
            # wb_ptr = tl.load(wb_ptrs + s).to(tl.float16)

            num_s_tiles = tl.cdiv(S, BLOCK_SIZE_S)
            # num_h1_tiles = tl.cdiv(H1, BLOCK_SIZE_H1)
            num_r_tiles = tl.cdiv(H1, BLOCK_SIZE_R)

            num_tiles = num_s_tiles * num_r_tiles
            # iterate through the tiles in the current gemm problem
            while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                # figure out tile coordinates
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_s_idx = tile_idx_in_gemm // num_r_tiles
                tile_r_idx = tile_idx_in_gemm % num_r_tiles
                # print(f"{tile_idx_in_gemm=} {tile_s_idx=} {tile_r_idx=}")

                # do regular gemm here
                offs_xs = tile_s_idx * BLOCK_SIZE_S + tl.arange(0, BLOCK_SIZE_S)
                offs_war = tile_r_idx * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
                offs_h1 = tl.arange(0, BLOCK_SIZE_H1)
                x_tile_ptr = x + offs_xs[:, None] * H1 + offs_h1[None, :]
                wa_tile_ptr = wa_ptr + offs_h1[:, None] * R + offs_war[None, :]
                accumulator = tl.zeros((BLOCK_SIZE_S, BLOCK_SIZE_R), dtype=tl.float32)
                for hh in range(0, tl.cdiv(H1, BLOCK_SIZE_H1)):
                    # hint to Triton compiler to do proper loop pipelining
                    tl.multiple_of(x_tile_ptr, [16, 16])
                    tl.multiple_of(wa_tile_ptr, [16, 16])
                    # assume full tile for now
                    x_tile = tl.load(x_tile_ptr)
                    wa_tile = tl.load(wa_tile_ptr)
                    accumulator += tl.dot(x_tile, wa_tile)
                    x_tile_ptr += BLOCK_SIZE_H1
                    wa_tile_ptr += BLOCK_SIZE_H1 * R
                y_tile = accumulator.to(tl.float16)

                offs_ys = tile_s_idx * BLOCK_SIZE_S + tl.arange(0, BLOCK_SIZE_S)
                offs_yr = tile_r_idx * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
                y_ptrs = y + R * offs_ys[:, None] + offs_yr[None, :]

                # assumes full tile for now
                tl.store(y_ptrs, y_tile)

                # go to the next tile by advancing NUM_SM
                tile_idx += NUM_SM

            # get ready to go to the next gemm problem
            last_problem_end = last_problem_end + num_tiles


def sgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    wa: List[torch.Tensor],
    wb: List[torch.Tensor],
    s_start: torch.IntTensor,
    s_end: torch.IntTensor,
    layer_idx: int,
):
    assert len(wa) == len(wb)
    assert len(wa) == s_start.shape[0]
    assert len(wa) == s_end.shape[0]

    device = x.device
    nsegments = s_start.shape[0]

    R = wa[0].shape[2]
    H1 = wa[0].shape[1]
    H2 = wb[0].shape[2]

    wa_ptrs = []
    wb_ptrs = []
    for i in range(nsegments):
        WA = wa[i]
        WB = wb[i]
        assert WA.shape[2] == WB.shape[1]
        # TODO(travis) remove this
        WA = WA[0, :, :].contiguous()
        WB = WB[0, :, :].contiguous()
        wa_ptrs.append(WA.data_ptr())
        wb_ptrs.append(WB.data_ptr())

    wa_ptrs = torch.tensor(wa_ptrs, device=device)
    wb_ptrs = torch.tensor(wb_ptrs, device=device)

    # we use a fixed number of CTA, and it's auto-tunable
    grid = lambda META: (META["NUM_SM"],)  # noqa: E731
    sgmv_kernel[grid](
        y,
        x,
        wa_ptrs,
        wb_ptrs,
        s_start,
        s_end,
        layer_idx,
        nsegments,
        R,
        H1,
        H2,
        BLOCK_SIZE_S=16,
        BLOCK_SIZE_H1=128,
        BLOCK_SIZE_R=16,
        NUM_SM=84,
    )

    return y


def lora_ref_impl(
    y: torch.Tensor,
    x: torch.Tensor,
    wa: List[torch.Tensor],
    wb: List[torch.Tensor],
    s_start: torch.IntTensor,
    s_end: torch.IntTensor,
    layer_idx: int,
):
    for i in range(len(wa)):
        if s_end[i] - s_start[i] <= 0:
            continue

        xi = x[s_start[i] : s_end[i]]
        wai = wa[i][layer_idx, :, :]
        wbi = wb[i][layer_idx, :, :]

        yi = y[s_start[i] : s_end[i]]
        tmp = xi @ wai
        y[s_start[i] : s_end[i]] = yi + tmp @ wbi


device = "cuda"

B = 3 * 16
H1 = 1024
H2 = 1024
R = 16

nlayers = 2
layer_idx = 0

torch.manual_seed(42)
X = torch.rand((B, H1), device=device, dtype=torch.float16)

s1, s2 = [0, 2 * 16], [1 * 16, 3 * 16]
s_start = torch.tensor(s1, dtype=torch.int32, device=device)
s_end = torch.tensor(s2, dtype=torch.int32, device=device)

wa = [
    torch.randn(nlayers, H1, R, dtype=torch.float16, device=device),
    torch.randn(nlayers, H1, R, dtype=torch.float16, device=device),
]

wb = [
    torch.randn(nlayers, R, H2, dtype=torch.float16, device=device),
    torch.randn(nlayers, R, H2, dtype=torch.float16, device=device),
]

y_ref = torch.zeros((B, H2), dtype=torch.float16, device=device)
lora_ref_impl(y_ref, X, wa, wb, s_start, s_end, layer_idx)
print("y_ref", y_ref, y_ref.shape)

y = torch.zeros((B, H2), dtype=torch.float16, device=device)
v = torch.zeros((B, R), dtype=torch.float16, device=device)
print("shapes", X.shape, wa[0].shape, v.shape)
print("strides", X.stride(0), wa[0][0, :, :].stride(0), v.stride(0))
sgmv(v, X, wa, wb, s_start, s_end, layer_idx)
print("v", v, v.shape)
# print("y", y, y.shape)


# group_r = [128, 64, 32, 16]
# group_X = []
# group_A = []
# assert len(group_b) == len(group_h1)
# assert len(group_h1) == len(group_r)
# group_size = len(group_b)
# for i in range(group_size):
#     B = group_b[i]
#     H1 = group_h1[i]
#     R = group_r[i]
#     A = torch.rand((H1, R), device="cuda", dtype=torch.float16)
#     group_X.append(X)
#     group_A.append(A)
#     print(f"{B=} {H1=} {R=} {X.shape=} {A.shape=}")

# ref_out = [torch.matmul(x, a) for x, a in zip(group_X, group_A)]
# for ref in ref_out:
#     print(f"{ref.shape=}")

# tri_out = group_gemm_fn(group_X, group_A)
# for i in range(group_size):
#     assert torch.allclose(ref_out[i], tri_out[i], atol=1e-2, rtol=0)
#     print(f"{tri_out[i].shape=}")


# # only launch the kernel, no tensor preparation here to remove all overhead
# def triton_perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size):
#     grid = lambda META: (META["NUM_SM"],)
#     grouped_matmul_kernel[grid](
#         a_ptrs,
#         b_ptrs,
#         c_ptrs,
#         sizes,
#         lds,
#         group_size,
#     )


# def torch_perf_fn(group_A, group_B):
#     for a, b in zip(group_A, group_B):
#         torch.matmul(a, b)


# @triton.testing.perf_report(
#     triton.testing.Benchmark(
#         # argument names to use as an x-axis for the plot
#         x_names=["N"],
#         x_vals=[2**i for i in range(7, 11)],  # different possible values for `x_name`
#         line_arg="provider",
#         # argument name whose value corresponds to a different line in the plot
#         # possible values for `line_arg``
#         line_vals=["cublas", "triton"],
#         # label name for the lines
#         line_names=["cuBLAS", "Triton"],
#         # line styles
#         styles=[("green", "-"), ("blue", "-")],
#         ylabel="runtime(ms)",  # label name for the y-axis
#         plot_name="group-gemm-performance",
#         # name for the plot. Used also as a file name for saving the plot.
#         args={},
#     )
# )
# def benchmark(N, provider):
#     group_size = 4
#     group_A = []
#     group_B = []
#     A_addrs = []
#     B_addrs = []
#     C_addrs = []
#     g_sizes = []
#     g_lds = []
#     group_C = []
#     for i in range(group_size):
#         A = torch.rand((N, N), device="cuda", dtype=torch.float16)
#         B = torch.rand((N, N), device="cuda", dtype=torch.float16)
#         C = torch.empty((N, N), device="cuda", dtype=torch.float16)
#         group_A.append(A)
#         group_B.append(B)
#         group_C.append(C)
#         A_addrs.append(A.data_ptr())
#         B_addrs.append(B.data_ptr())
#         C_addrs.append(C.data_ptr())
#         g_sizes += [N, N, N]
#         g_lds += [N, N, N]

#     d_a_ptrs = torch.tensor(A_addrs, device="cuda")
#     d_b_ptrs = torch.tensor(B_addrs, device="cuda")
#     d_c_ptrs = torch.tensor(C_addrs, device="cuda")
#     d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device="cuda")
#     d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device="cuda")

#     quantiles = [0.5, 0.2, 0.8]
#     if provider == "cublas":
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_perf_fn(group_A, group_B), quantiles=quantiles)
#     if provider == "triton":
#         ms, min_ms, max_ms = triton.testing.do_bench(
#             lambda: triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size), quantiles=quantiles
#         )
#     return ms, max_ms, min_ms


# benchmark.run(show_plots=True, print_data=True)
