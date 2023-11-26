// Adapted from cutlass
// https://github.com/NVIDIA/cutlass/blob/7d8317a63e0a978a8dbb3c1fb7af4dbe4f286616/tools/util/include/cutlass/util/device_rmsnorm.h
/******************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <float.h>

#include <algorithm>
#include <type_traits>

template <typename T, int NUM>
__inline__ __device__ T warpReduceSum(T *val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(0xffffffff, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSum(T *val) {
  __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSum<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSum<T, NUM>(val);
  return (T)0.0f;
}

template <typename half, typename half2>
__global__ void rmsnorm_twoPassAlgo_e8(float4 *__restrict__ output,
                                       const float4 *__restrict__ input,
                                       const float4 *__restrict__ weight, int m,
                                       int n, float epsilon) {
  const int m_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int bdimx = blockDim.x;
  __shared__ float s_mean;
  float local_sums[1] = {0.0f};
  const int n_8 = n / 8;
  int offset = m_idx * n_8;
  input += offset;
  output += offset;

  for (int index = tid; index < n_8; index += bdimx) {
    const float4 local_val = input[index];
    const half2 *h1 = (half2 *)&local_val.x;
    const half2 *h2 = (half2 *)&local_val.y;
    const half2 *h3 = (half2 *)&local_val.z;
    const half2 *h4 = (half2 *)&local_val.w;
    local_sums[0] += static_cast<float>(h1->x) * static_cast<float>(h1->x) +
                     static_cast<float>(h1->y) * static_cast<float>(h1->y) +
                     static_cast<float>(h2->x) * static_cast<float>(h2->x) +
                     static_cast<float>(h2->y) * static_cast<float>(h2->y) +
                     static_cast<float>(h3->x) * static_cast<float>(h3->x) +
                     static_cast<float>(h3->y) * static_cast<float>(h3->y) +
                     static_cast<float>(h4->x) * static_cast<float>(h4->x) +
                     static_cast<float>(h4->y) * static_cast<float>(h4->y);
  }

  blockReduceSum<float, 1>(local_sums);
  if (threadIdx.x == 0) {
    s_mean = rsqrtf(local_sums[0] / n + epsilon);
  }
  __syncthreads();

  for (int index = tid; index < n_8; index += bdimx) {
    const float4 local_val = input[index];
    const float4 weight_val = weight[index];

    const half2 *l1 = (half2 *)&local_val.x;
    const half2 *l2 = (half2 *)&local_val.y;
    const half2 *l3 = (half2 *)&local_val.z;
    const half2 *l4 = (half2 *)&local_val.w;

    const half2 *g1 = (half2 *)&weight_val.x;
    const half2 *g2 = (half2 *)&weight_val.y;
    const half2 *g3 = (half2 *)&weight_val.z;
    const half2 *g4 = (half2 *)&weight_val.w;

    float4 tmp;
    half2 *h1 = (half2 *)&tmp.x;
    half2 *h2 = (half2 *)&tmp.y;
    half2 *h3 = (half2 *)&tmp.z;
    half2 *h4 = (half2 *)&tmp.w;

    h1->x = static_cast<half>(static_cast<float>(l1->x) * s_mean *
                              static_cast<float>(g1->x));
    h1->y = static_cast<half>(static_cast<float>(l1->y) * s_mean *
                              static_cast<float>(g1->y));
    h2->x = static_cast<half>(static_cast<float>(l2->x) * s_mean *
                              static_cast<float>(g2->x));
    h2->y = static_cast<half>(static_cast<float>(l2->y) * s_mean *
                              static_cast<float>(g2->y));
    h3->x = static_cast<half>(static_cast<float>(l3->x) * s_mean *
                              static_cast<float>(g3->x));
    h3->y = static_cast<half>(static_cast<float>(l3->y) * s_mean *
                              static_cast<float>(g3->y));
    h4->x = static_cast<half>(static_cast<float>(l4->x) * s_mean *
                              static_cast<float>(g4->x));
    h4->y = static_cast<half>(static_cast<float>(l4->y) * s_mean *
                              static_cast<float>(g4->y));

    output[index] = tmp;
  }
}

template <typename T>
bool rms_norm(T *__restrict__ output, const T *__restrict__ input,
              const T *__restrict__ weight, int rows, int columns,
              float epsilon) {
  if (columns % 8 != 0) {
    return false;
  }

  dim3 grid(rows);
  dim3 block(std::min(1024, (columns / 8 + 31) / 32 * 32));

  if (std::is_same<T, nv_half>::value) {
    rmsnorm_twoPassAlgo_e8<nv_half, nv_half2>
        <<<grid, block>>>((float4 *)output, (float4 *)input, (float4 *)weight,
                          rows, columns, epsilon);
    return true;
  } else if (std::is_same<T, nv_bfloat16>::value) {
    rmsnorm_twoPassAlgo_e8<nv_bfloat16, nv_bfloat162>
        <<<grid, block>>>((float4 *)output, (float4 *)input, (float4 *)weight,
                          rows, columns, epsilon);
    return true;
  }
  return false;
}

template bool rms_norm(nv_half *__restrict__ output,
                       const nv_half *__restrict__ input,
                       const nv_half *__restrict__ weight, int rows,
                       int columns, float epsilon);
template bool rms_norm(nv_bfloat16 *__restrict__ output,
                       const nv_bfloat16 *__restrict__ input,
                       const nv_bfloat16 *__restrict__ weight, int rows,
                       int columns, float epsilon);
