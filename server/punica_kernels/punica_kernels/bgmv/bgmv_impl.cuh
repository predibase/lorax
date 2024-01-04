#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "flashinfer/vec_dtypes.cuh"

namespace cg = cooperative_groups;

// nthrs = (32, 4)
template <int feat_in, int feat_out, size_t vec_size, size_t X_copy_size,
          size_t W_copy_size, int tx, int ty, int tz, typename T>
__global__ void
bgmv_shrink_kernel(T* __restrict__ Y, const T* __restrict__ X,
                   T** __restrict__ W,
                   const int64_t* __restrict__ indicies, int64_t y_offset,
                   int64_t full_y_size, int64_t layer_idx,
                   float scale) {
  size_t batch_idx = blockIdx.y;
  int64_t idx = indicies[batch_idx];
  if (idx < 0) {
    return;
  }

  auto block = cg::this_thread_block();
  size_t j = blockIdx.x;
  constexpr size_t num_pipeline_stages = 2;
  constexpr size_t tile_size = tx * ty * vec_size;
  __shared__ T W_shared[num_pipeline_stages * tile_size];
  __shared__ T X_shared[num_pipeline_stages * tile_size];
  __shared__ float y_warpwise[ty];

  size_t W_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
  size_t X_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
  auto pipe = cuda::make_pipeline();

  const T* W_ptr = W[idx];

  // pipeline load W/X and compute WX;
  pipe.producer_acquire();
  cuda::memcpy_async(W_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
                     W_ptr + (layer_idx * feat_out + j) * feat_in +
                         (threadIdx.y * tx + threadIdx.x) * vec_size,
                     cuda::aligned_size_t<W_copy_size>(W_copy_size), pipe);
  cuda::memcpy_async(X_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
                     X + (batch_idx * feat_in) +
                         (threadIdx.y * tx + threadIdx.x) * vec_size,
                     cuda::aligned_size_t<X_copy_size>(X_copy_size), pipe);
  pipe.producer_commit();
  size_t copy_idx, compute_idx;
  float y = 0.f;
  flashinfer::vec_t<T, vec_size> x_vec;
  flashinfer::vec_t<T, vec_size> w_vec;
  size_t tile_idx;

#pragma unroll
  for (tile_idx = 1; tile_idx < (feat_in + tile_size - 1) / tile_size;
       ++tile_idx) {
    copy_idx = tile_idx % num_pipeline_stages;
    // pipeline stage: async copy W fragment
    pipe.producer_acquire();
    if (tile_idx * tile_size + threadIdx.y * tx * vec_size < feat_in) {
      cuda::memcpy_async(W_shared + W_shared_offset[copy_idx] +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         W_ptr + (layer_idx * feat_out + j) * feat_in +
                             tile_idx * tile_size +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         cuda::aligned_size_t<W_copy_size>(W_copy_size), pipe);
      cuda::memcpy_async(X_shared + X_shared_offset[copy_idx] +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         X + (batch_idx * feat_in) + tile_idx * tile_size +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         cuda::aligned_size_t<X_copy_size>(X_copy_size), pipe);
    }
    pipe.producer_commit();

    compute_idx = (tile_idx - 1) % num_pipeline_stages;
    // pipeline stage: compute WX
    pipe.consumer_wait();
    block.sync();
    x_vec.load(X_shared + X_shared_offset[compute_idx] +
               (threadIdx.y * tx + threadIdx.x) * vec_size);
    w_vec.load(W_shared + W_shared_offset[compute_idx] +
               (threadIdx.y * tx + threadIdx.x) * vec_size);
    float sum = 0.f;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      sum += float(w_vec[i]) * float(x_vec[i]) * scale;
    }
#pragma unroll
    for (size_t offset = tx / 2; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    y_warpwise[threadIdx.y] = sum;
    block.sync();
#pragma unroll
    for (size_t i = 0; i < ty; ++i) {
      y += y_warpwise[i];
    }

    block.sync();
    pipe.consumer_release();
  }

  compute_idx = (tile_idx - 1) % num_pipeline_stages;
  // final pipeline stage
  pipe.consumer_wait();
  block.sync();
  x_vec.load(X_shared + X_shared_offset[compute_idx] +
             (threadIdx.y * tx + threadIdx.x) * vec_size);
  w_vec.load(W_shared + W_shared_offset[compute_idx] +
             (threadIdx.y * tx + threadIdx.x) * vec_size);
  float sum = 0.f;
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    sum += float(w_vec[i]) * float(x_vec[i]) * scale;
  }
#pragma unroll
  for (size_t offset = tx / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  y_warpwise[threadIdx.y] =
      ((tile_idx - 1) * tile_size + threadIdx.y * tx * vec_size < feat_in)
          ? sum
          : 0.f;
  block.sync();
#pragma unroll
  for (size_t i = 0; i < ty; ++i) {
    y += y_warpwise[i];
  }

  block.sync();
  pipe.consumer_release();

  // write Y;
  if (block.thread_rank() == 0) {
    Y[batch_idx * full_y_size + y_offset + j] += static_cast<T>(y);
  }
}

// nthrs = (2, 16, 4)
template <int feat_in, int feat_out, size_t vec_size, int tx, int ty, int tz,
          typename T>
__global__ void
bgmv_expand_kernel(T* __restrict__ Y, const T* __restrict__ X,
                   T** __restrict__ W,
                   const int64_t* __restrict__ indicies, int64_t y_offset,
                   int64_t full_y_size, int64_t layer_idx,
                   float scale) {
  size_t batch_idx = blockIdx.y;
  int64_t idx = indicies[batch_idx];

  if (idx < 0) {
    return;
  }

  auto block = cg::this_thread_block();
  size_t tile_idx = blockIdx.x;

  const T* W_ptr = W[idx];

  // load X;
  flashinfer::vec_t<T, vec_size> x_vec;
  x_vec.load(X + batch_idx * feat_in + threadIdx.x * vec_size);

  // load W;
  flashinfer::vec_t<T, vec_size> w_vec;
  w_vec.load(W_ptr + (layer_idx * feat_out + tile_idx * tz * ty) * feat_in +
             block.thread_rank() * vec_size);

  float sum = 0.f;
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    sum += float(w_vec[i]) * float(x_vec[i]) * scale;
  }

  cg::thread_block_tile g = cg::tiled_partition<tx>(block);
#pragma unroll
  for (size_t offset = tx / 2; offset > 0; offset /= 2) {
    sum += g.shfl_down(sum, offset);
  }
  sum = g.shfl(sum, 0);

  if (threadIdx.x == 0) {
    Y[batch_idx * full_y_size + y_offset + tile_idx * (tz * ty) +
      threadIdx.z * ty + threadIdx.y] += static_cast<T>(sum);
  }
}

template <int feat_in, int feat_out, typename T>
void bgmv_kernel(T* __restrict__ Y, const T* __restrict__ X,
                 T** __restrict__ W,
                 const int64_t* __restrict__ indicies, int64_t y_offset,
                 int64_t full_y_size, int64_t batch_size,
                 int64_t layer_idx, float scale) {
  constexpr size_t vec_size = 8;
  constexpr int tz = 4;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if constexpr (feat_in < feat_out) {
    static_assert(feat_in % vec_size == 0);
    constexpr int tx = feat_in / vec_size;

    static_assert((32 % tx == 0 && feat_out % (32 / tx * tz) == 0) ||
                  (16 % tx == 0 && feat_out % (16 / tx * tz) == 0) ||
                  (8 % tx == 0 && feat_out % (8 / tx * tz) == 0));

    if constexpr (32 % tx == 0 && feat_out % (32 / tx * tz) == 0) {
      constexpr int ty = 32 / tx;
      dim3 nblks(feat_out / (ty * tz), batch_size);
      dim3 nthrs(tx, ty, tz);

      bgmv_expand_kernel<feat_in, feat_out, vec_size, tx, ty, tz>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, layer_idx,
                                        scale);
    } else if (16 % tx == 0 && feat_out % (16 / tx * tz) == 0) {
      constexpr int ty = 16 / tx;
      dim3 nblks(feat_out / (ty * tz), batch_size);
      dim3 nthrs(tx, ty, tz);

      bgmv_expand_kernel<feat_in, feat_out, vec_size, tx, ty, tz>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, layer_idx,
                                        scale);
    } else {
      constexpr int ty = 8 / tx;
      dim3 nblks(feat_out / (ty * tz), batch_size);
      dim3 nthrs(tx, ty, tz);

      bgmv_expand_kernel<feat_in, feat_out, vec_size, tx, ty, tz>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, layer_idx,
                                        scale);
    }
  } else {
    static_assert(feat_in % (vec_size * 32) == 0 ||
                  feat_in % (vec_size * 16) == 0 ||
                  feat_in % (vec_size * 8) == 0);

    if constexpr (feat_in % (vec_size * 32) == 0) {
      constexpr int tx = 32;
      constexpr int ty = 4;

      dim3 nblks(feat_out, batch_size);
      dim3 nthrs(tx, ty);

      bgmv_shrink_kernel<feat_in, feat_out, vec_size, vec_size * sizeof(T),
                         vec_size * sizeof(T), tx, ty, tz>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, layer_idx,
                                        scale);
    } else if constexpr (feat_in % (vec_size / 2 * 32) == 0) {
      constexpr int tx = 32;
      constexpr int ty = 4;

      dim3 nblks(feat_out, batch_size);
      dim3 nthrs(tx, ty);

      bgmv_shrink_kernel<feat_in, feat_out, vec_size / 2,
                         vec_size * sizeof(T) / 2,
                         vec_size * sizeof(T) / 2, tx, ty, tz>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, layer_idx,
                                        scale);
    } else if constexpr (feat_in % (vec_size / 2 * 16) == 0) {
      constexpr int tx = 16;
      constexpr int ty = 4;

      dim3 nblks(feat_out, batch_size);
      dim3 nthrs(tx, ty);

      bgmv_shrink_kernel<feat_in, feat_out, vec_size / 2,
                         vec_size * sizeof(T) / 2,
                         vec_size * sizeof(T) / 2, tx, ty, tz>
          <<<nblks, nthrs, 0, stream>>>(Y, X, W, indicies, y_offset,
                                        full_y_size, layer_idx,
                                        scale);
    }
  }
}

#define INST_BGMV(feat_in, feat_out, T)                         \
  template void bgmv_kernel<feat_in, feat_out>(                                \
      T*  __restrict__ Y, const T* __restrict__ X,                      \
      T** __restrict__ W, const int64_t* __restrict__ indicies,         \
      int64_t y_offset, int64_t full_y_size, int64_t batch_size,               \
      int64_t layer_idx, float scale);

#define INST_BGMV_TWOSIDE(T, narrow, wide)                      \
  INST_BGMV(narrow, wide, T)                                    \
  INST_BGMV(wide, narrow, T)