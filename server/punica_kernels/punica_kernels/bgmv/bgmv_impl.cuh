#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>
#include <iostream>

#include "../flashinfer/vec_dtypes.cuh"

namespace cg = cooperative_groups;

// nthrs = (32, 4)
template <int feat_in, int feat_out, typename T>
__global__ void bgmv_shrink_kernel(T* __restrict__ Y, const T* __restrict__ X,
                                   const T* __restrict__ W,
                                   const int64_t* __restrict__ indicies,
                                   int64_t num_layers, int64_t layer_idx,
                                   float scale) {
  auto block = cg::this_thread_block();
  size_t j = blockIdx.x;
  size_t batch_idx = blockIdx.y;
  constexpr size_t vec_size = 16 / sizeof(T);
  constexpr size_t tx = 32;
  constexpr size_t ty = 4;
  constexpr size_t num_pipeline_stages = 2;
  constexpr size_t tile_size = tx * ty * vec_size;
  __shared__ T W_shared[num_pipeline_stages * tile_size];
  __shared__ T X_shared[num_pipeline_stages * tile_size];
  __shared__ float y_warpwise[ty];

  int64_t idx = indicies[batch_idx] * num_layers + layer_idx;

  size_t W_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
  size_t X_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
  auto pipe = cuda::make_pipeline();

  // pipeline load W/X and compute WX;
  pipe.producer_acquire();
  cuda::memcpy_async(W_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
                     W + (idx * feat_out + j) * feat_in +
                         (threadIdx.y * tx + threadIdx.x) * vec_size,
                     cuda::aligned_size_t<16>(16), pipe);
  cuda::memcpy_async(
      X_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
      X + (batch_idx * feat_in) + (threadIdx.y * tx + threadIdx.x) * vec_size,
      cuda::aligned_size_t<16>(16), pipe);
  pipe.producer_commit();
  size_t copy_idx, compute_idx;
  float y = 0.f;
  flashinfer::vec_t<T, vec_size> x_vec, w_vec;
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
                         W + (idx * feat_out + j) * feat_in +
                             tile_idx * tile_size +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         cuda::aligned_size_t<16>(16), pipe);
      cuda::memcpy_async(X_shared + X_shared_offset[copy_idx] +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         X + (batch_idx * feat_in) + tile_idx * tile_size +
                             (threadIdx.y * tx + threadIdx.x) * vec_size,
                         cuda::aligned_size_t<16>(16), pipe);
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
    Y[batch_idx * feat_out + j] += y;
  }
}

// nthrs = (2, 16, 4)
template <int feat_in, int feat_out, typename T>
__global__ void bgmv_expand_kernel(T* __restrict__ Y, const T* __restrict__ X,
                                   const T* __restrict__ W,
                                   const int64_t* __restrict__ indicies,
                                   int64_t num_layers, int64_t layer_idx,
                                   float scale) {
  auto block = cg::this_thread_block();
  constexpr size_t vec_size = 16 / sizeof(T);
  constexpr size_t tx = feat_in / vec_size;
  static_assert(feat_in % vec_size == 0);
  constexpr size_t ty = 32 / tx;
  static_assert(32 % tx == 0);
  constexpr size_t tz = 4;
  size_t tile_idx = blockIdx.x;
  size_t batch_idx = blockIdx.y;
  int64_t idx = indicies[batch_idx] * num_layers + layer_idx;

  // load X;
  flashinfer::vec_t<T, vec_size> x_vec;
  x_vec.load(X + batch_idx * feat_in + threadIdx.x * vec_size);

  // load W;
  flashinfer::vec_t<T, vec_size> w_vec;
  w_vec.load(W + (idx * feat_out + tile_idx * tz * ty) * feat_in +
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
    Y[batch_idx * feat_out + tile_idx * (tz * ty) + threadIdx.z * ty +
      threadIdx.y] += sum;
  }
}

template <int feat_in, int feat_out, typename T>
void bgmv_kernel(T* __restrict__ Y, const T* __restrict__ X,
                 const T* __restrict__ W, const int64_t* __restrict__ indicies,
                 int64_t batch_size, int64_t num_layers, int64_t layer_idx,
                 float scale) {
  size_t vec_size = 16 / sizeof(T);
  if constexpr (feat_in < feat_out) {
    size_t tx = feat_in / vec_size;
    size_t ty = 32 / tx;
    size_t tz = 4;
    dim3 nblks(feat_out / (ty * tz), batch_size);
    dim3 nthrs(tx, ty, tz);

    bgmv_expand_kernel<feat_in, feat_out>
        <<<nblks, nthrs>>>(Y, X, W, indicies, num_layers, layer_idx, scale);
  } else {
    assert(feat_in % (vec_size * 32) == 0);
    dim3 nblks(feat_out, batch_size);
    dim3 nthrs(32, 4);
    bgmv_shrink_kernel<feat_in, feat_out>
        <<<nblks, nthrs>>>(Y, X, W, indicies, num_layers, layer_idx, scale);
  }
}

#define INST_BGMV(feat_in, feat_out, T)                                    \
  template void bgmv_kernel<feat_in, feat_out>(                            \
      T* __restrict__ Y, const T* __restrict__ X, const T* __restrict__ W, \
      const int64_t* __restrict__ indicies, int64_t batch_size,            \
      int64_t num_layers, int64_t layer_idx, float scale);

#define INST_BGMV_TWOSIDE(T, narrow, wide) \
  INST_BGMV(narrow, wide, T)               \
  INST_BGMV(wide, narrow, T)
