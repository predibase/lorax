#pragma once
#include <cooperative_groups.h>

#include "flashinfer/cp_async.cuh"
#include "flashinfer/mma.cuh"
#include "flashinfer/permuted_smem.cuh"
#include "flashinfer/vec_dtypes.cuh"

namespace flashinfer {

namespace sgmv {

template <bool cooperative, typename T, typename IdType, uint32_t num_warps,
          uint32_t d_out>
__global__ void sgmv_shrink(T* y, T* x, T** w, IdType* s_starts, IdType* s_ends, float* tmp,
                            uint32_t num_problems, uint32_t d_in,
                            uint32_t layer_idx, uint32_t chunk_size) {
  auto block = cooperative_groups::this_thread_block();
  auto grid = cooperative_groups::this_grid();
  constexpr auto fill_mode = cp_async::SharedMemFillMode::kFillZero;
  const uint32_t problem_id = blockIdx.y;
  const uint32_t bx = blockIdx.x;

  constexpr uint32_t num_stages = 2;
  constexpr uint32_t num_k_frags = 8;
  constexpr uint32_t num_cells_k = (num_k_frags * 16) / cell_capacity<T>();
  constexpr uint32_t num_blocks_n = d_out / 16;
  const uint32_t num_chunks = gridDim.x;
  const uint32_t chunk_start = chunk_size * bx;
  const uint32_t num_iterations =
      (chunk_size + (num_k_frags * 16 - 1)) / (num_k_frags * 16);
  constexpr uint32_t num_cells_n =
      (d_out < 32 ? 32 : d_out) / cell_capacity<T>();
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;

  extern __shared__ uint8_t smem[];

  smem_t x_smem[2]{smem, smem + sizeof(T) * num_warps * 16 * 16 * num_k_frags};
  smem_t w_smem[2]{smem + sizeof(T) * 2 * num_warps * 16 * 16 * num_k_frags,
                   smem + sizeof(T) * 16 * 16 * num_k_frags *
                              (2 * num_warps + num_blocks_n)};
  smem_t y_smem(smem);

  uint32_t x_frag[num_k_frags][4];
  uint32_t w_frag[num_k_frags][num_blocks_n][4];
  float y_frag[num_blocks_n][8];

  const uint32_t s_start = s_starts[problem_id], s_end = s_ends[problem_id];
  const uint32_t num_steps = (s_start < s_end) ? (s_end - s_start + (num_warps * 16 - 1)) / (num_warps * 16) : 0;
  for (uint32_t i = 0; i < num_steps; ++i) {
    // init y_frag
    if (bx == 0) {
      if constexpr (num_blocks_n == 1) {
        uint32_t row_idx = s_start + (i * num_warps + ty) * 16 + tx / 2;
        T* y_ptr = y + row_idx * d_out + (tx % 2) * cell_capacity<T>();
        auto offset =
            smem_t::get_permuted_offset<num_cells_n>(ty * 16 + tx / 2, tx % 2);
        y_smem.load_128b_async<fill_mode>(offset, y_ptr, row_idx < s_end);
      } else {
        uint32_t row_idx = s_start + (i * num_warps + ty) * 16 + tx / 4;
        T* y_ptr = y + row_idx * d_out + (tx % 4) * cell_capacity<T>();
        auto offset =
            smem_t::get_permuted_offset<num_cells_n>(ty * 16 + tx / 4, tx % 4);
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
          for (uint32_t fno = 0; fno < num_blocks_n / 2; ++fno) {
            y_smem.load_128b_async<fill_mode>(offset, y_ptr, row_idx < s_end);
            y_ptr += 4 * cell_capacity<T>();
            offset += 8;
          }
          row_idx += 8;
          y_ptr += 8 * d_out - 2 * num_blocks_n * cell_capacity<T>();
          offset += 8 * num_cells_n - 4 * num_blocks_n;
        }
      }
      cp_async::commit_group();
      cp_async::wait_group<0>();
      block.sync();

      auto offset =
          smem_t::get_permuted_offset<num_cells_n>(ty * 16 + tx % 16, tx / 16);
#pragma unroll
      for (uint32_t fn = 0; fn < num_blocks_n; ++fn) {
        uint32_t tmp[4];
        y_smem.ldmatrix_m8n8x4(offset, tmp);
        vec_cast<float, T, 8>(y_frag[fn], (T*)tmp);
        offset = (offset ^ 0x2) + (fn & 0x1) * 8;
      }
    } else {
#pragma unroll
      for (uint32_t fn = 0; fn < num_blocks_n; ++fn) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          y_frag[fn][reg_id] = 0.f;
        }
      }
    }

    // preload x_smem, w_smem
#pragma unroll
    for (uint32_t iter = 0; iter < num_stages; ++iter) {
      uint32_t row_idx = s_start + (i * num_warps + ty) * 16 + tx / 4;
      T* x_ptr = x + row_idx * d_in + chunk_start +
                 (2 * num_k_frags * iter + tx % 4) * cell_capacity<T>();
      T* x_ptr_max = x + row_idx * d_in + min(d_in, chunk_start + chunk_size);
      auto offset =
          smem_t::get_permuted_offset<num_cells_k>(ty * 16 + tx / 4, tx % 4);
      // pre-load x_smem, w_smem
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
        for (uint32_t fko = 0; fko < num_k_frags / 2; ++fko) {
          x_smem[iter].load_128b_async<fill_mode>(
              offset, x_ptr, row_idx < s_end && x_ptr < x_ptr_max);
          x_ptr += 4 * cell_capacity<T>();
          offset += 8;
        }
        row_idx += 8;
        x_ptr += 8 * d_in - 2 * cell_capacity<T>() * num_k_frags;
        x_ptr_max += 8 * d_in;
        offset += 8 * num_cells_k - 4 * num_k_frags;
      }
      row_idx -= 8;

      static_assert(num_k_frags % (num_warps * 2) == 0);
      constexpr uint32_t num_fko_iters_per_warp = num_k_frags / (num_warps * 2);
#pragma unroll
      for (uint32_t fn = 0; fn < num_blocks_n; ++fn) {
        T* w_ptr = w[problem_id] + layer_idx * d_in * d_out +
                   (fn * 16 + tx / 4) * d_in + chunk_start +
                   (2 * num_k_frags * iter + ty * num_fko_iters_per_warp * 4 +
                    tx % 4) *
                       cell_capacity<T>();
        T* w_ptr_max =
            w[problem_id] + layer_idx * d_in * d_out +
            min((fn * 16 + tx / 4 + 1) * d_in,
                (fn * 16 + tx / 4) * d_in + chunk_start + chunk_size);
        auto offset = smem_t::get_permuted_offset<num_cells_k>(
            fn * 16 + tx / 4, ty * num_fko_iters_per_warp * 4 + tx % 4);
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
          for (uint32_t fko = 0; fko < num_fko_iters_per_warp; ++fko) {
            w_smem[iter].load_128b_async<fill_mode>(offset, w_ptr,
                                                    w_ptr < w_ptr_max);
            w_ptr += 4 * cell_capacity<T>();
            offset += 8;
          }
          w_ptr += 8 * d_in - 4 * cell_capacity<T>() * num_fko_iters_per_warp;
          w_ptr_max += 8 * d_in;
          offset += 8 * num_cells_k - 8 * num_fko_iters_per_warp;
        }
      }
      cp_async::commit_group();
    }

#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      const uint32_t stage_idx = iter % 2;
      cp_async::wait_group<1>();
      block.sync();

      auto offset =
          smem_t::get_permuted_offset<num_cells_k>(ty * 16 + tx % 16, tx / 16);
#pragma unroll
      for (uint32_t fk = 0; fk < num_k_frags; ++fk) {
        x_smem[stage_idx].ldmatrix_m8n8x4(offset, x_frag[fk]);
        offset = (offset ^ 0x2) + (fk & 0x1) * 8;
      }

#pragma unroll
      for (uint32_t fn = 0; fn < num_blocks_n; ++fn) {
        auto offset = smem_t::get_permuted_offset<num_cells_k>(
            fn * 16 + 8 * (tx / 16) + tx % 8, (tx % 16) / 8);
#pragma unroll
        for (uint32_t fk = 0; fk < num_k_frags; ++fk) {
          w_smem[stage_idx].ldmatrix_m8n8x4(offset, w_frag[fk][fn]);
          offset = (offset ^ 0x2) + (fk & 0x1) * 8;
        }
        offset += 16 * num_cells_k - 4 * num_k_frags;
      }

      // compute y_frag
#pragma unroll
      for (uint32_t fk = 0; fk < num_k_frags; ++fk) {
#pragma unroll
        for (uint32_t fn = 0; fn < num_blocks_n; ++fn) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<T>(y_frag[fn], x_frag[fk],
                                                       w_frag[fk][fn]);
        }
      }
      block.sync();

      // load next stage
      if (iter + num_stages < num_iterations) {
        uint32_t row_idx = s_start + (i * num_warps + ty) * 16 + tx / 4;
        T* x_ptr = x + row_idx * d_in + chunk_start +
                   (2 * num_k_frags * (iter + num_stages) + tx % 4) *
                       cell_capacity<T>();
        T* x_ptr_max = x + row_idx * d_in + min(d_in, chunk_start + chunk_size);
        auto offset =
            smem_t::get_permuted_offset<num_cells_k>(ty * 16 + tx / 4, tx % 4);
        // pre-load x_smem, w_smem
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
          for (uint32_t fko = 0; fko < num_k_frags / 2; ++fko) {
            x_smem[stage_idx].load_128b_async<fill_mode>(
                offset, x_ptr, row_idx < s_end && x_ptr < x_ptr_max);
            x_ptr += 4 * cell_capacity<T>();
            offset += 8;
          }
          row_idx += 8;
          x_ptr += 8 * d_in - 2 * cell_capacity<T>() * num_k_frags;
          x_ptr_max += 8 * d_in;
          offset += 8 * num_cells_k - 4 * num_k_frags;
        }
        row_idx -= 8;

        constexpr uint32_t num_fko_iters_per_warp =
            num_k_frags / (num_warps * 2);
#pragma unroll
        for (uint32_t fn = 0; fn < num_blocks_n; ++fn) {
          T* w_ptr = w[problem_id] + layer_idx * d_in * d_out +
                     (fn * 16 + tx / 4) * d_in + chunk_start +
                     (2 * num_k_frags * (iter + num_stages) +
                      ty * num_fko_iters_per_warp * 4 + tx % 4) *
                         cell_capacity<T>();
          T* w_ptr_max =
              w[problem_id] + layer_idx * d_in * d_out +
              min((fn * 16 + tx / 4 + 1) * d_in,
                  (fn * 16 + tx / 4) * d_in + chunk_start + chunk_size);
          auto offset = smem_t::get_permuted_offset<num_cells_k>(
              fn * 16 + tx / 4, ty * num_fko_iters_per_warp * 4 + tx % 4);
#pragma unroll
          for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
            for (uint32_t fko = 0; fko < num_fko_iters_per_warp; ++fko) {
              w_smem[stage_idx].load_128b_async<fill_mode>(offset, w_ptr,
                                                           w_ptr < w_ptr_max);
              w_ptr += 4 * cell_capacity<T>();
              offset += 8;
            }
            w_ptr += 8 * d_in - 4 * cell_capacity<T>() * num_fko_iters_per_warp;
            w_ptr_max += 8 * d_in;
            offset += 8 * num_cells_k - 8 * num_fko_iters_per_warp;
          }
        }
      }
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    block.sync();

    if constexpr (cooperative) {
#pragma unroll
      for (uint32_t fn = 0; fn < num_blocks_n; ++fn) {
        vec_t<float, 8>::memcpy(
            tmp + (fn * grid.size() +
                   (problem_id * num_chunks + bx) * block.num_threads() +
                   block.thread_rank()) *
                      8,
            y_frag[fn]);
      }
      grid.sync();

#pragma unroll
      for (uint32_t fn = 0; fn < num_blocks_n; ++fn) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          y_frag[fn][reg_id] = 0.f;
        }
        for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
          vec_t<float, 8> y_other;
          y_other.load(tmp + (fn * grid.size() +
                              (problem_id * num_chunks + chunk_idx) *
                                  block.num_threads() +
                              block.thread_rank()) *
                                 8);
#pragma unroll
          for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
            y_frag[fn][reg_id] += y_other[reg_id];
          }
        }
      }
    }

    if (bx == 0) {
      // store y_frag
      auto offset =
          smem_t::get_permuted_offset<num_cells_n>(ty * 16 + tx / 4, 0);
#pragma unroll
      for (uint32_t fn = 0; fn < num_blocks_n; ++fn) {
        vec_cast<T, float, 2>((T*)(y_smem.base + offset) + (tx % 4) * 2,
                              &y_frag[fn][0]);
        vec_cast<T, float, 2>(
            (T*)(y_smem.base + offset + 8 * num_cells_n) + (tx % 4) * 2,
            &y_frag[fn][2]);
        vec_cast<T, float, 2>((T*)(y_smem.base + (offset ^ 0x1)) + (tx % 4) * 2,
                              &y_frag[fn][4]);
        vec_cast<T, float, 2>(
            (T*)(y_smem.base + (offset ^ 0x1) + 8 * num_cells_n) + (tx % 4) * 2,
            &y_frag[fn][6]);
        offset = (offset ^ 0x2) + (fn & 0x1) * 8;
      }

      // store y
      if constexpr (num_blocks_n == 1) {
        uint32_t row_idx = s_start + (i * num_warps + ty) * 16 + tx / 2;
        T* y_ptr = y + row_idx * d_out + (tx % 2) * cell_capacity<T>();
        auto offset =
            smem_t::get_permuted_offset<num_cells_n>(ty * 16 + tx / 2, tx % 2);
        if (row_idx < s_end) {
          y_smem.store_128b(offset, y_ptr);
        }
      } else {
        uint32_t row_idx = s_start + (i * num_warps + ty) * 16 + tx / 4;
        T* y_ptr = y + row_idx * d_out + (tx % 4) * cell_capacity<T>();
        auto offset =
            smem_t::get_permuted_offset<num_cells_n>(ty * 16 + tx / 4, tx % 4);
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
          for (uint32_t fno = 0; fno < num_blocks_n / 2; ++fno) {
            if (row_idx < s_end) {
              y_smem.store_128b(offset, y_ptr);
            }
            y_ptr += 4 * cell_capacity<T>();
            offset += 8;
          }
          row_idx += 8;
          y_ptr += 8 * d_out - 2 * num_blocks_n * cell_capacity<T>();
          offset += 8 * num_cells_n - 4 * num_blocks_n;
        }
      }
    }
  }

  // handle the case where one of the segments needs more steps than this one
  // to avoid deadlock
  if constexpr (cooperative) {
    uint32_t max_segment_size = 0;
    for (uint32_t i = 0; i < num_problems; ++i) {
      max_segment_size = max(max_segment_size, s_ends[i] - s_starts[i]);
    }

    const uint32_t max_steps = (max_segment_size + (num_warps * 16 - 1)) / (num_warps * 16);
    for (uint32_t i = 0; i < max_steps - num_steps; ++i) {
      grid.sync();
    }
  }
}

}  // namespace sgmv
}  // namespace flashinfer
