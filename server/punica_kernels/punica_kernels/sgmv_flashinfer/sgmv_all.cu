#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "sgmv_config.h"
#include "sgmv_flashinfer.cuh"

template <typename T, uint32_t d_out>
bool sgmv_shrink(T* y, T* x, T** w, int32_t* s_start, int32_t* s_end, void* tmp,
                 uint32_t num_problems, uint32_t d_in, uint32_t layer_idx, cudaStream_t stream) {
  static_assert(d_out % 16 == 0);

  constexpr uint32_t num_warps = 4;
  constexpr uint32_t num_stages = 2;
  constexpr uint32_t num_k_frags_per_stage = 8;
  constexpr uint32_t num_blocks_n = d_out / 16;
  uint32_t smem = num_stages * sizeof(T) * num_k_frags_per_stage * 16 * 16 *
                  (num_warps + num_blocks_n);
  auto cooperative_kernel =
      flashinfer::sgmv::sgmv_shrink<true, T, int, num_warps, d_out>;
  auto kernel = flashinfer::sgmv::sgmv_shrink<false, T, int, num_warps, d_out>;

  int dev_id = 0;
  int num_blocks_per_sm = 0;
  int num_sm = 0;
  bool use_cooperative = true;
  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_sm, cooperative_kernel, num_warps * 32, smem);

  const uint32_t max_grid_size = num_sm * num_blocks_per_sm;

  uint32_t chunk_size = 256;
  uint32_t num_chunks = (d_in + chunk_size - 1) / chunk_size;
  if (num_chunks * num_problems > max_grid_size) {
    use_cooperative = false;
    chunk_size = d_in;
    num_chunks = 1;
  }

  dim3 nthrs(32, num_warps);
  dim3 nblks(num_chunks, num_problems);

  void* args[] = {(void*)&y,    (void*)&x,         (void*)&w,
                  (void*)&s_start,    (void*)&s_end, (void*)&tmp,       (void*)&num_problems,
                  (void*)&d_in, (void*)&layer_idx, (void*)&chunk_size};

  cudaError_t status;
  if (use_cooperative) {
    if (smem > 46 * 1024) {
      cudaFuncSetAttribute(cooperative_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    status = cudaLaunchCooperativeKernel((void*)cooperative_kernel, nblks,
                                         nthrs, args, smem, stream);
  } else {
    if (smem > 46 * 1024) {
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    status = cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem, stream);
  }
  return status == cudaSuccess;
}

#define INST(T, d_out)                                                   \
  template bool sgmv_shrink<T, d_out>(T * y, T * x, T * *w, int32_t * s_start, int32_t * s_end, \
                                      void* tmp, uint32_t num_problems,  \
                                      uint32_t d_in, uint32_t layer_idx, cudaStream_t stream);

FOR_SGMV_NARROW(INST, nv_half);
FOR_SGMV_NARROW(INST, nv_bfloat16);