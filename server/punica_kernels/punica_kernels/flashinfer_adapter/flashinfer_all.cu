#include <algorithm>
#include <cmath>
#include <cstdint>

#include "../flashinfer/decode.cuh"
#include "../flashinfer/page.cuh"
#include "flashinfer_config.h"

template <typename T>
void FlashInferBatchDecodeKernel(T* o, T* q, T* kv_data, int32_t* kv_indptr,
                                 int32_t* kv_indicies,
                                 int32_t* last_page_offset, int head_dim,
                                 int num_layers, int layer_idx,
                                 int num_qo_heads, int num_kv_heads,
                                 int page_size, int batch_size) {
  flashinfer::paged_kv_t<T, int32_t> paged_kv(
      num_layers, layer_idx, num_kv_heads, page_size, head_dim, batch_size,
      kv_data, kv_indptr, kv_indicies, last_page_offset);
  flashinfer::BatchDecodeWithPagedKVCache(q, paged_kv, o, nullptr, num_qo_heads,
                                          flashinfer::RotaryMode::kLlama);
}

template <int head_dim, typename T>
void FlashInferInitKvKernel(T* kv_data, int32_t* kv_indptr,
                            int32_t* kv_indicies, int32_t* last_page_offset,
                            T* key, T* value, int32_t* seqlen_indptr,
                            int num_layers, int layer_idx, int num_kv_heads,
                            int page_size, int batch_size) {
  flashinfer::paged_kv_t<T, int32_t> paged_kv(
      num_layers, layer_idx, num_kv_heads, page_size, head_dim, batch_size,
      kv_data, kv_indptr, kv_indicies, last_page_offset);

  constexpr size_t vec_size =
      std::max(16 / sizeof(T), static_cast<size_t>(head_dim / 32));
  constexpr size_t bdx = head_dim / vec_size;
  constexpr size_t bdy = 128 / bdx;
  dim3 nblks(paged_kv.batch_size * paged_kv.num_heads / bdy);
  dim3 nthrs(bdx, bdy);
  flashinfer::AppendPagedKVCachePrefillKernel<head_dim, vec_size, bdx, bdy, T,
                                              int32_t>
      <<<nblks, nthrs>>>(paged_kv, key, value, seqlen_indptr);
}

template <int head_dim, typename T>
void FlashInferAppendKvKernel(T* kv_data, int32_t* kv_indptr,
                              int32_t* kv_indicies, int32_t* last_page_offset,
                              T* key, T* value, int num_layers, int layer_idx,
                              int num_kv_heads, int page_size, int batch_size) {
  flashinfer::paged_kv_t<T, int32_t> paged_kv(
      num_layers, layer_idx, num_kv_heads, page_size, head_dim, batch_size,
      kv_data, kv_indptr, kv_indicies, last_page_offset);

  constexpr size_t vec_size =
      std::max(16 / sizeof(T), static_cast<size_t>(head_dim / 32));
  constexpr size_t bdx = head_dim / vec_size;
  constexpr size_t bdy = 128 / bdx;
  dim3 nblks(paged_kv.batch_size * paged_kv.num_heads / bdy);
  dim3 nthrs(bdx, bdy);
  flashinfer::AppendPagedKVCacheDecodeKernel<head_dim, vec_size, bdx, bdy, T,
                                             int32_t>
      <<<nblks, nthrs>>>(paged_kv, key, value);
}

#define INST_FlashInferBatchDecodeKernel(T)                                    \
  template void FlashInferBatchDecodeKernel<T>(                                \
      T * o, T * q, T * kv_data, int32_t * kv_indptr, int32_t * kv_indicies,   \
      int32_t * last_page_offset, int head_dim, int num_layers, int layer_idx, \
      int num_qo_heads, int num_kv_heads, int page_size, int batch_size);

INST_FlashInferBatchDecodeKernel(nv_half);
INST_FlashInferBatchDecodeKernel(nv_bfloat16);

#define INST_FlashInferInitKvKernel(head_dim, T)                               \
  template void FlashInferInitKvKernel<head_dim, T>(                           \
      T * kv_data, int32_t * kv_indptr, int32_t * kv_indicies,                 \
      int32_t * last_page_offset, T * key, T * value, int32_t * seqlen_indptr, \
      int num_layers, int layer_idx, int num_kv_heads, int page_size,          \
      int batch_size);

FOR_FlashInferBatchDecode_D(INST_FlashInferInitKvKernel, nv_half);
FOR_FlashInferBatchDecode_D(INST_FlashInferInitKvKernel, nv_bfloat16);

#define INST_FlashInferAppendKvKernel(head_dim, T)                    \
  template void FlashInferAppendKvKernel<head_dim, T>(                \
      T * kv_data, int32_t * kv_indptr, int32_t * kv_indicies,        \
      int32_t * last_page_offset, T * key, T * value, int num_layers, \
      int layer_idx, int num_kv_heads, int page_size, int batch_size);
FOR_FlashInferBatchDecode_D(INST_FlashInferAppendKvKernel, nv_half);
FOR_FlashInferBatchDecode_D(INST_FlashInferAppendKvKernel, nv_bfloat16);
