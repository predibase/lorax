#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "flashinfer/page.cuh"
#include "flashinfer_config.h"
#include "flashinfer_decl.h"
#include "generated/dispatch.inc"

using flashinfer::paged_kv_t;
using flashinfer::PageStorage;
using flashinfer::RotaryMode;

#define _DISPATCH_SWITCH(cond, ...) \
  [&]() -> bool {                   \
    switch (cond) {                 \
      __VA_ARGS__                   \
      default:                      \
        return false;               \
    }                               \
  }()

#define _DISPATCH_CASE(case_expr, var, ...) \
  case case_expr: {                         \
    constexpr auto var = case_expr;         \
    return __VA_ARGS__();                   \
  }

#define DISPATCH_group_size(expr, ...) \
  _DISPATCH_SWITCH(expr, _DISPATCH_CASES_group_size(__VA_ARGS__))

#define DISPATCH_page_size(expr, ...) \
  _DISPATCH_SWITCH(expr, _DISPATCH_CASES_page_size(__VA_ARGS__))

#define DISPATCH_head_dim(expr, ...) \
  _DISPATCH_SWITCH(expr, _DISPATCH_CASES_head_dim(__VA_ARGS__))

namespace {
template <typename T>
inline T* alloc_from_buf(void** buf, int n) {
  auto* p = (T*)*buf;
  *buf = (void*)(p + n);
  return p;
}
}  // namespace

template <typename T>
bool FlashInferBatchPrefillKernel(T* o, T* q, int32_t* qo_indptr, T** kv_ptrs,
                                  int32_t* kv_indptr, int32_t* last_page_offset,
                                  void* tmpbuf, int head_dim, int num_layers,
                                  int layer_idx, int group_size,
                                  int num_kv_heads, int page_size,
                                  int batch_size) {
  return DISPATCH_page_size(page_size, [&] {
    return DISPATCH_group_size(group_size, [&] {
      return DISPATCH_head_dim(head_dim, [&] {
        auto kv_aux = alloc_from_buf<int32_t>(&tmpbuf, 4 * (batch_size + 1));
        paged_kv_t<PageStorage::kPointer, T, int32_t> paged_kv(
            num_layers, layer_idx, num_kv_heads, page_size, head_dim,
            batch_size, kv_ptrs, kv_indptr, last_page_offset, kv_aux);
        int num_qo_heads = num_kv_heads * group_size;
        constexpr bool allow_fp16_qk_reduction = false;
        constexpr bool causal = true;
        constexpr auto rotary = RotaryMode::kLlama;
        float rope_scale = 1.f;
        float rope_theta = 1e4;
        cudaStream_t stream = nullptr;
        auto status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<
            PAGE_SIZE, GROUP_SIZE, HEAD_DIM, PageStorage::kPointer, rotary,
            allow_fp16_qk_reduction, causal>(q, paged_kv, qo_indptr, o,
                                             (float*)tmpbuf, num_qo_heads,
                                             rope_scale, rope_theta, stream);
        if (status != cudaSuccess) {
          fprintf(stderr, "batch_prefill failed: %s\n",
                  cudaGetErrorString(status));
        }
        return true;
      });
    });
  });
}

template <typename T>
bool FlashInferBatchDecodeKernel(T* o, T* q, T** kv_ptrs, int32_t* kv_indptr,
                                 int32_t* last_page_offset, void* tmpbuf,
                                 int head_dim, int num_layers, int layer_idx,
                                 int group_size, int num_kv_heads,
                                 int page_size, int batch_size) {
  return DISPATCH_page_size(page_size, [&] {
    return DISPATCH_group_size(group_size, [&] {
      return DISPATCH_head_dim(head_dim, [&] {
        auto kv_aux = alloc_from_buf<int32_t>(&tmpbuf, 4 * (batch_size + 1));
        paged_kv_t<PageStorage::kPointer, T, int32_t> paged_kv(
            num_layers, layer_idx, num_kv_heads, page_size, head_dim,
            batch_size, kv_ptrs, kv_indptr, last_page_offset, kv_aux);
        constexpr auto rotary = RotaryMode::kLlama;
        float rope_scale = 1.f;
        float rope_theta = 1e4;
        cudaStream_t stream = nullptr;
        auto status = flashinfer::BatchDecodeWithPagedKVCacheDispatched<
            PAGE_SIZE, GROUP_SIZE, HEAD_DIM, PageStorage::kPointer, rotary>(
            q, paged_kv, o, nullptr, rope_scale, rope_theta, stream);
        if (status != cudaSuccess) {
          fprintf(stderr, "batch_decode failed: %s\n",
                  cudaGetErrorString(status));
        }
        return true;
      });
    });
  });
}

template <int head_dim, typename T>
void FlashInferInitKvKernel(T** kv_ptrs, int32_t* kv_indptr,
                            int32_t* last_page_offset, T* key, T* value,
                            int32_t* seqlen_indptr, int num_layers,
                            int layer_idx, int num_kv_heads, int page_size,
                            int batch_size) {
  paged_kv_t<PageStorage::kPointer, T, int32_t> paged_kv(
      num_layers, layer_idx, num_kv_heads, page_size, head_dim, batch_size,
      kv_ptrs, kv_indptr, last_page_offset);

  constexpr size_t vec_size =
      std::max(16 / sizeof(T), static_cast<size_t>(head_dim / 32));
  constexpr size_t bdx = head_dim / vec_size;
  constexpr size_t bdy = 1;
  dim3 nblks(paged_kv.batch_size * paged_kv.num_heads / bdy);
  dim3 nthrs(bdx, bdy);
  flashinfer::AppendPagedKVCachePrefillKernel<head_dim, vec_size, bdx, bdy,
                                              PageStorage::kPointer, T, int32_t>
      <<<nblks, nthrs>>>(paged_kv, key, value, seqlen_indptr);
}

template <int head_dim, typename T>
void FlashInferAppendKvKernel(T** kv_ptrs, int32_t* kv_indptr,
                              int32_t* last_page_offset, T* key, T* value,
                              int num_layers, int layer_idx, int num_kv_heads,
                              int page_size, int batch_size) {
  paged_kv_t<PageStorage::kPointer, T, int32_t> paged_kv(
      num_layers, layer_idx, num_kv_heads, page_size, head_dim, batch_size,
      kv_ptrs, kv_indptr, last_page_offset);

  constexpr size_t vec_size =
      std::max(16 / sizeof(T), static_cast<size_t>(head_dim / 32));
  constexpr size_t bdx = head_dim / vec_size;
  constexpr size_t bdy = 1;
  dim3 nblks(paged_kv.batch_size * paged_kv.num_heads / bdy);
  dim3 nthrs(bdx, bdy);
  flashinfer::AppendPagedKVCacheDecodeKernel<head_dim, vec_size, bdx, bdy,
                                             PageStorage::kPointer, T, int32_t>
      <<<nblks, nthrs>>>(paged_kv, key, value);
}

#define INST_FlashInferBatchPrefillKernel(T)                                  \
  template bool FlashInferBatchPrefillKernel<T>(                              \
      T * o, T * q, int32_t * qo_indptr, T * *kv_ptrs, int32_t * kv_indptr,   \
      int32_t * last_page_offset, void* tmpbuf, int head_dim, int num_layers, \
      int layer_idx, int group_size, int num_kv_heads, int page_size,         \
      int batch_size);
INST_FlashInferBatchPrefillKernel(nv_half);
INST_FlashInferBatchPrefillKernel(nv_bfloat16);

#define INST_FlashInferBatchDecodeKernel(T)                                   \
  template bool FlashInferBatchDecodeKernel<T>(                               \
      T * o, T * q, T * *kv_ptrs, int32_t * kv_indptr,                        \
      int32_t * last_page_offset, void* tmpbuf, int head_dim, int num_layers, \
      int layer_idx, int group_size, int num_kv_heads, int page_size,         \
      int batch_size);
INST_FlashInferBatchDecodeKernel(nv_half);
INST_FlashInferBatchDecodeKernel(nv_bfloat16);

#define INST_FlashInferInitKvKernel(head_dim, T)                              \
  template void FlashInferInitKvKernel<head_dim, T>(                          \
      T * *kv_ptrs, int32_t * kv_indptr, int32_t * last_page_offset, T * key, \
      T * value, int32_t * seqlen_indptr, int num_layers, int layer_idx,      \
      int num_kv_heads, int page_size, int batch_size);
FOR_FlashInferBatchDecode_D(INST_FlashInferInitKvKernel, nv_half);
FOR_FlashInferBatchDecode_D(INST_FlashInferInitKvKernel, nv_bfloat16);

#define INST_FlashInferAppendKvKernel(head_dim, T)                            \
  template void FlashInferAppendKvKernel<head_dim, T>(                        \
      T * *kv_ptrs, int32_t * kv_indptr, int32_t * last_page_offset, T * key, \
      T * value, int num_layers, int layer_idx, int num_kv_heads,             \
      int page_size, int batch_size);
FOR_FlashInferBatchDecode_D(INST_FlashInferAppendKvKernel, nv_half);
FOR_FlashInferBatchDecode_D(INST_FlashInferAppendKvKernel, nv_bfloat16);
