#pragma once
#include "flashinfer/page.cuh"
#include "flashinfer/rope.cuh"

namespace flashinfer {
template <uint32_t PAGE_SIZE, uint32_t GROUP_SIZE, uint32_t HEAD_DIM,
          PageStorage page_storage, RotaryMode ROTARY_MODE,
          bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL, typename DTypeIn,
          typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(
    DTypeIn* q, paged_kv_t<page_storage, DTypeIn, IdType> paged_kv,
    IdType* qo_indptr, DTypeOut* o, float* tmp, uint32_t num_qo_heads,
    float rope_scale, float rope_theta, cudaStream_t stream);
}

#define INST_BatchPrefill(T, PAGE_SIZE, GROUP_SIZE, HEAD_DIM)        \
  namespace flashinfer {                                             \
  template cudaError_t BatchPrefillWithPagedKVCacheDispatched<       \
      PAGE_SIZE, GROUP_SIZE, HEAD_DIM, PageStorage::kPointer,        \
      RotaryMode::kLlama, /* ALLOW_FP16_QK_REDUCTION= */ false,      \
      /* CAUSAL= */ true, T, T, int32_t>(                            \
      T * q, paged_kv_t<PageStorage::kPointer, T, int32_t> paged_kv, \
      int32_t* qo_indptr, T* o, float* tmp, uint32_t num_qo_heads,   \
      float rope_scale, float rope_theta, cudaStream_t stream);      \
  }

namespace flashinfer {
template <uint32_t PAGE_SIZE, uint32_t GROUP_SIZE, uint32_t HEAD_DIM,
          PageStorage page_storage, RotaryMode ROTARY_MODE, typename DTypeIn,
          typename DTypeOut, typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(
    DTypeIn* q, paged_kv_t<page_storage, DTypeIn, IdType> paged_kv, DTypeOut* o,
    float* tmp, float rope_scale, float rope_theta, cudaStream_t stream);
}
#define INST_BatchDecode(T, PAGE_SIZE, GROUP_SIZE, HEAD_DIM)                \
  namespace flashinfer {                                                    \
  template cudaError_t BatchDecodeWithPagedKVCacheDispatched<               \
      PAGE_SIZE, GROUP_SIZE, HEAD_DIM, PageStorage::kPointer,               \
      RotaryMode::kLlama, T, T, int32_t>(                                   \
      T * q, paged_kv_t<PageStorage::kPointer, T, int32_t> paged_kv, T* o,  \
      float* tmp, float rope_scale, float rope_theta, cudaStream_t stream); \
  }
