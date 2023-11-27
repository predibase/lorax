#pragma once
#include <cstdint>

template <typename T>
bool FlashInferBatchPrefillKernel(T* o, T* q, int32_t* qo_indptr, T** kv_ptrs,
                                  int32_t* kv_indptr, int32_t* last_page_offset,
                                  void* tmpbuf, int head_dim, int num_layers,
                                  int layer_idx, int group_size,
                                  int num_kv_heads, int page_size,
                                  int batch_size);

template <typename T>
bool FlashInferBatchDecodeKernel(T* o, T* q, T** kv_ptrs, int32_t* kv_indptr,
                                 int32_t* last_page_offset, void* tmpbuf,
                                 int head_dim, int num_layers, int layer_idx,
                                 int group_size, int num_kv_heads,
                                 int page_size, int batch_size);

template <int head_dim, typename T>
void FlashInferInitKvKernel(T** kv_ptrs, int32_t* kv_indptr,
                            int32_t* last_page_offset, T* key, T* value,
                            int32_t* seqlen_indptr, int num_layers,
                            int layer_idx, int num_kv_heads, int page_size,
                            int batch_size);

template <int head_dim, typename T>
void FlashInferAppendKvKernel(T** kv_ptrs, int32_t* kv_indptr,
                              int32_t* last_page_offset, T* key, T* value,
                              int num_layers, int layer_idx, int num_kv_heads,
                              int page_size, int batch_size);

// clang-format off

#define FOR_FlashInferBatchDecode_D(f, ...) \
    f(64, __VA_ARGS__) \
    f(128, __VA_ARGS__)

// clang-format on
