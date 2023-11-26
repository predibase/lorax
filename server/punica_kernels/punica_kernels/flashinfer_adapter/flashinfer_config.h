#pragma once

template <typename T>
void FlashInferBatchDecodeKernel(T* o, T* q, T* kv_data, int32_t* kv_indptr,
                                 int32_t* kv_indicies,
                                 int32_t* last_page_offset, int head_dim,
                                 int num_layers, int layer_idx,
                                 int num_qo_heads, int num_kv_heads,
                                 int page_size, int batch_size);

template <int head_dim, typename T>
void FlashInferInitKvKernel(T* kv_data, int32_t* kv_indptr,
                            int32_t* kv_indicies, int32_t* last_page_offset,
                            T* key, T* value, int32_t* seqlen_indptr,
                            int num_layers, int layer_idx, int num_kv_heads,
                            int page_size, int batch_size);

template <int head_dim, typename T>
void FlashInferAppendKvKernel(T* kv_data, int32_t* kv_indptr,
                              int32_t* kv_indicies, int32_t* last_page_offset,
                              T* key, T* value, int num_layers, int layer_idx,
                              int num_kv_heads, int page_size, int batch_size);

// clang-format off

#define FOR_FlashInferBatchDecode_D(f, ...) \
    f(64, __VA_ARGS__) \
    f(128, __VA_ARGS__)

// clang-format on
