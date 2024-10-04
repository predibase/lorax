#pragma once

template <int feat_in, int feat_out, typename T>
void bgmv_kernel(T *__restrict__ Y, const T *__restrict__ X,
                 T **__restrict__ W,
                 const int64_t *__restrict__ indicies, int64_t y_offset,
                 int64_t full_y_size, int64_t batch_size,
                 int64_t layer_idx, float scale);

// clang-format off

#define FOR_BGMV_WIDE(f, T, narrow) \
    f(T, narrow, 256) \
    f(T, narrow, 512) \
    f(T, narrow, 640) \
    f(T, narrow, 768) \
    f(T, narrow, 1024) \
    f(T, narrow, 1152) \
    f(T, narrow, 1280) \
    f(T, narrow, 1536) \
    f(T, narrow, 1728) \
    f(T, narrow, 1792) \
    f(T, narrow, 2048) \
    f(T, narrow, 2304) \
    f(T, narrow, 2560) \
    f(T, narrow, 2752) \
    f(T, narrow, 2816) \
    f(T, narrow, 3072) \
    f(T, narrow, 3456) \
    f(T, narrow, 3584) \
    f(T, narrow, 4096) \
    f(T, narrow, 4480) \
    f(T, narrow, 4608) \
    f(T, narrow, 5120) \
    f(T, narrow, 5504) \
    f(T, narrow, 5632) \
    f(T, narrow, 6144) \
    f(T, narrow, 6848) \
    f(T, narrow, 6912) \
    f(T, narrow, 7168) \
    f(T, narrow, 8192) \
    f(T, narrow, 8960) \
    f(T, narrow, 9216) \
    f(T, narrow, 9472) \
    f(T, narrow, 10240) \
    f(T, narrow, 11008) \
    f(T, narrow, 12288) \
    f(T, narrow, 13696) \
    f(T, narrow, 13824) \
    f(T, narrow, 14336) \
    f(T, narrow, 15360) \
    f(T, narrow, 16384) \
    f(T, narrow, 18944) \
    f(T, narrow, 20480) \
    f(T, narrow, 22016) \
    f(T, narrow, 24576) \
    f(T, narrow, 27392) \
    f(T, narrow, 28672) \
    f(T, narrow, 32000) \
    f(T, narrow, 32256) \
    f(T, narrow, 32512) \
    f(T, narrow, 32768) \
    f(T, narrow, 33024) \
    f(T, narrow, 36864) \
    f(T, narrow, 43264) \
    f(T, narrow, 49152) \
    f(T, narrow, 64000) \
    f(T, narrow, 64256) \
    f(T, narrow, 64512) \
    f(T, narrow, 102400) \
    f(T, narrow, 102656) \
    f(T, narrow, 102912) \
    f(T, narrow, 128000) \
    f(T, narrow, 128256) \
    f(T, narrow, 128512) \

#define FOR_BGMV_WIDE_NARROW(f, T) \
    FOR_BGMV_WIDE(f, T, 8) \
    FOR_BGMV_WIDE(f, T, 16) \
    FOR_BGMV_WIDE(f, T, 32) \
    FOR_BGMV_WIDE(f, T, 64) \
    FOR_BGMV_WIDE(f, T, 128)

// clang-format on
