#pragma once
#include <cstdint>

template <typename T, uint32_t d_out>
bool sgmv_shrink(T* y, T* x, T** w, int32_t* s, void* tmp,
                 uint32_t num_problems, uint32_t d_in, uint32_t layer_idx);

// clang-format off

#define FOR_SGMV_NARROW(f, T) \
    f(T, 16) \
    f(T, 32) \
    f(T, 64)

// clang-format on
