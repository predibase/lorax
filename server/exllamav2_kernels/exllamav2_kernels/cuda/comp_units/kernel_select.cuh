#ifndef _kernel_select_cuh
#define _kernel_select_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#include "../q_gemm_kernel.cuh"
#include "../q_gemm_kernel_gptq.cuh"

fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel(const int m_count, bool r_weights, bool mul_r_weights);
fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel_1(const int m_count, bool r_weights, bool mul_r_weights);
fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel_2(const int m_count, bool r_weights, bool mul_r_weights);
fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel_3(const int m_count, bool r_weights, bool mul_r_weights);

fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel(const int m_count, bool r_weights, bool mul_r_weights);
fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel_1a(const int m_count, bool r_weights, bool mul_r_weights);
fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel_1b(const int m_count, bool r_weights, bool mul_r_weights);
fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel_2a(const int m_count, bool r_weights, bool mul_r_weights);
fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel_2b(const int m_count, bool r_weights, bool mul_r_weights);
fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel_3a(const int m_count, bool r_weights, bool mul_r_weights);
fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel_3b(const int m_count, bool r_weights, bool mul_r_weights);

template <bool use_r_weights, bool mul_r_weights>
struct map_m_count_gptq {
    static constexpr fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel(int m_count)
    {
        if (m_count == GPTQ_BLOCK_M_SIZE_MAX) return gemm_half_q_half_gptq_kernel<GPTQ_BLOCK_M_SIZE_MAX, use_r_weights, mul_r_weights>;
        printf(" ## No GPTQ kernel found for block size %i\n", m_count);
        return NULL;
    }
};

template <bool use_r_weights, bool mul_r_weights>
struct map_m_count_exl2_a {
    static constexpr fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel(const int perm)
    {
        switch (perm)
        {
            case 0b00000010: return gemm_half_q_half_kernel<0b00000010, use_r_weights, mul_r_weights>;
            case 0b00000110: return gemm_half_q_half_kernel<0b00000110, use_r_weights, mul_r_weights>;
            case 0b00000100: return gemm_half_q_half_kernel<0b00000100, use_r_weights, mul_r_weights>;
            case 0b00001110: return gemm_half_q_half_kernel<0b00001110, use_r_weights, mul_r_weights>;
            case 0b00001100: return gemm_half_q_half_kernel<0b00001100, use_r_weights, mul_r_weights>;
            case 0b00001000: return gemm_half_q_half_kernel<0b00001000, use_r_weights, mul_r_weights>;
            case 0b00011000: return gemm_half_q_half_kernel<0b00011000, use_r_weights, mul_r_weights>;
            case 0b00010000: return gemm_half_q_half_kernel<0b00010000, use_r_weights, mul_r_weights>;
            case 0b00110000: return gemm_half_q_half_kernel<0b00110000, use_r_weights, mul_r_weights>;
            case 0b00100000: return gemm_half_q_half_kernel<0b00100000, use_r_weights, mul_r_weights>;
            default:
                return NULL;
        }
    }
};

template <bool use_r_weights, bool mul_r_weights>
struct map_m_count_exl2_b {
    static constexpr fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel(const int perm)
    {
        switch (perm)
        {
            case 0b10000000: return gemm_half_q_half_kernel<0b10000000, use_r_weights, mul_r_weights>;
            case 0b00100110: return gemm_half_q_half_kernel<0b00100110, use_r_weights, mul_r_weights>;
            case 0b00010100: return gemm_half_q_half_kernel<0b00010100, use_r_weights, mul_r_weights>;
            case 0b10001100: return gemm_half_q_half_kernel<0b10001100, use_r_weights, mul_r_weights>;
            case 0b10011000: return gemm_half_q_half_kernel<0b10011000, use_r_weights, mul_r_weights>;
            case 0b10110000: return gemm_half_q_half_kernel<0b10110000, use_r_weights, mul_r_weights>;
            case 0b10101100: return gemm_half_q_half_kernel<0b10101100, use_r_weights, mul_r_weights>;
            case 0b00001010: return gemm_half_q_half_kernel<0b00001010, use_r_weights, mul_r_weights>;
            case 0b00101000: return gemm_half_q_half_kernel<0b00101000, use_r_weights, mul_r_weights>;
            case 0b10100000: return gemm_half_q_half_kernel<0b10100000, use_r_weights, mul_r_weights>;
            default:         return gemm_half_q_half_kernel<0b10111110, use_r_weights, mul_r_weights>;
                // printf(" ## No kernel found for permutation %x\n", perm);
                // return NULL;
        }
    }
};

#endif