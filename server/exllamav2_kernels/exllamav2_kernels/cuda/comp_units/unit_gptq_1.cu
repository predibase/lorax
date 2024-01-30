#include "kernel_select.cuh"

fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel_1(const int m_count, bool r_weights, bool mul_r_weights)
{
    if (!r_weights && !mul_r_weights) return map_m_count_gptq<false, false>::pick_gemm_half_q_half_gptq_kernel(m_count);
    // if (!r_weights &&  mul_r_weights) return map_m_count_gptq<false,  true>::pick_gemm_half_q_half_gptq_kernel(m_count);
    // if ( r_weights && !mul_r_weights) return map_m_count_gptq< true, false>::pick_gemm_half_q_half_gptq_kernel(m_count);
    // if ( r_weights &&  mul_r_weights) return map_m_count_gptq< true,  true>::pick_gemm_half_q_half_gptq_kernel(m_count);
    return NULL;
}
