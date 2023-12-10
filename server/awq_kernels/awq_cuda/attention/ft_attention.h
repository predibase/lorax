#pragma once
#include <torch/extension.h>


torch::Tensor single_query_attention(const torch::Tensor q,
                                     const torch::Tensor k,
                                     const torch::Tensor v,
                                     torch::Tensor k_cache,
                                     torch::Tensor v_cache,
                                     c10::optional<const torch::Tensor> length_per_sample_,
                                     c10::optional<const torch::Tensor> alibi_slopes_,
                                     const int timestep,
                                     const int rotary_embedding_dim = 0,
                                     const float rotary_base = 10000.0f,
                                     const bool neox_rotary_style=true);