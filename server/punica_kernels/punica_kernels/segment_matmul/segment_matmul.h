#pragma once

#include <ATen/ATen.h>
#include <torch/script.h>

// Performs matrix multiplication according to segments.
// TODO (matthias) Support `out` argument.
at::Tensor segment_matmul(const at::Tensor &input,
                          const at::Tensor &ptr,
                          const at::Tensor &other);
