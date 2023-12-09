#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "layernorm/layernorm.h"
#include "quantization/gemm_cuda.h"
#include "quantization/gemv_cuda.h"
#include "position_embedding/pos_encoding.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("layernorm_forward_cuda", &layernorm_forward_cuda, "FasterTransformer layernorm kernel");
    m.def("gemm_forward_cuda", &gemm_forward_cuda, "Quantized GEMM kernel.");
    m.def("gemmv2_forward_cuda", &gemmv2_forward_cuda, "Quantized v2 GEMM kernel.");
    m.def("gemv_forward_cuda", &gemv_forward_cuda, "Quantized GEMV kernel.");
    m.def("rotary_embedding_neox", &rotary_embedding_neox, "Apply GPT-NeoX style rotary embedding to query and key");
}