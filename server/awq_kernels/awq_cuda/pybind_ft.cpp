#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "attention/ft_attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("single_query_attention", &single_query_attention, "Attention with a single query",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("k_cache"), py::arg("v_cache"),
          py::arg("length_per_sample_"), py::arg("alibi_slopes_"), py::arg("timestep"), py::arg("rotary_embedding_dim")=0,
          py::arg("rotary_base")=10000.0f, py::arg("neox_rotary_style")=true);
}