#include "matmul.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

// Performs matrix multiplication according to segments.
at::Tensor segment_matmul(const at::Tensor &input,
                          const at::Tensor &ptr,
                          const at::Tensor &other)
{
    at::TensorArg input_arg{input, "input", 0};
    at::TensorArg ptr_arg{ptr, "ptr", 1};
    at::TensorArg other_arg{other, "other", 2};
    at::CheckedFrom c{"segment_matmul"};

    at::checkAllDefined(c, {input_arg, ptr_arg, other_arg});
    at::checkSameType(c, input_arg, other_arg);
    at::checkDim(c, input_arg, 2);
    at::checkDim(c, ptr_arg, 1);
    at::checkDim(c, other_arg, 3);
    at::checkSize(c, other_arg, 1, input_arg->size(-1));
    at::checkNumel(c, ptr_arg, other_arg->size(0) + 1);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("segment_matmul", "")
                         .typed<decltype(segment_matmul)>();
    return op.call(input, ptr, other);
}

TORCH_LIBRARY_FRAGMENT(pyg, m)
{
    m.def(TORCH_SELECTIVE_SCHEMA(
        "segment_matmul(Tensor input, Tensor ptr, Tensor other) -> Tensor"));
}
