import itertools
import os
import pathlib
from typing import List

import setuptools
import torch.utils.cpp_extension as torch_cpp_ext

root = pathlib.Path(__name__).parent


def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


remove_unwanted_pytorch_nvcc_flags()


def generate_flashinfer_cu() -> List[str]:
    page_sizes = os.environ.get("PUNICA_PAGE_SIZES", "16").split(",")
    group_sizes = os.environ.get("PUNICA_GROUP_SIZES", "1,2,4,8").split(",")
    head_dims = os.environ.get("PUNICA_HEAD_DIMS", "128").split(",")
    page_sizes = [int(x) for x in page_sizes]
    group_sizes = [int(x) for x in group_sizes]
    head_dims = [int(x) for x in head_dims]
    dtypes = {"fp16": "nv_half", "bf16": "nv_bfloat16"}
    funcs = ["prefill", "decode"]
    prefix = "punica_kernels/flashinfer_adapter/generated"
    (root / prefix).mkdir(parents=True, exist_ok=True)
    files = []

    # dispatch.inc
    path = root / prefix / "dispatch.inc"
    if not path.exists():
        with open(root / prefix / "dispatch.inc", "w") as f:
            f.write("#define _DISPATCH_CASES_page_size(...)       \\\n")
            for x in page_sizes:
                f.write(f"  _DISPATCH_CASE({x}, PAGE_SIZE, __VA_ARGS__) \\\n")
            f.write("// EOL\n")

            f.write("#define _DISPATCH_CASES_group_size(...)      \\\n")
            for x in group_sizes:
                f.write(f"  _DISPATCH_CASE({x}, GROUP_SIZE, __VA_ARGS__) \\\n")
            f.write("// EOL\n")

            f.write("#define _DISPATCH_CASES_head_dim(...)        \\\n")
            for x in head_dims:
                f.write(f"  _DISPATCH_CASE({x}, HEAD_DIM, __VA_ARGS__) \\\n")
            f.write("// EOL\n")

            f.write("\n")

    # impl
    for func, page_size, group_size, head_dim, dtype in itertools.product(
        funcs, page_sizes, group_sizes, head_dims, dtypes
    ):
        fname = f"batch_{func}_p{page_size}_g{group_size}_h{head_dim}_{dtype}.cu"
        files.append(prefix + "/" + fname)
        if (root / prefix / fname).exists():
            continue
        with open(root / prefix / fname, "w") as f:
            f.write('#include "../flashinfer_decl.h"\n\n')
            f.write(f'#include "flashinfer/{func}.cuh"\n\n')
            f.write(
                f"INST_Batch{func.capitalize()}({dtypes[dtype]}, {page_size}, {group_size}, {head_dim})\n"
            )

    return files


setuptools.setup(
    name="punica_kernels",
    ext_modules=[
        torch_cpp_ext.CUDAExtension(
            name="punica_kernels",
            sources=[
                "punica_kernels/punica_ops.cc",
                "punica_kernels/bgmv/bgmv_all.cu",
                "punica_kernels/flashinfer_adapter/flashinfer_all.cu",
                "punica_kernels/rms_norm/rms_norm_cutlass.cu",
                "punica_kernels/sgmv/sgmv_cutlass.cu",
                "punica_kernels/sgmv_flashinfer/sgmv_all.cu",
            ] + generate_flashinfer_cu(),
            include_dirs=[
                str(root.resolve() / "third_party/cutlass/include"),
                str(root.resolve() / "third_party/flashinfer/include"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
)
