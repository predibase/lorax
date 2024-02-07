import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

common_setup_kwargs = {
}


def get_generator_flag():
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag = ["-DOLD_GENERATOR_PATH"]
    
    return generator_flag


def get_compute_capabilities():
    # Collect the compute capabilities of all available GPUs.
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cc = major * 10 + minor

        if cc < 75:
            raise RuntimeError("GPUs with compute capability less than 7.5 are not supported.")

    # figure out compute capability
    compute_capabilities = {75, 80, 86, 89, 90}

    capability_flags = []
    for cap in compute_capabilities:
        capability_flags += ["-gencode", f"arch=compute_{cap},code=sm_{cap}"]

    return capability_flags

generator_flags = get_generator_flag()
arch_flags = get_compute_capabilities()


extra_compile_args={
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
    "nvcc": [
        "-O3", 
        "-std=c++17",
        "-DENABLE_BF16",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ] + arch_flags + generator_flags
}

extensions = [
    CUDAExtension(
        "awq_inference_engine",
        [
            "awq_cuda/pybind_awq.cpp",
            "awq_cuda/quantization/gemm_cuda_gen.cu",
            "awq_cuda/layernorm/layernorm.cu",
            "awq_cuda/position_embedding/pos_encoding_kernels.cu",
            "awq_cuda/quantization/gemv_cuda.cu"
        ], extra_compile_args=extra_compile_args
    )
]

extensions.append(
    CUDAExtension(
        "ft_inference_engine",
        [
            "awq_cuda/pybind_ft.cpp",
            "awq_cuda/attention/ft_attention.cpp",
            "awq_cuda/attention/decoder_masked_multihead_attention.cu"
        ], extra_compile_args=extra_compile_args
    )
)

additional_setup_kwargs = {
    "ext_modules": extensions,
    "cmdclass": {'build_ext': BuildExtension}
}

common_setup_kwargs.update(additional_setup_kwargs)

setup(
    **common_setup_kwargs
)