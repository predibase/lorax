import pathlib

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
            ],
            include_dirs=[str(root.resolve() / "third_party/cutlass/include")],
        )
    ],
    cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
)
