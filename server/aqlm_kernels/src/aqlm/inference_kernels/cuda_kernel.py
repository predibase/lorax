import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load

CUDA_FOLDER = os.path.dirname(os.path.abspath(__file__))
CUDA_KERNEL = load(
    name="codebook_cuda",
    sources=[os.path.join(CUDA_FOLDER, "cuda_kernel.cpp"), os.path.join(CUDA_FOLDER, "cuda_kernel.cu")],
)
