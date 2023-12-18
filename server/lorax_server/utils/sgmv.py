from functools import lru_cache
import os
import warnings
from typing import Tuple

import torch

try:
    import punica_kernels as _kernels
    HAS_SGMV = not bool(os.environ.get("DISABLE_SGMV", ""))
except ImportError:
    warnings.warn("Could not import SGMV kernel from Punica, falling back to loop.")
    _kernels = None
    HAS_SGMV = False


MIN_SGMV_RANK = 8
MIN_RANK_CUSTOM = 16
MAX_RANK_CUSTOM = 128


def has_sgmv() -> bool:
    return HAS_SGMV


def orient_for_rank(t: torch.Tensor, rank: int) -> torch.Tensor:
    if MIN_RANK_CUSTOM <= rank <= MAX_RANK_CUSTOM:
        return t.transpose(0, 1)
    return t


# Source: https://github.com/punica-ai/punica/blob/master/src/punica/ops/__init__.py
def add_lora_sgmv_cutlass(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s_start: torch.Tensor,
    s_end: torch.Tensor,
    layer_idx: int,
    lora_rank: int,
):
    """
    Semantics:
        y[s[i]:s[i+1]] += x[s[i]:s[i+1]] @ deref(wa_ptr[i]).T @ deref(wb_ptr[i])

    Args:
        y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
        x: Shape: `[B, H1]`. Input vectors.
        wa_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
            Weight matrix shape: `[num_layers, R, H1]`.
        wb_ptr: Shape: `[S]`. DType: torch.int64. Pointer to the weight matrices.\
            Weight matrix shape: `[num_layers, R, H2]`.
        s_start: Shape: `[S]`, DType: torch.int32. Indptr of the weight matrices start indices.
        s_end: Shape: `[S]`, DType: torch.int32. Indptr of the weight matrices end indices.
        layer_idx: Layer index of the weight matrices.
    """
    if lora_rank < MIN_RANK_CUSTOM or lora_rank > MAX_RANK_CUSTOM:
        # Custom SGMV shrink only supports rank 16, 32, 64, 128
        _add_lora_sgmv_cutlass_legacy(y, x, wa_ptr, wb_ptr, s_start, s_end, layer_idx, lora_rank)
        return
    
    tmp1 = torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=x.device)
    tmp2_size = _kernels.sgmv_cutlass_tmp_size(wa_ptr.size(0))
    tmp2 = torch.empty((tmp2_size,), dtype=torch.uint8, device=x.device)
    v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
    _kernels.sgmv_shrink(v, x, wa_ptr, s_start, s_end, tmp1, layer_idx)
    _kernels.sgmv_cutlass(y, v, wb_ptr, s_start, s_end, tmp2, layer_idx)


def _add_lora_sgmv_cutlass_legacy(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    wb_ptr: torch.Tensor,
    s_start: torch.IntTensor,
    s_end: torch.IntTensor,
    layer_idx: int,
    lora_rank: int,
):
    tmp_size = _kernels.sgmv_cutlass_tmp_size(wa_ptr.size(0))
    tmp = torch.empty((tmp_size,), dtype=torch.uint8, device=x.device)
    v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
    _kernels.sgmv_cutlass(v, x, wa_ptr, s_start, s_end, tmp, layer_idx)
    _kernels.sgmv_cutlass(y, v, wb_ptr, s_start, s_end, tmp, layer_idx)


@lru_cache(maxsize=1)
def get_tmp_tensor(device: torch.device) -> torch.Tensor:
    return torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=device)


@lru_cache(maxsize=1)
def get_tmp_tensor_for_size(size: int, device: torch.device) -> torch.Tensor:
    tmp_size = _kernels.sgmv_cutlass_tmp_size(size)
    return torch.empty((tmp_size,), dtype=torch.uint8, device=device)


def lora_a_sgmv_cutlass(
    x: torch.Tensor,
    wa_ptr: torch.Tensor,
    s_start: torch.IntTensor,
    s_end: torch.IntTensor,
    layer_idx: int,
    lora_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
    if MIN_RANK_CUSTOM <= lora_rank <= MAX_RANK_CUSTOM:
        tmp1 = get_tmp_tensor(x.device)
        tmp = get_tmp_tensor_for_size(wa_ptr.size(0), x.device)
        _kernels.sgmv_shrink(v, x, wa_ptr, s_start, s_end, tmp1, layer_idx)
    else:
        tmp = get_tmp_tensor_for_size(wa_ptr.size(0), x.device)
        _kernels.sgmv_cutlass(v, x, wa_ptr, s_start, s_end, tmp, layer_idx)
    return v, tmp


def lora_b_sgmv_cutlass(
    y: torch.Tensor,
    v: torch.Tensor,
    tmp: torch.Tensor,
    wb_ptr: torch.Tensor,
    s_start: torch.IntTensor,
    s_end: torch.IntTensor,
    layer_idx: int,
):
    _kernels.sgmv_cutlass(y, v, wb_ptr, s_start, s_end, tmp, layer_idx)
