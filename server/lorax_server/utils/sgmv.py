from functools import lru_cache
import os
import warnings
from typing import Tuple

import torch
import torch.nn.functional as F

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
SGMV_BLOCK_SIZE = 16


def has_sgmv() -> bool:
    return HAS_SGMV


def use_cutlass_shrink(lora_rank: int) -> bool:
    return lora_rank < MIN_RANK_CUSTOM


def orient_for_rank(t: torch.Tensor, rank: int) -> torch.Tensor:
    if MIN_RANK_CUSTOM <= rank <= MAX_RANK_CUSTOM:
        return t.transpose(0, 1)
    return t


def pad_sgmv(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Pad a tensor to the minimum rank for SGMV and the nearest multiple of the SGMV block size."""
    if not has_sgmv():
        return t
    
    current_rank = t.size(dim)
    nearest_rank = (current_rank + SGMV_BLOCK_SIZE - 1) // SGMV_BLOCK_SIZE * SGMV_BLOCK_SIZE
    if current_rank == nearest_rank:
        return t

    pad_size = nearest_rank - current_rank
    
    # see complicatd pad syntax here: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    pad = [0, 0] * t.dim()
    pad[(t.dim() - dim - 1) * 2 + 1] = pad_size
    pad = tuple(pad)
    
    return F.pad(t, pad, mode="constant", value=0.0)



@lru_cache(maxsize=1)
def get_tmp_tensor(device: torch.device) -> torch.Tensor:
    return torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=device)


@lru_cache(maxsize=32)
def get_tmp_tensor_for_size(size: int, device: torch.device) -> torch.Tensor:
    tmp_size = _kernels.sgmv_cutlass_tmp_size(size)
    return torch.empty((tmp_size,), dtype=torch.uint8, device=device)


def get_tmp_expand_size(size: int) -> int:
    return _kernels.sgmv_cutlass_tmp_size(size)


def get_tmp_tensors(nsegments: int, lora_rank: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_cutlass_shrink(lora_rank):
        tmp = get_tmp_tensor_for_size(nsegments, device)
        return tmp, tmp
    else:
        tmp_shrink = get_tmp_tensor(device)
        tmp_expand = get_tmp_tensor_for_size(nsegments, device)
        return tmp_shrink, tmp_expand


def lora_a_sgmv(
    x: torch.Tensor,
    tmp: torch.Tensor,
    wa_ptr: torch.Tensor,
    s_start: torch.IntTensor,
    s_end: torch.IntTensor,
    ranks: torch.IntTensor,
    max_rank: int,
    layer_idx: int,
) -> torch.Tensor:
    v = torch.zeros((x.size(0), max_rank), dtype=x.dtype, device=x.device)
    if MIN_RANK_CUSTOM <= max_rank <= MAX_RANK_CUSTOM:
        _kernels.sgmv_shrink(v, x, wa_ptr, s_start, s_end, ranks, tmp, layer_idx)
    else:
        _kernels.sgmv_cutlass(v, x, wa_ptr, s_start, s_end, ranks, tmp, layer_idx)
    return v


def lora_b_sgmv(
    y: torch.Tensor,
    v: torch.Tensor,
    tmp: torch.Tensor,
    wb_ptr: torch.Tensor,
    s_start: torch.IntTensor,
    s_end: torch.IntTensor,
    ranks: torch.IntTensor,
    layer_idx: int,
):
    _kernels.sgmv_cutlass(y, v, wb_ptr, s_start, s_end, ranks, tmp, layer_idx)


"""
Semantics:
    y[i] += (
        x[i].unsqueeze(0)
        @ wa_T_all[indices[i], layer_idx, :, :].transpose(-1, -2)
        @ wb_T_all[indices[i], layer_idx, :, :].transpose(-1, -2)
        * scale
    ).squeeze(0)

Args:
    y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
    v: Shape: `[B, R]`. Temporary vector.
    x: Shape: `[B, H1]`. Input vectors.
    wa_T_all: Shape: `[None, L, R, H1]`. All of the transposed LoRA A matrices.
    wb_T_all: Shape: `[None, L, H2, R]`. All of the transposed LoRA B matrices.
    indicies: Shape: `[B]`. Indices of the LoRA weights.
    layer_idx: Layer index of LoRA weights.
    scale: Scaling factor.
"""


def add_lora_a_bgmv(
    v: torch.Tensor,
    x: torch.Tensor,
    wa_T_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
):
    _kernels.dispatch_bgmv(v, x, wa_T_all, indicies, layer_idx, 1.0)


def add_lora_b_bgmv(
    y: torch.Tensor,
    v: torch.Tensor,
    wb_T_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
):
    _kernels.dispatch_bgmv(y, v, wb_T_all, indicies, layer_idx, 1.0)
