from typing import List
import pytest
import torch

from lorax_server.utils.sgmv import (
    get_tmp_tensors,
    lora_a_sgmv_cutlass,
    lora_b_sgmv_cutlass,
    has_sgmv,
    use_cutlass_shrink,
)


def lora_ref_impl(
    y: torch.Tensor,
    x: torch.Tensor,
    wa: List[torch.Tensor],
    wb: List[torch.Tensor],
    s_start: torch.IntTensor,
    s_end: torch.IntTensor,
    layer_idx: int,
    lora_rank: int,
):
    for i in range(len(wa)):
        xi = x[s_start[i]:s_end[i]]
        wai = wa[i][layer_idx, :, :]
        wbi = wb[i][layer_idx, :, :]

        if not use_cutlass_shrink(lora_rank):
            wai = wai.t()

        yi = y[s_start[i]:s_end[i]]
        tmp = (xi @ wai)
        y[s_start[i]:s_end[i]] = (yi + tmp @ wbi)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not has_sgmv(), reason="SGMV not available")
@pytest.mark.parametrize("lora_rank", [8, 16, 32, 64, 128])
def test_add_lora_sgmv_cutlass(lora_rank: int):
    torch.manual_seed(42)

    B = 3
    H = 1024
    r = lora_rank
    nlayers = 2

    device = torch.device("cuda:0")
    
    y = torch.zeros((B, H), dtype=torch.float16, device=device)
    x = torch.randn((B, H), dtype=torch.float16, device=device)
    wa = torch.randn(nlayers, r, H, dtype=torch.float16, device=device)
    if use_cutlass_shrink(r):
        # cutlass uses (H, r) layout
        wa = wa.transpose(1, 2).contiguous()

    # TODO(travis): transpose (r, H) -> (H, r) when not using cutlass
    wb = torch.randn(nlayers, r, H, dtype=torch.float16, device=device)

    wa_ptr = torch.tensor([wa.data_ptr(), wa.data_ptr()], dtype=torch.int64, device=device)
    wb_ptr = torch.tensor([wb.data_ptr(), wb.data_ptr()], dtype=torch.int64, device=device)

    s_start = torch.tensor([0, 2], dtype=torch.int32, device=device)
    s_end = torch.tensor([1, 3], dtype=torch.int32, device=device)

    layer_idx = 0

    y_ref = y.clone()
    lora_ref_impl(y_ref, x, [wa, wa], [wb, wb], s_start, s_end, layer_idx, r)

    v = torch.zeros((x.size(0), r), dtype=x.dtype, device=x.device)
    tmp_shrink, tmp_expand = get_tmp_tensors(wa_ptr.size(0), r, x.device)

    y_ours = torch.zeros((B, H), dtype=torch.float16, device=device)
    lora_a_sgmv_cutlass(v, x, tmp_shrink, wa_ptr, s_start, s_end, layer_idx, r)
    lora_b_sgmv_cutlass(y_ours, v, tmp_expand, wb_ptr, s_start, s_end, layer_idx)

    assert torch.allclose(y_ref, y_ours, rtol=1e-2, atol=1e-3)
