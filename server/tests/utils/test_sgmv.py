import pytest
import torch

from lorax_server.utils.sgmv import add_lora_sgmv_cutlass, has_sgmv, orient_for_rank


def lora_ref_impl(
    y: torch.Tensor,
    x: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    s_start: torch.IntTensor,
    s_end: torch.IntTensor,
    layer_idx: int,
):
    for i in range(len(wa)):
        xi = x[s_start[i] : s_end[i]]
        wai = wa[i][layer_idx, :, :]
        wbi = wb[i][layer_idx, :, :]
        yi = y[s_start[i] : s_end[i]]
        tmp = xi @ wai
        y[s_start[i] : s_end[i]] = yi + tmp @ wbi


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not has_sgmv(), reason="SGMV not available")
@pytest.mark.parametrize("lora_rank", [4, 8, 16, 32, 64, 128])
def test_add_lora_sgmv_cutlass(lora_rank: int):
    B = 3
    H = 1024
    r = lora_rank
    nlayers = 2

    device = torch.device("cuda:0")

    y = torch.zeros((B, H), dtype=torch.float16, device=device)
    x = torch.randn((B, H), dtype=torch.float16, device=device)
    wa = torch.randn(nlayers, H, r, dtype=torch.float16, device=device)
    wb = torch.randn(nlayers, r, H, dtype=torch.float16, device=device)

    wa_sgmv = orient_for_rank(wa, lora_rank)
    wa_ptr = torch.tensor(
        [wa_sgmv.data_ptr(), wa_sgmv.data_ptr()], dtype=torch.int64, device=device
    )
    wb_ptr = torch.tensor(
        [wb.data_ptr(), wb.data_ptr()], dtype=torch.int64, device=device
    )

    s_start = torch.tensor([0, 2], dtype=torch.int32, device=device)
    s_end = torch.tensor([1, 3], dtype=torch.int32, device=device)

    layer_idx = 0

    y_ref = y.clone()
    lora_ref_impl(y_ref, x, [wa, wa], [wb, wb], s_start, s_end, layer_idx)
    # print(y_ref)

    lora_a = wa[0, :, :]
    lora_b = wb[0, :, :]
    mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float16, device=device)
    # out = torch.matmul(torch.matmul(x, lora_a), lora_b)
    out = ((x @ lora_a) @ lora_b) * mask.view(-1, 1)
    # print(x @ lora_a)
    # print(out)

    # assert torch.allclose(y_ref, out)

    y_ours = torch.zeros((B, H), dtype=torch.float16, device=device)
    add_lora_sgmv_cutlass(
        y_ours,
        x,
        wa_ptr,
        wb_ptr,
        s_start,
        s_end,
        layer_idx,
        r,
    )
    # print(y_ours)

    # assert torch.allclose(y_ref, y_ours)
    assert torch.allclose(out, y_ours)
