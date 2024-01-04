from typing import List, Tuple
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
        if s_end[i] - s_start[i] <= 0:
            continue
        
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
@pytest.mark.parametrize("segments", [
    ([0, 2], [1, 3]),
    ([0, -1], [1, -1]),
])
@pytest.mark.parametrize("lora_rank", [8, 16, 32, 64, 128])
def test_add_lora_sgmv(lora_rank: int, segments: Tuple[List[int], List[int]]):
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

    s1, s2 = segments
    s_start = torch.tensor(s1, dtype=torch.int32, device=device)
    s_end = torch.tensor(s2, dtype=torch.int32, device=device)

    wa_list = [wa if y - x > 0 else None for x, y in zip(s1, s2)]
    wb_list = [wb if y - x > 0 else None for x, y in zip(s1, s2)]

    wa_ptr = torch.tensor([wa.data_ptr() if wa is not None else 0 for wa in wa_list], dtype=torch.int64, device=device)
    wb_ptr = torch.tensor([wb.data_ptr() if wb is not None else 0 for wb in wb_list], dtype=torch.int64, device=device)

    layer_idx = 0

    y_ref = y.clone()
    lora_ref_impl(y_ref, x, wa_list, wb_list, s_start, s_end, layer_idx, r)

    tmp_shrink, tmp_expand = get_tmp_tensors(wa_ptr.size(0), r, x.device)
    y_ours = torch.zeros((B, H), dtype=torch.float16, device=device)

    v = lora_a_sgmv_cutlass(x, tmp_shrink, wa_ptr, s_start, s_end, layer_idx, r)
    lora_b_sgmv_cutlass(y_ours, v, tmp_expand, wb_ptr, s_start, s_end, layer_idx)

    assert torch.allclose(y_ref, y_ours, rtol=1e-2, atol=1e-3)

    # graph trace
    tmp_shrink, tmp_expand = get_tmp_tensors(wa_ptr.size(0), r, x.device)
    y_ours_graph = torch.zeros((B, H), dtype=torch.float16, device=device)

    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=None):
        v = lora_a_sgmv_cutlass(x, tmp_shrink, wa_ptr, s_start, s_end, layer_idx, r)
        lora_b_sgmv_cutlass(y_ours_graph, v, tmp_expand, wb_ptr, s_start, s_end, layer_idx)

    torch.cuda.synchronize(device)
    graph.replay()

    assert torch.allclose(y_ours, y_ours_graph, rtol=1e-2, atol=1e-3)