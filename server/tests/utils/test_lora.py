from typing import Dict, List, Optional, Tuple, Type
from unittest import mock

import pytest
import torch
from peft import LoraConfig

from lorax_server.adapters.lora import LoraWeights
from lorax_server.adapters.types import LORA
from lorax_server.adapters.weights import AdapterBatchMetadata, AdapterWeights, BatchAdapterWeights, LayerAdapterWeights
from lorax_server.utils.segments import find_segments
from lorax_server.utils.sgmv import MIN_RANK_CUSTOM


class FakeAdapterWeights(AdapterWeights):
    @classmethod
    def get_batch_types(cls) -> List[Type["FakeBatchAdapterWeights"]]:
        return [FakeBatchAdapterWeights]

    @property
    def speculative_tokens(self) -> int:
        return 0


class FakeBatchAdapterWeights(BatchAdapterWeights):
    @classmethod
    def has_adapter(self, adapter_index: int) -> bool:
        False

    @classmethod
    def key(cls) -> str:
        "fake"

    @classmethod
    def load(
        cls,
        adapter_weights: Dict[int, AdapterWeights],
        meta: "AdapterBatchMetadata",
        prefill: bool,
        prefill_head_indices: torch.Tensor,
    ) -> Optional["BatchAdapterWeights"]:
        return None


@pytest.mark.parametrize(
    "lora_ranks,adapter_indices,expected",
    [
        (
            [8, 8, 16], # ranks of adapters
            [0, 0, 1, 1, 0, 0, 1, 1, 2, 2], # adapter indices for each token in the batch
            {
                8: ( # rank
                    [0, 2, 4, 6], # expected segment starts
                    [2, 4, 6, 8], # expected segment ends
                    [0, 1, 0, 1], # expected adapter indices
                ),
                16: ([8], [10], [2]),
            }
        ),
        (
            [4, 8, 16],
            [0, 0, 1, 1, 0, 0, 1, 1, 2, 2],
            {
                4: ([0, 4], [2, 6], [0, 0]),
                8: ([2, 6], [4, 8], [1, 1]),
                16: ([8], [10], [2]),
            }
        ),
    ],
)
def test_batched_lora_weights(
    lora_ranks: List[int],
    adapter_indices: List[int],
    expected: Dict[int, Tuple[List[int], Tuple[int], Tuple[int]]]
):
    num_adapters = len(lora_ranks)
    batched_weights = LayerAdapterWeights()
    assert batched_weights.is_empty()

    h = 1024
    for idx, lora_rank in enumerate(lora_ranks):
        weights = LoraWeights(
            weights_a=[torch.randn((h, lora_rank), dtype=torch.float16)],
            weights_b=[torch.randn((lora_rank, h), dtype=torch.float16)],
            adapter_config=LoraConfig(r=lora_rank),
        )
        assert weights.lora_a_r == lora_rank
        assert weights.lora_b_r == lora_rank

        batched_weights.add_adapter(idx, weights)

    assert not batched_weights.is_empty()
    assert len(batched_weights.adapter_weights) == num_adapters

    segments, segment_indices = find_segments(adapter_indices)

    meta = AdapterBatchMetadata(
        adapter_indices=torch.tensor(adapter_indices, dtype=torch.int64),
        adapter_set=set(adapter_indices),
        adapter_segments=torch.tensor(segments, dtype=torch.int64),
        segment_indices=segment_indices,
    )

    with mock.patch("lorax_server.adapters.lora.get_tmp_tensors", return_value=(torch.empty(0), torch.empty(0))):
        data = batched_weights.get_data(meta, prefill=True, prefill_head_indices=None).get(LORA)

    assert len(data.lora_a) == num_adapters
    assert len(data.lora_b) == num_adapters
    assert data.lora_a.keys() == meta.adapter_set
    assert data.lora_b.keys() == meta.adapter_set
    for i in range(num_adapters):
        assert data.lora_a[i].shape == (
            (1, h, lora_ranks[i]) if lora_ranks[i] < MIN_RANK_CUSTOM else (1, lora_ranks[i], h)
        )
        assert data.lora_b[i].shape == (1, lora_ranks[i], h)

    for lora_rank, rd in data.rank_data.items():
        assert rd.rank == lora_rank
        expected_lora_a_ptr = []
        expected_lora_b_ptr = []
        for adapter_idx in expected[lora_rank][2]:
            expected_lora_a_ptr.append(batched_weights.adapter_weights[adapter_idx].weights_a.data_ptr())
            expected_lora_b_ptr.append(batched_weights.adapter_weights[adapter_idx].weights_b.data_ptr())
        expected_lora_a_ptr = torch.tensor(expected_lora_a_ptr, dtype=rd.lora_a_ptr.dtype, device=rd.lora_a_ptr.device)
        expected_lora_b_ptr = torch.tensor(expected_lora_b_ptr, dtype=rd.lora_b_ptr.dtype, device=rd.lora_b_ptr.device)
        assert all(rd.lora_a_ptr == expected_lora_a_ptr)
        assert all(rd.lora_b_ptr == expected_lora_b_ptr)

        expected_segment_starts = torch.tensor(
            expected[lora_rank][0], dtype=rd.segment_starts.dtype, device=rd.segment_starts.device
        )
        expected_segment_ends = torch.tensor(
            expected[lora_rank][1], dtype=rd.segment_ends.dtype, device=rd.segment_ends.device
        )
        assert all(rd.segment_ends == expected_segment_ends)
        assert all(rd.segment_starts == expected_segment_starts)


@pytest.mark.parametrize(
    "lora_ranks,adapter_indices,expected",
    [
        (
            [8, 8, 16], # ranks of adapters
            [0, 0, 1, 1, 0, 0, 1, 1, 2, 2], # adapter indices for each token in the batch
            {
                8: ( # rank
                    [0, 1], # expected adapter indices
                    [0, 0, 1, 1, 0, 0, 1, 1, -1, -1] # expected indices
                ),
                16: ([2], [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0]),
            }
        ),
        (
            [4, 8, 16],
            [0, 0, 1, 1, 0, 0, 1, 1, 2, 2],
            {
                4: ([0], [0, 0, -1, -1, 0, 0, -1, -1, -1, -1]),
                8: ([1], [-1, -1, 0, 0, -1, -1, 0, 0, -1, -1]),
                16: ([2], [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0]),
            }
        ),
    ],
)
def test_batched_lora_weights_decode(
    lora_ranks: List[int],
    adapter_indices: List[int],
    expected: Dict[int, Tuple[List[int], List[int]]]
):
    batched_weights = LayerAdapterWeights()
    assert batched_weights.is_empty()

    h = 1024
    for idx, lora_rank in enumerate(lora_ranks):
        weights = LoraWeights(
            weights_a=[torch.randn((h, lora_rank), dtype=torch.float16)],
            weights_b=[torch.randn((lora_rank, h), dtype=torch.float16)],
            adapter_config=LoraConfig(r=lora_rank),
        )
        batched_weights.add_adapter(idx, weights)

    segments, segment_indices = find_segments(adapter_indices)

    meta = AdapterBatchMetadata(
        adapter_indices=torch.tensor(adapter_indices, dtype=torch.int64),
        adapter_set=set(adapter_indices),
        adapter_segments=torch.tensor(segments, dtype=torch.int64),
        segment_indices=segment_indices,
    )

    with mock.patch("lorax_server.adapters.lora.get_tmp_tensors", return_value=(torch.empty(0), torch.empty(0))):
        data = batched_weights.get_data(meta, prefill=False, prefill_head_indices=None).get(LORA)

    for lora_rank, rd in data.rank_data.items():
        expected_lora_a_ptr = []
        expected_lora_b_ptr = []
        for adapter_idx in expected[lora_rank][0]:
            expected_lora_a_ptr.append(batched_weights.adapter_weights[adapter_idx].weights_a_t.data_ptr())
            expected_lora_b_ptr.append(batched_weights.adapter_weights[adapter_idx].weights_b_t.data_ptr())
        expected_lora_a_ptr = torch.tensor(expected_lora_a_ptr, dtype=rd.lora_a_ptr.dtype, device=rd.lora_a_ptr.device)
        expected_lora_b_ptr = torch.tensor(expected_lora_b_ptr, dtype=rd.lora_b_ptr.dtype, device=rd.lora_b_ptr.device)
        assert all(rd.lora_a_ptr == expected_lora_a_ptr)
        assert all(rd.lora_b_ptr == expected_lora_b_ptr)

        expected_indices = torch.tensor(expected[lora_rank][1], dtype=rd.indices.dtype, device=rd.indices.device)
        assert all(rd.indices == expected_indices)

        assert rd.segment_starts is None
        assert rd.segment_ends is None
        assert rd.tmp_shrink is None
        assert rd.tmp_expand is None

def test_batched_lora_weights_no_segments():
    batched_weights = LayerAdapterWeights()
    assert batched_weights.is_empty()

    h = 1024

    # fake weights
    idx = 0
    weights = FakeAdapterWeights()
    batched_weights.add_adapter(idx, weights)

    # lora weights
    idx = 1
    lora_rank = 16
    weights = LoraWeights(
        weights_a=[torch.randn((h, lora_rank), dtype=torch.float16)],
        weights_b=[torch.randn((lora_rank, h), dtype=torch.float16)],
        adapter_config=LoraConfig(r=lora_rank),
    )
    batched_weights.add_adapter(idx, weights)

    assert not batched_weights.is_empty()
    assert len(batched_weights.adapter_weights) == 2

    meta = AdapterBatchMetadata(
        adapter_indices=torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        adapter_set={0, 1},
        adapter_segments=torch.tensor([0, 4], dtype=torch.int64),
        segment_indices=[0],
    )

    with mock.patch("lorax_server.adapters.lora.get_tmp_tensors", return_value=(torch.empty(0), torch.empty(0))):
        data = batched_weights.get_data(meta, prefill=True, prefill_head_indices=None).get(LORA)

    print(data)
