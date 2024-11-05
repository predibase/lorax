from typing import Dict, List, Optional, Tuple, Type
from unittest import mock

import pytest
import torch
from peft import LoraConfig

from lorax_server.adapters.lora import LoraWeights
from lorax_server.adapters.types import LORA
from lorax_server.adapters.weights import AdapterBatchMetadata, AdapterWeights, BatchAdapterWeights, LayerAdapterWeights
from lorax_server.utils.lora import LM_HEAD
from lorax_server.utils.punica import MIN_RANK_CUSTOM


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
        layer_name: str,
        meta: "AdapterBatchMetadata",
        prefill: bool,
        prefill_head_indices: torch.Tensor,
    ) -> Optional["BatchAdapterWeights"]:
        return None


@pytest.mark.parametrize(
    "lora_ranks",
    [
        [8, 16],
        [32, 64],
    ],
)
def test_batched_lora_weights(lora_ranks: List[int]):
    # batch meta is hardcoded with this assumption below
    assert len(lora_ranks) == 2

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
    assert len(batched_weights.adapter_weights) == 2

    meta = AdapterBatchMetadata(
        adapter_indices=torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.int64),
        adapter_list=[0, 1, 0, 1],
        adapter_set={0, 1},
        adapter_segments=torch.tensor([0, 2, 4, 6, 8], dtype=torch.int64),
        segment_indices=[0, 1, 0, 1],
    )

    with mock.patch("lorax_server.adapters.lora.get_tmp_tensors", return_value=(torch.empty(0), torch.empty(0))):
        data = batched_weights.get_data(meta, LM_HEAD, prefill=True, prefill_head_indices=None).get(LORA)

    assert len(data.lora_a) == 2
    assert data.lora_a.keys() == meta.adapter_set
    assert data.lora_a[0].shape == ((1, h, lora_ranks[0]) if lora_ranks[0] < MIN_RANK_CUSTOM else (1, lora_ranks[0], h))
    assert data.lora_a[1].shape == ((1, h, lora_ranks[1]) if lora_ranks[1] < MIN_RANK_CUSTOM else (1, lora_ranks[1], h))

    assert len(data.lora_b) == 2
    assert data.lora_b.keys() == meta.adapter_set
    assert data.lora_b[0].shape == (1, lora_ranks[0], h)
    assert data.lora_b[1].shape == (1, lora_ranks[1], h)

    assert len(data.rank_data) == 2
    assert data.rank_data.keys() == set(lora_ranks)
    for lora_rank, rd in data.rank_data.items():
        assert rd.rank == lora_rank

        # shape in all cases is the number of segments with this rank
        assert rd.lora_a_ptr.shape == (2,)
        assert rd.lora_b_ptr.shape == (2,)
        assert rd.segment_starts.shape == (2,)
        assert rd.segment_ends.shape == (2,)



@pytest.mark.parametrize(
    "lora_ranks,adapter_indices,expected",
    [
        (
            [8, 8, 16],
            [0, 0, 1, 1, 0, 0, 1, 1, 2, 2],
            {
                8: (4, [0, 0, 1, 1, 0, 0, 1, 1, -1, -1]),
                16: (1, [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0])
            }
        ),
        (
            [4, 8, 16],
            [0, 0, 1, 1, 0, 0, 1, 1, 2, 2],
            {
                4: (2, [0, 0, -1, -1, 0, 0, -1, -1, -1, -1]),
                8: (2, [-1, -1, 0, 0, -1, -1, 0, 0, -1, -1]),
                16: (1, [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0]),
            }
        ),
    ],
)
def test_batched_lora_weights_decode(
    lora_ranks: List[int],
    adapter_indices: List[int],
    expected: Dict[int, Tuple[int, List[int]]]
):
    from lorax_server.utils.segments import find_segments
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
        adapter_list=adapter_indices,
        adapter_set=set(adapter_indices),
        adapter_segments=torch.tensor(segments, dtype=torch.int64),
        segment_indices=segment_indices,
    )

    with mock.patch("lorax_server.adapters.lora.get_tmp_tensors", return_value=(torch.empty(0), torch.empty(0))):
        data = batched_weights.get_data(meta, LM_HEAD, prefill=False, prefill_head_indices=None).get(LORA)

    for lora_rank, rd in data.rank_data.items():
        expected_indices = torch.tensor(expected[lora_rank][1], dtype=rd.indices.dtype, device=rd.indices.device)
        assert rd.lora_a_ptr.shape == (expected[lora_rank][0],)
        assert rd.lora_b_ptr.shape == (expected[lora_rank][0],)
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
        adapter_list=[0],
        adapter_set={0},
        adapter_segments=torch.tensor([0, 4], dtype=torch.int64),
        segment_indices=[0],
    )

    with mock.patch("lorax_server.adapters.lora.get_tmp_tensors", return_value=(torch.empty(0), torch.empty(0))):
        data = batched_weights.get_data(meta, LM_HEAD, prefill=True, prefill_head_indices=None).get(LORA)

    print(data)
