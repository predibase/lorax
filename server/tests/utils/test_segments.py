import pytest
import torch

from lorax_server.utils.segments import SegmentConcatBuilder, find_segments


@pytest.mark.parametrize(
    "adapter_indices,expected_segments,expected_segment_indices",
    [
        (
            torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 1, 1]),
            [0, 3, 5, 10, 12],
            [0, 1, 2, 1],
        ),
        (torch.tensor([]), [0], []),
        (torch.tensor([0]), [0, 1], [0]),
        (torch.tensor([1]), [0, 1], [1]),
    ],
)
def test_find_segments(adapter_indices, expected_segments, expected_segment_indices):
    segments, segment_indices = find_segments(adapter_indices)
    assert segments == expected_segments
    assert segment_indices == expected_segment_indices


@pytest.mark.parametrize(
    "batches,expected_segments,expected_segment_indices",
    [
        (
            [
                (torch.tensor([0, 1, 4, 7, 8]), [2, 1, 2, 1]),
                (torch.tensor([0, 2, 5]), [1, 2]),
            ],
            [0, 1, 4, 7, 10, 13],
            [2, 1, 2, 1, 2],
        ),
        (
            [
                (torch.tensor([0, 1, 4, 7]), [2, 1, 2]),
                (torch.tensor([0, 2, 5]), [1, 2]),
            ],
            [0, 1, 4, 7, 9, 12],
            [2, 1, 2, 1, 2],
        ),
    ],
)
def test_concat_segments(batches, expected_segments, expected_segment_indices):
    builder = SegmentConcatBuilder()
    for segment, indices in batches:
        builder.concat(segment, indices)

    segments, segment_indices = builder.build()
    assert segments.tolist() == expected_segments
    assert segment_indices == expected_segment_indices
