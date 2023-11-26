import pytest
import torch

from lorax_server.utils.segments import find_segments



@pytest.mark.parametrize(
    "adapter_indices,segments,segment_indices",
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
def test_find_segments(adapter_indices, segments, segment_indices):
    adapter_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 1, 1])
    segments, segment_indices = find_segments(adapter_indices)
    assert segments == [0, 3, 5, 10, 12]
    assert segment_indices == [0, 1, 2, 1]
