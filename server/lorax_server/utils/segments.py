from typing import List, Tuple

import torch


def find_segments(adapter_indices: torch.Tensor) -> Tuple[List[int], List[int]]:
    segments = [0]
    segment_indices = []

    start_index = 0
    for i in range(1, adapter_indices.shape[0]):
        if adapter_indices[i] != adapter_indices[i - 1]:
            segments.append(i)
            segment_indices.append(adapter_indices[i - 1].item())
            start_index = i

    # Handle the last segment
    if start_index < len(adapter_indices):
        segments.append(len(adapter_indices))
        segment_indices.append(adapter_indices[-1].item())

    return segments, segment_indices
