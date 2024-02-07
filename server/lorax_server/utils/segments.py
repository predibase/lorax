from typing import List, Tuple

import torch


def find_segments(adapter_indices: torch.Tensor) -> Tuple[List[int], List[int]]:
    segments = [0]
    segment_indices = []

    # Calling .item() repeatedly on CUDA tensor is very slow, so we move it to CPU first
    adapter_indices = adapter_indices.cpu()

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


class SegmentConcatBuilder:
    def __init__(self):
        self.adapter_segment_indices = []
        self.adapter_segment_tensors = []

    def concat(self, adapter_segments: torch.Tensor, segment_indices: List[int]):
        # Update adapter segments
        if self.adapter_segment_tensors:
            # Because we have already processed at least one batch, remove the 0 start index
            # from this batch denoting the beginning of the segment, then offset all segment
            # positions by the value of the last segment in the previous batch to account for
            # the concatenation.
            adapter_segments = adapter_segments[1:] + self.adapter_segment_tensors[-1][-1]
        
        if self.adapter_segment_indices and self.adapter_segment_indices[-1] == segment_indices[0]:
            # If the last segment in the previous batch is the same as the first segment in this batch,
            # then we merge them together into a single segment. In effect, this means removing it from
            # the segment indices of this batch, and extending the segment span by removing the segment
            # end index from the previous batch.
            segment_indices = segment_indices[1:]
            self.adapter_segment_tensors[-1] = self.adapter_segment_tensors[-1][:-1]
        
        self.adapter_segment_indices.extend(segment_indices)
        self.adapter_segment_tensors.append(adapter_segments)
    
    def build(self) -> Tuple[torch.Tensor, List[int]]:
        return torch.concat(self.adapter_segment_tensors), self.adapter_segment_indices
