from typing import List, Tuple


def find_segments(adapter_indixes: List[int]) -> Tuple[List[int], List[int]]:
    segments = [0]
    segment_indices = []

    start_index = 0
    for i in range(1, len(adapter_indixes)):
        if adapter_indixes[i] != adapter_indixes[i - 1]:
            segments.append(i - 1)
            segment_indices.append(adapter_indixes[i - 1])
            start_index = i

    # Handle the last segment
    if start_index < len(adapter_indixes):
        segments.append(len(adapter_indixes))
        segment_indices.append(adapter_indixes[-1])

    return segments, segment_indices
