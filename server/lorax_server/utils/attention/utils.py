from typing import List

import torch


def block_tables_to_ragged(
    *, block_tables: torch.Tensor, input_lengths: List[int], cache_lengths: List[int]
) -> torch.Tensor:
    """Convert block table to ragged format compatible with FlashInfer."""
    assert len(input_lengths) == len(cache_lengths)

    total_len = sum(input_lengths) + sum(cache_lengths)
    block_tables_ragged = torch.empty(total_len, dtype=torch.int32, device=block_tables.device)

    offset = 0
    for i, (input_length, cache_length) in enumerate(zip(input_lengths, cache_lengths)):
        seq_len = cache_length + input_length
        block_tables_ragged[offset : offset + seq_len] = block_tables[i][:seq_len]
        offset += seq_len

    return block_tables_ragged
