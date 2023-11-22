import logging
import torch

try:
    # Hack to ignore ray warnings from vLLM, which are not relevant to us.
    logging.disable(logging.WARNING)
    from vllm import cache_ops
    from vllm import attention_ops
finally:
    logging.disable(logging.NOTSET)


_PARTITION_SIZE = 512


def reshape_and_cache(
    key: torch.Tensor,           # [num_tokens, num_heads, head_size]
    value: torch.Tensor,         # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,     # [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache: torch.Tensor,   # [num_blocks, num_heads, head_size, block_size]
    slot_mapping: torch.Tensor,  # [num_tokens]
):
    cache_ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)


# Source: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention.py
def single_query_cached_kv_attention(
    output: torch.Tensor,         # [num_tokens, num_heads, head_size]
    query: torch.Tensor,          # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,      # [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache: torch.Tensor,    # [num_blocks, num_heads, head_size, block_size]
    kv_head_mapping: torch.Tensor,
    softmax_scale: float,
    block_tables: torch.Tensor,   # [num_blocks, block_size]
    input_lengths: torch.Tensor,  # [num_blocks]
    max_s: int,
):
    block_size = value_cache.shape[3]
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = (max_s + _PARTITION_SIZE - 1) // _PARTITION_SIZE
    
    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    # TODO(woosuk): Tune this heuristic.
    # For context len > 8192, use V2 kernel to avoid shared memory shortage.
    use_v1 = max_s <= 8192 and (max_num_partitions == 1 or num_seqs * num_heads > 512)
    if use_v1:
        # Run PagedAttention V1.
        attention_ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            kv_head_mapping,
            softmax_scale,
            block_tables,
            input_lengths,
            block_size,
            max_s,
            None,
        )
    else:
        # Run PagedAttention V2.
        assert _PARTITION_SIZE % block_size == 0
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)
        attention_ops.paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            kv_head_mapping,
            softmax_scale,
            block_tables,
            input_lengths,
            block_size,
            max_s,
            None,
        )