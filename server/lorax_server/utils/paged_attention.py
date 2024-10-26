from typing import Optional

import torch

from lorax_server.utils.attention.common import Seqlen
from lorax_server.utils.import_utils import SYSTEM
from lorax_server.utils.state import FLASH_INFER

_PARTITION_SIZE = 512

if SYSTEM == "xpu":
    import intel_extension_for_pytorch as ipex
else:
    try:
        import torch
        import vllm._custom_ops as ops
    except Exception as e:
        raise ImportError(
            f"Could not import vllm paged attention. Make sure your installation is correct. Error: {e}"
        ) from e


def static_per_tensor_quantize(tensor: torch.Tensor, inv_scale: float) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    fp8_kv: bool = False,
):
    if FLASH_INFER:
        if fp8_kv:
            key = static_per_tensor_quantize(key, k_scale).view(torch.uint8)
            value = static_per_tensor_quantize(value, v_scale).view(torch.uint8)
            key_cache = key_cache.view(torch.uint8)
            value_cache = value_cache.view(torch.uint8)
        shape = key_cache.shape
        key_cache.view(-1, shape[-2], shape[-1])[slots] = key
        value_cache.view(-1, shape[-2], shape[-1])[slots] = value
    elif SYSTEM == "xpu":
        ipex.llm.modules.PagedAttention.reshape_and_cache(key, value, key_cache, value_cache, slots)
    else:
        torch.ops._C_cache_ops.reshape_and_cache(key, value, key_cache, value_cache, slots, 'auto', 1.0, 1.0)


def attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    kv_head_mapping: torch.Tensor,
    softmax_scale: float,
    block_tables: torch.Tensor,
    seqlen: Seqlen,
    max_s: int,
    softcap: Optional[float] = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
):
    if FLASH_INFER:
        from lorax_server.utils.flashinfer_attention import decode_state

        return decode_state.get().forward(
            query.contiguous(),
            paged_kv_cache=(key_cache, value_cache),
            logits_soft_cap=softcap,
            sm_scale=softmax_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    # Adapted from: https://github.com/vllm-project/vllm/blob/f8a1e39fae05ca610be8d5a78be9d40f5274e5fc/vllm/model_executor/layers/attention.py
    # Copyright 2023 The vLLM team. All rights
    # reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    #

    # value_cache => [num_blocks, num_heads, head_size, block_size]
    input_lengths = seqlen.input_lengths + seqlen.cache_lengths
    block_size = value_cache.shape[3]
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = (max_s + _PARTITION_SIZE - 1) // _PARTITION_SIZE

    out = torch.empty_like(query)

    if SYSTEM == "xpu":
        query = query.contiguous()
        ipex.llm.modules.PagedAttention.single_query_cached_kv_attention(
            out,
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
        return out

    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    use_v1 = max_s <= 8192 and (max_num_partitions == 1 or num_seqs * num_heads > 512)
    if use_v1:
        ops.paged_attention_v1(
            out,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            softmax_scale,
            block_tables,
            input_lengths,
            block_size,
            max_s,
            None,
            'auto',
            1.0,
            1.0,
        )
    else:
        # Run PagedAttention V2.
        assert _PARTITION_SIZE % block_size == 0
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=out.dtype,
            device=out.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=out.device,
        )
        max_logits = torch.empty_like(exp_sums)

        ops.paged_attention_v2(
            out,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            softmax_scale,
            block_tables,
            input_lengths,
            block_size,
            max_s,
            None,
            'auto',
            1.0,
            1.0,
        )

    return out
