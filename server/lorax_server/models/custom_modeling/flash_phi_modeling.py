# Adapted from
# https://huggingface.co/microsoft/phi-2/blob/main/modeling_phi.py
# and
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/phi_1_5.py
#
# Copyright 2023 Predibase.
# Copyright 2023 The vLLM team.
# Copyright (c) Microsoft Corporation.
#
# LICENSE: https://huggingface.co/microsoft/phi-2/blob/main/LICENSE

import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from transformers.models.phi import PhiConfig
from typing import Optional, List, Tuple

from lorax_server.utils import flash_attn
from lorax_server.utils import paged_attn
from lorax_server.utils.layers import (
    FastLayerNorm,
    TensorParallelAdapterRowLinear,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelMultiAdapterLinear,
    PositionRotaryEmbedding,
    TensorParallelHead,
)
from lorax_server.utils.lora import LM_HEAD, AdapterBatchData


ATTN_WQKV = "mixer.Wqkv"
ATTN_OUT_PROJ = "mixer.out_proj"
MLP_FC1 = "mlp.fc1"
MLP_FC2 = "mlp.fc2"
        

def load_attention(config, prefix, weights, layer_id, head_dim, n_head, n_head_kv):
    op_size = head_dim * (n_head + 2 * n_head_kv)
    base_layer = load_attention_multi(config, prefix, weights, head_dim, n_head, n_head_kv)
    return TensorParallelMultiAdapterLinear.load(
        base_layer, layer_id, [ATTN_WQKV], sizes=[op_size], process_group=weights.process_group
    )


def load_attention_multi(config, prefix, weights, head_dim, n_head, n_head_kv):
    return TensorParallelColumnLinear.load_multi(
        config,
        prefixes=[
            (f"{prefix}.Wqkv", (0, head_dim * n_head)),
            (f"{prefix}.Wqkv", (head_dim * n_head, head_dim * n_head_kv)),
            (f"{prefix}.Wqkv", ((head_dim * n_head) + (head_dim * n_head_kv), head_dim * n_head_kv)),
        ],
        dim=0,
        weights=weights,
        bias=True,
    )


class FlashPhiAttention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
        layer_id: int,
    ):
        super().__init__()
        self.num_heads = config.n_head
        self.hidden_size = config.n_embd
        self.head_size = self.hidden_size // self.num_heads
        self.process_group = weights.process_group

        rope_theta = 10000
        config.max_position_embeddings = getattr(config, "n_positions", 2048)

        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=config.rotary_dim,
            base=rope_theta,
            device=weights.device,
            dtype=weights.dtype,
        )

        self.softmax_scale = self.head_size**-0.5

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_key_value_heads = getattr(config, "n_head_kv", None) or self.num_heads

        self.Wqkv = load_attention(config, prefix, weights, layer_id, self.head_size, self.num_heads, self.num_key_value_heads)
        self.out_proj = TensorParallelAdapterRowLinear.load(TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.out_proj",
            weights=weights,
            bias=True,
        ), layer_id, ATTN_OUT_PROJ, process_group=weights.process_group)

        # After initializing layers, scale num heads by num shards for use in forward() to split outputs
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = self.num_key_value_heads // weights.process_group.size()

        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)
        self.layer_id = layer_id

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
        adapter_data,
    ):
        qkv = self.Wqkv(hidden_states, adapter_data)
        query, kv = qkv.split(
            [
                self.head_size * self.num_heads,
                2 * self.head_size * self.num_key_value_heads,
            ],
            dim=1,
        )
        query = query.view(-1, self.num_heads, self.head_size)
        kv = kv.view(-1, 2, self.num_key_value_heads, self.head_size)

        self.rotary_emb(query, cos, sin)
        self.rotary_emb(torch.select(kv, dim=1, index=0), cos, sin)

        paged_attn.reshape_and_cache(
            kv[:, 0], kv[:, 1], kv_cache[0], kv_cache[1], slots
        )

        # output tensor
        attn_output = torch.empty_like(query)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            flash_attn.attention(
                query,
                torch.select(kv, dim=1, index=0),
                torch.select(kv, dim=1, index=1),
                attn_output,
                cu_seqlen_prefill,
                max_s,
                self.softmax_scale,
            )
        # Decode
        else:
            # kv_cache[1] => [num_blocks, num_heads, head_size, block_size]
            paged_attn.single_query_cached_kv_attention(
                attn_output,
                query,
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                input_lengths,
                max_s,
            )

        return self.out_proj(attn_output.view(-1, self.num_heads * self.head_size), adapter_data)


class PhiMLP(nn.Module):
    def __init__(self, prefix, config, weights, layer_id):
        super().__init__()
        act = config.activation_function
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh"
                if act in ["gelu_fast", "gelu_pytorch_tanh"]
                else "none",
            )
        )

        fc1 = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.fc1"],
            weights=weights,
            dim=0,
            bias=True,
        )

        out_size = fc1.linear.weight.shape[-1] * weights.process_group.size()
        self.fc1 = TensorParallelMultiAdapterLinear.load(
            fc1, layer_id, [MLP_FC1], sizes=[out_size], process_group=weights.process_group
        )
        self.fc2 = TensorParallelAdapterRowLinear.load(TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.fc2",
            weights=weights,
            bias=True,
        ), layer_id, MLP_FC2, process_group=weights.process_group)
        
    def forward(self, hidden_states, adapter_data):
        hidden_states = self.fc1(hidden_states, adapter_data)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states, adapter_data)
        return hidden_states


class FlashPhiLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"transformer.h.{layer_id}"
        
        self.ln = FastLayerNorm.load(
            prefix=f"{prefix}.ln", weights=weights, eps=config.layer_norm_epsilon
        )
        self.mixer = FlashPhiAttention(
            prefix=f"{prefix}.mixer", config=config, weights=weights, layer_id=layer_id,
        )
        self.mlp = PhiMLP(prefix=f"{prefix}.mlp", config=config, weights=weights, layer_id=layer_id)
        self.process_group = weights.process_group

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
        adapter_data,
    ):
        normed_hidden_states, _ = self.ln(hidden_states, residual=None)

        attn_output = self.mixer(
            normed_hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
            adapter_data,
        )

        mlp_output = self.mlp(normed_hidden_states, adapter_data)
        intermediate = mlp_output + attn_output

        if self.process_group.size() > 1:
            torch.distributed.all_reduce(intermediate, group=self.process_group)

        return intermediate + hidden_states, None


class FlashPhiModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embd = TensorParallelEmbedding(
            prefix="transformer.embd.wte", weights=weights
        )
        self.h = nn.ModuleList(
            [
                FlashPhiLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.n_layer)
            ]
        )

        self.gradient_checkpointing = False

        self.head_size = self.h[0].mixer.head_size
        self.num_heads = self.h[0].mixer.num_heads
        self.num_key_value_heads = self.h[0].mixer.num_key_value_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        adapter_data: AdapterBatchData,
    ) -> torch.Tensor:
        hidden_states = self.embd(input_ids)

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.h[0].mixer.rotary_emb.get_cos_sin(
            position_ids, max_s, hidden_states.dtype
        )

        residual = None
        for i, layer in enumerate(self.h):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache[i],
                block_tables,
                slots,
                input_lengths,
                max_s,
                adapter_data,
            )

        return hidden_states


class PhiCausalLMHead(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        prefix = "lm_head"
        self.ln = FastLayerNorm.load(
            prefix=f"{prefix}.ln", weights=weights, eps=config.layer_norm_epsilon
        )
        self.linear = TensorParallelAdapterRowLinear.load(TensorParallelHead.load(
            config,
            prefix=f"{prefix}.linear",
            weights=weights,
        ), 0, LM_HEAD, process_group=weights.process_group)
    
    def forward(self, hidden_states, adapter_data):
        hidden_states, _ = self.ln(hidden_states)
        hidden_states = self.linear(hidden_states, adapter_data)
        return hidden_states


class FlashPhiForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.transformer = FlashPhiModel(config, weights)
        self.lm_head = PhiCausalLMHead(config, weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        adapter_data: AdapterBatchData,
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.transformer(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
            adapter_data,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states, adapter_data)
        return logits
