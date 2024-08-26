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

from typing import List, Optional, Tuple

import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN

from lorax_server.adapters import AdapterBatchData
from lorax_server.utils import flash_attn, paged_attention
from lorax_server.utils.layers import (
    FastLayerNorm,
    MultiAdapterHead,
    PositionRotaryEmbedding,
    TensorParallelAdapterRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelHead,
    TensorParallelMultiAdapterLinear,
    TensorParallelRowLinear,
)
from lorax_server.utils.lora import LM_HEAD

ATTN_Q_PROJ = "self_attn.q_proj"
ATTN_K_PROJ = "self_attn.k_proj"
ATTN_V_PROJ = "self_attn.v_proj"
ATTN_DENSE = "self_attn.dense"
MLP_FC1 = "mlp.fc1"
MLP_FC2 = "mlp.fc2"


def load_attention(config, prefix, weights, layer_id, head_dim, n_head, n_head_kv):
    base_layer = load_attention_multi(config, prefix, weights, head_dim, n_head, n_head_kv)
    return TensorParallelMultiAdapterLinear.load(
        base_layer,
        layer_id,
        [ATTN_Q_PROJ, ATTN_K_PROJ, ATTN_V_PROJ],
        sizes=[
            head_dim * n_head,
            head_dim * n_head_kv,
            head_dim * n_head_kv,
        ],
        process_group=weights.process_group,
    )


def load_attention_multi(config, prefix, weights, head_dim, n_head, n_head_kv):
    return TensorParallelColumnLinear.load_multi(
        config,
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
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
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        self.process_group = weights.process_group

        rope_theta = 10000
        config.max_position_embeddings = getattr(config, "n_positions", 2048)

        rotary_dim = int(config.partial_rotary_factor * (config.hidden_size // config.num_attention_heads))
        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=rotary_dim,
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

        self.qkv_proj = load_attention(
            config,
            prefix,
            weights,
            layer_id,
            self.head_size,
            self.num_heads,
            self.num_key_value_heads,
        )
        self.dense = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.dense",
                weights=weights,
                bias=True,
            ),
            layer_id,
            ATTN_DENSE,
            process_group=weights.process_group,
        )

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
        qkv = self.qkv_proj(hidden_states, adapter_data)
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

        paged_attention.reshape_and_cache(kv[:, 0], kv[:, 1], kv_cache[0], kv_cache[1], slots)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            attn_output = flash_attn.attention(
                query,
                torch.select(kv, dim=1, index=0),
                torch.select(kv, dim=1, index=1),
                kv_cache[0],
                kv_cache[1],
                cu_seqlen_prefill,
                max_s,
                self.softmax_scale,
            )
        # Decode
        else:
            # kv_cache[1] => [num_blocks, num_heads, head_size, block_size]
            attn_output = paged_attention.attention(
                query,
                kv_cache[0],
                kv_cache[1],
                self.num_key_value_heads,
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                input_lengths,
                max_s,
            )

        return self.dense(attn_output.view(-1, self.num_heads * self.head_size), adapter_data)


class PhiMLP(nn.Module):
    def __init__(self, prefix, config, weights, layer_id):
        super().__init__()
        act = config.hidden_act
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none",
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
        self.fc2 = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.fc2",
                weights=weights,
                bias=True,
            ),
            layer_id,
            MLP_FC2,
            process_group=weights.process_group,
        )

    def forward(self, hidden_states, adapter_data):
        hidden_states = self.fc1(hidden_states, adapter_data)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states, adapter_data)
        return hidden_states


class FlashPhiLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"

        self.input_layernorm = FastLayerNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.layer_norm_eps
        )
        self.self_attn = FlashPhiAttention(
            prefix=f"{prefix}.self_attn",
            config=config,
            weights=weights,
            layer_id=layer_id,
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
        normed_hidden_states, _ = self.input_layernorm(hidden_states, residual=None)

        attn_output = self.self_attn(
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
        self.embed_tokens = TensorParallelEmbedding(prefix="model.embed_tokens", weights=weights)
        self.layers = nn.ModuleList(
            [
                FlashPhiLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.final_layernorm = FastLayerNorm.load(
            prefix="model.final_layernorm", weights=weights, eps=config.layer_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

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
        hidden_states = self.embed_tokens(input_ids)

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(position_ids, max_s, hidden_states.dtype)

        residual = None
        for i, layer in enumerate(self.layers):
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

        hidden_states, _ = self.final_layernorm(hidden_states)
        return hidden_states


class FlashPhiForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.config = config

        self.model = FlashPhiModel(config, weights)
        self.lm_head = MultiAdapterHead.load(
            TensorParallelHead.load(
                config,
                prefix="lm_head",
                weights=weights,
            ),
            0,
            LM_HEAD,
            process_group=weights.process_group,
        )

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
        prefill_cache_indices: Optional[torch.Tensor] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.model(
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
        logits, speculative_logits = self.lm_head(hidden_states, adapter_data)
        return logits, speculative_logits
