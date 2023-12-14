# Adapted from
# https://huggingface.co/microsoft/phi-1_5/blob/main/modeling_phi.py
# Copyright 2023 The vLLM team.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# BSD 3-Clause License
#
# Copyright (c) 2022, Tri Dao, trid@cs.stanford.edu.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from typing import Optional, List, Tuple

# Flash attention imports
import dropout_layer_norm

from lorax_server.utils import flash_attn
from lorax_server.utils import paged_attn
from lorax_server.utils.layers import (
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


class PhiConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        n_head=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_scaling=None,
        rope_theta=10000.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.n_head = n_head

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = n_head

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class PhiRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        PhiRMSNorm is equivalent to LlamaLayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if hidden_states.shape[-1] > 8192:
            if residual is not None:
                hidden_states += residual
            residual = hidden_states

            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )

            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)

            return self.weight * hidden_states, residual
        else:
            # faster post attention rms norm
            normed_hidden_states, res, *rest = dropout_layer_norm.dropout_add_ln_fwd(
                hidden_states,
                residual,
                self.weight,
                None,
                None,
                None,
                None,
                None,
                0.0,
                self.variance_epsilon,
                1.0,
                0,
                None,
                False,
                True,  # Activate RMSNorm
            )
            if res is None:
                res = hidden_states

            return normed_hidden_states, res
        

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
        self.projection_size = (self.head_size * config.n_head) // weights.process_group.size()
        self.process_group = weights.process_group

        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=self.head_size,
            base=config.rope_theta,
            device=weights.device,
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


        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = self.num_key_value_heads // weights.process_group.size()

        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

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

        n_inner = getattr(config, "n_inner", None)
        n_inner = n_inner if n_inner is not None else 4 * config.hidden_size

        fc1 = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.fc1"],
            weights=weights,
            dim=0,
            bias=True,
        )

        self.fc1 = TensorParallelMultiAdapterLinear.load(
            fc1, layer_id, [MLP_FC1], sizes=[n_inner], process_group=weights.process_group
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
        
        self.ln = PhiRMSNorm(
            prefix=f"{prefix}.ln", weights=weights, eps=config.layer_norm_epsilon
        )
        self.mixer = FlashPhiAttention(
            prefix=f"{prefix}.mixer", config=config, weights=weights, layer_id=layer_id,
        )
        self.mlp = PhiMLP(prefix=f"{prefix}.mlp", config=config, weights=weights, layer_id=layer_id)

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
        residual = hidden_states
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
        hidden_states = attn_output + mlp_output + residual

        return mlp_output, residual


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
                for layer_id in range(config.num_hidden_layers)
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
        self.ln = PhiRMSNorm(
            prefix=f"{prefix}.ln", weights=weights, eps=config.layer_norm_epsilon
        )
        self.linear = TensorParallelAdapterRowLinear.load(TensorParallelHead.load(
            config,
            prefix=f"{prefix}.linear",
            weights=weights,
        ), 0, LM_HEAD, process_group=weights.process_group)
    
    def forward(self, hidden_states, adapter_data):
        hidden_states = self.ln(hidden_states)
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
