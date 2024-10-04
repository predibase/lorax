# coding=utf-8
# Copyright 2024 Cohere team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from typing import List, Optional, Tuple

import dropout_layer_norm
import rotary_emb
import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN

from lorax_server.adapters.weights import AdapterBatchData
from lorax_server.models.custom_modeling.utils import prepend
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
    get_linear,
)
from lorax_server.utils.lora import DOWN_PROJ, GATE_PROJ, K_PROJ, LM_HEAD, O_PROJ, Q_PROJ, UP_PROJ, V_PROJ


class CohereRotary(PositionRotaryEmbedding):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        q1 = query[..., ::2]
        q2 = query[..., 1::2]

        rotary_emb.apply_rotary(q1, q2, cos, sin, q1, q2, False)

        k1 = key[..., ::2]
        k2 = key[..., 1::2]

        rotary_emb.apply_rotary(k1, k2, cos, sin, k1, k2, False)


class CohereLayerNorm(nn.Module):
    def __init__(self, prefix, weights, eps):
        super().__init__()
        weight = weights.get_sharded(f"{prefix}.weight", dim=0)
        self.weight = nn.Parameter(weight)
        # Fake weights
        self.ones = weight.new_ones(weight.shape[1])
        self.eps = eps

    def forward(self, hidden_states):
        if hidden_states.shape[-1] > 8192:
            hidden_states = hidden_states.reshape(-1, self.weight.shape[0], self.weight.shape[1])
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            mean = hidden_states.mean(-1, keepdim=True)
            hidden_states_minus_mean = hidden_states - mean
            variance = hidden_states_minus_mean.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states_minus_mean * torch.rsqrt(variance + self.eps)
            hidden_states = self.weight.to(torch.float32) * hidden_states
            hidden_states = hidden_states.view(-1, self.weight.shape[1])
            return hidden_states.to(input_dtype)

        (
            hidden_states,
            *rest,
        ) = dropout_layer_norm.dropout_add_ln_fwd(
            hidden_states,
            None,
            self.ones,
            None,
            None,
            None,
            None,
            None,
            0.0,
            self.eps,
            1.0,
            0,
            None,
            False,
            False,
        )

        # Required to apply one weight matrix per head
        hidden_states = hidden_states.view(-1, self.weight.shape[0], self.weight.shape[1])
        hidden_states = self.weight * hidden_states
        hidden_states = hidden_states.view(-1, self.weight.shape[1])

        return hidden_states


def load_attention(config, prefix, weights, layer_id):
    base_layer = load_attention_multi(config, prefix, weights)
    head_size = config.hidden_size // config.num_attention_heads
    return TensorParallelMultiAdapterLinear.load(
        base_layer,
        layer_id,
        [Q_PROJ, K_PROJ, V_PROJ],
        sizes=[
            head_size * config.num_attention_heads,
            head_size * config.num_key_value_heads,
            head_size * config.num_key_value_heads,
        ],
        process_group=weights.process_group,
    )


def load_attention_multi(config, prefix, weights):
    if config.num_attention_heads != config.num_key_value_heads:
        return _load_gqa(config, prefix, weights)
    else:
        return TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=config.attention_bias,
        )


def _load_gqa(config, prefix: str, weights):
    assert config.hidden_size % config.num_attention_heads == 0
    assert config.num_attention_heads % weights.process_group.size() == 0

    weight = weights.get_multi_weights_col(
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        quantize=config.quantize,
        dim=0,
    )

    if config.quantize not in ["gptq", "awq"]:
        weight = weight.to(dtype=weights.dtype).to(device=weights.device)

        head_size = config.hidden_size // config.num_attention_heads
        num_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.num_key_value_heads // weights.process_group.size()
        assert list(weight.shape) == [
            (num_heads + 2 * num_key_value_heads) * head_size,
            config.hidden_size,
        ], f"{list(weight.shape)} != {[(num_heads + 2 * num_key_value_heads) * head_size, config.hidden_size]}"

    if config.attention_bias:
        w = [
            weights.get_sharded(f"{p}.bias", dim=0)
            for p in [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"]
        ]
        bias = torch.cat(w, dim=0).to(dtype=weights.dtype).to(device=weights.device)
    else:
        bias = None

    return TensorParallelColumnLinear(get_linear(weight, bias=bias, quantize=config.quantize))


class FlashCohereAttention(torch.nn.Module):
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

        self.rotary_emb = CohereRotary.static(
            config=config,
            dim=self.head_size,
            base=config.rope_theta,
            device=weights.device,
            dtype=weights.dtype,
        )

        self.softmax_scale = self.head_size**-0.5

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = config.num_key_value_heads // weights.process_group.size()

        self.query_key_value = load_attention(config, prefix, weights, layer_id)

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = CohereLayerNorm(
                prefix=f"{prefix}.q_norm",
                weights=weights,
                eps=config.layer_norm_eps,
            )
            self.k_norm = CohereLayerNorm(
                prefix=f"{prefix}.k_norm",
                weights=weights,
                eps=config.layer_norm_eps,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.o_proj = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.o_proj",
                weights=weights,
                bias=config.attention_bias,
                all_reduce=False,
            ),
            layer_id,
            O_PROJ,
            process_group=weights.process_group,
        )
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
        qkv = self.query_key_value(hidden_states, adapter_data)
        query, key, value = qkv.split(
            [
                self.head_size * self.num_heads,
                self.head_size * self.num_key_value_heads,
                self.head_size * self.num_key_value_heads,
            ],
            dim=1,
        )

        if self.use_qk_norm:
            query = query.reshape(-1, self.head_size)
            key = key.reshape(-1, self.head_size)
            query = self.q_norm(query.contiguous())
            key = self.k_norm(key.contiguous())

        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_key_value_heads, self.head_size)
        value = value.view(-1, self.num_key_value_heads, self.head_size)

        self.rotary_emb(query, key, cos, sin)

        paged_attention.reshape_and_cache(key, value, kv_cache[0], kv_cache[1], slots)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            attn_output = flash_attn.attention(
                query,
                key,
                value,
                kv_cache[0],
                kv_cache[1],
                cu_seqlen_prefill,
                max_s,
                self.softmax_scale,
            )
        # Decode
        else:
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

        return self.o_proj(attn_output.view(-1, self.num_heads * self.head_size), adapter_data)


class CohereMLP(nn.Module):
    def __init__(self, prefix, config, weights, layer_id: int):
        super().__init__()
        act = config.hidden_act
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate=("tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"),
            )
        )
        # Fuse gate and up proj
        gate_up_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            dim=0,
            bias=False,
        )
        self.gate_up_proj = TensorParallelMultiAdapterLinear.load(
            gate_up_proj,
            layer_id,
            [GATE_PROJ, UP_PROJ],
            sizes=[
                config.intermediate_size,
                config.intermediate_size,
            ],
            process_group=weights.process_group,
        )

        self.down_proj = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.down_proj",
                weights=weights,
                bias=False,
                all_reduce=False,
            ),
            layer_id,
            DOWN_PROJ,
            process_group=weights.process_group,
        )
        self.intermediate_size = config.intermediate_size // weights.process_group.size()

    def forward(self, hidden_states, adapter_data):
        gate_up_states = self.gate_up_proj(hidden_states, adapter_data)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.down_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1], adapter_data)


class FlashCohereLayer(nn.Module):
    def __init__(self, prefix: str, layer_id, config, weights):
        super().__init__()
        prefix = prepend(prefix, f"model.layers.{layer_id}")
        self.self_attn = FlashCohereAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights, layer_id=layer_id
        )
        self.mlp = CohereMLP(prefix=f"{prefix}.mlp", config=config, weights=weights, layer_id=layer_id)

        self.input_layernorm = FastLayerNorm.load_no_bias(
            prefix=f"{prefix}.input_layernorm",
            weights=weights,
            eps=config.layer_norm_eps,
        )
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
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
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
        output = attn_output + mlp_output

        if self.process_group.size() > 1:
            torch.distributed.all_reduce(output, group=self.process_group)

        return output, res


class FlashCohereModel(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorParallelEmbedding(prefix=prepend(prefix, "model.embed_tokens"), weights=weights)
        self.layers = nn.ModuleList(
            [
                FlashCohereLayer(
                    prefix,
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = FastLayerNorm.load_no_bias(
            prefix=prepend(prefix, "model.norm"), weights=weights, eps=config.layer_norm_eps
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

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class FlashCohereForCausalLM(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()
        self.config = config

        self.model = FlashCohereModel(prefix, config, weights)
        try:
            lm_head = TensorParallelHead.load(
                config,
                prefix=prepend(prefix, "lm_head"),
                weights=weights,
            )
        except RuntimeError:
            lm_head = TensorParallelHead.load(
                config,
                prefix=prepend(prefix, "model.embed_tokens"),
                weights=weights,
            )
        self.lm_head = MultiAdapterHead.load(
            lm_head,
            0,
            LM_HEAD,
            process_group=weights.process_group,
        )
        self.logit_scale = config.logit_scale

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
        logits *= self.logit_scale
        if speculative_logits is not None:
            speculative_logits *= self.logit_scale
        return logits, speculative_logits
