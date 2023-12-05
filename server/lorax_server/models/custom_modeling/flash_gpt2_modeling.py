# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2 import GPT2Config
from typing import Optional, List, Tuple

from lorax_server.utils import flash_attn
from lorax_server.utils import paged_attn
from lorax_server.utils.layers import (
    FastConv1D,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelHead,
    FastLayerNorm,
    PositionRotaryEmbedding,
    get_linear,
)

from lorax_server.utils.lora import AdapterBatchData


class FlashGPT2Attention(torch.nn.Module):
    def __init__(self, config, prefix, weights, layer_id):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        if self.scale_attn_weights:
            self.softmax_scale = self.head_dim ** -0.5
        else:
            self.softmax_scale = 1.0
        
        if config.add_cross_attention:
            raise ValueError("Cross attention in GPT-2 is not supported.")

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_id
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        self.c_attn = FastConv1D.load(config, prefix=f"{prefix}.c_attn", weights=weights)
        self.c_proj = FastConv1D.load(config, prefix=f"{prefix}.c_proj", weights=weights)

        self.pruned_heads = set()

        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()

        self.kv_head_mapping = torch.arange(
            0, self.num_heads, dtype=torch.int32, device=weights.device
        )
        self.num_key_value_heads = self.num_heads


    def forward(
        self,
        hidden_states,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
    ):
        qkv = self.c_attn(hidden_states)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_size)

        paged_attn.reshape_and_cache(
            qkv[:, 1], qkv[:, 2], kv_cache[0], kv_cache[1], slots
        )

        # output tensor
        attn_output = torch.empty_like(qkv[:, 0])

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            flash_attn.attention(
                qkv[:, 0],
                qkv[:, 1],
                qkv[:, 2],
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
                qkv[:, 0],
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                input_lengths,
                max_s,
            )

        attn_output = attn_output.view(-1, self.num_heads * self.head_size)
        out = self.c_proj(attn_output)
        return out


class GPT2MLP(nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        self.c_fc = FastConv1D.load(config, prefix=f"{prefix}.c_fc", weights=weights)
        self.c_proj = FastConv1D.load(config, prefix=f"{prefix}.c_proj", weights=weights)
        self.act = ACT2FN[config.activation_function]

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()

        layer_norm_eps = config.layer_norm_epsilon
        prefix = f"h.{layer_id}"

        self.ln_1 = FastLayerNorm.load(
            prefix=f"{prefix}.ln_1", weights=weights, eps=layer_norm_eps
        )
        self.attn = FlashGPT2Attention(
            config, prefix=f"{prefix}.attn", weights=weights, layer_id=layer_id
        )
        self.ln_2 = FastLayerNorm.load(
            prefix=f"{prefix}.ln_2", weights=weights, eps=layer_norm_eps
        )

        self.mlp = GPT2MLP(config, prefix=f"{prefix}.mlp", weights=weights)
        self.process_group = weights.process_group

    def forward(
        self,
        hidden_states,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
    ):
        residual = hidden_states
        hidden_states, _ = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )

        # residual connection
        hidden_states = attn_outputs + residual

        residual = hidden_states
        hidden_states, _ = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = feed_forward_hidden_states + residual

        return hidden_states


class FlashGPT2PreTrainedModel(PreTrainedModel):
    config_class = GPT2Config
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = None


class FlashGPT2Model(FlashGPT2PreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.config = config

        self.embed_dim = config.hidden_size

        self.wte = TensorParallelEmbedding(prefix="wte", weights=weights)
        self.wpe = TensorParallelEmbedding(prefix="wpe", weights=weights)

        self.layers = nn.ModuleList(
            [
                GPT2Block(layer_id, config, weights)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = FastLayerNorm.load(
            prefix="ln_f",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].attn.head_size
        self.num_heads = self.layers[0].attn.num_heads
        self.num_key_value_heads = self.layers[0].attn.num_key_value_heads

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
    ) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                cu_seqlen_prefill,
                kv_cache[i],
                block_tables,
                slots,
                input_lengths,
                max_s,
            )

        hidden_states, _ = self.ln_f(hidden_states)
        return hidden_states


class FlashGPT2ForCausalLM(FlashGPT2PreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.model = FlashGPT2Model(config, weights)

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
        adapter_data: AdapterBatchData,  # TODO: plumb this through
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]

        # lm_head reuses the weights of the embedding layer
        # https://github.com/huggingface/transformers/issues/6291
        logits = hidden_states @ self.model.wte.weight.T
        logits = logits[:, :self.model.config.vocab_size]
        return logits
