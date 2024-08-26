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

from typing import List, Optional, Tuple

import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2 import GPT2Config

from lorax_server.adapters import AdapterBatchData
from lorax_server.utils import flash_attn, paged_attention
from lorax_server.utils.layers import (
    FastLayerNorm,
    TensorParallelAdapterRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelMultiAdapterLinear,
    TensorParallelRowLinear,
)

ATTN_C_ATTN = "attn.c_attn"
ATTN_C_PROJ = "attn.c_proj"
MLP_C_FC = "mlp.c_fc"
MLP_C_PROJ = "mlp.c_proj"
LM_HEAD = "lm_head"


def load_attention_multi(config, prefix, weights, fan_in_fan_out=False):
    return TensorParallelColumnLinear.load_multi(
        config,
        prefixes=[f"{prefix}.c_attn"],
        dim=0,
        weights=weights,
        bias=True,
        fan_in_fan_out=fan_in_fan_out,
    )


def load_attention(config, prefix, weights, layer_id, layer_names, fan_in_fan_out=False):
    base_layer = load_attention_multi(config, prefix, weights, fan_in_fan_out=fan_in_fan_out)
    projection_size = config.n_embd
    return TensorParallelMultiAdapterLinear.load(
        base_layer,
        layer_id,
        layer_names,
        sizes=[
            3 * projection_size,
        ],
        process_group=weights.process_group,
    )


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
            self.softmax_scale = self.head_dim**-0.5
        else:
            self.softmax_scale = 1.0

        if config.add_cross_attention:
            raise ValueError("Cross attention in GPT-2 is not supported.")

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_id
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        self.c_attn = load_attention(config, prefix, weights, layer_id, [ATTN_C_ATTN], fan_in_fan_out=True)
        self.c_proj = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.c_proj",
                weights=weights,
                bias=True,
                fan_in_fan_out=True,
            ),
            layer_id,
            ATTN_C_PROJ,
            process_group=weights.process_group,
        )

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

        self.kv_head_mapping = torch.arange(0, self.num_heads, dtype=torch.int32, device=weights.device)
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
        adapter_data,
    ):
        qkv = self.c_attn(hidden_states, adapter_data)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_size)

        paged_attention.reshape_and_cache(qkv[:, 1], qkv[:, 2], kv_cache[0], kv_cache[1], slots)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            attn_output = flash_attn.attention(
                qkv[:, 0],
                qkv[:, 1],
                qkv[:, 2],
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
                qkv[:, 0],
                kv_cache[0],
                kv_cache[1],
                self.num_key_value_heads,
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                input_lengths,
                max_s,
            )

        attn_output = attn_output.view(-1, self.num_heads * self.head_size)
        out = self.c_proj(attn_output, adapter_data)
        return out


class GPT2MLP(nn.Module):
    def __init__(self, config, prefix, weights, layer_id):
        super().__init__()

        c_fc = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.c_fc",
            weights=weights,
            bias=True,
            fan_in_fan_out=True,
        )
        # https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Config.n_inner
        n_inner = config.n_inner if config.n_inner is not None else config.n_embd * 4
        self.c_fc = TensorParallelMultiAdapterLinear.load(
            c_fc, layer_id, [MLP_C_FC], sizes=[n_inner], process_group=weights.process_group
        )

        self.c_proj = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.c_proj",
                weights=weights,
                bias=True,
                fan_in_fan_out=True,
            ),
            layer_id,
            MLP_C_PROJ,
            process_group=weights.process_group,
        )

        self.act = ACT2FN[config.activation_function]

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        adapter_data: AdapterBatchData,
    ) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states, adapter_data)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states, adapter_data)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()

        layer_norm_eps = config.layer_norm_epsilon
        prefix = f"h.{layer_id}"

        self.ln_1 = FastLayerNorm.load(prefix=f"{prefix}.ln_1", weights=weights, eps=layer_norm_eps)
        self.attn = FlashGPT2Attention(config, prefix=f"{prefix}.attn", weights=weights, layer_id=layer_id)
        self.ln_2 = FastLayerNorm.load(prefix=f"{prefix}.ln_2", weights=weights, eps=layer_norm_eps)

        self.mlp = GPT2MLP(config, prefix=f"{prefix}.mlp", weights=weights, layer_id=layer_id)
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
        adapter_data,
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
            adapter_data,
        )

        # residual connection
        hidden_states = attn_outputs + residual

        residual = hidden_states
        hidden_states, _ = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states, adapter_data)
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

        self.h = nn.ModuleList([GPT2Block(layer_id, config, weights) for layer_id in range(config.num_hidden_layers)])
        self.ln_f = FastLayerNorm.load(
            prefix="ln_f",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )

        self.gradient_checkpointing = False

        self.head_size = self.h[0].attn.head_size
        self.num_heads = self.h[0].attn.num_heads
        self.num_key_value_heads = self.h[0].attn.num_key_value_heads

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
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        for i, layer in enumerate(self.h):
            hidden_states = layer(
                hidden_states,
                cu_seqlen_prefill,
                kv_cache[i],
                block_tables,
                slots,
                input_lengths,
                max_s,
                adapter_data,
            )

        hidden_states, _ = self.ln_f(hidden_states)
        return hidden_states


class FlashGPT2ForCausalLM(FlashGPT2PreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.config = config
        self.transformer = FlashGPT2Model(config, weights)
        self.wte_t = self.transformer.wte.weight.T.contiguous()

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

        # lm_head reuses the weights of the embedding layer
        # https://github.com/huggingface/transformers/issues/6291
        logits = hidden_states @ self.wte_t
        logits = logits[:, : self.transformer.config.vocab_size]
        return logits, None
