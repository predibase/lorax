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

# Flash attention imports
import torch
import torch.distributed
from transformers.modeling_rope_utils import rope_config_validation

from lorax_server.adapters import AdapterBatchData
from lorax_server.models.custom_modeling.flash_llama_modeling import (
    FlashLlamaAttention,
    FlashLlamaModel, 
    LlamaConfig, 
    FlashLlamaLayer,
)
from lorax_server.utils.attention.common import Seqlen
from lorax_server.utils.layers import (
    MultiAdapterHead,
    TensorParallelEmbedding,
    TensorParallelHead,
)
from lorax_server.utils.lora import (
    LM_HEAD,
)


class GraniteConfig(LlamaConfig):
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        embedding_multiplier=1.0,
        logits_scaling=1.0,
        residual_multiplier=1.0,
        attention_multiplier=1.0,
        **kwargs,
    ):
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        self.embedding_multiplier = embedding_multiplier
        self.logits_scaling = logits_scaling
        self.residual_multiplier = residual_multiplier
        self.attention_multiplier = attention_multiplier
        
        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            rope_scaling=rope_scaling,
            rope_theta=rope_theta,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        rope_config_validation(self)


class FlashGraniteAttention(FlashLlamaAttention):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
        layer_id: int,
    ):
        super().__init__(prefix=prefix, config=config, weights=weights, layer_id=layer_id)
        self.softmax_scale = config.attention_multiplier


class FlashGraniteLayer(FlashLlamaLayer):
    def __init__(self, layer_id, prefix: str, config, weights):
        super().__init__(layer_id=layer_id, prefix=prefix, config=config, weights=weights)
        self.residual_multiplier = config.residual_multiplier
        self.self_attn = FlashGraniteAttention(
            prefix=f"{prefix}.self_attn",
            config=config,
            weights=weights,
            layer_id=layer_id,
        )
        

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
        seqlen,
        max_s,
        adapter_data,
        cross_attention_states,
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            seqlen,
            max_s,
            adapter_data,
        )

        attn_output = res + attn_output * self.residual_multiplier

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(attn_output, res)

        mlp_output = self.mlp(normed_attn_res_output, adapter_data)
        mlp_output = attn_res + mlp_output * self.residual_multiplier  # main diff with Llama

        return mlp_output, attn_res


class FlashGraniteModel(FlashLlamaModel):
    def __init__(self, prefix: str, config, weights, create_layer_fn):
        super().__init__(prefix=prefix, config=config, weights=weights, create_layer_fn=FlashGraniteLayer)
        self.embedding_multiplier = config.embedding_multiplier


    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        max_s: int,
        adapter_data: AdapterBatchData,
        prefill_cache_indices: Optional[torch.Tensor],
        cross_attention_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        inputs_embeds = inputs_embeds * self.embedding_multiplier  # main diff with Llama
        hidden_states = inputs_embeds

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
                seqlen,
                max_s,
                adapter_data,
                cross_attention_states,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class FlashGraniteForCausalLM(torch.nn.Module):
    def __init__(self, prefix: str, config, weights, create_layer_fn=None):
        super().__init__()
        self.config = config

        self.embed_tokens = TensorParallelEmbedding(
            prefix=("model.embed_tokens" if not prefix else f"{prefix}.model.embed_tokens"),
            weights=weights,
        )
        self.model = FlashGraniteModel(prefix, config, weights, create_layer_fn)
        if config.tie_word_embeddings:
            suffix = "model.embed_tokens"
        else:
            suffix = "lm_head"

        self.lm_head = MultiAdapterHead.load(
            TensorParallelHead.load(
                config,
                prefix=suffix if not prefix else f"{prefix}.{suffix}",
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
        seqlen: Seqlen,
        max_s: int,
        adapter_data: AdapterBatchData,
        prefill_cache_indices: Optional[torch.Tensor] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        skip_lm_head: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.model(
            inputs_embeds,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            seqlen,
            max_s,
            adapter_data,
            prefill_cache_indices,
            cross_attention_states,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]

        if skip_lm_head:
            return hidden_states, None

        logits, speculative_logits = self.lm_head(hidden_states, adapter_data)
        logits = logits / self.config.logits_scaling  # main diff with Llama
        return logits, speculative_logits
