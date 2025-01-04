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
    FlashLlamaForCausalLM,
    FlashLlamaLayer,
    LlamaConfig,
    LlamaMLP,
)
from lorax_server.utils.attention.common import Seqlen
from lorax_server.utils.layers import (
    TensorParallelAdapterRowLinear,
    TensorParallelColumnLinear,
    TensorParallelMultiAdapterLinear,
    TensorParallelRowLinear,
)
from lorax_server.utils.lora import (
    DOWN_PROJ,
    GATE_PROJ,
    UP_PROJ,
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


def load_attention(config, prefix: str, weights, layer_id):
    # Only defined in granite.
    bias = getattr(config, "attention_bias", False)

    head_size = config.hidden_size // config.num_attention_heads
    sizes = None
    prefixes = None

    prefixes = ["q_proj", "k_proj", "v_proj"]
    sizes = [
        head_size * config.num_attention_heads,
        head_size * config.num_key_value_heads,
        head_size * config.num_key_value_heads,
    ]
    base_layer = TensorParallelColumnLinear.load_multi(
        config,
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        dim=0,
        weights=weights,
        bias=bias,
    )

    return TensorParallelMultiAdapterLinear.load(
        base_layer=base_layer,
        layer_id=layer_id,
        layer_names=prefixes,
        sizes=sizes,
        process_group=weights.process_group,
    )


class FlashGraniteAttention(FlashLlamaAttention):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
        layer_id: int,
    ):
        super().__init__(prefix=prefix, config=config, weights=weights, layer_id=layer_id)

        self.softmax_scale = getattr(config, "attention_multiplier", self.head_size**-0.5)

        self.query_key_value = load_attention(config, prefix, weights, layer_id)


class GraniteMLP(LlamaMLP):
    def __init__(self, prefix, config, weights, layer_id):
        super().__init__(prefix, config, weights, layer_id)

        bias = getattr(config, "mlp_bias", False)

        # Fuse gate and up proj
        gate_up_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            dim=0,
            bias=bias,
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
                bias=bias,
            ),
            layer_id,
            DOWN_PROJ,
            process_group=weights.process_group,
        )


class FlashGraniteLayer(FlashLlamaLayer):
    def __init__(self, layer_id, prefix: str, config, weights):
        super().__init__(layer_id=layer_id, prefix=prefix, config=config, weights=weights)

        self.residual_multiplier = getattr(config, "residual_multiplier", None)

        self.self_attn = FlashGraniteAttention(
            prefix=f"{prefix}.self_attn",
            config=config,
            weights=weights,
            layer_id=layer_id,
        )

        self.mlp = GraniteMLP(prefix=f"{prefix}.mlp", config=config, weights=weights, layer_id=layer_id)

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
            seqlen,
            max_s,
            adapter_data,
        )

        if self.residual_multiplier is not None:
            attn_output *= self.residual_multiplier

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(attn_output, res)

        mlp_output = self.mlp(normed_attn_res_output, adapter_data)

        if self.residual_multiplier is not None:
            mlp_output *= self.residual_multiplier

        return mlp_output, attn_res


class FlashGraniteForCausalLM(FlashLlamaForCausalLM):
    def __init__(self, prefix: str, config, weights, create_layer_fn=None):
        super().__init__(prefix, config, weights, create_layer_fn=FlashGraniteLayer)

        embedding_multiplier = getattr(config, "embedding_multiplier", None)
        if embedding_multiplier is not None:
            self.embed_tokens.weight.data *= embedding_multiplier

        self.logits_scaling = getattr(config, "logits_scaling", None)

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
        logits, speculative_logits = super().forward(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            seqlen,
            max_s,
            adapter_data,
            prefill_cache_indices,
            lm_head_indices,
            cross_attention_states,
            skip_lm_head,
        )

        if self.logits_scaling is not None:
            logits /= self.logits_scaling
            if speculative_logits is not None:
                speculative_logits /= self.logits_scaling

        return logits, speculative_logits
