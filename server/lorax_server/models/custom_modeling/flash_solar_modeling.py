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

# from typing import List, Optional, Tuple

# Flash attention imports
import dropout_layer_norm
import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

# from lorax_server.adapters import AdapterBatchData
# from lorax_server.utils import flash_attn, paged_attn
# from lorax_server.utils.attention.common import Seqlen
from lorax_server.utils.layers import (
    # MultiAdapterHead,
    # PositionRotaryEmbedding,
    TensorParallelAdapterRowLinear,
    TensorParallelColumnLinear,
    # TensorParallelEmbedding,
    # TensorParallelHead,
    TensorParallelMultiAdapterLinear,
    TensorParallelRowLinear,
    # get_linear,
)
from lorax_server.utils.lora import (
    DOWN_PROJ,
    GATE_PROJ,
    # K_PROJ,
    # O_PROJ,
    # Q_PROJ,
    UP_PROJ,
    # V_PROJ,
)


class SolarConfig(PretrainedConfig):
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
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_scaling=None,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        sliding_window=2047,
        bskcn_1=[12, 20, 32, 44],
        bskcn_2=[20, 32],
        bskcn_3=[16, 24, 36, 48],
        bskcn_4=[28, 40],
        bskcn_tv=[0.9, 0.8],
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.sliding_window = sliding_window
        self.bskcn_1 = bskcn_1
        self.bskcn_2 = bskcn_2
        self.bskcn_3 = bskcn_3
        self.bskcn_4 = bskcn_4
        self.bskcn_tv = bskcn_tv

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class SolarRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        SolarRMSNorm is equivalent to T5LayerNorm
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
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

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


class SolarMLP(nn.Module):
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
            ),
            layer_id,
            DOWN_PROJ,
            process_group=weights.process_group,
        )
        self.intermediate_size = config.intermediate_size // weights.process_group.size()
