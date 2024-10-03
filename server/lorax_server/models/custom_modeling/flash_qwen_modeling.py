# coding=utf-8
# Adapted from
# https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
# Copyright (c) Alibaba Cloud.
# LICENSE: https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE

from typing import List, Optional, Tuple

# Flash attention imports
import dropout_layer_norm
import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from lorax_server.adapters import AdapterBatchData
from lorax_server.models.custom_modeling.utils import prepend
from lorax_server.utils import flash_attn, paged_attention
from lorax_server.utils.layers import (
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

ATTN_C_ATTN = "attn.c_attn"
ATTN_C_PROJ = "attn.c_proj"
MLP_W1 = "mlp.w1"
MLP_W2 = "mlp.w2"
MLP_C_PROJ = "mlp.c_proj"


class QwenConfig(PretrainedConfig):
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
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class QwenRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        QwenRMSNorm is equivalent to LlamaLayerNorm
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


def load_attention(config, prefix, weights, layer_id):
    projection_size = (config.hidden_size // config.num_attention_heads) * config.num_attention_heads
    base_layer = load_attention_multi(config, prefix, weights, projection_size)
    return TensorParallelMultiAdapterLinear.load(
        base_layer,
        layer_id,
        [ATTN_C_ATTN],
        sizes=[
            3 * projection_size,
        ],
        process_group=weights.process_group,
    )


def load_attention_multi(config, prefix, weights, projection_size):
    return TensorParallelColumnLinear.load_multi(
        config,
        prefixes=[
            (f"{prefix}.c_attn", (0, projection_size)),
            (f"{prefix}.c_attn", (projection_size, projection_size)),
            (f"{prefix}.c_attn", (2 * projection_size, projection_size)),
        ],
        dim=0,
        weights=weights,
        bias=True,
    )


class FlashQwenAttention(torch.nn.Module):
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
        self.projection_size = (self.head_size * config.num_attention_heads) // weights.process_group.size()
        self.process_group = weights.process_group

        self.rotary_emb = PositionRotaryEmbedding.static(
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
        self.num_key_value_heads = self.num_heads

        self.c_attn = load_attention(config, prefix, weights, layer_id)

        self.c_proj = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.c_proj",
                weights=weights,
                bias=False,
            ),
            layer_id,
            ATTN_C_PROJ,
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
        qkv = self.c_attn(hidden_states, adapter_data)
        query, kv = qkv.split(
            [
                self.projection_size,
                2 * self.projection_size,
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

        return self.c_proj(attn_output.view(-1, self.num_heads * self.head_size), adapter_data)


class QwenMLP(nn.Module):
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
            prefixes=[f"{prefix}.w2", f"{prefix}.w1"],
            weights=weights,
            dim=0,
            bias=False,
        )
        self.gate_up_proj = TensorParallelMultiAdapterLinear.load(
            gate_up_proj,
            layer_id,
            [MLP_W2, MLP_W1],
            sizes=[
                config.intermediate_size // 2,
                config.intermediate_size // 2,
            ],
            process_group=weights.process_group,
        )

        self.c_proj = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.c_proj",
                weights=weights,
                bias=False,
            ),
            layer_id,
            MLP_C_PROJ,
            process_group=weights.process_group,
        )
        self.intermediate_size = config.intermediate_size // weights.process_group.size()

    def forward(self, hidden_states, adapter_data):
        gate_up_states = self.gate_up_proj(hidden_states, adapter_data)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size // 2)
        return self.c_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1], adapter_data)


class FlashQwenLayer(nn.Module):
    def __init__(self, prefix: str, layer_id, config, weights):
        super().__init__()
        prefix = prepend(prefix, f"transformer.h.{layer_id}")
        self.attn = FlashQwenAttention(
            prefix=f"{prefix}.attn",
            config=config,
            weights=weights,
            layer_id=layer_id,
        )
        self.mlp = QwenMLP(prefix=f"{prefix}.mlp", config=config, weights=weights, layer_id=layer_id)

        self.ln_1 = QwenRMSNorm(prefix=f"{prefix}.ln_1", weights=weights, eps=config.layer_norm_epsilon)
        self.ln_2 = QwenRMSNorm(
            prefix=f"{prefix}.ln_2",
            weights=weights,
            eps=config.layer_norm_epsilon,
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
        input_lengths,
        max_s,
        adapter_data,
    ):
        normed_hidden_states, res = self.ln_1(hidden_states, residual)

        # Self Attention
        attn_output = self.attn(
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

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.ln_2(attn_output, res)

        mlp_output = self.mlp(normed_attn_res_output, adapter_data)

        return mlp_output, attn_res


class FlashQwenModel(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.wte = TensorParallelEmbedding(prefix=prepend(prefix, "transformer.wte"), weights=weights)
        self.h = nn.ModuleList(
            [
                FlashQwenLayer(
                    prefix,
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = QwenRMSNorm(
            prefix=prepend(prefix, "transformer.ln_f"), weights=weights, eps=config.layer_norm_epsilon
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
        hidden_states = self.wte(input_ids)

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.h[0].attn.rotary_emb.get_cos_sin(position_ids, max_s, hidden_states.dtype)

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

        hidden_states, _ = self.ln_f(hidden_states, residual)

        return hidden_states


class FlashQwenForCausalLM(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()
        self.config = config

        self.transformer = FlashQwenModel(prefix, config, weights)
        self.lm_head = MultiAdapterHead.load(
            TensorParallelHead.load(
                config,
                prefix=prepend(prefix, "lm_head"),
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
        logits, speculative_logits = self.lm_head(hidden_states, adapter_data)
        return logits, speculative_logits
