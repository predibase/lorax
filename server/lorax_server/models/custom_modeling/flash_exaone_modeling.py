
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.configuration_utils import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from lorax_server.utils.attention.common import Seqlen
from lorax_server.adapters.weights import AdapterBatchData
from transformers.modeling_utils import PreTrainedModel

from lorax_server.utils.layers import (
    MultiAdapterHead,
    PositionRotaryEmbedding,
    TensorParallelAdapterRowLinear,
    TensorParallelEmbedding,
    TensorParallelHead,
    TensorParallelRowLinear,
)
from lorax_server.utils.lora import (
    DOWN_PROJ,
    GATE_PROJ,
    K_PROJ,
    LM_HEAD,
    O_PROJ,
    Q_PROJ,
    UP_PROJ,
    V_PROJ,
)
from lorax_server.utils import flash_attn, paged_attention


class ExaoneConfig(PretrainedConfig):
    model_type = "exaone"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=102400,
        max_position_embeddings=2048,
        hidden_size=2048,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        intermediate_size=None,
        activation_function="silu",
        rope_theta=10000.0,
        rope_scaling=None,
        embed_dropout=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        if intermediate_size:
            self.intermediate_size = intermediate_size
        else:
            self.intermediate_size = hidden_size * 4
        self.activation_function = activation_function
        self.embed_dropout = embed_dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class ExaoneRMSNorm(torch.nn.Module):
    def __init__(self, prefix: str, weights, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(weights.get_tensor(f"{prefix}.weight"))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class ExaoneFlashAttention(nn.Module):
    def __init__(self, prefix: str, config: ExaoneConfig, weights, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout_rate = config.attention_dropout
        self.softmax_scale = self.head_dim**-0.5

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.k_proj = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.k_proj",
                weights=weights,
                bias=False,
            ),
            layer_idx,
            K_PROJ,
            process_group=weights.process_group,
        )
        self.v_proj = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.v_proj",
                weights=weights,
                bias=False,
            ),
            layer_idx,
            V_PROJ,
            process_group=weights.process_group,
        )
        self.q_proj = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.q_proj",
                weights=weights,
                bias=False,
            ),
            layer_idx,
            Q_PROJ,
            process_group=weights.process_group,
        )
        self.out_proj = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.out_proj",
                weights=weights,
                bias=False,
            ),
            layer_idx,
            O_PROJ,
            process_group=weights.process_group,
        )
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_key_value_groups)
        self.rotary = PositionRotaryEmbedding.static(
            config=config,
            dim=config.head_dim,
            base=config.rope_theta,
            device=weights.device,
            dtype=weights.dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        seqlen,
        max_s,
        adapter_data,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        query = self.q_proj(hidden_states, adapter_data)
        key = self.k_proj(hidden_states, adapter_data)
        value = self.v_proj(hidden_states, adapter_data)
        query = query.view(-1, self.num_heads, self.head_dim)
        key = key.view(-1, self.num_key_value_heads, self.head_dim)
        value  = value.view(-1, self.num_key_value_heads, self.head_dim)
        query = self.rotary(query, cos, sin)
        key = self.rotary(key, cos, sin)
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
                seqlen,
                max_s,
            )

        return self.out_proj(attn_output.view(-1, self.num_heads * self.head_dim), adapter_data)


class ExaoneAttention(nn.Module):
    def __init__(self, prefix: str, config, weights, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        prefix = f"{prefix}.attention"
        self.attention = ExaoneFlashAttention(prefix, config, weights, self.layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: list[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        max_s: int,
        adapter_data: AdapterBatchData,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        return self.attention(
            hidden_states,
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


class ExaoneGatedMLP(nn.Module):
    def __init__(self, prefix: str, config, weights, layer_idx: int):
        super().__init__()
        self.config = config
        self.c_fc_0 = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.c_fc_0",
                weights=weights,
                bias=False,
            ),
            layer_idx,
            GATE_PROJ,
            process_group=weights.process_group,
        )
        self.c_fc_1 = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.c_fc_1",
                weights=weights,
                bias=False,
            ),
            layer_idx,
            UP_PROJ,
            process_group=weights.process_group,
        )
        self.c_proj = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.c_proj",
                weights=weights,
                bias=False,
            ),
            layer_idx,
            DOWN_PROJ,
            process_group=weights.process_group,
        )
        self.act = ACT2FN[config.activation_function]

    def forward(self, hidden_states, adapter_data):
        output_proj = self.c_proj(self.act(self.c_fc_0(hidden_states, adapter_data)) * self.c_fc_1(hidden_states, adapter_data), adapter_data)
        return output_proj


class ExaoneBlock(nn.Module):
    def __init__(self, prefix: str, config, weights, layer_id):
        super().__init__()
        self.config = config
        self.ln_1 = ExaoneRMSNorm(f"{prefix}.ln_1", weights, eps=config.layer_norm_epsilon)
        self.attn = ExaoneAttention(f"{prefix}.attn", config, weights, layer_id)
        self.ln_2 = ExaoneRMSNorm(f"{prefix}.ln_2", weights, eps=config.layer_norm_epsilon)
        self.mlp = ExaoneGatedMLP(f"{prefix}.mlp", config, weights, layer_idx=layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: list[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        max_s: int,
        adapter_data: AdapterBatchData,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        hidden_states = self.attn(
            hidden_states,
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
        # residual connection
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states, adapter_data)

        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs


class ExaonePreTrainedModel(PreTrainedModel):
    config_class = ExaoneConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["ExaoneBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


class ExaoneModel(ExaonePreTrainedModel):
    def __init__(self, prefix: str, config: ExaoneConfig, weights):
        super().__init__(config)
        self.config = config
        self.embed_dim = config.hidden_size
        self.wte = TensorParallelEmbedding(
            prefix=f"{prefix}.wte",
            weights=weights,
        )
        self.drop = nn.Dropout(float(config.embed_dropout))
        self.h = nn.ModuleList([ExaoneBlock(f"{prefix}.h.{i}", config, weights, layer_id=i)
                                for i in range(config.num_layers)])
        self.ln_f = ExaoneRMSNorm(f"{prefix}.ln_f", weights, eps=config.layer_norm_epsilon)   
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: list[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        max_s: int,
        adapter_data: AdapterBatchData,
    ) -> torch.Tensor:

        inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.h[0].attn.attention.rotary.get_cos_sin(position_ids, max_s, hidden_states.dtype)

        for i, block in enumerate(self.h):
            outputs = block(
                hidden_states,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache[i],
                block_tables,
                slots,
                seqlen,
                max_s,
                adapter_data,
            )

            hidden_states = outputs

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class ExaoneForCausalLM(ExaonePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, prefix: str, config, weights):
        super().__init__(config)
        self.transformer = ExaoneModel(f"{prefix}.transformer" if prefix else "transformer", config, weights)
        self.lm_head = MultiAdapterHead.load(
            TensorParallelHead.load(
                config,
                prefix=f"{prefix}.transformer.wte" if prefix else "transformer.wte",
                weights=weights,
            ),
            0,
            LM_HEAD,
            process_group=weights.process_group,
        )
        self.config = config
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: list[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        max_s: int,
        adapter_data: AdapterBatchData,
        prefill_cache_indices: Optional[torch.Tensor] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
        skip_lm_head: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        transformer_outputs = self.transformer(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            seqlen,
            max_s,
            adapter_data,
        )
        hidden_states = transformer_outputs
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        if skip_lm_head:
            return hidden_states, None
        logits, speculative_logits = self.lm_head(hidden_states, adapter_data)
        return logits, speculative_logits
