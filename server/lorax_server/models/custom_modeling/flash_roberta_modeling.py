import torch
from torch import nn
from transformers.activations import ACT2FN

from lorax_server.utils.flash_attn import attention
from lorax_server.utils.layers import (
    FastLayerNorm,
    TensorParallelColumnLinear,
    TensorParallelMultiAdapterLinear,
)

ATTN_Q = "self.query"
ATTN_K = "self.key"
ATTN_V = "self.value"


class RobertaEmbeddings:
    def __init__(self, prefix, weights, device, dtype, config):
        self.word_embeddings_weight = weights.get_tensor(f"{prefix}.word_embeddings.weight").to(dtype).to(device)
        self.token_type_embeddings_weight = (
            weights.get_tensor(f"{prefix}.token_type_embeddings.weight").to(dtype).to(device)
        )

        if config.position_embedding_type == "absolute":
            self.position_embeddings_weight = (
                weights.get_tensor(f"{prefix}.position_embeddings.weight").to(dtype).to(device)
            )
        else:
            raise NotImplementedError("FlashRoberta only supports absolute position embeddings")
        self.pad_token_id = config.pad_token_id

        self.layer_norm = FastLayerNorm.load(prefix=f"{prefix}.LayerNorm", weights=weights, eps=config.layer_norm_eps)

    def forward(self, input_ids, token_type_ids, position_ids):
        # position numbers begin at pad_token_id + 1
        # see transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
        position_ids += self.pad_token_id + 1

        inputs_embeds = nn.functional.embedding(input_ids, self.word_embeddings_weight)
        token_type_embeds = nn.functional.embedding(token_type_ids, self.token_type_embeddings_weight)
        position_embeds = nn.functional.embedding(position_ids, self.position_embeddings_weight)

        inputs_embeds += position_embeds

        embeddings, _ = self.layer_norm.forward(inputs_embeds, token_type_embeds)
        return embeddings


class RobertaAttention:
    def __init__(self, prefix, layer_id, weights, device, dtype, config):
        self.query_key_value = RobertaAttention.load_attention(config, prefix, weights, layer_id)

        self.dense_weight = weights.get_tensor(f"{prefix}.output.dense.weight").T.to(dtype).to(device)
        self.dense_bias = weights.get_tensor(f"{prefix}.output.dense.bias").to(dtype).to(device)

        self.layer_norm = FastLayerNorm.load(
            prefix=f"{prefix}.output.LayerNorm", weights=weights, eps=config.layer_norm_eps
        )

        self.head_size = config.hidden_size // config.num_attention_heads
        self.softmax_scale = self.head_size**-0.5
        self.num_heads = config.num_attention_heads
        self.layer_id = layer_id

    @staticmethod
    def load_attention(config, prefix, weights, layer_id):
        config.quantize = None
        base_layer = RobertaAttention.load_attention_multi(config, prefix, weights)
        return TensorParallelMultiAdapterLinear.load(
            base_layer,
            layer_id,
            [ATTN_Q, ATTN_K, ATTN_V],
            sizes=[
                config.hidden_size,
                config.hidden_size,
                config.hidden_size,
            ],
            process_group=weights.process_group,
        )

    @staticmethod
    def load_attention_multi(config, prefix, weights):
        prefixes = [f"{prefix}.{ATTN_Q}", f"{prefix}.{ATTN_K}", f"{prefix}.{ATTN_V}"]
        return TensorParallelColumnLinear.load_multi(
            config,
            prefixes=prefixes,
            dim=0,
            weights=weights,
            bias=True,
        )

    def forward(self, hidden_states, cu_seqlens, max_s, adapter_data):
        residual = hidden_states

        qkv = self.query_key_value(hidden_states, adapter_data)
        q, k, v = qkv.view(-1, self.num_heads * 3, self.head_size).split(self.num_heads, dim=1)

        attn_output = attention(q, k, v, None, None, cu_seqlens, max_s, self.softmax_scale, causal=False)

        hidden_states = torch.addmm(
            self.dense_bias,
            attn_output.view(-1, self.num_heads * self.head_size),
            self.dense_weight,
        )
        hidden_states, _ = self.layer_norm.forward(hidden_states, residual)

        return hidden_states


class RobertaLayer:
    def __init__(self, prefix, layer_id, weights, device, dtype, config):
        self.attention = RobertaAttention(f"{prefix}.attention", layer_id, weights, device, dtype, config)

        self.intermediate_weight = weights.get_tensor(f"{prefix}.intermediate.dense.weight").T.to(dtype).to(device)
        self.intermediate_bias = weights.get_tensor(f"{prefix}.intermediate.dense.bias").to(dtype).to(device)

        act = config.hidden_act
        self.intermediate_act_fn = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none",
            )
        )

        self.output_weight = weights.get_tensor(f"{prefix}.output.dense.weight").T.to(dtype).to(device)
        self.output_bias = weights.get_tensor(f"{prefix}.output.dense.bias").to(dtype).to(device)
        self.layer_norm = FastLayerNorm.load(
            prefix=f"{prefix}.output.LayerNorm", weights=weights, eps=config.layer_norm_eps
        )

    def forward(self, hidden_states, cu_seqlens, max_s, adapter_data):
        hidden_states = self.attention.forward(hidden_states, cu_seqlens, max_s, adapter_data)
        residual = hidden_states

        hidden_states = torch.addmm(self.intermediate_bias, hidden_states, self.intermediate_weight)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = torch.addmm(
            self.output_bias,
            hidden_states,
            self.output_weight,
        )
        hidden_states, _ = self.layer_norm.forward(hidden_states, residual)
        return hidden_states
