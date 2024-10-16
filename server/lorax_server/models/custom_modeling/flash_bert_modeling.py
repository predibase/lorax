import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.bert import BertConfig
from transformers.models.distilbert import DistilBertConfig

from lorax_server.utils.flash_attn import attention
from lorax_server.utils.layers import FastLayerNorm

# NOTE: This implementation of flashbert was based on the
# huggingface/text-embeddings-inference implementation of flashbert here:
# https://github.com/huggingface/text-embeddings-inference/blob/cb802a25d43fe6078c715b49652a3bc8a7d5aac8/backends/python/server/text_embeddings_server/models/flash_bert.py


class DistilBertEmbeddings:
    def __init__(self, prefix, weights, device, dtype, config: DistilBertConfig):
        self.word_embeddings_weight = weights.get_tensor(f"{prefix}.word_embeddings.weight").to(dtype).to(device)
        self.position_embeddings_weight = (
            weights.get_tensor(f"{prefix}.position_embeddings.weight").to(dtype).to(device)
        )

        self.layer_norm = FastLayerNorm.load(prefix=f"{prefix}.LayerNorm", weights=weights, eps=1e-12)

    def forward(self, input_ids, token_type_ids, position_ids):
        inputs_embeds = nn.functional.embedding(input_ids, self.word_embeddings_weight)
        position_embeds = nn.functional.embedding(position_ids, self.position_embeddings_weight)

        inputs_embeds += position_embeds

        embeddings, _ = self.layer_norm.forward(inputs_embeds)
        return embeddings


class DistilBertAttention:
    def __init__(self, prefix, weights, device, dtype, config: DistilBertConfig):
        query_weight = weights.get_tensor(f"{prefix}.attention.q_lin.weight")
        query_bias = weights.get_tensor(f"{prefix}.attention.q_lin.bias")
        key_weight = weights.get_tensor(f"{prefix}.attention.k_lin.weight")
        key_bias = weights.get_tensor(f"{prefix}.attention.k_lin.bias")
        value_weight = weights.get_tensor(f"{prefix}.attention.v_lin.weight")
        value_bias = weights.get_tensor(f"{prefix}.attention.v_lin.bias")

        self.qkv_weight = torch.cat([query_weight, key_weight, value_weight]).T.to(dtype).to(device)
        self.qkv_bias = torch.cat([query_bias, key_bias, value_bias]).to(dtype).to(device)

        self.dense_weight = weights.get_tensor(f"{prefix}.attention.out_lin.weight").T.to(dtype).to(device)
        self.dense_bias = weights.get_tensor(f"{prefix}.attention.out_lin.bias").to(dtype).to(device)

        self.layer_norm = FastLayerNorm.load(prefix=f"{prefix}.sa_layer_norm", weights=weights, eps=1e-12)

        self.head_size = config.hidden_size // config.num_attention_heads
        self.softmax_scale = self.head_size**-0.5
        self.num_heads = config.num_attention_heads

    def forward(self, hidden_states, cu_seqlens, max_s):
        residual = hidden_states

        qkv = torch.addmm(self.qkv_bias, hidden_states, self.qkv_weight)
        q, k, v = qkv.view(-1, self.num_heads * 3, self.head_size).split(self.num_heads, dim=1)

        attn_output = attention(q, k, v, None, None, cu_seqlens, max_s, self.softmax_scale, causal=False)

        hidden_states = torch.addmm(
            self.dense_bias,
            attn_output.view(-1, self.num_heads * self.head_size),
            self.dense_weight,
        )
        hidden_states, _ = self.layer_norm.forward(hidden_states, residual)

        return hidden_states


class DistilBertLayer:
    def __init__(self, prefix, weights, device, dtype, config: DistilBertConfig):
        self.attention = DistilBertAttention(prefix, weights, device, dtype, config)

        self.intermediate_weight = weights.get_tensor(f"{prefix}.ffn.lin1.weight").T.to(dtype).to(device)
        self.intermediate_bias = weights.get_tensor(f"{prefix}.ffn.lin1.bias").to(dtype).to(device)

        act = config.activation
        self.intermediate_act_fn = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none",
            )
        )

        self.output_weight = weights.get_tensor(f"{prefix}.ffn.lin2.weight").T.to(dtype).to(device)
        self.output_bias = weights.get_tensor(f"{prefix}.ffn.lin2.bias").to(dtype).to(device)
        self.layer_norm = FastLayerNorm.load(prefix=f"{prefix}.output_layer_norm", weights=weights, eps=1e-12)

    def forward(self, hidden_states, cu_seqlens, max_s):
        hidden_states = self.attention.forward(hidden_states, cu_seqlens, max_s)
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


class BertEmbeddings:
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        self.word_embeddings_weight = weights.get_tensor(f"{prefix}.word_embeddings.weight").to(dtype).to(device)
        self.token_type_embeddings_weight = (
            weights.get_tensor(f"{prefix}.token_type_embeddings.weight").to(dtype).to(device)
        )

        if config.position_embedding_type == "absolute":
            self.position_embeddings_weight = (
                weights.get_tensor(f"{prefix}.position_embeddings.weight").to(dtype).to(device)
            )
        else:
            raise NotImplementedError("FlashBert only supports absolute position embeddings")

        self.layer_norm = FastLayerNorm.load(prefix=f"{prefix}.LayerNorm", weights=weights, eps=config.layer_norm_eps)

    def forward(self, input_ids, token_type_ids, position_ids):
        inputs_embeds = nn.functional.embedding(input_ids, self.word_embeddings_weight)
        token_type_embeds = nn.functional.embedding(token_type_ids, self.token_type_embeddings_weight)
        position_embeds = nn.functional.embedding(position_ids, self.position_embeddings_weight)

        inputs_embeds += position_embeds

        embeddings, _ = self.layer_norm.forward(inputs_embeds, token_type_embeds)
        return embeddings


class BertAttention:
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        query_weight = weights.get_tensor(f"{prefix}.self.query.weight")
        query_bias = weights.get_tensor(f"{prefix}.self.query.bias")
        key_weight = weights.get_tensor(f"{prefix}.self.key.weight")
        key_bias = weights.get_tensor(f"{prefix}.self.key.bias")
        value_weight = weights.get_tensor(f"{prefix}.self.value.weight")
        value_bias = weights.get_tensor(f"{prefix}.self.value.bias")

        self.qkv_weight = torch.cat([query_weight, key_weight, value_weight]).T.to(dtype).to(device)
        self.qkv_bias = torch.cat([query_bias, key_bias, value_bias]).to(dtype).to(device)

        self.dense_weight = weights.get_tensor(f"{prefix}.output.dense.weight").T.to(dtype).to(device)
        self.dense_bias = weights.get_tensor(f"{prefix}.output.dense.bias").to(dtype).to(device)

        self.layer_norm = FastLayerNorm.load(
            prefix=f"{prefix}.output.LayerNorm", weights=weights, eps=config.layer_norm_eps
        )

        self.head_size = config.hidden_size // config.num_attention_heads
        self.softmax_scale = self.head_size**-0.5
        self.num_heads = config.num_attention_heads

    def forward(self, hidden_states, cu_seqlens, max_s):
        residual = hidden_states

        qkv = torch.addmm(self.qkv_bias, hidden_states, self.qkv_weight)
        q, k, v = qkv.view(-1, self.num_heads * 3, self.head_size).split(self.num_heads, dim=1)

        attn_output = attention(q, k, v, None, None, cu_seqlens, max_s, self.softmax_scale, causal=False)

        hidden_states = torch.addmm(
            self.dense_bias,
            attn_output.view(-1, self.num_heads * self.head_size),
            self.dense_weight,
        )
        hidden_states, _ = self.layer_norm.forward(hidden_states, residual)

        return hidden_states


class BertLayer:
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        self.attention = BertAttention(f"{prefix}.attention", weights, device, dtype, config)

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

    def forward(self, hidden_states, cu_seqlens, max_s):
        hidden_states = self.attention.forward(hidden_states, cu_seqlens, max_s)
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
