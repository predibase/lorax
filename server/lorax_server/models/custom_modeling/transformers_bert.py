from typing import Optional, Tuple, Union, List

import torch
import torch.utils.checkpoint

from transformers.activations import ACT2FN
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert import BertConfig

import torch
import torch.nn.functional as F
from transformers import BertConfig
from typing import Optional, Tuple, Union
from lorax_server.utils.layers import FastLayerNorm
from lorax_server.utils.flash_attn import attention

class BertEmbeddings:
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        self.word_embeddings_weight = weights.get_tensor(f"{prefix}.word_embeddings.weight").to(dtype).to(device)
        self.token_type_embeddings_weight = weights.get_tensor(f"{prefix}.token_type_embeddings.weight").to(dtype).to(device)
        self.position_embeddings_weight = weights.get_tensor(f"{prefix}.position_embeddings.weight").to(dtype).to(device)
        self.layer_norm = FastLayerNorm.load(prefix=f"{prefix}.LayerNorm", weights=weights, eps=config.layer_norm_eps)

    def forward(self, input_ids, token_type_ids, position_ids):
        inputs_embeds = F.embedding(input_ids, self.word_embeddings_weight)
        token_type_embeds = F.embedding(token_type_ids, self.token_type_embeddings_weight)
        position_embeds = F.embedding(position_ids, self.position_embeddings_weight)
        embeddings = inputs_embeds + token_type_embeds + position_embeds
        embeddings, _ = self.layer_norm.forward(embeddings)
        embeddings = embeddings.unsqueeze(0)
        return embeddings

class BertSdpaSelfAttention:
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_weight = weights.get_tensor(f"{prefix}.self.query.weight").to(dtype).to(device)
        self.query_bias = weights.get_tensor(f"{prefix}.self.query.bias").to(dtype).to(device)
        self.key_weight = weights.get_tensor(f"{prefix}.self.key.weight").to(dtype).to(device)
        self.key_bias = weights.get_tensor(f"{prefix}.self.key.bias").to(dtype).to(device)
        self.value_weight = weights.get_tensor(f"{prefix}.self.value.weight").to(dtype).to(device)
        self.value_bias = weights.get_tensor(f"{prefix}.self.value.bias").to(dtype).to(device)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None) -> Tuple[torch.Tensor]:
        bsz, tgt_len, _ = hidden_states.size()
        query = F.linear(hidden_states, self.query_weight, self.query_bias)
        key = F.linear(hidden_states, self.key_weight, self.key_bias)
        value = F.linear(hidden_states, self.value_weight, self.value_bias)

        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # attn_output = attention(query, key, value, None, None, cu_seqlens, max_s, self.softmax_scale, causal=False)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            dropout_p=0,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)

        return attn_output

class BertSelfOutput:
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        self.dense_weight = weights.get_tensor(f"{prefix}.output.dense.weight").to(dtype).to(device)
        self.dense_bias = weights.get_tensor(f"{prefix}.output.dense.bias").to(dtype).to(device)
        self.layer_norm = FastLayerNorm.load(prefix=f"{prefix}.output.LayerNorm", weights=weights, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = F.linear(hidden_states, self.dense_weight, self.dense_bias)
        hidden_states, _ = self.layer_norm.forward(hidden_states.squeeze(0), input_tensor.squeeze(0))
        hidden_states = hidden_states.unsqueeze(0)
        return hidden_states

class BertAttention:
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        self.self = BertSdpaSelfAttention(prefix, weights, device, dtype, config)
        self.output = BertSelfOutput(prefix, weights, device, dtype, config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        self_output = self.self.forward(hidden_states, attention_mask)
        attention_output = self.output.forward(self_output, hidden_states)
        return attention_output

class BertIntermediate:
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        self.dense_weight = weights.get_tensor(f"{prefix}.intermediate.dense.weight").to(dtype).to(device)
        self.dense_bias = weights.get_tensor(f"{prefix}.intermediate.dense.bias").to(dtype).to(device)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.linear(hidden_states, self.dense_weight, self.dense_bias)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput:
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        self.dense_weight = weights.get_tensor(f"{prefix}.output.dense.weight").to(dtype).to(device)
        self.dense_bias = weights.get_tensor(f"{prefix}.output.dense.bias").to(dtype).to(device)
        self.layer_norm = FastLayerNorm.load(prefix=f"{prefix}.output.LayerNorm", weights=weights, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = F.linear(hidden_states, self.dense_weight, self.dense_bias)
        hidden_states, _ = self.layer_norm.forward(hidden_states.squeeze(0), input_tensor.squeeze(0))
        hidden_states = hidden_states.unsqueeze(0)
        return hidden_states

class BertLayer:
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        self.attention = BertAttention(f"{prefix}.attention", weights, device, dtype, config)
        self.intermediate = BertIntermediate(prefix, weights, device, dtype, config)
        self.output = BertOutput(prefix, weights, device, dtype, config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        attention_output = self.attention.forward(hidden_states, attention_mask)
        intermediate_output = self.intermediate.forward(attention_output)
        layer_output = self.output.forward(intermediate_output, attention_output)
        return layer_output

class BertEncoder:
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        self.layers = [BertLayer(f"{prefix}.layer.{i}", weights, device, dtype, config) for i in range(config.num_hidden_layers)]

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, attention_mask)
        return hidden_states

class BertModel:
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        self.embeddings = BertEmbeddings(f"{prefix}.embeddings", weights, device, dtype, config)
        self.encoder = BertEncoder(f"{prefix}.encoder", weights, device, dtype, config)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        cu_seqlens: List[int],
        max_s: int,
    ) -> Tuple[torch.Tensor]:
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)

        attention_mask = torch.ones_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings.forward(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder.forward(embedding_output, extended_attention_mask)
        encoder_outputs = encoder_outputs.squeeze(0)
        batch_size = encoder_outputs.shape[0] // max_s
        encoder_outputs = encoder_outputs.reshape(batch_size, max_s, -1)

        return encoder_outputs

class BertForTokenClassification(torch.nn.Module):
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        super().__init__()
        self.bert = BertModel(prefix, weights, device, dtype, config)
        self.config = config
        self.classifier_weight = weights.get_tensor(f"classifier.weight").to(dtype).to(device)
        self.classifier_bias = weights.get_tensor(f"classifier.bias").to(dtype).to(device)
        self.num_labels = config.num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        cu_seqlens: List[int],
        max_s: int,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        encoder_outputs = self.bert.forward(input_ids, token_type_ids, position_ids, cu_seqlens, max_s)
        logits = F.linear(encoder_outputs, self.classifier_weight, self.classifier_bias)
        return logits