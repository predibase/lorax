from typing import Optional, Type

import torch
from opentelemetry import trace
from torch import nn
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
from transformers.models.bert import BertConfig

from lorax_server.models import Model
from lorax_server.models.types import FlashEmbeddingBatch
from lorax_server.pb.generate_pb2 import Embedding
from lorax_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
)
from lorax_server.utils.flash_attn import attention
from lorax_server.utils.layers import FastLayerNorm

tracer = trace.get_tracer(__name__)


# NOTE: This implementation of flashbert was based on the
# huggingface/text-embeddings-inference implementation of flashbert here:
# https://github.com/huggingface/text-embeddings-inference/blob/cb802a25d43fe6078c715b49652a3bc8a7d5aac8/backends/python/server/text_embeddings_server/models/flash_bert.py


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

        attn_output = torch.empty_like(q)
        attention(q, k, v, attn_output, cu_seqlens, max_s, self.softmax_scale)

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


class BertEncoder:
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        self.layers = [
            BertLayer(f"{prefix}.layer.{i}", weights, device, dtype, config) for i in range(config.num_hidden_layers)
        ]

    def forward(self, hidden_states, cu_seqlens, max_s):
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, cu_seqlens, max_s)
        return hidden_states


class FlashBertModel(torch.nn.Module):
    def __init__(self, weights, device, dtype, config: BertConfig):
        super().__init__()
        self.embeddings = BertEmbeddings("embeddings", weights, device, dtype, config)
        self.encoder = BertEncoder("encoder", weights, device, dtype, config)

    def forward(self, input_ids, token_type_ids, position_ids, cu_seqlens, max_s):
        embeddings = self.embeddings.forward(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder.forward(embeddings, cu_seqlens, max_s)

        return encoder_outputs[cu_seqlens[:-1]]


class FlashBert(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashSantacoderSharded is only available on GPU")

        self.device = device
        self.dtype = dtype

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer = tokenizer

        config = BertConfig.from_pretrained(model_id)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames,
            device,
            dtype,
            process_group=self.process_group,
        )
        model = FlashBertModel(weights, device, dtype, config)

        self.hidden_size = config.hidden_size

        super(FlashBert, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            requires_padding=False,
        )

    @property
    def batch_type(self) -> Type[FlashEmbeddingBatch]:
        return FlashEmbeddingBatch

    @property
    def supports_embeddings(self) -> bool:
        return True

    @property
    def supports_text_generation(self) -> bool:
        return False

    def warmup(self, batch: FlashEmbeddingBatch, max_new_tokens: int) -> int | None:
        return 42  # no-op for now

    def generate_token(self, batch: FlashEmbeddingBatch) -> None:
        if not self.supports_text_generation:
            raise NotImplementedError("This model does not support text generation")
        return None

    def forward(self, batch: FlashEmbeddingBatch):
        return self.embed(batch)

    @tracer.start_as_current_span("embed")
    def embed(self, batch: FlashEmbeddingBatch) -> Embedding:
        embedding = self.model.forward(
            input_ids=batch.input_ids,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
            cu_seqlens=batch.cu_seqlens,
            max_s=batch.max_s,
        )
        cpu_results = embedding.view(-1).tolist()

        return Embedding(values=cpu_results[: self.hidden_size])

    def tokenize_to_batch(self, inputs) -> FlashEmbeddingBatch:
        tokens = self.tokenizer(inputs, return_token_type_ids=True)
        num_tokens = len(tokens["input_ids"])
        position_ids = range(num_tokens)
        return FlashEmbeddingBatch(
            input_ids=torch.tensor(tokens["input_ids"], dtype=torch.int32, device=self.device),
            token_type_ids=torch.tensor(tokens["token_type_ids"], dtype=torch.int32, device=self.device),
            position_ids=torch.tensor(position_ids, dtype=torch.int32, device=self.device),
            cu_seqlens=torch.tensor([0, num_tokens], dtype=torch.int32, device=self.device),
            max_s=num_tokens,
            size=1,
        )
