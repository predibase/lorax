from contextlib import nullcontext
from typing import Any, ContextManager, Optional, Type

import torch
from opentelemetry import trace
from transformers import AutoTokenizer
from transformers.models.bert import BertConfig

from lorax_server.models import Model
from lorax_server.models.custom_modeling.flash_bert_modeling import BertEmbeddings, BertLayer
from lorax_server.models.types import FlashEmbeddingClassificationBatch
from lorax_server.pb.generate_pb2 import Embedding
from lorax_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
)
from lorax_server.utils.state import FLASH_INFER

tracer = trace.get_tracer(__name__)


def _format_prefix(prefix, name):
    if prefix is None:
        return name
    return f"{prefix}.{name}"


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
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(_format_prefix(prefix, "embeddings"), weights, device, dtype, config)
        self.encoder = BertEncoder(_format_prefix(prefix, "encoder"), weights, device, dtype, config)

    def forward(self, input_ids, token_type_ids, position_ids, cu_seqlens, max_s):
        embeddings = self.embeddings.forward(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder.forward(embeddings, cu_seqlens, max_s)

        return encoder_outputs[cu_seqlens[:-1]]


class FlashBertModelForClassification(torch.nn.Module):
    def __init__(self, prefix, weights, device, dtype, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(_format_prefix(prefix, "embeddings"), weights, device, dtype, config)
        self.encoder = BertEncoder(_format_prefix(prefix, "encoder"), weights, device, dtype, config)
        self.classifier_weight = weights.get_tensor("classifier.weight").to(dtype).to(device)
        self.classifier_bias = weights.get_tensor("classifier.bias").to(dtype).to(device)

    def forward(self, input_ids, token_type_ids, position_ids, cu_seqlens, max_s):
        embeddings = self.embeddings.forward(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder.forward(embeddings, cu_seqlens, max_s)
        batch_size = encoder_outputs.shape[0] // max_s
        encoder_outputs = encoder_outputs.reshape(batch_size, max_s, -1)
        logits = torch.nn.functional.linear(encoder_outputs, self.classifier_weight, self.classifier_bias)
        return logits


class FlashBert(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        classifcation_head: bool = False,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashBert is only available on GPU")

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
        prefix = "bert"
        if model_id in ["WhereIsAI/UAE-Large-V1", "BAAI/bge-base-en-v1.5"]:
            prefix = None
        if classifcation_head:
            model = FlashBertModelForClassification(prefix, weights, device, dtype, config)
        else:
            model = FlashBertModel(prefix, weights, device, dtype, config)

        self.classification_head_enabled = classifcation_head
        self.hidden_size = config.hidden_size
        self.config = config

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        if FLASH_INFER:
            from lorax_server.utils.flashinfer_attention import create_prefill_state

            self.prefill_state = create_prefill_state(device=device)

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
    def batch_type(self) -> Type[FlashEmbeddingClassificationBatch]:
        return FlashEmbeddingClassificationBatch

    @property
    def supports_embeddings(self) -> bool:
        return True

    @property
    def supports_text_generation(self) -> bool:
        return False

    @property
    def supports_classification(self) -> bool:
        return self.classification_head_enabled

    def warmup(self, batch: FlashEmbeddingClassificationBatch, max_new_tokens: int) -> int | None:
        # Note: This is meant to 1) preallocate the memory by doing a forward pass
        # and then just returning the max seqlen since for embeddings we are never generating
        _ = self.embed(batch)
        return batch.max_s

    def generate_token(self, batch: FlashEmbeddingClassificationBatch) -> None:
        if not self.supports_text_generation:
            raise NotImplementedError("This model does not support text generation")
        return None

    def _forward_context(
        self,
        *,
        cu_seqlens: torch.Tensor,
        state: Optional[Any] = None,
    ) -> ContextManager:
        if not FLASH_INFER:
            return nullcontext()

        from lorax_server.utils.flashinfer_attention import use_prefill_state

        return use_prefill_state(
            state=(state if state is not None else self.prefill_state),
            cu_seqlens=cu_seqlens,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
        )

    def forward(self, batch: FlashEmbeddingClassificationBatch):
        return self.embed(batch)

    @tracer.start_as_current_span("embed")
    def embed(self, batch: FlashEmbeddingClassificationBatch) -> Embedding:
        with self._forward_context(cu_seqlens=batch.cu_seqlens):
            embedding: torch.Tensor = self.model.forward(
                input_ids=batch.input_ids,
                token_type_ids=batch.token_type_ids,
                position_ids=batch.position_ids,
                cu_seqlens=batch.cu_seqlens,
                max_s=batch.max_s,
            )
        embedding = embedding.reshape(embedding.shape[0], -1)[:, : self.hidden_size]

        cpu_results = embedding.cpu().tolist()
        return cpu_results

    @tracer.start_as_current_span("classify")
    def classify(self, batch: FlashEmbeddingClassificationBatch):
        with self._forward_context(cu_seqlens=batch.cu_seqlens):
            logits: torch.Tensor = self.model.forward(
                input_ids=batch.input_ids,
                token_type_ids=batch.token_type_ids,
                position_ids=batch.position_ids,
                cu_seqlens=batch.cu_seqlens,
                max_s=batch.max_s,
            )
        probabilities = torch.nn.functional.softmax(logits, dim=2)
        confidence_scores, predictions = torch.max(probabilities, dim=2)
        predicted_token_class = [[self.config.id2label[t.item()] for t in prediction] for prediction in predictions]
        return predicted_token_class, confidence_scores.cpu().tolist()
