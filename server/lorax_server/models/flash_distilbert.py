from typing import Optional, Type

import torch
from opentelemetry import trace
from transformers import AutoTokenizer
from transformers.models.distilbert import DistilBertConfig

from lorax_server.models import Model
from lorax_server.models.custom_modeling.flash_bert_modeling import DistilBertEmbeddings, DistilBertLayer
from lorax_server.models.types import FlashEmbeddingBatch
from lorax_server.pb.generate_pb2 import Embedding
from lorax_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
)

tracer = trace.get_tracer(__name__)


class DistilBertEncoder:
    def __init__(self, prefix, weights, device, dtype, config: DistilBertConfig):
        self.layers = [
            DistilBertLayer(f"{prefix}.layer.{i}", weights, device, dtype, config)
            for i in range(config.num_hidden_layers)
        ]

    def forward(self, hidden_states, cu_seqlens, max_s):
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, cu_seqlens, max_s)
        return hidden_states


class FlashDistilBertModel(torch.nn.Module):
    def __init__(self, weights, device, dtype, config: DistilBertConfig):
        super().__init__()
        self.embeddings = DistilBertEmbeddings("distilbert.embeddings", weights, device, dtype, config)
        self.encoder = DistilBertEncoder("distilbert.transformer", weights, device, dtype, config)

    def forward(self, input_ids, token_type_ids, position_ids, cu_seqlens, max_s):
        embeddings = self.embeddings.forward(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder.forward(embeddings, cu_seqlens, max_s)

        return encoder_outputs[cu_seqlens[:-1]]

class FlashDistilBertModelForClassification(torch.nn.Module):
    def __init__(self, weights, device, dtype, config: DistilBertConfig):
        super().__init__()
        self.embeddings = DistilBertEmbeddings("distilbert.embeddings", weights, device, dtype, config)
        self.encoder = DistilBertEncoder("distilbert.transformer", weights, device, dtype, config)
        self.classifier_weight = weights.get_tensor("classifier.weight").to(dtype).to(device)
        self.classifier_bias = weights.get_tensor("classifier.bias").to(dtype).to(device)

    def forward(self, input_ids, token_type_ids, position_ids, cu_seqlens, max_s):
        embeddings = self.embeddings.forward(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder.forward(embeddings, cu_seqlens, max_s)
        logits = torch.nn.functional.linear(encoder_outputs, self.classifier_weight, self.classifier_bias)
        return logits


class FlashDistilBert(Model):
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
            raise NotImplementedError("FlashDistilBert is only available on GPU")

        self.device = device
        self.dtype = dtype

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer = tokenizer

        config = DistilBertConfig.from_pretrained(model_id)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames,
            device,
            dtype,
            process_group=self.process_group,
        )
        if classifcation_head:
            model = FlashDistilBertModelForClassification(weights, device, dtype, config)
        else:
            model = FlashDistilBertModel(weights, device, dtype, config)

        self.hidden_size = config.hidden_size
        self.classification_head_enabled = classifcation_head
        self.config = config

        super(FlashDistilBert, self).__init__(
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

    @property
    def supports_classification(self) -> bool:
        return self.classification_head_enabled

    def warmup(self, batch: FlashEmbeddingBatch, max_new_tokens: int) -> int | None:
        # Note: This is meant to 1) preallocate the memory by doing a forward pass
        # and then just returning the max seqlen since for embeddings we are never generating
        # _ = self.embed(batch)
        return batch.max_s

    def generate_token(self, batch: FlashEmbeddingBatch) -> None:
        if not self.supports_text_generation:
            raise NotImplementedError("This model does not support text generation")
        return None

    def forward(self, batch: FlashEmbeddingBatch):
        return self.embed(batch)
    
    def construct_labels(self, batch: FlashEmbeddingBatch, logits, labels):
        sentence = batch.strings[0]
        input_ids = batch.input_ids[0]
        scores = logits
        pre_entities = []
        for idx, token_scores in enumerate(scores):
            # Filter special_tokens
            if special_tokens_mask[idx]:
                continue

            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            start_ind = None
            end_ind = None
            is_subword = False

            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            pre_entities.append(pre_entity)
        return pre_entities


    @tracer.start_as_current_span("embed")
    def embed(self, batch: FlashEmbeddingBatch) -> Embedding:
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
    def classify(self, batch: FlashEmbeddingBatch):
        logits: torch.Tensor = self.model.forward(
            input_ids=batch.input_ids,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
            cu_seqlens=batch.cu_seqlens,
            max_s=batch.max_s,
        )

 
        predictions = torch.argmax(logits, dim=1)
        predicted_token_class = [self.config.id2label[t.item()] for t in predictions]

        return predicted_token_class
