from contextlib import nullcontext
from typing import Any, ContextManager, Optional, Type

import torch
from loguru import logger
from opentelemetry import trace
from transformers import AutoTokenizer
from transformers.models.xlm_roberta import XLMRobertaConfig

from lorax_server.adapters import AdapterBatchData
from lorax_server.models import Model
from lorax_server.models.custom_modeling.flash_roberta_modeling import (
    ATTN_K,
    ATTN_Q,
    ATTN_V,
    RobertaEmbeddings,
    RobertaLayer,
)
from lorax_server.models.types import FlashEmbeddingClassificationBatch
from lorax_server.pb.generate_pb2 import Embedding
from lorax_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
)
from lorax_server.utils.adapter import create_merged_weight_files
from lorax_server.utils.state import FLASH_INFER

tracer = trace.get_tracer(__name__)


class RobertaEncoder:
    def __init__(self, prefix, weights, device, dtype, config):
        self.layers = [
            RobertaLayer(f"{prefix}.layer.{i}", i, weights, device, dtype, config)
            for i in range(config.num_hidden_layers)
        ]

    def forward(self, hidden_states, cu_seqlens, max_s, adapter_data):
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, cu_seqlens, max_s, adapter_data)
        return hidden_states


class FlashRobertaModel(torch.nn.Module):
    def __init__(self, prefix, weights, device, dtype, config):
        super().__init__()
        self.config = config
        self.embeddings = RobertaEmbeddings(f"{prefix}.embeddings", weights, device, dtype, config)
        self.encoder = RobertaEncoder(f"{prefix}.encoder", weights, device, dtype, config)

    def forward(self, input_ids, token_type_ids, position_ids, cu_seqlens, max_s, adapter_data):
        embeddings = self.embeddings.forward(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder.forward(embeddings, cu_seqlens, max_s, adapter_data)
        return encoder_outputs[cu_seqlens[:-1]]


class FlashXlmRoberta(Model):
    def __init__(
        self,
        model_id: str,
        adapter_id: str,
        adapter_source: str,
        revision: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        merge_adapter_weights: bool = False,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashXlmRoberta is only available on GPU")

        self.device = device
        self.dtype = dtype

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer = tokenizer

        config = XLMRobertaConfig.from_pretrained(model_id)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        merged_weight_filenames = None
        if merge_adapter_weights:
            if len(adapter_id) > 0:
                logger.info(f"Merging adapter weights from adapter_id {adapter_id} into model weights.")
                # Need to pass the adapter source here
                merged_weight_filenames = create_merged_weight_files(
                    adapter_id, model_id, model_weight_filenames=filenames, adapter_source=adapter_source
                )
                self.dynamic_adapter_loading_enabled = False
                self.adapter_id = adapter_id
            else:
                raise ValueError("Cannot merge adapter weights without an adapter_id")

        weights = Weights(
            filenames,
            device,
            dtype,
            process_group=self.process_group,
            merged_weight_filenames=merged_weight_filenames,
        )
        prefix = "roberta"
        model = FlashRobertaModel(prefix, weights, device, dtype, config)

        self.hidden_size = config.hidden_size
        self.config = config

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        if FLASH_INFER:
            from lorax_server.utils.flashinfer_attention import create_prefill_state

            self.prefill_state = create_prefill_state(device=device)

        super(FlashXlmRoberta, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            rank=rank,
            world_size=world_size,
            requires_padding=False,
        )

    @property
    def batch_type(self) -> Type[FlashEmbeddingClassificationBatch]:
        return FlashEmbeddingClassificationBatch

    @property
    def supports_adapter_loading(self) -> bool:
        return True

    @property
    def supports_embeddings(self) -> bool:
        return True

    @property
    def supports_text_generation(self) -> bool:
        return False

    def warmup(self, batch: FlashEmbeddingClassificationBatch, max_new_tokens: int) -> int | None:
        # Note: This is meant to 1) preallocate the memory by doing a forward pass
        # and then just returning the max seqlen since for embeddings we are never generating
        _ = self.embed(batch)
        return batch.max_s

    def generate_token(self, batch: FlashEmbeddingClassificationBatch) -> None:
        if not self.supports_text_generation:
            raise NotImplementedError("This model does not support text generation")
        return None

    def adapter_target_to_layer(self) -> dict[str, tuple[str, torch.Tensor]]:
        layer_weights = {}

        prefix = "roberta.encoder.layer"
        for i, layer in enumerate(self.model.encoder.layers):
            layer_weights[(i, ATTN_Q)] = (
                f"{prefix}.{i}.attention.{ATTN_Q}",
                layer.attention.query_key_value,
            )
            layer_weights[(i, ATTN_K)] = (
                f"{prefix}.{i}.attention.{ATTN_K}",
                layer.attention.query_key_value,
            )
            layer_weights[(i, ATTN_V)] = (
                f"{prefix}.{i}.attention.{ATTN_V}",
                layer.attention.query_key_value,
            )
        return layer_weights

    @property
    def adapter_layers(self) -> list[str]:
        return [ATTN_Q, ATTN_V, ATTN_K]

    @property
    def default_traced_adapter_layers(self) -> list[str]:
        return [ATTN_Q, ATTN_V]

    def get_num_layers_for_type(self, layer_type: str) -> int:
        return len(self.model.encoder.layers)

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
        adapter_data = AdapterBatchData.from_meta(
            meta=batch.adapter_meta,
            weights=self.layer_to_adapter_weights,
            layer_to_lora_weights={},
            punica_wrapper=None,
            prefill=False,
            prefill_head_indices=None,
        )

        with self._forward_context(cu_seqlens=batch.cu_seqlens):
            embedding: torch.Tensor = self.model.forward(
                input_ids=batch.input_ids,
                token_type_ids=batch.token_type_ids,
                position_ids=batch.position_ids,
                cu_seqlens=batch.cu_seqlens,
                max_s=batch.max_s,
                adapter_data=adapter_data,
            )
        embedding = embedding.reshape(embedding.shape[0], -1)[:, : self.hidden_size]

        cpu_results = embedding.cpu().tolist()
        return cpu_results
