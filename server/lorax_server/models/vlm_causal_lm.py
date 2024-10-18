from io import BytesIO
from typing import Iterable, List, Optional, Tuple, Type

import torch
import torch.distributed
from loguru import logger
from opentelemetry import trace
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerBase
from transformers.image_processing_utils import select_best_resolution

from lorax_server.adapters.weights import AdapterBatchData
from lorax_server.models.flash_causal_lm import (
    FlashCausalLM,
    FlashCausalLMBatch,
    block_tables_to_ragged,
)
from lorax_server.pb import generate_pb2
from lorax_server.utils.attention.common import Seqlen
from lorax_server.utils.state import PREFIX_CACHING
from lorax_server.utils.tokenizer import TokenizerManager

tracer = trace.get_tracer(__name__)

IDEFICS2_FAKE_TOKEN = "<fake_token_around_image>"
IDEFICS2_IMAGE_TOKEN = "<image>"


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (height, width).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise ValueError("grid_pinpoints should be a list of tuples or lists")

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def image_text_replacement(processor, image_input, config, image_id: int) -> str:
    if config.model_type == "idefics2":
        image_seq_len = 64
        image_str = f"{IDEFICS2_FAKE_TOKEN}{IDEFICS2_IMAGE_TOKEN * image_seq_len}{IDEFICS2_FAKE_TOKEN}"
        if processor.image_processor.do_image_splitting:
            image_str *= 5
        return image_str
    elif config.model_type == "llava_next":
        height, width = image_input["image_sizes"][image_id]
        num_features = get_number_of_features(height, width, config)

        logger.info(
            f"Found {num_features} features in image of resolution {height}x{width}",
        )
        return "<image>" * num_features

    elif config.model_type == "paligemma":
        return "<image>" * config.text_config.num_image_tokens
    else:
        raise RuntimeError(f"Unknown config {config.model_type} for multimodal")


def image_text_replacement_fixup(config, text: str) -> str:
    if config.model_type == "idefics2":
        return text.replace(f"{IDEFICS2_FAKE_TOKEN}{IDEFICS2_FAKE_TOKEN}", IDEFICS2_FAKE_TOKEN)
    return text


def get_unpadded_features(
    original_height: int,
    original_width: int,
    npatches: int,
    num_patch_height: int,
    num_patch_width: int,
) -> Tuple[int, int]:
    current_height = npatches * num_patch_height
    current_width = npatches * num_patch_width

    aspect_ratio: float = original_width / original_height
    current_aspect_ratio: float = current_width / current_height

    if aspect_ratio > current_aspect_ratio:
        new_height = (original_height * current_width) // original_width
        padding = (current_height - new_height) // 2
        current_height = current_height - (2 * padding)
    else:
        new_width = (original_width * current_height) // original_height
        padding = (current_width - new_width) // 2
        current_width = current_width - (2 * padding)

    unpadded_features = current_height * current_width
    newline_features = current_height
    return (unpadded_features, newline_features)


def get_number_of_features(height: int, width: int, config) -> int:
    # From config
    # Hardcoded for CLIP for now
    # image_grid_pinpoints = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
    image_grid_pinpoints = config.image_grid_pinpoints
    image_size = config.vision_config.image_size
    patch_size = config.vision_config.patch_size

    assert image_size % patch_size == 0

    npatches = image_size // patch_size

    # Dimensions are intentionally swapped to be bug-compatible with
    # upstream: https://github.com/LLaVA-VL/LLaVA-NeXT/issues/59
    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
        [height, width],
        image_grid_pinpoints,
        image_size,
    )
    unpadded_features, newline_features = get_unpadded_features(
        height, width, npatches, num_patch_height, num_patch_width
    )
    # The base patch covers the entire image
    base_features = npatches**2
    return unpadded_features + newline_features + base_features


class VlmCausalLMBatch(FlashCausalLMBatch):
    pixel_values: Optional[List[torch.Tensor]]
    pixel_attention_mask: Optional[List[torch.Tensor]]
    image_sizes: Optional[List[Tuple[int, int]]]

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches):
        batch = super(VlmCausalLMBatch, cls).concatenate(batches)
        batch.pixel_values = None
        batch.pixel_attention_mask = None
        batch.image_sizes = None
        return batch

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]):
        batch = super().filter(request_ids)
        batch.pixel_values = None
        batch.pixel_attention_mask = None
        batch.image_sizes = None
        return batch

    @classmethod
    def batch_tokenized_inputs(cls, requests: Iterable[generate_pb2.Request], tokenizer, processor, config):
        # Process images first. We need all of them so that the processor
        # can make the image splits the same size. And we need the final
        # sizes to insert correct number of image tokens.
        images = []
        for r in requests:
            for chunk in r.tokenized_inputs.input_chunks:
                chunk_type = chunk.WhichOneof("chunk")
                if chunk_type == "text":
                    pass
                elif chunk_type == "image":
                    image = Image.open(BytesIO(chunk.image.data))
                    if config.model_type == "llava_next":
                        images.append(image)
                    else:
                        images.append([image])
                else:
                    raise RuntimeError(f"Invalid chunk type {chunk_type}")

        if images:
            image_inputs = processor.image_processor(images, return_tensors="pt")
        else:
            image_inputs = None

        batch_inputs = []
        max_truncation = 0
        image_id = 0
        for r in requests:
            full_text = ""
            for chunk in r.tokenized_inputs.input_chunks:
                chunk_type = chunk.WhichOneof("chunk")
                if chunk_type == "text":
                    full_text += chunk.text
                elif chunk_type == "image":
                    full_text += image_text_replacement(processor, image_inputs, config, image_id)
                    image_id += 1

            full_text = image_text_replacement_fixup(config, full_text)

            batch_inputs.append(full_text)
            max_truncation = max(max_truncation, r.truncate)

        batch_tokenized_inputs = tokenizer(
            batch_inputs,
            truncation=True,
            max_length=max_truncation,
            add_special_tokens=not config.model_type == "paligemma",
        )["input_ids"]

        return batch_tokenized_inputs, image_inputs

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        tokenizers: TokenizerManager,
        processor,
        config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "VlmCausalLMBatch":
        batch_tokenized_inputs, image_inputs = cls.batch_tokenized_inputs(pb.requests, tokenizer, processor, config)

        batch = super().from_pb(
            pb, tokenizer, tokenizers, processor, config, dtype, device, batch_tokenized_inputs=batch_tokenized_inputs
        )
        if image_inputs is not None:
            batch.pixel_values = image_inputs["pixel_values"].to(device=device)
            if "pixel_attention_mask" in image_inputs:
                batch.pixel_attention_mask = image_inputs["pixel_attention_mask"].to(device=device)
            else:
                batch.pixel_attention_mask = None
            if "image_sizes" in image_inputs:
                batch.image_sizes = image_inputs["image_sizes"].to(device=device)
            else:
                batch.image_sizes = None
        else:
            batch.pixel_values = None
            batch.pixel_attention_mask = None
            batch.image_sizes = None
        return batch


class VlmCausalLM(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        model_class,
        *,
        processor_class=AutoProcessor,
        processor_kwargs=None,
        batch_class=VlmCausalLMBatch,
        adapter_id: str,
        adapter_source: str,
        revision: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        if PREFIX_CACHING:
            raise NotImplementedError("Vlm do not work with prefix caching yet")

        if processor_kwargs is None:
            processor_kwargs = {}

        processor = processor_class.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **processor_kwargs,
        )

        self.batch_class = batch_class

        super().__init__(
            model_id=model_id,
            model_cls=model_class,
            dtype=dtype,
            revision=revision,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            processor=processor,
            trust_remote_code=trust_remote_code,
            # FIXME: VLM do not work with context chunking yet
            supports_chunking=False,
            **kwargs,
        )

    @property
    def batch_type(self) -> Type[VlmCausalLMBatch]:
        return self.batch_class

    def max_past(self) -> Optional[int]:
        return getattr(self.model.text_model, "max_past", None)

    def forward(
        self,
        batch: VlmCausalLMBatch,
        adapter_data: AdapterBatchData,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Model Forward
        if batch.speculative_ids is not None:
            input_ids = batch.input_ids
            position_ids = batch.position_ids
            cu_seqlen_prefill = batch.cu_seqlen_prefill
            block_tables = batch.block_tables_tensor
            slots = batch.slots[batch.slot_indices]
            input_lengths = batch.input_lengths_tensor
            max_s = batch.max_current_length

            speculative_ids = batch.speculative_ids

            B, speculative_length = speculative_ids.shape
            new_length = speculative_length + 1
            new_input_ids = torch.cat([input_ids.unsqueeze(-1), speculative_ids], dim=1).reshape(-1)
            arange = torch.arange(new_length, device=position_ids.device).unsqueeze(0)
            arange_int = arange.to(dtype=torch.int32)
            new_position_ids = (position_ids.unsqueeze(-1).expand(B, new_length) + arange).view(-1)
            slots = (slots.unsqueeze(-1).expand(B, new_length) + arange_int).view(-1)
            input_lengths = (input_lengths.unsqueeze(-1).expand(B, new_length) + arange_int).view(-1)
            cache_lengths_tensor = (batch.cache_lengths_tensor.unsqueeze(-1).expand(B, new_length)).reshape(-1)

            # Add Copy the block tables for all members
            block_tables = block_tables.unsqueeze(1).expand(B, new_length, -1).reshape(B * new_length, -1).contiguous()
            max_s = max_s + speculative_length

            input_ids = new_input_ids
            position_ids = new_position_ids
        else:
            input_ids = batch.input_ids
            position_ids = batch.position_ids
            cu_seqlen_prefill = batch.cu_seqlen_prefill
            block_tables = batch.block_tables_tensor
            slots = batch.slots[batch.slot_indices]
            input_lengths = batch.input_lengths_tensor
            cache_lengths_tensor = batch.cache_lengths_tensor
            max_s = batch.max_current_length

        if cu_seqlen_prefill is None and self.max_past() is not None:
            # In decode, not prefill, we're actually overwriting the KV-cache
            # in a circular buffer mode.
            # This makes sure the max_s for the decode pass is correct.
            max_s = min(self.max_past(), max_s)

        prefill = batch.cu_seqlen_prefill is not None
        model = self.model
        use_graph = False
        if (
            self.model_graph_wrapper is not None
            and not prefill
            and self.model_graph_wrapper.can_use_graph(batch, adapter_data)
        ):
            use_graph = True
            model = self.model_graph_wrapper

        seqlen = Seqlen(
            input_lengths=input_lengths,
            cache_lengths=cache_lengths_tensor,
            cu_seqlen_q=None,
            max_q=batch.max_input_length,
            max_k=batch.max_current_length,
        )

        if not use_graph:
            input_lengths = input_lengths + cache_lengths_tensor
            if PREFIX_CACHING:
                block_tables = block_tables_to_ragged(
                    block_tables=block_tables,
                    input_lengths=batch.input_lengths,
                    cache_lengths=batch.cache_lengths,
                )
            with self._forward_context(
                block_tables=block_tables,
                cu_seqlen_prefill=cu_seqlen_prefill,
                input_lengths=batch.input_lengths,
                input_lengths_tensor=input_lengths,
                cache_lengths=batch.cache_lengths,
                cache_lengths_tensor=cache_lengths_tensor,
            ):
                # input_lengths = Seqlen(input_lengths=input_lengths)
                out = model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cu_seqlen_prefill=batch.cu_seqlen_prefill,
                    kv_cache=self.kv_cache,
                    block_tables=block_tables,
                    slots=slots,
                    seqlen=seqlen,
                    max_s=max_s,
                    adapter_data=adapter_data,
                    prefill_cache_indices=batch.prefill_cache_indices,
                    lm_head_indices=batch.prefill_head_indices,
                    pixel_values=batch.pixel_values,
                    pixel_attention_mask=batch.pixel_attention_mask,
                    image_sizes=batch.image_sizes,
                )
        else:
            # CUDA graph mode
            out = model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seqlen_prefill=batch.cu_seqlen_prefill,
                kv_cache=self.kv_cache,
                block_tables=block_tables,
                slots=slots,
                seqlen=seqlen,
                cache_lengths=batch.cache_lengths,
                cache_lengths_tensor=cache_lengths_tensor,
                max_s=max_s,
                adapter_data=adapter_data,
                prefill_cache_indices=batch.prefill_cache_indices,
                lm_head_indices=batch.prefill_head_indices,
            )

        if batch.prefill_cache_indices is not None:
            batch.prefill_cache_indices = None
        if batch.pixel_values is not None:
            batch.pixel_values = None
        if batch.pixel_attention_mask is not None:
            batch.pixel_attention_mask = None
        if batch.image_sizes is not None:
            batch.image_sizes = None
        return out
