from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Type

import torch
from opentelemetry import trace
from PIL import Image
from transformers import (
    PreTrainedTokenizerBase,
)

from lorax_server.adapters.weights import AdapterBatchData, AdapterBatchMetadata
from lorax_server.models import Model
from lorax_server.models.types import (
    Batch,
    GeneratedText,
    Generation,
    NextTokens,
    PrefillTokens,
)
from lorax_server.pb import generate_pb2
from lorax_server.utils import NextTokenChooser, Sampling, StoppingCriteria
from lorax_server.utils.segments import find_segments
from lorax_server.utils.tokenizer import TokenizerManager

tracer = trace.get_tracer(__name__)


@dataclass
class MultimodalCausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    requests_idx_mapping: Dict[int, int]

    # Decoder values
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    pixel_values: Optional[torch.Tensor]
    aspect_ratio_ids: Optional[torch.Tensor]
    aspect_ratio_mask: Optional[torch.Tensor]
    cross_attention_mask: Optional[torch.Tensor]
    image_hidden_states: Optional[torch.Tensor]
    image_attention_mask: Optional[torch.Tensor]
    past_key_values: Optional[List[Tuple]]

    # All tokens
    all_input_ids: List[torch.Tensor]

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    prefix_offsets: List[int]
    read_offsets: List[int]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]
    stopping_criterias: List[StoppingCriteria]

    # Adapter metadata for each request
    adapter_meta: AdapterBatchMetadata

    # Metadata used for padding
    max_input_length: int
    padding_right_offset: int

    # Maximum number of tokens this batch will grow to
    max_tokens: int

    # Past metadata
    keys_head_dim_last: bool = True

    def to_pb(self) -> generate_pb2.CachedBatch:
        return generate_pb2.CachedBatch(
            id=self.batch_id,
            request_ids=[r.id for r in self.requests],
            size=len(self),
            max_tokens=self.max_tokens,
        )

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
    ) -> "MultimodalCausalLMBatch":
        inputs = []
        next_token_choosers = []
        stopping_criterias = []
        prefix_offsets = []
        read_offsets = []
        requests_idx_mapping = {}

        adapter_indices_list = []
        adapter_set = set()

        # Parse batch
        max_truncation = 0
        padding_right_offset = 0
        max_decode_tokens = 0
        for i, r in enumerate(pb.requests):
            requests_idx_mapping[r.id] = i
            inputs.append(r.tokenized_inputs.input_chunks)
            next_token_choosers.append(NextTokenChooser.from_pb(r.parameters, device, tokenizer))
            stopping_criteria = StoppingCriteria.from_pb(r.stopping_parameters, tokenizer)
            stopping_criterias.append(stopping_criteria)
            max_truncation = max(max_truncation, r.truncate)
            max_decode_tokens += stopping_criteria.max_new_tokens
            padding_right_offset = max(padding_right_offset, stopping_criteria.max_new_tokens)
            adapter_indices_list.append(r.adapter_index)
            adapter_set.add(r.adapter_index)

        adapter_indices = torch.tensor(adapter_indices_list, dtype=torch.int64, device=device)

        # TODO Check impact on idefics

        if config.model_type == "idefics":
            prompts = []
            for inp in inputs:
                # Each input is encoded into a list, where each element of this input list is either a string or a URL
                prompt = []
                for chunk in inp:
                    chunk_type = chunk.WhichOneof("chunk")
                    if chunk_type == "text":
                        prompt.append(chunk.text)
                    elif chunk_type == "image":
                        image = Image.open(BytesIO(chunk.image.data))
                        prompt.append(image)
                    else:
                        raise RuntimeError(f"Invalid chunk type {chunk_type}")
                prompts.append(prompt)

            # The processor replaces the call to tokenizer, and
            # a/ takes care of fetching images from the URL
            # b/ generate the correct input_ids, attention_mask, pixel_values, image_attention_mask to feed to the model
            tokenized_inputs = processor(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_truncation,
                # TODO Check impact on idefics
                # add_end_of_utterance_token=False,  # Already taken care of inside the prompts, so bypassing the processor's handling of this token
            ).to(device)
        else:
            images = []
            texts = []
            for inp in inputs:
                # Each input is encoded into a list, where each element of this input list is either a string or a URL
                curr_images = []
                curr_text = ""
                for chunk in inp:
                    chunk_type = chunk.WhichOneof("chunk")
                    if chunk_type == "text":
                        curr_text += chunk.text
                    elif chunk_type == "image":
                        image = Image.open(BytesIO(chunk.image.data))
                        curr_images.append(image)
                        # TODO unsure about BOS
                        curr_text += "<|image|>"
                    else:
                        raise RuntimeError(f"Invalid chunk type {chunk_type}")
                images.append(curr_images)
                texts.append(curr_text)

            # The processor replaces the call to tokenizer, and
            # a/ takes care of fetching images from the URL
            # b/ generate the correct input_ids, attention_mask, pixel_values, image_attention_mask to feed to the model
            if all(len(im) == 0 for im in images):
                images = None
            tokenized_inputs = processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_truncation,
            ).to(device)
        for _ in pb.requests:
            input_len = tokenized_inputs["input_ids"].shape[1]
            prefix_offsets.append(input_len - 5)  # To decode without potential fallbacks errors
            read_offsets.append(input_len)  # To decode without potential fallbacks errors

        input_lengths = tokenized_inputs["attention_mask"].sum(1)
        max_input_length = input_lengths.max()

        input_ids = tokenized_inputs["input_ids"]
        pixel_values = tokenized_inputs.get("pixel_values", None)
        image_hidden_states = None
        # Allocate maximum attention_mask
        attention_mask = input_ids.new_zeros((pb.size, max_input_length + padding_right_offset))
        # Copy tokenizer attention_mask into fully allocated attention_mask
        attention_mask[:, :max_input_length] = tokenized_inputs["attention_mask"]
        # Do the same for image_attention_mask
        if pixel_values is None:
            image_attention_mask = None
            aspect_ratio_ids = None
            aspect_ratio_mask = None
            cross_attention_mask = None
        elif "image_attention_mask" in tokenized_inputs:
            image_attention_mask = input_ids.new_zeros(
                (
                    pb.size,
                    max_input_length + padding_right_offset,
                    pixel_values.size(1),
                )
            )
            image_attention_mask[:, :max_input_length, :] = tokenized_inputs["image_attention_mask"]
            aspect_ratio_ids = None
            aspect_ratio_mask = None
            cross_attention_mask = None
        elif "cross_attention_mask" in tokenized_inputs:
            image_attention_mask = None
            aspect_ratio_ids = tokenized_inputs["aspect_ratio_ids"]
            aspect_ratio_mask = tokenized_inputs["aspect_ratio_mask"]
            cross_attention_mask = tokenized_inputs["cross_attention_mask"]
            pixel_values = pixel_values.to(dtype=dtype)
            # XXX: <|image|> token is actually out of bounds and bugs out the logit processors.
            tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"].clamp(max=processor.tokenizer.vocab_size - 1)
        else:
            raise RuntimeError("Unhandled state for idefics/mllama")

        position_ids = tokenized_inputs["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(tokenized_inputs["attention_mask"] == 0, 1)
        all_input_ids = tokenized_inputs[
            "input_ids"
        ].T.split(
            1, dim=1
        )  # It's input_ids but splitted into a tuple of tensors where each tensor is (seq_len, 1) size. It is then transformed into a list

        max_tokens = len(inputs) * (max_input_length + max_decode_tokens)

        adapter_segments, adapter_segment_indices = find_segments(adapter_indices)
        adapter_segments = torch.tensor(adapter_segments, dtype=torch.int32, device=device)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_hidden_states=image_hidden_states,
            image_attention_mask=image_attention_mask,
            past_key_values=None,
            all_input_ids=list(all_input_ids),
            input_lengths=input_lengths.tolist(),
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            max_input_length=max_input_length.item(),
            padding_right_offset=padding_right_offset,
            max_tokens=max_tokens,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            cross_attention_mask=cross_attention_mask,
            adapter_meta=AdapterBatchMetadata(
                adapter_indices=adapter_indices,
                adapter_set=adapter_set,
                adapter_segments=adapter_segments,
                segment_indices=adapter_segment_indices,
            ),
        )

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]) -> Optional["MultimodalCausalLMBatch"]:
        # It deletes requests from the batch. For instance when client lost connection
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")
        if len(request_ids) == len(self):
            return self

        keep_indices = []

        # New values after filtering
        requests_idx_mapping = {}
        requests = []
        input_lengths = []
        prefix_offsets = []
        read_offsets = []
        all_input_ids = []
        max_input_length = 0

        next_token_choosers = []
        stopping_criterias = []

        total_remaining_decode_tokens = 0
        new_padding_right_offset = 0

        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            requests_idx_mapping[request_id] = i
            keep_indices.append(idx)

            requests.append(self.requests[idx])
            prefix_offsets.append(self.prefix_offsets[idx])
            read_offsets.append(self.read_offsets[idx])
            all_input_ids.append(self.all_input_ids[idx])

            request_input_length = self.input_lengths[idx]
            input_lengths.append(request_input_length)
            max_input_length = max(max_input_length, request_input_length)

            next_token_choosers.append(self.next_token_choosers[idx])
            stopping_criteria = self.stopping_criterias[idx]
            stopping_criterias.append(stopping_criteria)
            remaining_decode_tokens = stopping_criteria.max_new_tokens - stopping_criteria.current_tokens
            total_remaining_decode_tokens += remaining_decode_tokens
            new_padding_right_offset = max(new_padding_right_offset, remaining_decode_tokens)

        # Apply indices to input_ids, attention mask, past key values and other items that need to be cached
        input_ids = self.input_ids[keep_indices]
        position_ids = self.position_ids[keep_indices]
        self.attention_mask = self.attention_mask[
            keep_indices,
            -(self.padding_right_offset + max_input_length) : (self.attention_mask.shape[1] - self.padding_right_offset)
            + new_padding_right_offset,
        ]
        # Do the same for pixel_values and image_attention_mask
        if self.pixel_values is not None:
            pixel_values = self.pixel_values[keep_indices]
        else:
            pixel_values = None

        if self.image_attention_mask is not None:
            self.image_attention_mask = self.image_attention_mask[
                keep_indices,
                -(self.padding_right_offset + max_input_length) : (
                    self.image_attention_mask.shape[1] - self.padding_right_offset
                )
                + new_padding_right_offset,
                :,
            ]

        if self.image_hidden_states is None:
            image_hidden_states = None
        else:
            image_hidden_states = self.image_hidden_states[keep_indices]

        # Ensure that past_key_values tensors can be updated in-place
        if type(self.past_key_values[0]) is tuple:
            self.past_key_values = [list(layer) for layer in self.past_key_values]

        # Update tensors in-place to allow incremental garbage collection
        past_kv_length = max_input_length - 1
        for layer in self.past_key_values:
            past_keys, past_values = layer
            if not isinstance(past_keys, torch.Tensor):
                past_keys = [k for i, k in enumerate(past_keys) if i in keep_indices]
                past_values = [k for i, k in enumerate(past_values) if i in keep_indices]
                layer[0] = past_keys
                layer[1] = past_values
                continue
            elif len(past_keys.shape) == 3:
                # Force past to be of dim [self_size, num_heads, ...] for easy indexing
                past_keys = past_keys.view(len(self), -1, *past_keys.shape[-2:])
                past_values = past_values.view(len(self), -1, *past_values.shape[-2:])
            if self.keys_head_dim_last:
                layer[0] = past_keys[keep_indices, :, -past_kv_length:, :]
            else:
                layer[0] = past_keys[keep_indices, :, :, -past_kv_length:]
            del past_keys
            layer[1] = past_values[keep_indices, :, -past_kv_length:, :]
            del past_values

        max_tokens = len(request_ids) * max_input_length + total_remaining_decode_tokens

        self.requests = requests
        self.requests_idx_mapping = requests_idx_mapping
        self.input_ids = input_ids
        self.pixel_values = pixel_values
        self.image_hidden_states = image_hidden_states
        self.position_ids = position_ids
        self.all_input_ids = all_input_ids
        self.input_lengths = input_lengths
        self.prefix_offsets = prefix_offsets
        self.read_offsets = read_offsets
        self.next_token_choosers = next_token_choosers
        self.stopping_criterias = stopping_criterias
        self.max_input_length = max_input_length
        self.padding_right_offset = new_padding_right_offset
        self.max_tokens = max_tokens
        self.aspect_ratio_ids = None
        self.aspect_ratio_mask = None
        self.cross_attention_mask = None

        return self

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["MultimodalCausalLMBatch"]) -> "MultimodalCausalLMBatch":
        # It adds new requests to the batch
        # Used for padding
        total_batch_size = 0
        max_input_length = 0
        max_num_images = 0
        padding_right_offset = 0
        for batch in batches:
            total_batch_size += len(batch)
            max_input_length = max(max_input_length, batch.max_input_length)
            if batch.pixel_values is not None:
                max_num_images = max(max_num_images, batch.pixel_values.size(1))
            padding_right_offset = max(padding_right_offset, batch.padding_right_offset)

        # Batch attributes
        requests = []
        requests_idx_mapping = {}
        input_lengths = []
        prefix_offsets = []
        read_offsets = []
        all_input_ids = []
        next_token_choosers = []
        stopping_criterias = []
        max_tokens = 0

        # Batch tensors
        input_ids = None
        attention_mask = None
        position_ids = None
        pixel_values = None
        image_hidden_states = None
        image_attention_mask = None
        past_key_values = []

        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        start_index = 0
        for i, batch in enumerate(batches):
            requests.extend(batch.requests)
            input_lengths.extend(batch.input_lengths)
            prefix_offsets.extend(batch.prefix_offsets)
            read_offsets.extend(batch.read_offsets)
            all_input_ids.extend(batch.all_input_ids)
            next_token_choosers.extend(batch.next_token_choosers)
            stopping_criterias.extend(batch.stopping_criterias)

            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                # We need to offset the mapping for each batch by the cumulative batch size
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + start_index

            # Slicing end index for this batch
            end_index = start_index + len(batch)

            # We only concatenate batches that did at least one step
            if batch.past_key_values is None:
                raise ValueError("only concatenate prefilled batches")

            # Create empty tensor
            # input_ids is always of shape [batch_size, 1]
            # We do not need to pad it
            if input_ids is None:
                input_ids = batch.input_ids.new_empty((total_batch_size, 1))
            # Copy to correct indices
            input_ids[start_index:end_index] = batch.input_ids

            # Create padded tensor
            if attention_mask is None:
                attention_mask = batch.attention_mask.new_zeros(
                    (total_batch_size, max_input_length + padding_right_offset),
                )

            if batch.pixel_values is not None:
                curr_batch_max_num_images = batch.pixel_values.size(1)
                if pixel_values is None:
                    pixel_values = batch.pixel_values.new_zeros((total_batch_size, max_num_images, 3, 224, 224))
                pixel_values[start_index:end_index, :curr_batch_max_num_images] = batch.pixel_values
            else:
                pixel_values = None

            if image_attention_mask is None and batch.image_attention_mask is not None:
                image_attention_mask = batch.image_attention_mask.new_zeros(
                    (
                        total_batch_size,
                        max_input_length + padding_right_offset,
                        max_num_images,
                    )
                )

            # We need to slice the attention mask to remove padding from previous steps
            # and to remove unused allocated space
            left_offset = max_input_length - batch.max_input_length
            batch_left_offset = batch.attention_mask.shape[1] - batch.max_input_length - batch.padding_right_offset
            attention_mask[
                start_index:end_index,
                left_offset:-padding_right_offset,
            ] = batch.attention_mask[
                :,
                batch_left_offset : -batch.padding_right_offset,
            ]
            if batch.image_attention_mask is not None:
                image_attention_mask[
                    start_index:end_index,
                    left_offset:-padding_right_offset,
                    :curr_batch_max_num_images,
                ] = batch.image_attention_mask[:, batch_left_offset : -batch.padding_right_offset, :]

            # Create empty tensor
            # position_ids is always of shape [batch_size, 1]
            if position_ids is None:
                position_ids = batch.position_ids.new_empty((total_batch_size, 1))
            position_ids[start_index:end_index] = batch.position_ids

            # Shenanigans to get dimensions because BLOOM outputs a past with a different shape
            # BLOOM Keys:   [batch_size * num_heads, head_dim, seq_length]
            # BLOOM Values: [batch_size * num_heads, seq_length, head_dim]
            # And ensure that we can update tensors in-place
            if isinstance(batch.past_key_values[0], tuple):
                batch.past_key_values = [
                    [(t.view(len(batch), -1, *t.shape[-2:]) if isinstance(t, torch.Tensor) else t) for t in layer]
                    for layer in batch.past_key_values
                ]
            elif len(batch.past_key_values[0][0].shape) == 3:
                for layer in batch.past_key_values:
                    for k, t in enumerate(layer):
                        layer[k] = t.view(len(batch), -1, *t.shape[-2:])

            # Add eventual padding tokens that were added while concatenating
            max_tokens += batch.max_tokens + (max_input_length - batch.max_input_length) * len(batch)

            start_index = end_index

        first_past_kvs = batches[0].past_key_values
        _, num_heads, padded_sequence_length, head_dim = first_past_kvs[0][1].shape

        padded_past_values_shape = (
            total_batch_size,
            num_heads,
            max_input_length - 1,
            head_dim,
        )

        if batches[0].keys_head_dim_last:
            padded_past_keys_shape = padded_past_values_shape
        else:
            # seq_length is last for BLOOM
            padded_past_keys_shape = (
                total_batch_size,
                num_heads,
                head_dim,
                max_input_length - 1,
            )

        # Iterate over attention layers
        # Concatenate past key values layer by layer to allow incremental garbage collection
        for j in range(len(first_past_kvs)):
            if any(not isinstance(batch.past_key_values[j][0], torch.Tensor) for batch in batches):
                # XXX: Special handling for cross attention for mllama
                padded_past_keys = [k for batch in batches for k in batch.past_key_values[j][0]]
                padded_past_values = [k for batch in batches for k in batch.past_key_values[j][1]]
                past_key_values.append([padded_past_keys, padded_past_values])
            else:
                _, _num_heads, seqlen, _head_dim = first_past_kvs[j][0].shape
                if seqlen > max_input_length:
                    # XXX: This is probably a cross attention key value
                    # If not this is ok
                    _padded_past_keys_shape = (
                        total_batch_size,
                        _num_heads,
                        seqlen,
                        _head_dim,
                    )
                else:
                    _padded_past_keys_shape = padded_past_keys_shape

                padded_past_keys = first_past_kvs[j][0].new_zeros(_padded_past_keys_shape)
                start_index = 0
                for batch in batches:
                    past_keys = batch.past_key_values[j][0]
                    # Clear reference to the original tensor
                    batch.past_key_values[j][0] = None

                    # Slicing end index for this batch
                    end_index = start_index + len(batch)
                    # We slice the keys to remove the padding from previous batches
                    past_seq_len = batch.max_input_length - 1
                    if past_keys.shape[2] > past_seq_len:
                        # XXX: This is a cross attention kv in mllama
                        past_seq_len = past_keys.shape[2]
                    if batch.keys_head_dim_last:
                        padded_past_keys[start_index:end_index, :, -past_seq_len:, :] = past_keys[
                            :, :, -past_seq_len:, :
                        ]
                    else:
                        # BLOOM case
                        padded_past_keys[start_index:end_index, :, :, -past_seq_len:] = past_keys[
                            :, :, :, -past_seq_len:
                        ]
                    del past_keys

                    start_index = end_index

                _, _num_heads, seqlen, _head_dim = first_past_kvs[j][1].shape
                if seqlen > max_input_length:
                    # XXX: This is probably a cross attention key value
                    # If not this is ok
                    _padded_past_values_shape = (
                        total_batch_size,
                        _num_heads,
                        seqlen,
                        _head_dim,
                    )
                else:
                    _padded_past_values_shape = padded_past_values_shape
                padded_past_values = first_past_kvs[j][1].new_zeros(_padded_past_values_shape)
                start_index = 0
                for batch in batches:
                    past_values = batch.past_key_values[j][1]
                    # Clear reference to the original tensor
                    batch.past_key_values[j][1] = None

                    # Slicing end index for this batch
                    end_index = start_index + len(batch)
                    # We slice the past values to remove the padding from previous batches
                    past_seq_len = batch.max_input_length - 1
                    if past_values.shape[2] > past_seq_len:
                        # XXX: This is a cross attention kv in mllama
                        past_seq_len = past_values.shape[2]
                    padded_past_values[start_index:end_index, :, -past_seq_len:, :] = past_values[
                        :, :, -past_seq_len:, :
                    ]
                    del past_values

                    # Update values
                    start_index = end_index

                past_key_values.append([padded_past_keys, padded_past_values])

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_hidden_states=image_hidden_states,
            image_attention_mask=image_attention_mask,
            past_key_values=past_key_values,
            all_input_ids=all_input_ids,
            input_lengths=input_lengths,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            max_input_length=max_input_length,
            padding_right_offset=padding_right_offset,
            keys_head_dim_last=batches[0].keys_head_dim_last,
            max_tokens=max_tokens,
            # No need to keep this around. for Mllamma
            aspect_ratio_ids=None,
            aspect_ratio_mask=None,
            cross_attention_mask=None,
        )

    def __len__(self):
        return len(self.requests)


class MultimodalCausalLM(Model):
    @property
    def batch_type(self) -> Type[MultimodalCausalLMBatch]:
        return MultimodalCausalLMBatch

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        pixel_values,
        image_hidden_states,
        image_attention_mask,
        past_key_values=None,
        aspect_ratio_ids=None,
        aspect_ratio_mask=None,
        cross_attention_mask=None,
        adapter_data=None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # Model Forward
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_hidden_states": image_hidden_states,
            "image_attention_mask": image_attention_mask,
            "past_key_values": past_key_values,
            "adapter_data": adapter_data,
        }
        if self.has_position_ids:
            kwargs["position_ids"] = position_ids
        if aspect_ratio_ids is not None:
            kwargs["aspect_ratio_ids"] = aspect_ratio_ids
        if aspect_ratio_mask is not None:
            kwargs["aspect_ratio_mask"] = aspect_ratio_mask
        if cross_attention_mask is not None:
            kwargs["cross_attention_mask"] = cross_attention_mask

        outputs, speculative_logits = self.model.forward(**kwargs)
        assert outputs.past_key_values is not None
        return (
            outputs.logits,
            speculative_logits,
            outputs.past_key_values,
            getattr(outputs, "image_hidden_states", None),
        )

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: MultimodalCausalLMBatch
    ) -> Tuple[List[Generation], Optional[MultimodalCausalLMBatch], Tuple[int, int]]:
        # slice the attention mask to the correct shape
        attention_mask = batch.attention_mask[:, : -batch.padding_right_offset]
        if batch.image_attention_mask is None:
            image_attention_mask = None
        else:
            if batch.input_ids.size(1) == 1:
                # THIS is a hack: when calling idefics.generate, the first time, we need the whole image_attention_mask (size bs x max_seq_len x max_num_images),
                # but the subsequent times, we only need the last attention mask along the `max_seq_len` dimension
                # this is due to the nature IDEFICS: it's an encoder decoder, and so when decoding, only the currently generated
                # token need to attend to the encoder hidden states (i.e. the vision encoder)
                # Also see seq2seq_lm.Seq2SeqLM.generate_token which has roughly the same logic
                image_attention_mask = batch.image_attention_mask[:, -(batch.padding_right_offset + 1)].unsqueeze(1)
            else:
                image_attention_mask = batch.image_attention_mask[:, : -batch.padding_right_offset]

        # Update adapter indices for speculative tokens (if present)
        prefill = batch.past_key_values is not None
        adapter_meta = batch.adapter_meta

        # Assign pointers to adapter weights
        # TODO(travis): don't update this if indices haven't changed
        adapter_data = AdapterBatchData.from_meta(adapter_meta, self.layer_to_adapter_weights, prefill, None)

        logits, speculative_logits, past, image_hidden_states = self.forward(
            input_ids=batch.input_ids,
            attention_mask=attention_mask,
            position_ids=batch.position_ids,
            pixel_values=batch.pixel_values,
            image_hidden_states=batch.image_hidden_states,
            image_attention_mask=image_attention_mask,
            past_key_values=batch.past_key_values,
            aspect_ratio_ids=batch.aspect_ratio_ids,
            aspect_ratio_mask=batch.aspect_ratio_mask,
            cross_attention_mask=batch.cross_attention_mask,
            adapter_data=adapter_data,
        )
        # Hardcoded remove image tokens
        if self.config.model_type == "idefics":
            logits[:, 32000:32001] = torch.finfo(logits.dtype).min

        # Results
        generations: List[Generation] = []
        stopped = True

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.prefix_offsets,
            batch.read_offsets,
            logits,
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.all_input_ids,
        )

        # For each member of the batch
        for i, (
            request,
            input_length,
            prefix_offset,
            read_offset,
            logits,
            next_token_chooser,
            stopping_criteria,
            all_input_ids,
        ) in enumerate(iterator):
            # Select next token
            next_token_id, logprobs = next_token_chooser(all_input_ids.view(1, -1), logits[-1:, :])

            # Append next token to all tokens
            all_input_ids = torch.cat([all_input_ids, next_token_id])
            new_input_length = input_length + 1

            # Generated token
            next_token_logprob = logprobs[-1, next_token_id]
            next_token_id_squeezed = next_token_id.squeeze()
            next_token_text, prefix_offset, read_offset = self.decode_token(
                all_input_ids[:, 0], prefix_offset, read_offset
            )

            # Evaluate stopping criteria
            stop, reason = stopping_criteria(
                next_token_id_squeezed,
                next_token_text,
            )

            if not stop:
                stopped = False

            # Shard generations
            # All generations will be appended in the rust sharded client
            if i % self.world_size == self.rank:
                if stop:
                    # Decode generated tokens
                    output_text, _, _ = self.decode_token(
                        all_input_ids[:, 0],
                        prefix_offset=len(all_input_ids) - stopping_criteria.current_tokens - 1,
                        read_offset=len(all_input_ids) - stopping_criteria.current_tokens,
                        skip_special_tokens=True,
                    )
                    # Get seed
                    if isinstance(next_token_chooser.choice, Sampling):
                        seed = next_token_chooser.choice.seed
                    else:
                        seed = None

                    generated_text = GeneratedText(output_text, stopping_criteria.current_tokens, reason, seed)
                else:
                    generated_text = None

                # Prefill
                if stopping_criteria.current_tokens == 1 and request.prefill_logprobs:
                    # Remove generated token to only have prefill and add nan for first prompt token
                    prefill_logprobs = [float("nan")] + torch.log_softmax(logits, -1).gather(
                        1, all_input_ids[1:]
                    ).squeeze(1)[-new_input_length:-1].tolist()
                    prefill_token_ids = all_input_ids[-new_input_length:-1]
                    prefill_texts = self.tokenizer.batch_decode(
                        prefill_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )
                    prefill_tokens = PrefillTokens(prefill_token_ids, prefill_logprobs, prefill_texts)
                    prefill_tokens_length = len(prefill_tokens.token_ids)
                else:
                    prefill_tokens = None
                    prefill_tokens_length = 0

                # TODO(travis): support all_alternative_tokens for multimodal
                generation = Generation(
                    request.id,
                    prefill_tokens,
                    prefill_tokens_length,
                    NextTokens(
                        [next_token_id_squeezed],
                        [next_token_logprob],
                        [next_token_text],
                        [next_token_id_squeezed.item() in self.all_special_ids],
                        None,
                    ),
                    generated_text,
                )

                generations.append(generation)

            # advance FSM state
            batch.next_token_choosers[i].next_state(next_token_id_squeezed.item())

            # Update values
            batch.input_ids[i, 0] = next_token_id
            batch.all_input_ids[i] = all_input_ids
            batch.input_lengths[i] = new_input_length
            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.max_input_length = max(batch.max_input_length, new_input_length)

        # We finished all generations in the batch; there is no next batch
        if stopped:
            return generations, None

        # Slice unused values from prefill
        batch.input_ids = batch.input_ids[:, :1]

        # Update attention_mask as we added a new token to input_ids
        batch.attention_mask[:, -batch.padding_right_offset] = 1
        if batch.image_attention_mask is not None:
            batch.image_attention_mask[:, -batch.padding_right_offset, :] = batch.image_attention_mask[
                :, -(batch.padding_right_offset + 1), :
            ]
        if batch.cross_attention_mask is not None:
            batch.cross_attention_mask = batch.cross_attention_mask[:, -1:]
        # Decrease right offset
        batch.padding_right_offset -= 1

        # Update position_ids
        batch.position_ids = batch.position_ids[:, -1:] + 1

        # Update past key values
        batch.past_key_values = past
        batch.image_hidden_states = image_hidden_states
        if self.model.config.model_type == "mllama":
            batch.pixel_values = None
        return generations, batch
