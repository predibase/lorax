# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Mllama model."""

from typing import List, Optional, Tuple

import flash_attn_2_cuda
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN

from lorax_server.adapters.weights import AdapterBatchData
from lorax_server.models.custom_modeling.flash_llama_modeling import (
    FlashLlamaForCausalLM,
    FlashLlamaLayer,
)
from lorax_server.utils.attention.common import Seqlen
from lorax_server.utils.layers import (
    FastLinear,
    TensorParallelAdapterRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelMultiAdapterLinear,
    TensorParallelRowLinear,
)
from lorax_server.utils.lora import (
    DOWN_PROJ,
    FC1,
    FC2,
    GATE_PROJ,
    K_PROJ,
    O_PROJ,
    Q_PROJ,
    UP_PROJ,
    V_PROJ,
)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _prepare_aspect_ratio_attention_mask(
    aspect_ratio_mask: torch.Tensor,
    num_patches: int,
    target_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Expand aspect ratio mask to target_length
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = aspect_ratio_mask.view(batch_size, max_num_tiles, 1, 1).to(dtype)
    attention_mask = attention_mask.repeat(1, 1, target_length, 1)

    # Mask padding patches
    pad_patches = target_length - num_patches
    attention_mask[:, :, -pad_patches:] = 0

    # Invert the mask (0 -> 1, 1 -> 0)
    attention_mask = 1 - attention_mask

    # Reshape to 2D and create 4D attention mask
    # (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length)
    attention_mask = attention_mask.reshape(batch_size, max_num_tiles * target_length, 1)
    attention_mask = attention_mask @ attention_mask.transpose(-1, -2) * torch.finfo(dtype).min
    attention_mask = attention_mask.unsqueeze(1)

    return attention_mask


# Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

    return causal_mask


def _prepare_cross_attention_mask(
    cross_attention_mask: torch.Tensor,
    num_vision_tokens: int,
    dtype: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape so it can be used by attn module
    batch_size, text_total_length, *_ = cross_attention_mask.shape
    cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=3)
    cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
    cross_attention_mask = cross_attention_mask.unsqueeze(1)

    # invert the mask
    inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
    cross_attention_mask = inverted_cross_attn_mask.masked_fill(
        inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min
    )

    # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
    # last dimension contains negative infinity values, otherwise it's 1
    negative_inf_value = torch.finfo(dtype).min
    full_text_row_masked_out_mask = (
        (cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None]
    )
    cross_attention_mask *= full_text_row_masked_out_mask

    return cross_attention_mask, full_text_row_masked_out_mask


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->MllamaVision
class MllamaVisionMLP(nn.Module):
    def __init__(self, *, prefix, config, weights, layer_id, model_type):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]

        fc1 = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.fc1"],
            weights=weights,
            dim=0,
            bias=True,
        )

        out_size = fc1.linear.weight.shape[-1] * weights.process_group.size()
        self.fc1 = TensorParallelMultiAdapterLinear.load(
            fc1, layer_id, [f"{model_type}_{FC1}"], sizes=[out_size], process_group=weights.process_group
        )
        self.fc2 = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.fc2",
                weights=weights,
                bias=True,
            ),
            layer_id,
            f"{model_type}_{FC2}",
            process_group=weights.process_group,
        )

    def forward(self, hidden_states: torch.Tensor, adapter_data: AdapterBatchData) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states, adapter_data)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states.view(-1, hidden_states.shape[-1]), adapter_data)
        return hidden_states


def load_attention(config, prefix, weights, layer_id, model_type, head_dim, n_head, n_head_kv):
    base_layer = TensorParallelColumnLinear.load_multi(
        config,
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        dim=0,
        weights=weights,
        bias=False,
    )
    return TensorParallelMultiAdapterLinear.load(
        base_layer,
        layer_id,
        [f"{model_type}_{Q_PROJ}", f"{model_type}_{K_PROJ}", f"{model_type}_{V_PROJ}"],
        sizes=[
            head_dim * n_head,
            head_dim * n_head_kv,
            head_dim * n_head_kv,
        ],
        process_group=weights.process_group,
    )


class MllamaVisionSdpaAttention(nn.Module):
    def __init__(self, *, prefix, config, weights, layer_id, model_type):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.head_dim = config.hidden_size // config.attention_heads
        self.num_heads = config.attention_heads // weights.process_group.size()
        self.head_size = config.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, "n_head_kv", None) or self.num_heads

        self.qkv_proj = load_attention(
            config,
            prefix,
            weights,
            layer_id,
            model_type,
            self.head_size,
            self.num_heads,
            self.num_key_value_heads,
        )
        self.o_proj = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.o_proj",
                weights=weights,
                bias=False,
            ),
            layer_id,
            f"{model_type}_{O_PROJ}",
            process_group=weights.process_group,
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        adapter_data: AdapterBatchData = None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_state, adapter_data)
        query, key, value = qkv.split(
            [
                self.head_dim * self.num_heads,
                self.head_dim * self.num_heads,
                self.head_dim * self.num_heads,
            ],
            dim=2,
        )

        batch_size, q_seq_len, _ = query.shape
        _, kv_seq_len, _ = key.shape

        query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size * q_seq_len, -1)

        output = self.o_proj(attn_output, adapter_data)
        return output


class MllamaVisionEncoderLayer(nn.Module):
    def __init__(self, *, prefix, config, weights, is_gated: bool, layer_id: int, model_type: str):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.attention_heads
        self.is_gated = is_gated
        self.intermediate_size = config.intermediate_size

        self.self_attn = MllamaVisionSdpaAttention(
            prefix=f"{prefix}.self_attn",
            config=config,
            weights=weights,
            layer_id=layer_id,
            model_type=model_type,
        )
        self.mlp = MllamaVisionMLP(
            prefix=f"{prefix}.mlp",
            config=config,
            weights=weights,
            layer_id=layer_id,
            model_type=model_type,
        )

        self.input_layernorm = nn.LayerNorm.load(prefix=f"{prefix}.input_layernorm", weights=weights, eps=1e-05)
        self.post_attention_layernorm = nn.LayerNorm.load(
            prefix=f"{prefix}.post_attention_layernorm", weights=weights, eps=1e-05
        )

        # there used to be an if else here, no code path
        if is_gated:
            self.gate_attn = nn.Parameter(weights.get_tensor(f"{prefix}.gate_attn"), requires_grad=False)
            self.gate_ffn = nn.Parameter(weights.get_tensor(f"{prefix}.gate_ffn"), requires_grad=False)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        adapter_data: AdapterBatchData = None,
    ):
        # Self Attention
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(hidden_state, attention_mask, adapter_data)
        gate_attn = 1 if not self.is_gated else self.gate_attn.tanh()
        hidden_state = residual + gate_attn * hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state, adapter_data)
        gate_ffn = 1 if not self.is_gated else self.gate_ffn.tanh()
        hidden_state = residual + gate_ffn * hidden_state
        return hidden_state


class MllamaVisionEncoder(nn.Module):
    def __init__(self, *, prefix, config, weights, is_gated: bool, num_layers: int, model_type: str):
        super().__init__()
        self.config = config
        self.layers = [
            MllamaVisionEncoderLayer(
                prefix=f"{prefix}.layers.{layer_id}",
                config=config,
                weights=weights,
                is_gated=is_gated,
                layer_id=layer_id,
                model_type=model_type,
            )
            for layer_id in range(num_layers)
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        adapter_data: AdapterBatchData = None,
    ):
        encoder_states = [hidden_states]
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                adapter_data,
            )

            hidden_states = layer_outputs
            encoder_states.append(hidden_states)

        return hidden_states, encoder_states


class MllamaPrecomputedAspectRatioEmbedding(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id

        self.embedding = TensorParallelEmbedding(prefix=f"{prefix}.embedding", weights=weights)
        self.gate = nn.Parameter(weights.get_tensor(f"{prefix}.gate"), requires_grad=False)

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

        # Always gated.
        embeddings = embeddings * self.gate.tanh()

        hidden_state = hidden_state + embeddings
        return hidden_state


class MllamaPrecomputedPositionEmbedding(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        self.hidden_size = config.hidden_size
        self.scale = config.hidden_size**-0.5

        self.gate = nn.Parameter(weights.get_tensor(f"{prefix}.gate"), requires_grad=False)

        # position embedding
        embedding = nn.Parameter(weights.get_tensor(f"{prefix}.embedding"), requires_grad=False)
        self.gated_position_embedding = (1 - self.gate.tanh()) * embedding
        self.tile_embedding = TensorParallelEmbedding(prefix=f"{prefix}.tile_embedding", weights=weights)

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        # position embeddings
        hidden_state = hidden_state + self.gated_position_embedding.view(1, 1, self.num_patches, self.hidden_size)

        # precomputed tile position embeddings
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(
            batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
        )
        gated_tile_position_embedding = self.gate.tanh() * tile_position_embedding
        hidden_state = hidden_state + gated_tile_position_embedding

        return hidden_state


class MllamaVisionModel(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        self.intermediate_layers_indices = config.intermediate_layers_indices

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size**-0.5
        self.dtype = weights.dtype

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
            bias=False,
        )
        self.patch_embedding.weight = nn.Parameter(
            weights.get_tensor(f"{prefix}.patch_embedding.weight"), requires_grad=False
        )

        self.class_embedding = nn.Parameter(weights.get_tensor(f"{prefix}.class_embedding"), requires_grad=False)

        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(
            prefix=f"{prefix}.gated_positional_embedding",
            config=config,
            weights=weights,
        )

        self.pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(
            prefix=f"{prefix}.pre_tile_positional_embedding",
            config=config,
            weights=weights,
        )
        self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(
            prefix=f"{prefix}.post_tile_positional_embedding",
            config=config,
            weights=weights,
        )

        ## layer norms
        self.layernorm_pre = nn.LayerNorm.load(
            prefix=f"{prefix}.layernorm_pre",
            weights=weights,
            # torch default
            eps=1e-05,
        )
        self.layernorm_post = nn.LayerNorm.load(
            prefix=f"{prefix}.layernorm_post",
            weights=weights,
            # torch default
            eps=1e-05,
        )

        ## encoders
        self.transformer = MllamaVisionEncoder(
            prefix=f"{prefix}.transformer",
            config=config,
            weights=weights,
            is_gated=False,
            num_layers=config.num_hidden_layers,
            model_type="VISION_TRANSFORMER",
        )
        self.global_transformer = MllamaVisionEncoder(
            prefix=f"{prefix}.global_transformer",
            config=config,
            weights=weights,
            is_gated=True,
            num_layers=config.num_global_layers,
            model_type="VISION_GLOBAL_TRANSFORMER",
        )

    def apply_class_embedding(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_state.shape
        class_embedding = self.class_embedding.expand(batch_size, 1, hidden_size)
        hidden_state = torch.cat([class_embedding, hidden_state], dim=1)
        return hidden_state

    def forward(
        self,
        pixel_values: torch.Tensor,
        aspect_ratio_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adapter_data: AdapterBatchData,
    ) -> torch.Tensor:
        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape

        pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
        aspect_ratio_ids = aspect_ratio_ids.reshape(batch_size * num_concurrent_media, -1)

        # patch embedding
        patch_embeds = self.patch_embedding(pixel_values)
        hidden_state = patch_embeds.flatten(2).transpose(1, 2)

        # tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, -1, dim)
        hidden_state = self.pre_tile_positional_embedding(hidden_state, aspect_ratio_ids)

        # apply cls token
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media * num_tiles, num_patches, dim)
        hidden_state = self.apply_class_embedding(hidden_state)
        num_patches += 1

        # apply position embeddings
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_patches, dim)
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)

        # apply encoder
        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
        # Compute padding tuple for pad function
        padding = (
            0,
            0,
            0,
            num_padding_patches,
        )  # (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        # Pad the tensor
        hidden_state = F.pad(hidden_state, padding, mode="constant", value=0)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(batch_size * num_concurrent_media, -1)
            attention_mask = _prepare_aspect_ratio_attention_mask(
                aspect_ratio_mask=attention_mask,
                num_patches=self.num_patches,
                target_length=hidden_state.shape[2],
                dtype=self.dtype,
            )

        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
        hidden_state, all_intermediate_hidden_states = self.transformer(
            hidden_state,
            attention_mask=attention_mask,
            adapter_data=adapter_data,
        )
        intermediate_hidden_states = [
            hidden_state
            for idx, hidden_state in enumerate(all_intermediate_hidden_states)
            if idx in self.intermediate_layers_indices
        ]
        intermediate_hidden_states = torch.stack(intermediate_hidden_states, dim=-1)

        # apply global encoder
        hidden_state = self.layernorm_post(hidden_state)
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            dim,
        )
        hidden_state = self.post_tile_positional_embedding(hidden_state, aspect_ratio_ids)
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media,
            num_tiles * (num_patches + num_padding_patches),
            dim,
        )
        hidden_state, _ = self.global_transformer(
            hidden_state,
            attention_mask=attention_mask,
            adapter_data=adapter_data,
        )
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            dim,
        )
        hidden_state = hidden_state[:, :, :slice_index]

        # adding intermediate layer outputs
        hidden_state = hidden_state.reshape(batch_size, num_concurrent_media, num_tiles, num_patches, dim)
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            -1,
        )
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, -1
        )
        hidden_state = torch.cat([hidden_state, intermediate_hidden_states], dim=-1)
        return hidden_state


class MllamaTextCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, *, prefix, config, weights, layer_idx):
        super().__init__()
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.head_size = config.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.layer_idx = layer_idx

        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = self.num_key_value_heads // weights.process_group.size()

        self.q_proj = TensorParallelMultiAdapterLinear.load(
            TensorParallelColumnLinear.load_multi(
                config,
                prefixes=[f"{prefix}.q_proj"],
                weights=weights,
                dim=0,
                bias=False,
            ),
            layer_idx,
            [Q_PROJ],
            sizes=[self.head_size * self.num_heads],
            process_group=weights.process_group,
        )
        self.k_proj = TensorParallelMultiAdapterLinear.load(
            TensorParallelColumnLinear.load_multi(
                config,
                prefixes=[f"{prefix}.k_proj"],
                weights=weights,
                dim=0,
                bias=False,
            ),
            layer_idx,
            [K_PROJ],
            sizes=[self.head_size * self.num_key_value_heads],
            process_group=weights.process_group,
        )
        self.v_proj = TensorParallelMultiAdapterLinear.load(
            TensorParallelColumnLinear.load_multi(
                config,
                prefixes=[f"{prefix}.v_proj"],
                weights=weights,
                dim=0,
                bias=False,
            ),
            layer_idx,
            [V_PROJ],
            sizes=[self.head_size * self.num_key_value_heads],
            process_group=weights.process_group,
        )
        self.o_proj = TensorParallelAdapterRowLinear.load(
            TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.o_proj",
                weights=weights,
                bias=False,
            ),
            layer_idx,
            O_PROJ,
            process_group=weights.process_group,
        )

        self.q_norm = MllamaTextRMSNorm.load(prefix=f"{prefix}.q_norm", weights=weights, eps=config.rms_norm_eps)
        self.k_norm = MllamaTextRMSNorm.load(prefix=f"{prefix}.k_norm", weights=weights, eps=config.rms_norm_eps)
        self.softmax_scale = self.head_size**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        # past_key_value=None,
        # attention_mask: Optional[torch.Tensor] = None,
        # cache_position: Optional[torch.LongTensor] = None,
        adapter_data: Optional[AdapterBatchData] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # hidden_states = hidden_states.unsqueeze(0)
        # bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states, adapter_data)
        query_states = query_states.view(-1, self.num_heads, self.head_size)
        query_states = self.q_norm(query_states)

        (
            cross_attention_states,
            cu_seqlen_q,
            cu_seqlen_k,
            max_q,
            max_k,
            indices,
        ) = cross_attention_states

        key_states = self.k_proj(cross_attention_states, adapter_data)
        value_states = self.v_proj(cross_attention_states, adapter_data)
        key_states = key_states.view(-1, self.num_key_value_heads, self.head_size)
        value_states = value_states.view(-1, self.num_key_value_heads, self.head_size)
        key_states = self.k_norm(key_states)

        # key_states = key_states.repeat(1, self.num_key_value_groups, 1)
        # value_states = value_states.repeat(1, self.num_key_value_groups, 1)

        causal = False
        # logger.info(
        #     f"Q: {query_states.shape} -K {key_states.shape} - V{value_states.shape}"
        # )
        out = torch.empty_like(query_states)
        attn_output = flash_attn_2_cuda.varlen_fwd(
            query_states,
            key_states,
            value_states,
            out,
            cu_seqlen_q,
            cu_seqlen_k,
            None,
            None,
            None,  # block_tables
            max_q,
            max_k,
            0.0,
            self.softmax_scale,
            False,
            causal,  # Causal
            -1,  # window_size_left,
            0,  # softcap
            False,
            None,
        )[0]
        attn_output = self.o_proj(attn_output.view(-1, self.num_heads * self.head_size), adapter_data)

        return attn_output


# Copied from transformers.models.gemma2.modeling_gemma2.Gemma2MLP with Gemma2->MllamaText
class MllamaTextMLP(nn.Module):
    def __init__(self, *, prefix, config, weights, layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size // weights.process_group.size()
        gate_up_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            dim=0,
            bias=False,
        )
        self.gate_up_proj = TensorParallelMultiAdapterLinear.load(
            gate_up_proj,
            layer_idx,
            [GATE_PROJ, UP_PROJ],
            sizes=[
                config.intermediate_size,
                config.intermediate_size,
            ],
            process_group=weights.process_group,
        )
        down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.down_proj = TensorParallelAdapterRowLinear.load(
            down_proj,
            layer_idx,
            DOWN_PROJ,
            process_group=weights.process_group,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, adapter_data):
        shape = x.shape
        gate_up_states = self.gate_up_proj(x, adapter_data)
        gate_up_states = gate_up_states.view(*shape[:-1], 2, self.intermediate_size)
        result = self.down_proj(self.act_fn(gate_up_states[:, 0]) * gate_up_states[:, 1], adapter_data)
        return result


class FlashLlamaCrossLayer(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention and feedforward."""

    def __init__(self, layer_id, prefix, config, weights) -> None:
        layer_idx = layer_id
        super().__init__()
        self.cross_attn = MllamaTextCrossAttention(
            prefix=f"{prefix}.cross_attn",
            config=config,
            weights=weights,
            layer_idx=layer_idx,
        )

        self.input_layernorm = MllamaTextRMSNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.cross_attn_attn_gate = torch.nn.Parameter(
            weights.get_tensor(f"{prefix}.cross_attn_attn_gate"), requires_grad=False
        )

        self.mlp = MllamaTextMLP(prefix=f"{prefix}.mlp", config=config, weights=weights, layer_idx=layer_idx)
        self.post_attention_layernorm = MllamaTextRMSNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )
        self.cross_attn_mlp_gate = torch.nn.Parameter(
            weights.get_tensor(f"{prefix}.cross_attn_mlp_gate"), requires_grad=False
        )
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        seqlen,
        max_s,
        adapter_data,
        cross_attention_states,  # [ IB, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cross_attention_states is None:
            return hidden_states, residual
        if residual is not None:
            hidden_states += residual

        indices = cross_attention_states[-1]
        out_hidden_states = hidden_states[:]
        if len(indices) > 0:
            assert max(indices) < hidden_states.shape[0]
        hidden_states = hidden_states[indices]
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            # attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            adapter_data=adapter_data,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, adapter_data)
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

        out_hidden_states[indices] = hidden_states
        hidden_states = out_hidden_states

        return hidden_states, None


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->MllamaText
class MllamaTextRMSNorm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    @classmethod
    def load(cls, *, prefix, weights, eps):
        weight = nn.Parameter(weights.get_tensor(f"{prefix}.weight"), requires_grad=False)
        return cls(weight=weight, eps=eps)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MllamaForConditionalGeneration(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        config.vision_config.quantize = None
        config.text_config.quantize = config.quantize
        config.text_config._attn_implementation = "sdpa"
        self.hidden_size = config.text_config.hidden_size
        cross_attention_layers = getattr(config.text_config, "cross_attention_layers", [])
        def create_layer(layer_id, prefix, config, weights):
            layer_cls = FlashLlamaCrossLayer if layer_id in cross_attention_layers else FlashLlamaLayer
            return layer_cls(
                layer_id,
                prefix=prefix,
                config=config,
                weights=weights,
            )

        self.vision_model = MllamaVisionModel(prefix="vision_model", config=config.vision_config, weights=weights)
        self.multi_modal_projector = FastLinear.load(
            prefix="multi_modal_projector", config=config, weights=weights, bias=True
        )
        self.text_model = FlashLlamaForCausalLM(
            prefix="language_model", config=config.text_config, weights=weights, create_layer_fn=create_layer
        )
        self.config = config
        self.dtype = weights.dtype
        self.device = weights.device

    def vision_forward(self, pixel_values, aspect_ratio_ids, aspect_ratio_mask, adapter_data):
        if aspect_ratio_ids is None:
            raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
        # logger.info(f"PIxel values {pixel_values.shape}")
        batch_size = pixel_values.shape[0]
        vision_states = self.vision_model(pixel_values, aspect_ratio_ids, aspect_ratio_mask, adapter_data)
        cross_attention_states = self.multi_modal_projector(vision_states).reshape(
            -1, vision_states.shape[-2], self.hidden_size
        )
        _, _, h = cross_attention_states.shape
        cross_attention_states = cross_attention_states.view(batch_size, -1, h)
        # logger.info(f"cross {cross_attention_states.shape}")
        return cross_attention_states

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        max_s: int,
        adapter_data: AdapterBatchData,
        prefill_cache_indices: Optional[torch.Tensor],
        lm_head_indices: Optional[torch.Tensor],
        # XXX: Putting these as optional so that the cuda warmup calls can go through.
        cross_attention_states: Optional[torch.Tensor] = None,
        image_indices=None,
        skip_lm_head: bool = False,
    ):
        if cross_attention_states is not None:
            seqlen_q = len(image_indices)
            n_images = cross_attention_states.shape[0]
            seqlen_k = cross_attention_states.shape[1]
            device = cross_attention_states.device
            if cu_seqlen_prefill is not None:
                offset = 0
                cu_q = []
                indices = []
                for index in image_indices:
                    cu_q.append(offset)
                    length = seqlen.input_lengths[index].item()
                    assert index < cu_seqlen_prefill.shape[0]
                    input_ids_offset = cu_seqlen_prefill[index]
                    indices.extend(range(input_ids_offset, input_ids_offset + length))
                    offset += length
                cu_q.append(offset)
                cu_seqlen_q = torch.Tensor(cu_q).to(device=device, dtype=torch.int32)

                assert max(indices) < input_ids.shape[0]

                cu_seqlen_k = (
                    torch.arange(
                        n_images + 1,
                        device=device,
                        dtype=torch.int32,
                    )
                    * seqlen_k
                )
                max_q = cu_seqlen_q[-1].item()
                max_k = seqlen_k
            else:
                cu_seqlen_q = torch.arange(seqlen_q + 1, device=device, dtype=torch.int32)
                seqlen_k = cross_attention_states.shape[1]
                n_images = cross_attention_states.shape[0]
                cu_seqlen_k = (
                    torch.arange(
                        n_images + 1,
                        device=device,
                        dtype=torch.int32,
                    )
                    * seqlen_k
                )
                max_q = seqlen_q
                max_k = seqlen_k
                indices = image_indices[:]

            cross_attention_states = (
                cross_attention_states,
                cu_seqlen_q,
                cu_seqlen_k,
                max_q,
                max_k,
                indices,
            )

        outputs = self.text_model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            seqlen=seqlen,
            max_s=max_s,
            adapter_data=adapter_data,
            prefill_cache_indices=prefill_cache_indices,
            lm_head_indices=lm_head_indices,
            cross_attention_states=cross_attention_states,
            skip_lm_head=skip_lm_head,
        )

        return outputs
