# transfromers version 4.36.2
# https://github.com/datamllab/LongLM/tree/master
# Should work for 'susnato/phi-2', a hf version of microsfot/phi-2, check the detail in Huggingface Hub. 
# Haven't tested it! ! !
import math
from typing import Optional, Tuple
from transformers.cache_utils import Cache
import torch
import torch.utils.checkpoint
from torch import nn




# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
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
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin) if q is not None else None
    k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None
    return q_embed, k_embed

def apply_group_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1, group_size_1=2, group_size_2=512):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
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
    q_pos = position_ids//group_size_1 + group_size_2 - group_size_2//group_size_1
    k_pos = position_ids//group_size_1 

    q_cos = cos[q_pos].unsqueeze(unsqueeze_dim)
    q_sin = sin[q_pos].unsqueeze(unsqueeze_dim)
    k_cos = cos[k_pos].unsqueeze(unsqueeze_dim)
    k_sin = sin[k_pos].unsqueeze(unsqueeze_dim)
    q_embed = (q * q_cos) + (rotate_half(q) * q_sin) if q is not None else None
    k_embed = (k * k_cos) + (rotate_half(k) * k_sin) if k is not None else None
    return q_embed, k_embed


def self_extend_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    group_size_1: int = 2,
    group_size_2: int = 512,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    # [batch_size, seq_length, 3 x hidden_size]
    fused_qkv = self.query_key_value(hidden_states)

    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_states, key_states, value_states) = self._split_heads(fused_qkv)

    if self.qk_layernorm:
        query_states = self.q_layernorm(query_states)
        key_states = self.k_layernorm(key_states)

    # [batch_size, num_heads, seq_length, head_dim] -> [batch_size, seq_length, num_heads, head_dim]
    query_states = query_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        #print(kv_seq_len)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)


    if past_key_value is not None:
        # Specific to RoPE models with partial rotation
        cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": self.rotary_emb.dim}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    # The new Cache class does not support add one more key_states, have to do RoPE computation later.

    # Partial rotary embedding
    query_rot, query_pass = (
        query_states[..., : self.rotary_emb.dim],
        query_states[..., self.rotary_emb.dim :],
    )
    key_rot, key_pass = (
        key_states[..., : self.rotary_emb.dim],
        key_states[..., self.rotary_emb.dim :],
    )
    # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
    k_pos = torch.arange(kv_seq_len, device=position_ids.device).view(bsz, kv_seq_len)
 # need to recompute 
    q_pos = position_ids

    neighbor_query_rot, _ = apply_rotary_pos_emb(query_rot, None, cos, sin, q_pos)
    _, neighbor_key_rot = apply_rotary_pos_emb(None, key_rot, cos, sin, k_pos)

    _re_group_size_2 = 0 if position_ids.max() < group_size_2 else group_size_2 # in case that, the smallest q position, g2-g2//g1 exceed the max position
    group_query_rot, _ = apply_group_rotary_pos_emb(query_rot, None, cos, sin, q_pos, group_size_1=group_size_1, group_size_2=_re_group_size_2)
    _, group_key_rot = apply_group_rotary_pos_emb(None, key_rot, cos, sin, k_pos, group_size_1=group_size_1, group_size_2=_re_group_size_2)

    # [batch_size, seq_length, num_heads, head_dim]
    neighbor_query_states = torch.cat((neighbor_query_rot, query_pass), dim=-1)
    neighbor_key_states = torch.cat((neighbor_key_rot, key_pass), dim=-1)

    group_query_states = torch.cat((group_query_rot, query_pass), dim=-1)
    group_key_states = torch.cat((group_key_rot, key_pass), dim=-1)


    neighbor_attn_weights = torch.matmul(neighbor_query_states, neighbor_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    group_attn_weights = torch.matmul(group_query_states, group_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)


    if neighbor_attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {neighbor_attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        neighbor_attn_weights = neighbor_attn_weights + attention_mask
        group_attn_weights = group_attn_weights + attention_mask
     
    if q_len == 1:
        neighbor_attention_mask = torch.zeros((q_len, kv_seq_len), device=neighbor_attn_weights.device)
        neighbor_attention_mask[:, -group_size_2:] = 1
    elif q_len == kv_seq_len:
        neighbor_attention_mask = torch.ones((q_len, kv_seq_len), device=neighbor_attn_weights.device)
        neighbor_attention_mask = torch.tril(neighbor_attention_mask)
        if q_len > group_size_2:
            # seq length is larger than group_size_2, should do replacement. 
            group_attention_mask =  torch.tril(torch.ones((q_len-group_size_2, kv_seq_len-group_size_2), device=group_attn_weights.device))
            neighbor_attention_mask[group_size_2:, :-group_size_2] -= group_attention_mask
    else:
        raise ValueError("q_len should be 1 or seq_len.")
    
    merged_attn_weights = torch.where(neighbor_attention_mask.bool(), neighbor_attn_weights, group_attn_weights) # replace the group attention with neighbor attention within the neighbor window. 
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(merged_attn_weights, dtype=torch.float32, dim=-1).to(query_states.dtype)
    attn_weights = self.attention_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.dense(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value