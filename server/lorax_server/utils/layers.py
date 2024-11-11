import math
import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.distributed
from accelerate import init_empty_weights
from torch import nn
from torch.nn import functional as F

from lorax_server.adapters.types import LORA, MEDUSA
from lorax_server.layers.linear import FastLinear, get_linear  # noqa: F401
from lorax_server.layers.tensor_parallel import SuperLayer, TensorParallelColumnLinear, TensorParallelHead  # noqa: F401
from lorax_server.utils.lora import LM_HEAD
from lorax_server.utils.punica import (
    add_lora_a_bgmv,
    add_lora_b_bgmv,
    has_sgmv,
    lora_a_sgmv_cutlass,
    lora_b_sgmv_cutlass,
    orient_for_rank,
)
from lorax_server.utils.state import get_speculative_tokens, is_warmup

if TYPE_CHECKING:
    from lorax_server.adapters import AdapterBatchData
    from lorax_server.adapters.lora import BatchLoraWeights
    from lorax_server.adapters.medusa import BatchMedusaWeights


# Monkey patching
@classmethod
def load_layer_norm(cls, prefix, weights, eps):
    weight = weights.get_tensor(f"{prefix}.weight")
    bias = weights.get_tensor(f"{prefix}.bias")
    with init_empty_weights():
        ln = cls(weight.shape, eps=eps)

    ln.weight = nn.Parameter(weight)
    ln.bias = nn.Parameter(bias)
    return ln


@classmethod
def load_layer_norm_no_bias(cls, prefix, weights, eps):
    weight = weights.get_tensor(f"{prefix}.weight")
    with init_empty_weights():
        ln = cls(weight.shape, eps=eps)

    ln.weight = nn.Parameter(weight)
    ln.bias = None
    return ln


torch.nn.LayerNorm.load = load_layer_norm
torch.nn.LayerNorm.load_no_bias = load_layer_norm_no_bias


class LoraLinear(nn.Module):
    def __init__(self, base_layer, layer_id, process_group):
        super().__init__()
        self.base_layer = base_layer
        self.layer_id = layer_id
        self.process_group = process_group

    def forward_layer_type(
        self,
        result: torch.Tensor,
        input: torch.Tensor,
        adapter_data: "AdapterBatchData",
        layer_type: str,
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor:
        data = adapter_data.data.get(layer_type)
        data: Optional["BatchLoraWeights"] = data.get(LORA) if data is not None else None
        can_vectorize = data is not None and data.can_vectorize(self.process_group)

        # Triton Punica kernels
        key = (layer_type, self.layer_id)
        if (
            adapter_data.punica_wrapper is not None
            and adapter_data.punica_wrapper.enabled
            and key in adapter_data.layer_to_lora_weights
            and input.shape[0] <= adapter_data.punica_wrapper.max_batch_size
            and can_vectorize
        ):
            if end_idx - start_idx != result.shape[1]:
                y_offset = start_idx
                y_slice_size = end_idx - start_idx
            else:
                y_offset = None
                y_slice_size = None

            lora_a_weights, lora_b_weights = adapter_data.layer_to_lora_weights[key]
            adapter_data.punica_wrapper.add_lora(
                result,
                input,
                lora_a_weights,
                lora_b_weights,
                1.0,
                y_offset,
                y_slice_size,
                callback=self.collect_lora_a if self.process_group.size() > 1 else None,
            )

        # Legacy Punica kernels
        elif has_sgmv() and can_vectorize:
            if end_idx - start_idx != result.shape[1]:
                proj = torch.zeros_like(result[:, start_idx:end_idx])
            else:
                proj = result

            for r, rank_segments in data.rank_data.items():
                lora_a_ptr = rank_segments.lora_a_ptr
                lora_b_ptr = rank_segments.lora_b_ptr

                if data.use_sgmv:
                    # Use SGMV for prefill
                    if lora_a_ptr is not None and lora_b_ptr is not None:
                        v = lora_a_sgmv_cutlass(
                            input,
                            rank_segments.tmp_shrink,
                            lora_a_ptr,
                            rank_segments.segment_starts,
                            rank_segments.segment_ends,
                            self.layer_id,
                            r,
                        )

                        if self.process_group.size() > 1:
                            v = self.collect_lora_a(v)

                        lora_b_sgmv_cutlass(
                            proj,
                            v,
                            rank_segments.tmp_expand,
                            lora_b_ptr,
                            rank_segments.segment_starts,
                            rank_segments.segment_ends,
                            self.layer_id,
                        )
                else:
                    # Use BGMV for decode
                    if lora_a_ptr is not None and lora_b_ptr is not None:
                        v = torch.zeros((input.size(0), r), dtype=input.dtype, device=input.device)
                        add_lora_a_bgmv(
                            v,
                            input,
                            lora_a_ptr,
                            rank_segments.indices,
                            self.layer_id,
                        )

                        if self.process_group.size() > 1:
                            v = self.collect_lora_a(v)

                        add_lora_b_bgmv(
                            proj,
                            v,
                            lora_b_ptr,
                            rank_segments.indices,
                            self.layer_id,
                        )

            if end_idx - start_idx != result.shape[1]:
                result[:, start_idx:end_idx] += proj

        # Vanilla PyTorch
        else:
            adapter_indices = adapter_data.meta.adapter_indices
            if data is not None and data.prefill_head_indices is not None and data.layer_name == LM_HEAD:
                # LM_HEAD inputs have different shape during prefill than other layers
                adapter_indices = adapter_indices[data.prefill_head_indices]

            speculative_tokens = get_speculative_tokens()
            for adapter_index in adapter_data.meta.adapter_set:
                if data is not None and data.has_adapter(adapter_index):
                    adapter_mask = (adapter_indices == adapter_index).to(input.dtype).view(-1, 1)

                    # If we're doing speculative decoding, then the input will have 3D shape:
                    # (batch_size, seq_len, hidden_size)
                    # If the input shape is not 3D though, then this means we skipped speculation because the
                    # batch size was too large
                    if speculative_tokens > 0 and len(input.shape) == 3:
                        # Expand adapter mask to cover the speculative tokens
                        adapter_mask = adapter_mask.repeat_interleave(speculative_tokens + 1, dim=1).unsqueeze(dim=2)

                    layer_result = self.forward_lora(input, data, adapter_index, adapter_mask)
                    result[:, start_idx:end_idx] += layer_result

        return result

    def forward_lora(
        self,
        input: torch.Tensor,
        data: "BatchLoraWeights",
        adapter_index: int,
        adapter_mask: torch.Tensor,
    ) -> torch.Tensor:
        lora_a = data.lora_a[adapter_index][self.layer_id, :, :]
        lora_b = data.lora_b[adapter_index][self.layer_id, :, :]

        lora_a = orient_for_rank(lora_a, lora_b.size(0))

        a_out = input @ lora_a
        if self.process_group.size() > 1:
            a_out = self.collect_lora_a(a_out)

        result = (a_out @ lora_b) * adapter_mask
        return result

    def collect_lora_a(self, a_out: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implemented in subclasses")


class TensorParallelMultiAdapterLinear(LoraLinear):
    def __init__(self, base_layer, layer_id, layer_names, sizes, process_group):
        super().__init__(base_layer, layer_id, process_group)
        self.layer_names = layer_names
        self.sizes = sizes

    @classmethod
    def load(cls, base_layer, layer_id, layer_names, sizes, process_group):
        return TensorParallelMultiAdapterLinear(base_layer, layer_id, layer_names, sizes, process_group)

    def forward(self, input: torch.Tensor, adapter_data: "AdapterBatchData") -> torch.Tensor:
        result = self.base_layer(input)

        # handle models like Bloom that have inputs of shape
        # (batch_size, sequence_length, hidden_size)
        # we need to reshape them to (batch_size * sequence_length, hidden_size)
        # for the LoRA computation, then reshape back
        prev_shape = result.shape
        is_3d = len(input.shape) >= 3
        if is_3d:
            input = input.reshape(-1, input.shape[-1])
            result = result.reshape(-1, result.shape[-1])

        offset = 0
        for i, layer_name in enumerate(self.layer_names):
            start_idx = offset // self.process_group.size()

            if self.sizes is not None:
                offset += self.sizes[i]
                end_idx = offset // self.process_group.size()
            else:
                end_idx = result.shape[1]

            result = self.forward_layer_type(result, input, adapter_data, layer_name, start_idx, end_idx)

        if is_3d:
            result = result.reshape(prev_shape)

        return result

    def collect_lora_a(self, a_out: torch.Tensor) -> torch.Tensor:
        # Tensor parallel implementation of X @ A@B, where A and B are sharded column-wise.
        # We use an all-gather between X@A and (X@A)@B to ensure alignment across ranks.
        #
        # TODO(travis): this is not very efficient as we do an all-gather for every adapter,
        #   instead we could pre-allocate a (B, a, r) tensor for all adapters with the same
        #   rank, compute `a_out` on each, and then slice them into the buffer as shown here:
        #   https://discuss.pytorch.org/t/concatenate-tensors-without-memory-copying/34609
        gathered_tensors = [torch.empty_like(a_out) for _ in range(self.process_group.size())]
        torch.distributed.all_gather(gathered_tensors, a_out)
        return torch.cat(gathered_tensors, dim=1)


class TensorParallelAdapterRowLinear(LoraLinear):
    def __init__(self, base_layer, layer_id, layer_name, process_group):
        super().__init__(base_layer, layer_id, process_group)
        self.layer_name = layer_name

    @classmethod
    def load(cls, base_layer, layer_id, layer_name, process_group):
        return cls(base_layer, layer_id, layer_name, process_group)

    def forward(self, input: torch.Tensor, adapter_data: "AdapterBatchData") -> torch.Tensor:
        result = self.base_layer(input)

        # Fused all-gather + all-reduce from S-LoRA paper: https://arxiv.org/abs/2311.03285
        stride = result.shape[-1] // self.process_group.size()
        start_idx = self.process_group.rank() * stride
        end_idx = (self.process_group.rank() + 1) * stride

        self.forward_layer_type(result, input, adapter_data, self.layer_name, start_idx, end_idx)

        return result

    def collect_lora_a(self, a_out: torch.Tensor) -> torch.Tensor:
        # Tensor parallel implementation of X @ A@B, where A and B are sharded row-wise.
        # We use an all-reduce between X@A and (X@A)@B to ensure alignment across ranks.
        #
        # TODO(travis): this is not very efficient as we do an all-reduce for every adapter,
        #   instead we could pre-allocate a (B, a, r) tensor for all adapters with the same
        #   rank, compute `a_out` on each, and then slice them into the buffer as shown here:
        #   https://discuss.pytorch.org/t/concatenate-tensors-without-memory-copying/34609
        torch.distributed.all_reduce(a_out, group=self.process_group)
        return a_out


class MultiAdapterHead(TensorParallelAdapterRowLinear):
    def forward(
        self, input: torch.Tensor, adapter_data: "AdapterBatchData"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Medusa
        data = adapter_data.data.get(self.layer_name)
        data: Optional["BatchMedusaWeights"] = data.get(MEDUSA) if data is not None else None

        speculative_logits = None
        if data is not None and data.default_medusa is not None:
            forward = super().forward
            lm_head = lambda x: forward(x, adapter_data)  # noqa: E731
            logits, speculative_logits = data(input, lm_head)

            # TODO(travis): support multiple medusa adapters with masking:
            # for adapter_index in adapter_data.meta.adapter_set:
            #     if data.has_adapter(adapter_index):
            #         adapter_mask = (adapter_data.meta.adapter_indices == adapter_index).to(input.dtype).view(-1, 1)
            #         speculative_logits = data.adapter_to_medusa[adapter_index].model(input)
            #         ...
        else:
            logits = super().forward(input, adapter_data)

        return logits, speculative_logits


class TensorParallelRowLinear(SuperLayer):
    def __init__(self, linear, process_group, all_reduce: bool = True):
        super().__init__(linear)
        self.process_group = process_group
        self.all_reduce = all_reduce

    @classmethod
    def load(
        cls,
        config,
        prefix: str,
        weights,
        bias: bool,
        fan_in_fan_out: bool = False,
        all_reduce: bool = True,
    ):
        weight = weights.get_multi_weights_row(prefix, quantize=config.quantize)

        input_scale, weight_scale = None, None
        if type(weight) is tuple:
            weight, input_scale, weight_scale = weight

        if bias and weights.process_group.rank() == 0:
            # Rank is only on the first rank process
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None

        return cls(
            get_linear(
                weight,
                bias,
                config.quantize,
                fan_in_fan_out=fan_in_fan_out,
                weight_scale=weight_scale,
                input_scale=input_scale,
            ),
            process_group=weights.process_group,
            all_reduce=all_reduce,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = super().forward(input)
        if self.process_group.size() > 1 and self.all_reduce:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out


class TensorParallelEmbedding(nn.Module):
    def __init__(self, prefix: str, weights, reduce=True):
        super().__init__()
        weight = weights.get_partial_sharded(f"{prefix}.weight", dim=0)
        num_embeddings = weights.get_shape(f"{prefix}.weight")[0]

        process_group = weights.process_group

        world_size = process_group.size()
        rank = process_group.rank()

        block_size = num_embeddings // world_size
        self.min_id = rank * block_size
        self.max_id = min(num_embeddings, (rank + 1) * block_size)
        self.null_idx = block_size
        self.process_group = weights.process_group
        self.reduce = reduce

        """Additional 0 entry used for masking"""
        self.weight = nn.Parameter(F.pad(weight, (0, 0, 0, 1)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # default all out of bounds values to `self.null_idx` that will then be mapped to 0
        # translate for [0, self.max_id - self.min_id[
        input = torch.where(
            (self.min_id > input) | (input >= self.max_id),
            self.null_idx,
            input - self.min_id,
        )
        out = torch.nn.functional.embedding(input, self.weight)
        if self.reduce and self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out


try:
    import dropout_layer_norm

    class FastLayerNorm(nn.LayerNorm):
        def forward(self, hidden_states, residual=None):
            if hidden_states.shape[-1] > 8192:
                if residual is not None:
                    hidden_states += residual
                residual = hidden_states

                return super(FastLayerNorm, self).forward(hidden_states), residual
            else:
                (
                    normed_hidden_states,
                    residual,
                    *rest,
                ) = dropout_layer_norm.dropout_add_ln_fwd(
                    hidden_states,
                    residual,
                    self.weight,
                    self.bias,
                    None,
                    None,
                    None,
                    None,
                    0.0,
                    self.eps,
                    1.0,
                    0,
                    None,
                    False,
                    False,
                )
                if residual is None:
                    residual = hidden_states

                return normed_hidden_states, residual

except ImportError:
    pass


try:
    import rotary_emb
    from flash_attn.layers.rotary import RotaryEmbedding  # noqa: F401

    def _create_inv_freq(dim, base, device):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        return inv_freq

    def _get_rope_config(config):
        if os.getenv("ROPE_SCALING", None) is not None:
            rope_scaling = {
                "type": os.environ["ROPE_SCALING"],
                "factor": float(os.environ["ROPE_FACTOR"]),
            }
            return rope_scaling
        return getattr(config, "rope_scaling", None)

    class PositionRotaryEmbedding(nn.Module):
        def __init__(self, inv_freq, scaling_factor, max_position_embeddings, device, dtype):
            super().__init__()
            self.inv_freq = inv_freq
            self._seq_len_cached = 0
            self._cos_cached = None
            self._sin_cached = None
            self._cos_k_cached = None
            self._sin_k_cached = None
            self.scaling_factor = scaling_factor
            self.dynamic_args = None
            self._update_cos_sin_cache(dtype, device, max_position_embeddings)

        @classmethod
        def static(cls, config, dim, base, device, dtype):
            inv_freq = _create_inv_freq(dim, base, device)
            scaling_factor = None
            rope_scaling = _get_rope_config(config)
            if rope_scaling is not None:
                rope_scaling = rope_scaling.copy()
                rope_type = rope_scaling.pop("rope_type", rope_scaling.pop("type", None))
                if rope_type == "linear":
                    pass
                elif rope_type == "dynamic":
                    scaling_factor = rope_scaling["factor"]
                    return DynamicPositionRotaryEmbedding(
                        dim=dim,
                        max_position_embeddings=config.max_position_embeddings,
                        base=base,
                        device=inv_freq.device,
                        dtype=dtype,
                        scaling_factor=scaling_factor,
                    )
                elif rope_type == "llama3":
                    inv_freq = apply_llama3_scaling(
                        inv_freq,
                        scaling_factor=rope_scaling["factor"],
                        low_freq_factor=rope_scaling["low_freq_factor"],
                        high_freq_factor=rope_scaling["high_freq_factor"],
                        original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
                    )
                    return cls(
                        inv_freq,
                        scaling_factor,
                        max_position_embeddings=config.max_position_embeddings,
                        device=inv_freq.device,
                        dtype=dtype,
                    )
                elif rope_type == "yarn":
                    scaling_factor = rope_scaling["factor"]
                    return YarnPositionRotaryEmbedding(
                        dim=dim,
                        max_position_embeddings=config.max_position_embeddings,
                        base=base,
                        device=inv_freq.device,
                        dtype=dtype,
                        **rope_scaling,
                    )
                elif rope_type in ["su", "longrope"]:
                    short_factor = torch.tensor(rope_scaling["short_factor"], dtype=torch.float32, device=device)
                    short_inv_freq = 1.0 / (
                        short_factor * base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
                    )
                    long_factor = torch.tensor(rope_scaling["long_factor"], dtype=torch.float32, device=device)
                    long_inv_freq = 1.0 / (
                        long_factor * base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
                    )

                    original_max_position_embeddings = config.original_max_position_embeddings
                    max_position_embeddings = config.max_position_embeddings
                    if max_position_embeddings <= original_max_position_embeddings:
                        scaling_factor = 1.0
                    else:
                        scale = max_position_embeddings / original_max_position_embeddings
                        scaling_factor = math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))

                    return SuRotaryEmbedding(
                        short_inv_freq=short_inv_freq,
                        long_inv_freq=long_inv_freq,
                        scaling_factor=scaling_factor,
                        original_max_position_embeddings=original_max_position_embeddings,
                    )
                else:
                    raise NotImplementedError(f"rope scaling type {rope_type} is not implemented or invalid")
            return cls(inv_freq, scaling_factor, config.max_position_embeddings, device, dtype)

        @classmethod
        def load(cls, config, prefix, weights):
            # XXX: Always load this in float32 !
            dtype = weights.dtype
            weights.dtype = torch.float32
            inv_freq = weights.get_tensor(f"{prefix}.inv_freq")
            weights.dtype = dtype

            scaling_factor = None
            rope_scaling = _get_rope_config(config)
            if rope_scaling is not None:
                rope_scaling = rope_scaling.copy()
                scaling_factor = rope_scaling["factor"]
                rope_type = rope_scaling.pop("type")
                if rope_type == "linear":
                    pass
                elif rope_type == "dynamic":
                    return DynamicPositionRotaryEmbedding(
                        dim=2 * inv_freq.shape[0],
                        max_position_embeddings=config.max_position_embeddings,
                        base=10000.0,
                        device=inv_freq.device,
                        dtype=dtype,
                        scaling_factor=scaling_factor,
                    )
                elif rope_type == "yarn":
                    return YarnPositionRotaryEmbedding(
                        dim=2 * inv_freq.shape[0],
                        max_position_embeddings=config.max_position_embeddings,
                        base=10000.0,
                        device=inv_freq.device,
                        dtype=dtype,
                        **rope_scaling,
                    )
                else:
                    raise NotImplementedError(f"rope scaling type {rope_type} is not implemented or invalid")
            return cls(inv_freq, scaling_factor)

        def _update_cos_sin_cache(self, dtype, device, seqlen):
            # Reset the tables if the sequence length has changed,
            # or if we're on a new device (possibly due to tracing for instance)
            if seqlen > self._seq_len_cached or self._cos_cached.device != device or self._cos_cached.dtype != dtype:
                self._seq_len_cached = seqlen
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                if self.scaling_factor is not None:
                    t /= self.scaling_factor
                # Don't do einsum, it converts fp32 to fp16
                # freqs = torch.einsum("i,j->ij", t, self.inv_freq)

                freqs = torch.outer(t, self.inv_freq.to(device=t.device))
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)

        def get_cos_sin(self, position_ids: torch.Tensor, max_s: int, dtype: torch.dtype):
            """
            Return cos and sin for the asked position ids
            """

            # When using dynamic position embeddings, the max sequence length might exceed
            # the max position embeddings of the base model, so we need to update our
            # cache during warmup.
            # This should never result in a change after warmup, otherwise we break
            # cuda graphs.
            if is_warmup():
                self._update_cos_sin_cache(dtype, position_ids.device, max_s)

            cos = torch.index_select(self._cos_cached, 0, position_ids)
            sin = torch.index_select(self._sin_cached, 0, position_ids)
            return cos.unsqueeze(1), sin.unsqueeze(1)

        def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
            rotary_dim = cos.shape[-1]
            x1 = x[..., :rotary_dim]
            x2 = x[..., rotary_dim : 2 * rotary_dim]

            rotary_emb.apply_rotary(x1, x2, cos, sin, x1, x2, False)
            return x

    class DynamicPositionRotaryEmbedding(PositionRotaryEmbedding):
        def __init__(self, dim, max_position_embeddings, base, device, dtype, scaling_factor):
            inv_freq = _create_inv_freq(dim, base, device)
            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base
            super().__init__(inv_freq, scaling_factor, max_position_embeddings, device, dtype)

        def _update_cos_sin_cache(self, dtype, device, seqlen):
            # Reset the tables if the sequence length has changed,
            # or if we're on a new device (possibly due to tracing for instance)
            if seqlen > self._seq_len_cached or self._cos_cached.device != device or self._cos_cached.dtype != dtype:
                if seqlen > self.max_position_embeddings:
                    newbase = self.base * (
                        (self.scaling_factor * seqlen / self.max_position_embeddings) - (self.scaling_factor - 1)
                    ) ** (self.dim / (self.dim - 2))
                    self.inv_freq = _create_inv_freq(self.dim, newbase, self.inv_freq.device)
                self._seq_len_cached = seqlen
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                # Don't do einsum, it converts fp32 to fp16
                # freqs = torch.einsum("i,j->ij", t, self.inv_freq)

                freqs = torch.outer(t, self.inv_freq.to(device=t.device))
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)

    class YarnPositionRotaryEmbedding(PositionRotaryEmbedding):
        """https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py"""

        def __init__(
            self,
            dim,
            max_position_embeddings=2048,
            base=10000,
            factor=1,
            original_max_position_embeddings=2048,
            extrapolation_factor=1,
            attn_factor=1,
            beta_fast=32,
            beta_slow=1,
            finetuned=True,
            device=None,
            dtype=None,
        ):
            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base
            self.original_max_position_embeddings = original_max_position_embeddings
            self.extrapolation_factor = extrapolation_factor
            self.attn_factor = attn_factor
            self.beta_fast = beta_fast
            self.beta_slow = beta_slow
            self.finetuned = finetuned

            self.yarn(device, factor)
            super().__init__(_create_inv_freq(dim, base, device), factor, max_position_embeddings, device, dtype)

        def _update_cos_sin_cache(self, dtype, device, seqlen):
            if seqlen > self._seq_len_cached or self._cos_cached.device != device or self._cos_cached.dtype != dtype:
                self._seq_len_cached = seqlen

                t = torch.arange(self._seq_len_cached, device=device, dtype=self.inv_freq.dtype)
                freqs = torch.outer(t, self.inv_freq.to(device=t.device))

                self._cos_cached = (torch.cos(freqs) * self.mscale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * self.mscale).to(dtype)

        def yarn(self, device, scaling_factor):
            pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            inv_freq_extrapolation = 1.0 / pos_freqs
            inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

            low, high = find_correction_range(
                self.beta_fast,
                self.beta_slow,
                self.dim,
                self.base,
                self.original_max_position_embeddings,
            )
            inv_freq_mask = (
                1 - linear_ramp_mask(low, high, self.dim // 2).float().to(device)
            ) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
            inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

            self.inv_freq = inv_freq
            self.mscale = float(
                get_mscale(scaling_factor) * self.attn_factor
            )  # Get n-d magnitude scaling corrected for interpolation

    class SuRotaryEmbedding(PositionRotaryEmbedding):
        def __init__(
            self,
            short_inv_freq,
            long_inv_freq,
            scaling_factor,
            original_max_position_embeddings,
        ):
            super(PositionRotaryEmbedding, self).__init__()
            self.short_inv_freq = short_inv_freq
            self.long_inv_freq = long_inv_freq
            self.scaling_factor = scaling_factor
            self.original_max_position_embeddings = original_max_position_embeddings
            self._seq_len_cached = 0
            self._cos_cached = None
            self._sin_cached = None
            self._cos_k_cached = None
            self._sin_k_cached = None
            self.dynamic_args = None

        def _update_cos_sin_cache(self, dtype, device, seqlen):
            # Reset the tables if the sequence length has changed,
            # or if we're on a new device (possibly due to tracing for instance)
            if seqlen > self._seq_len_cached or self._cos_cached.device != device or self._cos_cached.dtype != dtype:
                self._seq_len_cached = seqlen
                if seqlen > self.original_max_position_embeddings:
                    inv_freq = self.long_inv_freq
                else:
                    inv_freq = self.short_inv_freq
                t = torch.arange(seqlen, device=device, dtype=inv_freq.dtype)
                if self.scaling_factor is not None:
                    t /= self.scaling_factor
                # Don't do einsum, it converts fp32 to fp16
                # freqs = torch.einsum("i,j->ij", t, self.inv_freq)

                freqs = torch.outer(t, inv_freq.to(device=t.device))
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)

    # Inverse dim formula to find dim based on number of rotations
    def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    # Find dim range bounds based on rotations
    def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)  # Clamp values just in case

    def linear_ramp_mask(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    def get_mscale(scale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

except ImportError:
    pass


def apply_llama3_scaling(
    freqs: torch.Tensor,
    *,
    scaling_factor: int,
    low_freq_factor: int,
    high_freq_factor: int,
    original_max_position_embeddings: int,
):
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    new_freqs = []

    for freq in freqs:
        wavelen = 2 * math.pi / freq

        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scaling_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scaling_factor + smooth * freq)

    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)
