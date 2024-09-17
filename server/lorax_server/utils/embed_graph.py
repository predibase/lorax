# CUDA Graph implementation modified from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py

from dataclasses import dataclass
from functools import lru_cache
from statistics import median
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from lorax_server.utils.sgmv import BGMV_MAX_RANK

if TYPE_CHECKING:
    from lorax_server.models.flash_causal_lm import FlashCausalLMBatch
    from lorax_server.models.model import Model


# TODO(travis): make this configurable by model / user
MAX_BATCH_SIZE = 256
MAX_RANK = BGMV_MAX_RANK

SLOT_PAD_VALUE = -1
SEGMENT_PAD_VALUE = -1

# Cached batch sizes used in vLLM. This and the helper function `get_cached_batch_size` below
# must be kept in sync.
BATCH_SIZE_INCREMENT = 32
CACHED_BATCH_SIZES = [1, 2, 4, 8, 16] + [BATCH_SIZE_INCREMENT * (i + 1) for i in range(32)]

MAX_SAMPLES = 3


def get_cached_batch_size(batch_size: int) -> int:
    if batch_size == 1:
        return 1
    if batch_size == 2:
        return 2
    if batch_size <= 4:
        return 4
    if batch_size <= 8:
        return 8
    if batch_size <= 16:
        return 16
    return (batch_size + BATCH_SIZE_INCREMENT - 1) // BATCH_SIZE_INCREMENT * BATCH_SIZE_INCREMENT


def pad_and_fill(dest: torch.Tensor, src: torch.Tensor, pad_value: int):
    dest[: src.shape[0]].copy_(src, non_blocking=True)
    dest[src.shape[0] :].fill_(pad_value)


def next_pow_2(x: int) -> int:
    assert x > 0
    return 1 << (x - 1).bit_length()


@dataclass
class GraphState:
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor


@lru_cache(maxsize=1)
def get_max_graph_state(
    device: torch.device,
    max_total_tokens: int,
) -> GraphState:
    input_ids = torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int64, device=device)
    token_type_ids = torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int64, device=device)
    position_ids = torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int32, device=device)

    return GraphState(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        cu_seqlens=cu_seqlens,
    )


class GraphWrapper:
    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        memory_pool: Tuple[int, int],
        input_state: GraphState,
        output_states: Tuple[torch.Tensor, Optional[torch.Tensor]],
        model: nn.Module,
        forward_context: Optional[Callable[..., Any]],
    ):
        self.graph = graph
        self.memory_pool = memory_pool
        self.input_state = input_state
        self.output_states = output_states
        self.model = model
        self.forward_context = forward_context

    @staticmethod
    def trace(
        model: nn.Module,
        device: torch.device,
        forward_context: Callable[..., Any],
        batch_size: int,
        memory_pool: Tuple[int, int],
        max_total_tokens: int,
        num_heads: int,
        num_kv_heads: int,
    ) -> "GraphWrapper":
        max_input_state = get_max_graph_state(device, max_total_tokens)

        input_state = GraphState(
            input_ids=max_input_state.input_ids[:batch_size],
            token_type_ids=max_input_state.token_type_ids[:batch_size],
            position_ids=max_input_state.position_ids[:batch_size],
            cu_seqlens=max_input_state.cu_seqlens[:batch_size],
        )

        torch.cuda.synchronize(device)

        with forward_context(cu_seqlens=input_state.cu_seqlens):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=memory_pool):  # noqa: SIM117
                output_states = model.forward(
                    input_ids=input_state.input_ids,
                    token_type_ids=input_state.token_type_ids,
                    position_ids=input_state.position_ids,
                    cu_seqlens=input_state.cu_seqlens,
                    max_s=max_total_tokens,
                )

        torch.cuda.synchronize(device)

        return GraphWrapper(graph, graph.pool(), input_state, output_states, model, forward_context)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor],
        max_s: int,
    ) -> None:
        pad_and_fill(self.input_state.input_ids, input_ids, 0)
        pad_and_fill(self.input_state.token_type_ids, token_type_ids, 0)
        pad_and_fill(self.input_state.position_ids, position_ids, 0)

        with self.forward_context(cu_seqlens=cu_seqlens):
            self.graph.replay()

        return tuple(state[: input_ids.shape[0]] if state is not None else None for state in self.output_states)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class EmbedGraphCache:
    def __init__(
        self,
        model: "Model",
        device: torch.device,
        forward_context: Callable[..., Any],
        max_total_tokens: int,
        num_heads: int,
        num_kv_heads: int,
    ):
        self.model = model
        self.device = device
        self.forward_context = forward_context
        self.memory_pool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.cache: Dict[Tuple[int, int], GraphWrapper] = {}
        self.max_total_tokens = max_total_tokens
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

    def can_use_graph(
        self,
        batch: "FlashCausalLMBatch",
    ) -> bool:
        batch_size = batch.input_ids.shape[0]
        max_s = batch.max_seqlen

        # TODO(travis): allow using CUDA graphs with multi-rank batches
        return (
            torch.cuda.is_available()
            and batch_size <= MAX_BATCH_SIZE
            and max_s <= self.max_total_tokens
        )

    def get_estimated_cache_memory(self) -> int:
        # Store off graphs into temporary cache to discard after estimation
        tmp_cache = {}
        pool = None

        # Use the largest batch size to overestimate memory overhead
        batch_size = CACHED_BATCH_SIZES[-1]

        samples = []
        torch.cuda.synchronize(self.device)

        key = (batch_size,)
        graph = GraphWrapper.trace(
            self.model,
            self.device,
            self.forward_context,
            batch_size,
            pool,
            self.max_total_tokens,
            self.num_heads,
            self.num_kv_heads,
        )
        tmp_cache[key] = graph
        pool = graph.memory_pool
        torch.cuda.synchronize(self.device)

        # Estimate memory usage for all batch sizes and ranks
        ngraphs = len(CACHED_BATCH_SIZES)
        per_graph_memory = median(samples)
        return ngraphs * per_graph_memory

    def warmup(self):
        ngraphs = len(CACHED_BATCH_SIZES)
        pool = None
        with tqdm(total=ngraphs, desc="Trace CUDA graphs") as pbar:
            for batch_size in reversed(CACHED_BATCH_SIZES):
                pbar.set_postfix({"batch_size": batch_size})
                key = (batch_size,)
                graph = GraphWrapper.trace(
                    self.model,
                    self.device,
                    self.forward_context,
                    batch_size,
                    pool,
                    self.max_total_tokens,
                    self.num_heads,
                    self.num_kv_heads,
                )
                self.cache[key] = graph
                pool = graph.memory_pool
                pbar.update(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor],
        max_s: int,
        **kwargs,
    ) -> None:
        batch_size = get_cached_batch_size(input_ids.shape[0])

        key = (batch_size,)
        graph = self.cache.get(key)
        if graph is None:
            graph = GraphWrapper.trace(
                self.model,
                self.device,
                self.forward_context,
                batch_size,
                self.memory_pool,
                self.max_total_tokens,
                self.num_heads,
                self.num_kv_heads,
            )
            self.cache[key] = graph

        output_states = graph.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_s=max_s,
        )

        return output_states

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
