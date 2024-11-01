# CUDA Graph implementation modified from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py

import os
from dataclasses import dataclass
from functools import lru_cache
from statistics import median
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from loguru import logger
from torch import nn
from tqdm import tqdm

from lorax_server.adapters import AdapterBatchData, AdapterBatchMetadata
from lorax_server.adapters.lora import BatchLoraWeights, RankSegments
from lorax_server.adapters.types import LORA
from lorax_server.models.metadata_kernels import block_tables_to_ragged
from lorax_server.utils.attention.common import Seqlen
from lorax_server.utils.punica import BGMV_MAX_RANK, PunicaWrapper
from lorax_server.utils.state import BLOCK_SIZE, FLASH_INFER, get_speculative_tokens

if TYPE_CHECKING:
    from lorax_server.models.flash_causal_lm import FlashCausalLMBatch
    from lorax_server.models.model import Model


MAX_BATCH_SIZE = int(os.environ.get("LORAX_COMPILE_MAX_BATCH_SIZE", 96))
MAX_RANK = int(os.environ.get("LORAX_COMPILE_MAX_RANK", 64))

SLOT_PAD_VALUE = -1
SEGMENT_PAD_VALUE = -1

# Cached batch sizes used in vLLM. This and the helper function `get_cached_batch_size` below
# must be kept in sync.
BATCH_SIZE_INCREMENT = 32

# set CACHED_BATCH_SIZES to 1, 2, 3, 4, 8, 16 and then increments of BATCH_SIZE_INCREMENT up to MAX_BATCH_SIZE
CACHED_BATCH_SIZES = [1, 2, 3, 4, 8, 16] + [
    BATCH_SIZE_INCREMENT * (i + 1) for i in range(MAX_BATCH_SIZE // BATCH_SIZE_INCREMENT)
]
CACHED_BATCH_SIZES = [b for b in CACHED_BATCH_SIZES if b <= MAX_BATCH_SIZE]

# Include 0 to ensure we can use cuda graphs without adapters
# TODO(travis): use padding to allow for more ranks without increasing memory usage
CACHED_MAX_RANKS = [0, 8, 16, 32, 64, 128]
CACHED_MAX_RANKS = [r for r in CACHED_MAX_RANKS if r <= MAX_RANK]
_allowed_ranks = set(CACHED_MAX_RANKS)

assert all([r <= BGMV_MAX_RANK for r in _allowed_ranks]), f"Invalid ranks: {_allowed_ranks}"

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
    position_ids: torch.Tensor
    block_tables: torch.Tensor
    slots: torch.Tensor
    seqlen: Seqlen
    input_lengths: List[int]
    cache_lengths: List[int]
    cache_lengths_tensor: torch.Tensor
    adapter_data: AdapterBatchData
    traced_adapter_layer_names: Set[str]
    state: Any = None


@lru_cache(maxsize=1)
def get_max_graph_state(
    device: torch.device,
    adapter_layers: Tuple[str],
    max_total_tokens: int,
    sliding_window_blocks: Optional[int] = None,
) -> GraphState:
    max_num_blocks = (max_total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    if sliding_window_blocks is not None:
        # Needed blocks can not go over SLIDING_WINDOW_BLOCKS
        max_num_blocks = max(max_num_blocks, sliding_window_blocks)

    block_tables_arr = np.zeros((MAX_BATCH_SIZE, max_num_blocks), dtype=np.int32)
    block_tables = torch.from_numpy(block_tables_arr).to(device=device)

    input_ids = torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int64, device=device)
    position_ids = torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int32, device=device)
    slots = torch.full((MAX_BATCH_SIZE,), SLOT_PAD_VALUE, dtype=torch.int64, device=device)
    input_lengths = torch.full((MAX_BATCH_SIZE,), max_total_tokens, dtype=torch.int32, device=device)
    cache_lengths = [0] * MAX_BATCH_SIZE
    cache_lengths_tensor = torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int32, device=device)

    adapter_weight_data = {}
    for layer_name in adapter_layers:
        adapter_weight_data[layer_name] = BatchLoraWeights(
            lora_a={},
            lora_b={},
            adapter_index_configs={},
            rank_data={
                MAX_RANK: RankSegments(
                    rank=MAX_RANK,
                    lora_a_ptr=torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int64, device=device),
                    lora_b_ptr=torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int64, device=device),
                    indices=torch.full((MAX_BATCH_SIZE,), SEGMENT_PAD_VALUE, dtype=torch.int64, device=device),
                    segment_starts=None,
                    segment_ends=None,
                    tmp_shrink=None,
                    tmp_expand=None,
                ),
            },
            use_sgmv=False,  # bgmv during decode
            layer_name=layer_name,
            prefill_head_indices=None,
        )

    return GraphState(
        input_ids=input_ids,
        position_ids=position_ids,
        block_tables=block_tables,
        slots=slots,
        seqlen=Seqlen(
            input_lengths=input_lengths,
            cache_lengths=cache_lengths_tensor,
            cu_seqlen_q=None,
            max_q=1,
            max_k=max_total_tokens,
        ),
        input_lengths=input_lengths.tolist(),
        cache_lengths=cache_lengths,
        cache_lengths_tensor=cache_lengths_tensor,
        adapter_data=AdapterBatchData(
            meta=AdapterBatchMetadata(
                adapter_indices=torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int64, device=device),
                adapter_list=[],
                adapter_set=set(),
                adapter_segments=torch.zeros((MAX_BATCH_SIZE,), dtype=torch.int64, device=device),
                segment_indices=[],
            ),
            layer_to_lora_weights={},
            punica_wrapper=None,
            data=adapter_weight_data,
            prefill=False,
        ),
        traced_adapter_layer_names=set(adapter_layers),
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
        kv_cache: Dict,
        adapter_layers: Tuple[str],
        forward_context: Callable[..., Any],
        batch_size: int,
        max_rank: int,
        memory_pool: Tuple[int, int],
        max_total_tokens: int,
        num_heads: int,
        num_kv_heads: int,
        sliding_window_blocks: Optional[int] = None,
        traced_adapter_layer_names: Optional[Set[str]] = None,
        layer_to_lora_weights: Dict[str, Dict[str, Any]] = {},
        punica_wrapper: Optional[PunicaWrapper] = None,
    ) -> "GraphWrapper":
        max_input_state = get_max_graph_state(device, adapter_layers, max_total_tokens, sliding_window_blocks)

        # WARNING: for some reason the SGMV kernel can hang if we don't use a power of 2
        # as the segment size. This is a workaround until we can figure out why.
        # Specifically, this issue has been observed with batch_size=96.
        # I suspect it is related to synchronization and the chunk size (256) used in the kernel.
        # But we need to investigate further.
        segment_size = next_pow_2(batch_size)

        traced_adapter_layer_names = traced_adapter_layer_names or set()

        adapter_weight_data = {}
        for layer_name, weight_data in max_input_state.adapter_data.data.items():
            if layer_name not in traced_adapter_layer_names:
                continue

            adapter_weight_data[layer_name] = {
                LORA: BatchLoraWeights(
                    lora_a={},
                    lora_b={},
                    adapter_index_configs={},
                    rank_data=(
                        {
                            max_rank: RankSegments(
                                rank=max_rank,
                                lora_a_ptr=weight_data.rank_data[MAX_RANK].lora_a_ptr[:segment_size],
                                lora_b_ptr=weight_data.rank_data[MAX_RANK].lora_b_ptr[:segment_size],
                                indices=weight_data.rank_data[MAX_RANK].indices[:batch_size],
                                segment_starts=None,
                                segment_ends=None,
                                tmp_shrink=None,
                                tmp_expand=None,
                            ),
                        }
                        if max_rank > 0
                        else {}
                    ),
                    use_sgmv=False,  # bgmv during decode
                    layer_name=layer_name,
                    prefill_head_indices=None,
                )
            }

        block_tables = max_input_state.block_tables[:batch_size]
        input_lengths = max_input_state.input_lengths[:batch_size]
        input_lengths_tensor = max_input_state.seqlen.input_lengths[:batch_size]
        cache_lengths = max_input_state.cache_lengths[:batch_size]
        cache_lengths_tensor = max_input_state.cache_lengths_tensor[:batch_size]
        state = None

        if FLASH_INFER:
            from lorax_server.utils.flashinfer_attention import (
                create_decode_state_cuda_graphs,
            )

            block_tables = block_tables_to_ragged(
                block_tables=block_tables,
                input_lengths=input_lengths,
                cache_lengths=cache_lengths,
                input_lengths_tensor=input_lengths_tensor,
                cache_lengths_tensor=cache_lengths_tensor,
                max_current_length=max_total_tokens,
            )

            block_tables_ptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
            last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

            state = create_decode_state_cuda_graphs(
                device=max_input_state.input_ids.device,
                block_tables=block_tables,
                block_tables_ptr=block_tables_ptr,
                last_page_len=last_page_len,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )

        meta = AdapterBatchMetadata(
            adapter_indices=max_input_state.adapter_data.meta.adapter_indices[:batch_size],
            adapter_list=max_input_state.adapter_data.meta.adapter_list,
            adapter_set=max_input_state.adapter_data.meta.adapter_set,
            adapter_segments=max_input_state.adapter_data.meta.adapter_segments[:batch_size],
            segment_indices=max_input_state.adapter_data.meta.segment_indices,
        )
        punica_wrapper.update_metadata(meta=meta, prefill=False)

        input_state = GraphState(
            input_ids=max_input_state.input_ids[:batch_size],
            position_ids=max_input_state.position_ids[:batch_size],
            block_tables=block_tables,
            slots=max_input_state.slots[:batch_size],
            seqlen=Seqlen(
                input_lengths=input_lengths_tensor,
                cache_lengths=cache_lengths_tensor,
                cu_seqlen_q=None,
                max_q=1,
                max_k=max_total_tokens,
            ),
            input_lengths=input_lengths,
            cache_lengths=cache_lengths,
            cache_lengths_tensor=cache_lengths_tensor,
            adapter_data=AdapterBatchData(
                meta=meta,
                layer_to_lora_weights=layer_to_lora_weights,
                punica_wrapper=punica_wrapper,
                data=adapter_weight_data,
                prefill=False,
            ),
            traced_adapter_layer_names=traced_adapter_layer_names,
            state=state,
        )

        torch.cuda.synchronize(device)

        with forward_context(
            block_tables=input_state.block_tables,
            cu_seqlen_prefill=None,
            input_lengths=input_lengths,
            input_lengths_tensor=input_state.seqlen.input_lengths,
            cache_lengths=cache_lengths,
            cache_lengths_tensor=cache_lengths_tensor,
            state=input_state.state,
        ):
            # warmup
            output_states = model.forward(
                input_ids=input_state.input_ids,
                position_ids=input_state.position_ids,
                cu_seqlen_prefill=None,
                kv_cache=kv_cache,
                block_tables=input_state.block_tables,
                slots=input_state.slots,
                seqlen=input_state.seqlen,
                max_s=max_total_tokens,
                adapter_data=input_state.adapter_data,
                prefill_cache_indices=None,
                lm_head_indices=None,
                skip_lm_head=get_speculative_tokens() > 0,
            )
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=memory_pool):  # noqa: SIM117
                output_states = model.forward(
                    input_ids=input_state.input_ids,
                    position_ids=input_state.position_ids,
                    cu_seqlen_prefill=None,
                    kv_cache=kv_cache,
                    block_tables=input_state.block_tables,
                    slots=input_state.slots,
                    seqlen=input_state.seqlen,
                    max_s=max_total_tokens,
                    adapter_data=input_state.adapter_data,
                    prefill_cache_indices=None,
                    lm_head_indices=None,
                    skip_lm_head=get_speculative_tokens() > 0,
                )

        torch.cuda.synchronize(device)

        return GraphWrapper(graph, graph.pool(), input_state, output_states, model, forward_context)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        cache_lengths: List[int],
        cache_lengths_tensor: torch.Tensor,
        max_s: int,
        adapter_data: AdapterBatchData,
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> None:
        pad_and_fill(self.input_state.input_ids, input_ids, 0)
        pad_and_fill(self.input_state.position_ids, position_ids, 0)
        pad_and_fill(self.input_state.slots, slots, SLOT_PAD_VALUE)
        pad_and_fill(self.input_state.seqlen.input_lengths, seqlen.input_lengths, 0)
        pad_and_fill(self.input_state.seqlen.cache_lengths, seqlen.cache_lengths, 0)
        self.input_state.cache_lengths[: len(cache_lengths)] = cache_lengths
        pad_and_fill(self.input_state.cache_lengths_tensor, cache_lengths_tensor, 0)

        if FLASH_INFER:
            block_tables = block_tables_to_ragged(
                block_tables=block_tables,
                input_lengths=seqlen.input_lengths,
                cache_lengths=seqlen.cache_lengths,
                input_lengths_tensor=seqlen.input_lengths,
                cache_lengths_tensor=cache_lengths_tensor,
                max_current_length=max_s,
            )
            self.input_state.block_tables[: block_tables.shape[0]] = block_tables
        else:
            self.input_state.block_tables[: block_tables.shape[0], : block_tables.shape[1]] = block_tables

        for layer_name, weight_data in self.input_state.adapter_data.data.items():
            # TODO(travis): generalize this to support other adapter types
            if LORA not in weight_data:
                continue

            lora_data = weight_data[LORA]
            if layer_name not in adapter_data.data:
                # zero out all the segments
                for rank_data in lora_data.rank_data.values():
                    rank_data.indices.fill_(SEGMENT_PAD_VALUE)
                continue

            if LORA not in adapter_data.data[layer_name]:
                continue

            source_data = adapter_data.data[layer_name][LORA]
            dest_data = lora_data
            for rank, source_rank_data in source_data.rank_data.items():
                dest_rank_data = dest_data.rank_data[rank]
                pad_and_fill(dest_rank_data.lora_a_ptr, source_rank_data.lora_a_ptr, 0)
                pad_and_fill(dest_rank_data.lora_b_ptr, source_rank_data.lora_b_ptr, 0)
                pad_and_fill(dest_rank_data.indices, source_rank_data.indices, SEGMENT_PAD_VALUE)

        self.input_state.adapter_data.punica_wrapper.update_metadata(meta=adapter_data.meta, prefill=False)

        with self.forward_context(
            block_tables=self.input_state.block_tables,
            cu_seqlen_prefill=None,
            input_lengths=seqlen.input_lengths,
            input_lengths_tensor=self.input_state.seqlen.input_lengths,
            cache_lengths=self.input_state.cache_lengths,
            cache_lengths_tensor=self.input_state.cache_lengths_tensor,
            state=self.input_state.state,
        ):
            self.graph.replay()

        return tuple(state[: input_ids.shape[0]] if state is not None else None for state in self.output_states)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class GraphCache:
    def __init__(
        self,
        model: "Model",
        device: torch.device,
        kv_cache: Dict,
        adapter_layers: List[str],
        default_traced_adapter_layers: List[str],
        forward_context: Callable[..., Any],
        max_total_tokens: int,
        num_heads: int,
        num_kv_heads: int,
        sliding_window_blocks: Optional[int] = None,
        layer_to_lora_weights: Dict[str, Dict[str, Any]] = {},
        punica_wrapper: Optional[PunicaWrapper] = None,
    ):
        self.model = model
        self.device = device
        self.kv_cache = kv_cache
        self.adapter_layers = tuple(adapter_layers)
        self.default_traced_adapter_layers = set(default_traced_adapter_layers)
        self.forward_context = forward_context
        self.memory_pool = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None
        self.cache: Dict[Tuple[int, int], GraphWrapper] = {}
        self.max_total_tokens = max_total_tokens
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.sliding_window_blocks = sliding_window_blocks
        self.layer_to_lora_weights = layer_to_lora_weights
        self.punica_wrapper = punica_wrapper

    def can_use_graph(
        self,
        batch: "FlashCausalLMBatch",
        adapter_data: AdapterBatchData,
    ) -> bool:
        ranks = adapter_data.ranks()
        nranks = len(ranks)
        max_rank = max(ranks) if len(ranks) > 0 else 0

        batch_size = batch.input_ids.shape[0]
        max_s = batch.max_current_length

        # TODO(travis): allow using CUDA graphs with multi-rank batches
        return (
            torch.cuda.is_available()
            and batch_size <= MAX_BATCH_SIZE
            and max_s <= self.max_total_tokens
            and max_rank <= MAX_RANK
            and nranks <= 1
            and max_rank in _allowed_ranks
        )

    def get_estimated_cache_memory(self) -> int:
        # Store off graphs into temporary cache to discard after estimation
        tmp_cache = {}
        pool = None

        # Use the largest batch size to overestimate memory overhead
        batch_size = CACHED_BATCH_SIZES[-1]

        # Need at least two samples to discard the first run
        ranks = CACHED_MAX_RANKS
        if len(ranks) == 1:
            ranks = ranks * 2

        samples = []
        for i, max_rank in enumerate(reversed(ranks)):
            torch.cuda.synchronize(self.device)
            free_memory_before, _ = torch.cuda.mem_get_info(self.device)

            key = (batch_size, max_rank)
            graph = GraphWrapper.trace(
                self.model,
                self.device,
                self.kv_cache,
                self.adapter_layers,
                self.forward_context,
                batch_size,
                max_rank,
                pool,
                self.max_total_tokens,
                self.num_heads,
                self.num_kv_heads,
                self.sliding_window_blocks,
                self.adapter_layers,  # estimate memory assuming all adapters are traced
                self.layer_to_lora_weights,
                self.punica_wrapper,
            )
            tmp_cache[key] = graph
            pool = graph.memory_pool

            torch.cuda.synchronize(self.device)
            free_memory_after, _ = torch.cuda.mem_get_info(self.device)

            # Measure memory difference after tracing the graph,
            # discard first sample to account for global state initialization
            delta_memory = free_memory_before - free_memory_after
            if i > 0:
                samples.append(delta_memory)

            # Tracing all graphs can take a while, so limit the number of samples
            if len(samples) == MAX_SAMPLES:
                break

        # Estimate memory usage for all batch sizes and ranks
        ngraphs = len(CACHED_BATCH_SIZES) * len(CACHED_MAX_RANKS)
        per_graph_memory = median(samples)
        return ngraphs * per_graph_memory

    def warmup(self):
        ngraphs = len(CACHED_BATCH_SIZES) * len(CACHED_MAX_RANKS)
        pool = None
        logger.info("Tracing CUDA graphs with initial adapter layers: {}", self.default_traced_adapter_layers)
        with tqdm(total=ngraphs, desc="Trace CUDA graphs") as pbar:
            for batch_size in reversed(CACHED_BATCH_SIZES):
                pbar.set_postfix({"batch_size": batch_size})
                for max_rank in reversed(CACHED_MAX_RANKS):
                    key = (batch_size, max_rank)
                    graph = GraphWrapper.trace(
                        self.model,
                        self.device,
                        self.kv_cache,
                        self.adapter_layers,
                        self.forward_context,
                        batch_size,
                        max_rank,
                        pool,
                        self.max_total_tokens,
                        self.num_heads,
                        self.num_kv_heads,
                        self.sliding_window_blocks,
                        self.default_traced_adapter_layers,
                        self.layer_to_lora_weights,
                        self.punica_wrapper,
                    )
                    self.cache[key] = graph
                    pool = graph.memory_pool
                    pbar.update(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        cache_lengths: List[int],
        cache_lengths_tensor: torch.Tensor,
        max_s: int,
        adapter_data: AdapterBatchData,
        lm_head_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        batch_size = get_cached_batch_size(input_ids.shape[0])
        max_rank = adapter_data.max_rank

        key = (batch_size, max_rank)
        graph = self.cache.get(key)
        if graph is None or not graph.input_state.traced_adapter_layer_names.issuperset(adapter_data.layer_names()):
            current_traced_adapter_layer_names = (
                graph.input_state.traced_adapter_layer_names if graph is not None else set()
            )
            logger.info(
                "batch_size={} -- retrace graph with new adapter layers: {} -> {}",
                batch_size,
                current_traced_adapter_layer_names,
                adapter_data.layer_names(),
            )
            graph = GraphWrapper.trace(
                self.model,
                self.device,
                self.kv_cache,
                self.adapter_layers,
                self.forward_context,
                batch_size,
                max_rank,
                self.memory_pool,
                self.max_total_tokens,
                self.num_heads,
                self.num_kv_heads,
                self.sliding_window_blocks,
                adapter_data.layer_names(),
                self.layer_to_lora_weights,
                self.punica_wrapper,
            )
            self.cache[key] = graph

        output_states = graph.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            seqlen=seqlen,
            cache_lengths=cache_lengths,
            cache_lengths_tensor=cache_lengths_tensor,
            max_s=max_s,
            adapter_data=adapter_data,
            lm_head_indices=lm_head_indices,
        )

        return output_states

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
