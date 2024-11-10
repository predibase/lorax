import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Type, TypeVar

import torch
from loguru import logger
from transformers import PreTrainedTokenizerBase

from lorax_server.adapters.lora import LoraWeights
from lorax_server.adapters.medusa_lora import MedusaLoraWeights
from lorax_server.adapters.utils import download_adapter_weights
from lorax_server.adapters.weights import LayerAdapterWeights
from lorax_server.models.types import Batch, GeneratedText
from lorax_server.pb import generate_pb2
from lorax_server.pb.generate_pb2 import AdapterParameters, AdapterSource, InfoResponse
from lorax_server.utils.adapter import (
    BASE_MODEL_ADAPTER_ID,
    load_and_merge_adapters,
)
from lorax_server.utils.punica import LORAX_PUNICA_TRITON_DISABLED, pad_to_min_rank, use_cutlass_shrink
from lorax_server.utils.sources import HUB
from lorax_server.utils.state import (
    BLOCK_SIZE,
    CHUNKED_PREFILL,
    FLASH_INFER,
    LORAX_PROFILER_DIR,
    get_speculative_tokens,
    set_supports_chunking,
)
from lorax_server.utils.tokenizer import TokenizerManager
from lorax_server.utils.weights import shard_on_dim

B = TypeVar("B", bound=Batch)


class Model(ABC):
    def __init__(
        self,
        model_id: str,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        requires_padding: bool,
        dtype: torch.dtype,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        sliding_window: Optional[int] = None,
        adapter_id: str = BASE_MODEL_ADAPTER_ID,
        adapter_source: str = HUB,
        dynamic_adapter_loading_enabled: bool = True,
        trust_remote_code: bool = False,
        processor=None,
        supports_chunking: bool = False,
    ):
        self.model_id = model_id
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.tokenizers = TokenizerManager()
        self.processor = processor

        self.all_special_ids = set(tokenizer.all_special_ids)

        # all_special_ids is not set correctly if the rust tokenizer is unpacked
        other_special_ids = {id for id, token in tokenizer.added_tokens_decoder.items() if token.special}
        self.all_special_ids.update(other_special_ids)

        self.requires_padding = requires_padding
        self.dtype = dtype
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.sliding_window = sliding_window

        # This may be set to False in the subclass constructor
        self.dynamic_adapter_loading_enabled = dynamic_adapter_loading_enabled
        self.layer_to_adapter_weights: Dict[str, LayerAdapterWeights] = defaultdict(LayerAdapterWeights)
        self.target_to_layer = self.adapter_target_to_layer()
        self.loaded_adapters = set()
        self.static_adapter_id = adapter_id
        self.preloaded_adapter_indices = set()
        self.preloaded_adapter_memory_fractions = {}
        self.preloaded_adapters = []
        self.layer_to_lora_weights = {}

        self.trust_remote_code = trust_remote_code

        speculation_tokens = get_speculative_tokens()

        supports_chunking = supports_chunking and CHUNKED_PREFILL
        if supports_chunking:
            if speculation_tokens != 0:
                logger.warning(
                    "Chunked prefill does not support speculation yet. " "Chunked prefill will be disabled",
                )
                supports_chunking = False
            if not FLASH_INFER:
                logger.warning(
                    "Chunked prefill is only supported with `flashinfer` backend.",
                )
                supports_chunking = False
            logger.info(f"Using experimental chunked prefill = {supports_chunking}")

        self.supports_chunking = supports_chunking
        set_supports_chunking(supports_chunking)

        self.has_position_ids = inspect.signature(model.forward).parameters.get("position_ids", None) is not None

        if dynamic_adapter_loading_enabled and adapter_id and adapter_id != BASE_MODEL_ADAPTER_ID:
            download_adapter_weights(adapter_id, adapter_source, api_token=None)
            self.load_adapter(
                AdapterParameters(adapter_ids=[adapter_id]),
                adapter_source,
                adapter_index=0,
                api_token=None,
                dynamic=False,
            )

        self.check_initialized()

        self.profiler = None
        if LORAX_PROFILER_DIR is not None:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(LORAX_PROFILER_DIR, use_gzip=True),
            )
            self.profiler_steps = 0

    @property
    def info(self) -> InfoResponse:
        if self.requires_padding and self.sliding_window is not None:
            raise NotImplementedError("sliding_window is not implemented with padding")

        return InfoResponse(
            requires_padding=self.requires_padding,
            dtype=str(self.dtype),
            device_type=self.device.type,
            window_size=self.sliding_window,
            block_size=self.block_size,
            speculate=get_speculative_tokens(),
            preloaded_adapters=self.preloaded_adapters,
            supports_generation=self.supports_text_generation,
            supports_embeddings=self.supports_embeddings,
            supports_classification=self.supports_classification,
            chunked_prefill=self.supports_chunking,
        )

    @property
    def block_size(self) -> int:
        # TODO: (magdy) revisit if this change works. For now this allows us
        # to not have to set it for embedding and NER models as well
        return BLOCK_SIZE

    @property
    def sliding_window_blocks(self) -> Optional[int]:
        return None

    @property
    def supports_embeddings(self) -> bool:
        return False

    @property
    def supports_classification(self) -> bool:
        return False

    @property
    def supports_text_generation(self) -> bool:
        return True

    @property
    @abstractmethod
    def batch_type(self) -> Type[B]:
        raise NotImplementedError

    def adapter_memory_size(self) -> int:
        return 0

    @abstractmethod
    def generate_token(self, batch: B) -> Tuple[List[GeneratedText], Optional[B]]:
        raise NotImplementedError

    def warmup(self, batch: B, max_new_tokens: int) -> Optional[int]:
        self.generate_token(batch)
        return None

    def decode_token(
        self,
        all_input_ids: List[int],
        prefix_offset: int = 0,
        read_offset: int = 0,
        skip_special_tokens: bool = False,
    ) -> Tuple[str, int, int]:
        """Hack to hopefully support generate_stream for the maximum number of tokenizers"""

        # The prefix text is necessary only to defeat cleanup algorithms in the decode
        # which decide to add a space or not depending on the surrounding ids.
        prefix_text = self.tokenizer.decode(
            all_input_ids[prefix_offset:read_offset], skip_special_tokens=skip_special_tokens
        )
        new_text = self.tokenizer.decode(all_input_ids[prefix_offset:], skip_special_tokens=skip_special_tokens)

        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            # utf-8 char at the end means it's a potential unfinished byte sequence
            # from byte fallback tokenization.
            # If it's in the middle, it's probably a real invalid id generated
            # by the model
            new_text = new_text[len(prefix_text) :]
            return new_text, read_offset, len(all_input_ids)
        else:
            return "", prefix_offset, read_offset

    def check_initialized(self):
        uninitialized_parameters = []
        for n, p in self.model.named_parameters():
            if p.data.device == torch.device("meta"):
                uninitialized_parameters.append(n)
        if uninitialized_parameters:
            raise RuntimeError(
                f"found uninitialized parameters in model {self.__class__.__name__}: {uninitialized_parameters}"
            )

    @property
    def supports_adapter_loading(self) -> bool:
        return False

    def adapter_target_to_layer(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        return {}

    @property
    def adapter_layers(self) -> List[str]:
        return []

    @property
    def traced_adapter_layers(self) -> List[str]:
        if self.layer_to_adapter_weights:
            return list(self.layer_to_adapter_weights.keys())
        return self.default_traced_adapter_layers

    @property
    def default_traced_adapter_layers(self) -> List[str]:
        return []

    def get_num_layers_for_type(self, layer_type: str) -> int:
        return 0

    def is_row_parallel(self, layer_type: str) -> bool:
        return False

    @property
    def max_speculative_tokens(self) -> int:
        return max(
            [weights.max_speculative_tokens for weights in self.layer_to_adapter_weights.values()],
            default=0,
        )

    def register_preloaded_adapters(
        self, preloaded_adapters: List[generate_pb2.PreloadedAdapter], adapter_memory_fractions: List[float]
    ):
        if preloaded_adapters is None:
            return

        self.dynamic_adapter_loading_enabled = False
        self.preloaded_adapter_indices.update({adapter.adapter_index for adapter in preloaded_adapters})
        self.preloaded_adapter_memory_fractions.update(
            {
                adapter.adapter_parameters.adapter_ids[0]: memory_fraction
                for adapter, memory_fraction in zip(preloaded_adapters, adapter_memory_fractions)
            }
        )
        self.preloaded_adapters.extend(preloaded_adapters)

        if LORAX_PUNICA_TRITON_DISABLED:
            # Following code is only applicable to Triton kernels
            return

        # For Triton kernels: need weights into contiguous tensor
        # dict of (layer_name, layer_id) -> (lora_a_weights, lora_b_weights)
        # where:
        #   lora_a_weights = [num_adapters, r, hidden_size]
        #   lora_b_weights = [num_adapters, hidden_size, r]
        for layer_name, layer_adapter_weights in self.layer_to_adapter_weights.items():
            layer_id_to_lora_a_weights = defaultdict(list)
            layer_id_to_lora_b_weights = defaultdict(list)
            for adapter in preloaded_adapters:
                adapter_index = adapter.adapter_index
                adapter_weights = layer_adapter_weights.adapter_weights.get(adapter_index)
                if not isinstance(adapter_weights, LoraWeights):
                    if isinstance(adapter_weights, MedusaLoraWeights):
                        # only use lora component
                        adapter_weights = adapter_weights.lora_weights
                    else:
                        # only applicable to lora for now
                        continue

                if adapter_weights is None:
                    # no weights for this layer
                    continue

                # transpose into col major
                lora_b = adapter_weights.weights_b.transpose(1, 2)
                lora_a = adapter_weights.weights_a
                if use_cutlass_shrink(lora_b.size(2)):
                    lora_a = lora_a.transpose(1, 2)

                nlayers = lora_a.size(0)
                for layer_id in range(nlayers):
                    layer_id_to_lora_a_weights[layer_id].append(lora_a[layer_id])
                    layer_id_to_lora_b_weights[layer_id].append(lora_b[layer_id])

            for layer_id, lora_a_weights in layer_id_to_lora_a_weights.items():
                lora_b_weights = layer_id_to_lora_b_weights[layer_id]

                # right pad every adapter to the max rank
                r = max([w.size(-1) for w in lora_b_weights])
                lora_a_weights = [pad_to_min_rank(w, 0, r) for w in lora_a_weights]
                lora_b_weights = [pad_to_min_rank(w, 1, r) for w in lora_b_weights]

                # stack into [num_adapters, r, hidden_size] and [num_adapters, hidden_size, r]
                lora_a_weights = torch.stack(lora_a_weights).to(self.device).contiguous()
                lora_b_weights = torch.stack(lora_b_weights).to(self.device).contiguous()
                self.layer_to_lora_weights[(layer_name, layer_id)] = (lora_a_weights, lora_b_weights)

    def load_adapter(
        self,
        adapter_parameters: AdapterParameters,
        adapter_source: AdapterSource,
        adapter_index: int,
        api_token: str,
        dynamic: bool = True,
    ):
        """Loads adapter weights from disk / host memory on the GPU.

        adapter_id must be `BASE_MODEL_ADAPTER_ID` if adapter statically loaded
        into model. Otherwise, the adapter weights are applied during the forward
        pass and stored separately from the base model parameters.
        """
        if adapter_index in self.loaded_adapters:
            # Adapter already loaded
            return

        if not self.supports_adapter_loading:
            raise ValueError("This model does not support adapter loading.")

        if dynamic and not self.dynamic_adapter_loading_enabled:
            raise ValueError(
                "This model does not support dynamic adapter loading. "
                "Please initialize a new model instance from the base model and remove preloaded adapters "
                "to use the dynamic adapter loading feature."
            )

        logger.info(f"Loading adapter weights into model: {','.join(adapter_parameters.adapter_ids)}")
        weight_names = tuple([v[0] for v in self.target_to_layer.values()])
        (
            module_map,
            adapter_config,
            adapter_weight_names,
            adapter_tokenizer,
        ) = load_and_merge_adapters(
            self.model_id,
            adapter_parameters,
            adapter_source,
            adapter_index,
            weight_names,
            api_token,
            self.trust_remote_code,
        )

        unused_weight_names = adapter_weight_names.copy()
        for layer_name in self.adapter_layers:
            adapter_weights = adapter_config.load_batched_adapter_weights(
                self,
                module_map,
                layer_name,
                unused_weight_names,
                dynamic,
            )

            if adapter_weights is None:
                continue

            layer_weights = self.layer_to_adapter_weights[layer_name]
            layer_weights.add_adapter(adapter_index, adapter_weights)

        if len(unused_weight_names) > 0:
            logger.warning(f"{','.join(adapter_parameters.adapter_ids)} unused adapter weights: {unused_weight_names}")

        if adapter_tokenizer is not None:
            self.tokenizers.add_tokenizer(adapter_index, adapter_tokenizer)

        self.loaded_adapters.add(adapter_index)

    def shard_lora_weights(
        self,
        weights_a: List[torch.Tensor],
        weights_b: List[torch.Tensor],
        layer_type: str,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # [hidden_size, r]
        split_dim = 0 if self.is_row_parallel(layer_type) else 1
        weights_a = [shard_on_dim(w, dim=split_dim, process_group=self.process_group) for w in weights_a]

        # [r, hidden_size]
        weights_b = [shard_on_dim(w, dim=1, process_group=self.process_group) for w in weights_b]

        return weights_a, weights_b

    def offload_adapter(
        self,
        adapter_parameters: AdapterParameters,
        adapter_source: AdapterSource,
        adapter_index: int,
    ) -> bool:
        """Offloads the adapter weights from GPU to CPU or disk."""
        if adapter_index not in self.loaded_adapters:
            # Adapter already offloaded
            return False

        if adapter_index in self.preloaded_adapter_indices:
            # Adapter was preloaded and should not be offloaded
            return False

        if not self.supports_adapter_loading:
            raise ValueError("This model does not support adapter loading.")

        if not self.dynamic_adapter_loading_enabled:
            raise ValueError(
                "This model does not support dynamic adapter loading. "
                "Please initialize a new model instance from the base model and remove preloaded adapters "
                "to use the dynamic adapter loading feature."
            )

        for layer_name in self.adapter_layers:
            if layer_name in self.layer_to_adapter_weights:
                self.layer_to_adapter_weights[layer_name].remove_adapter(adapter_index)

        self.loaded_adapters.remove(adapter_index)
        return True
