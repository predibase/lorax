import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Type, TypeVar

import torch
from loguru import logger
from transformers import PreTrainedTokenizerBase

from lorax_server.adapters.utils import download_adapter_weights
from lorax_server.adapters.weights import LayerAdapterWeights
from lorax_server.models.types import Batch, GeneratedText
from lorax_server.pb import generate_pb2
from lorax_server.pb.generate_pb2 import AdapterParameters, AdapterSource, InfoResponse
from lorax_server.utils.adapter import (
    BASE_MODEL_ADAPTER_ID,
    load_and_merge_adapters,
)
from lorax_server.utils.sources import HUB
from lorax_server.utils.state import get_speculative_tokens
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
    ):
        self.model_id = model_id
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.tokenizers = TokenizerManager()

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

        self.trust_remote_code = trust_remote_code

        self.has_position_ids = inspect.signature(model.forward).parameters.get("position_ids", None) is not None

        if adapter_id and adapter_id != BASE_MODEL_ADAPTER_ID:
            download_adapter_weights(adapter_id, adapter_source, api_token=None)
            self.load_adapter(
                AdapterParameters(adapter_ids=[adapter_id]),
                adapter_source,
                adapter_index=0,
                api_token=None,
                dynamic=False,
            )

        self.check_initialized()

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
        )

    @property
    def block_size(self) -> int:
        return 0

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
    ) -> Tuple[str, int, int]:
        """Hack to hopefully support generate_stream for the maximum number of tokenizers"""

        # The prefix text is necessary only to defeat cleanup algorithms in the decode
        # which decide to add a space or not depending on the surrounding ids.
        prefix_text = self.tokenizer.decode(all_input_ids[prefix_offset:read_offset], skip_special_tokens=False)
        new_text = self.tokenizer.decode(all_input_ids[prefix_offset:], skip_special_tokens=False)

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
        self.preloaded_adapter_indices.update({adapter.adapter_index for adapter in preloaded_adapters})
        self.preloaded_adapter_memory_fractions.update(
            {
                adapter.adapter_parameters.adapter_ids[0]: memory_fraction
                for adapter, memory_fraction in zip(preloaded_adapters, adapter_memory_fractions)
            }
        )
        self.preloaded_adapters.extend(preloaded_adapters)

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
                f"This model was initialized with the adapter {self.static_adapter_id} "
                f"and therefore does not support dynamic adapter loading. "
                f"Please initialize a new model instance from the base model in "
                f"order to use the dynamic adapter loading feature."
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
                f"This model was initialized with the adapter {self.static_adapter_id} "
                f"and therefore does not support dynamic adapter loading. "
                f"Please initialize a new model instance from the base model in "
                f"order to use the dynamic adapter loading feature."
            )

        for layer_name in self.adapter_layers:
            if layer_name in self.layer_to_adapter_weights:
                self.layer_to_adapter_weights[layer_name].remove_adapter(adapter_index)

        self.loaded_adapters.remove(adapter_index)
        return True
