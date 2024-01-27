from collections import defaultdict
import inspect
import torch

from abc import ABC, abstractmethod
from loguru import logger
from peft import LoraConfig
from typing import Dict, List, Set, Tuple, Optional, TypeVar, Type
from transformers import PreTrainedTokenizerBase

from lorax_server.models.types import Batch, GeneratedText
from lorax_server.pb.generate_pb2 import AdapterParameters, AdapterSource, InfoResponse
from lorax_server.utils.adapter import BASE_MODEL_ADAPTER_ID, load_and_merge_adapters
from lorax_server.utils.tokenizer import TokenizerManager
from lorax_server.utils.lora import BatchedLoraWeights, MergedLoraWeights
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
        dynamic_adapter_loading_enabled: bool = True,
    ):
        self.model_id = model_id
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.tokenizers = TokenizerManager()
        self.all_special_ids = set(tokenizer.all_special_ids)
        self.requires_padding = requires_padding
        self.dtype = dtype
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.sliding_window = sliding_window

        # This may be set to False in the subclass constructor
        self.dynamic_adapter_loading_enabled = dynamic_adapter_loading_enabled
        self.batched_lora_weights: Dict[str, BatchedLoraWeights] = defaultdict(BatchedLoraWeights)
        self.target_to_layer = self.adapter_target_to_layer()
        self.loaded_adapters = set()
        self.static_adapter_id = adapter_id

        self.has_position_ids = (
            inspect.signature(model.forward).parameters.get("position_ids", None)
            is not None
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
        )

    @property
    @abstractmethod
    def batch_type(self) -> Type[B]:
        raise NotImplementedError

    @abstractmethod
    def generate_token(self, batch: B) -> Tuple[List[GeneratedText], Optional[B]]:
        raise NotImplementedError

    def warmup(self, batch: B) -> Optional[int]:
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
        prefix_text = self.tokenizer.decode(
            all_input_ids[prefix_offset:read_offset], skip_special_tokens=False
        )
        new_text = self.tokenizer.decode(
            all_input_ids[prefix_offset:], skip_special_tokens=False
        )

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
    
    def get_num_layers_for_type(self, layer_type: str) -> int:
        return 0
    
    def is_row_parallel(self, layer_type: str) -> bool:
        return False

    def load_adapter(
        self,
        adapter_parameters: AdapterParameters,
        adapter_source: AdapterSource,
        adapter_index: int,
        api_token: str,
    ):
        """Physically loads the adapter weights into the model.

        adapter_id must be `BASE_MODEL_ADAPTER_ID` if adapter statically loaded 
        into model. Otherwise, the adapter weights are merged into the model 
        weights on the fly.
        """
        if adapter_index in self.loaded_adapters:
            # Adapter already loaded
            return
        
        if not self.supports_adapter_loading:
            raise ValueError("This model does not support adapter loading.")
        
        if not self.dynamic_adapter_loading_enabled:
            raise ValueError(f"This model was initialized with the adapter {self.static_adapter_id} "
                             f"and therefore does not support dynamic adapter loading. "
                             f"Please initialize a new model instance from the base model in "
                             f"order to use the dynamic adapter loading feature.")

        logger.info(f"Loading adapter weights into model: {','.join(adapter_parameters.adapter_ids)}")
        weight_names = tuple([v[0] for v in self.target_to_layer.values()])
        module_map, adapter_config, adapter_weight_names, adapter_tokenizer = load_and_merge_adapters(
            self.model_id, adapter_parameters, adapter_source, adapter_index, weight_names, api_token
        )

        unused_weight_names = adapter_weight_names.copy()
        for layer_name in self.adapter_layers:
            self.load_batched_adapter_weights(
                module_map, adapter_config, adapter_index, layer_name, unused_weight_names
            )
        
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
        weights_a = [
            shard_on_dim(w, dim=split_dim, process_group=self.process_group)
            for w in weights_a
        ]

        # [r, hidden_size]
        weights_b = [
            shard_on_dim(w, dim=1, process_group=self.process_group)
            for w in weights_b
        ]

        return weights_a, weights_b

    def load_batched_adapter_weights(
        self, 
        module_map: Dict[str, Dict], 
        adapter_config: LoraConfig, 
        adapter_index: int, 
        layer_type: str,
        unused_weight_names: Set[str],
    ):
        nlayers = self.get_num_layers_for_type(layer_type)
        lora_a_list = [None] * nlayers
        lora_b_list = [None] * nlayers
        
        for layer_id in range(nlayers):
            key = (layer_id, layer_type)
            weight_name, layer = self.target_to_layer[key]
        
            base_weight = layer.base_layer.linear.weight
            base_device = base_weight.device

            if weight_name not in module_map:
                # There is no LoRA weight for this layer type in the adapter
                return
            
            lora_a, lora_a_name = module_map[weight_name]["lora_A"]
            lora_a = lora_a.to(base_device, self.dtype)

            lora_b, lora_b_name = module_map[weight_name]["lora_B"]
            lora_b = lora_b.to(base_device, self.dtype)

            scale = adapter_config.lora_alpha / adapter_config.r

            unused_weight_names.discard(lora_a_name)
            unused_weight_names.discard(lora_b_name)

            # Merge scaling factor into lora_b due to associativity of matrix multiplication:
            # (A * B) * C = A * (B * C)
            lora_a_list[layer_id] = lora_a.transpose(0, 1)
            lora_b_list[layer_id] = lora_b.transpose(0, 1) * scale

        q_lora_merged = MergedLoraWeights(
            *self.shard_lora_weights(lora_a_list, lora_b_list, layer_type), adapter_config,
        )
        q_lora_weights = self.batched_lora_weights[layer_type]
        q_lora_weights.add_adapter(adapter_index, q_lora_merged)
    
    def offload_adapter(
        self,
        adapter_parameters: AdapterParameters,
        adapter_source: AdapterSource,
        adapter_index: int,
    ):
        """Offloads the adapter weights from GPU to CPU or disk."""
        if adapter_index not in self.loaded_adapters:
            # Adapter already offloaded
            return
        
        if not self.supports_adapter_loading:
            raise ValueError("This model does not support adapter loading.")
        
        if not self.dynamic_adapter_loading_enabled:
            raise ValueError(f"This model was initialized with the adapter {self.static_adapter_id} "
                             f"and therefore does not support dynamic adapter loading. "
                             f"Please initialize a new model instance from the base model in "
                             f"order to use the dynamic adapter loading feature.")

        for layer_name in self.adapter_layers:
            if layer_name in self.batched_lora_weights:
                self.batched_lora_weights[layer_name].remove_adapter(adapter_index)

        self.loaded_adapters.remove(adapter_index)
