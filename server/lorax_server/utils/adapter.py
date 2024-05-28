import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Set, Tuple

from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

from lorax_server.pb import generate_pb2
from lorax_server.utils.merges.strategies import merge_adapters
from lorax_server.utils.sources import get_config_path, get_model_source

if TYPE_CHECKING:
    from lorax_server.adapters.config import AdapterConfig, ModuleMap


BASE_MODEL_ADAPTER_ID = "__base_model__"


@dataclass
class AdapterParametersContainer:
    adapter_parameters: generate_pb2.AdapterParameters
    adapter_source: str
    adapter_index: int

    def __hash__(self) -> int:
        return self.adapter_index


def is_base_model(adapter_parameters: generate_pb2.AdapterParameters) -> bool:
    if len(adapter_parameters.adapter_ids) != 1:
        return False
    return adapter_parameters.adapter_ids[0] == BASE_MODEL_ADAPTER_ID


def load_and_merge_adapters(
    model_id: str,
    adapter_parameters: generate_pb2.AdapterParameters,
    adapter_source: str,
    adapter_index: int,
    weight_names: Tuple[str],
    api_token: str,
    trust_remote_code: bool = False,
) -> Tuple["ModuleMap", "AdapterConfig", Set[str], PreTrainedTokenizer]:
    if len(adapter_parameters.adapter_ids) == 1:
        return load_module_map(
            model_id, adapter_parameters.adapter_ids[0], adapter_source, weight_names, api_token, trust_remote_code
        )

    adapter_params = AdapterParametersContainer(adapter_parameters, adapter_source, adapter_index)
    return _load_and_merge(model_id, adapter_params, weight_names, api_token, trust_remote_code)


@lru_cache(maxsize=32)
def _load_and_merge(
    model_id: str,
    adapter_params: AdapterParametersContainer,
    weight_names: Tuple[str],
    api_token: str,
    trust_remote_code: bool = False,
) -> Tuple["ModuleMap", "AdapterConfig", Set[str], PreTrainedTokenizer]:
    params = adapter_params.adapter_parameters

    adapters_to_merge = []
    merged_weight_names = set()
    tokenizer = None
    for adapter_id in params.adapter_ids:
        if adapter_id == BASE_MODEL_ADAPTER_ID:
            raise ValueError("Base model adapter cannot be merged.")

        module_map, adapter_config, adapter_weight_names, adapter_tokenizer = load_module_map(
            model_id,
            adapter_id,
            adapter_params.adapter_source,
            weight_names,
            api_token,
            trust_remote_code,
        )

        adapters_to_merge.append((module_map, adapter_config))
        merged_weight_names = merged_weight_names.union(adapter_weight_names)
        if tokenizer is None:
            tokenizer = adapter_tokenizer

    if len(adapters_to_merge) == 0:
        raise ValueError("No adapters to merge.")

    module_map, adapter_config = merge_adapters(adapters_to_merge, params)
    return module_map, adapter_config, merged_weight_names, tokenizer


def check_architectures(
    model_id: str,
    adapter_id: str,
    adapter_config: "AdapterConfig",
    trust_remote_code: bool = False,
):
    try:
        if not adapter_config.base_model_name_or_path:
            # Avoid execuation latency caused by the network connection retrying for AutoConfig.from_pretrained(None)
            return

        expected_config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model_config = AutoConfig.from_pretrained(
            adapter_config.base_model_name_or_path, trust_remote_code=trust_remote_code
        )
    except Exception as e:
        warnings.warn(
            f"Unable to check architecture compatibility for adapter '{adapter_id}' "
            f"against model '{model_id}'. Assuming they are compatible. Error: {e}"
        )
        return

    if model_config.architectures == expected_config.architectures:
        warnings.warn(
            f"Adapter '{adapter_id}' was not trained on base model '{model_id}'. "
            f"If you encounter issues, use --model-id '{adapter_config.base_model_name_or_path}' instead."
        )
    else:
        # TODO(travis): revisit this when we support clasification heads which will not use CausalLM
        raise ValueError(
            f"Adapter '{adapter_id}' is not compatible with model '{model_id}'. "
            f"Architectures differ: {model_config.architectures} != {expected_config.architectures}. "
            f"Use --model-id '{adapter_config.base_model_name_or_path}' instead."
        )


@lru_cache(maxsize=128)
def load_module_map(
    model_id: str,
    adapter_id: str,
    adapter_source: str,
    weight_names: Tuple[str],
    api_token: str,
    trust_remote_code: bool = False,
) -> Tuple["ModuleMap", "AdapterConfig", Set[str], PreTrainedTokenizer]:
    # TODO(geoffrey): refactor this and merge parts of this function with
    # lorax_server/utils/adapter.py::create_merged_weight_files
    source = get_model_source(adapter_source, adapter_id, extension=".safetensors", api_token=api_token)
    config_path = get_config_path(adapter_id, adapter_source)
    adapter_config = source.load_config()
    if adapter_config.base_model_name_or_path != model_id:
        check_architectures(model_id, adapter_id, adapter_config, trust_remote_code)

    try:
        adapter_tokenizer = AutoTokenizer.from_pretrained(
            config_path, token=api_token, trust_remote_code=trust_remote_code
        )
    except Exception:
        # Adapter does not have a tokenizer, so fallback to base model tokenizer
        adapter_tokenizer = None

    # load adapter weights from all shards (should have relatively small memory footprint)
    adapter_filenames = source.weight_files()
    adapter_weights = {}
    for filename in adapter_filenames:
        adapter_weights.update(load_file(filename))

    # map the model weights to the relevant adapter weights (LoRA A and B matrices)
    module_map, adapter_weight_names = adapter_config.map_weights_for_model(adapter_weights, weight_names)
    return module_map, adapter_config, adapter_weight_names, adapter_tokenizer
