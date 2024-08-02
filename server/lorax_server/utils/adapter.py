import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Set, Tuple, Union

from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

from lorax_server.adapters.utils import download_adapter_weights
from lorax_server.pb import generate_pb2
from lorax_server.utils.merges.strategies import merge_adapters
from lorax_server.utils.sources import HUB, LOCAL, PBASE, S3, get_config_path, get_model_source

if TYPE_CHECKING:
    from lorax_server.adapters.config import AdapterConfig, ModuleMap
    from lorax_server.models.model import Model


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
            False,
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
    lazy_load_weights: bool = True,
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
        if lazy_load_weights:
            result = {}
            # just fetching the layer names of the module
            with safe_open(filename, framework="pt") as f:
                for k in f.keys():
                    result[k] = filename
            adapter_weights.update(result)
        else:
            adapter_weights.update(load_file(filename))

    # map the model weights to the relevant adapter weights (LoRA A and B matrices)
    module_map, adapter_weight_names = adapter_config.map_weights_for_model(adapter_weights, weight_names)
    return module_map, adapter_config, adapter_weight_names, adapter_tokenizer


def download_adapter(
    request: generate_pb2.DownloadAdapterRequest, model: "Model"
) -> generate_pb2.DownloadAdapterResponse:
    adapter_parameters = request.adapter_parameters
    if is_base_model(adapter_parameters):
        logger.info("No adapter to download for base model. Skipping.")
        return generate_pb2.DownloadAdapterResponse(downloaded=False)

    adapter_bytes = 0
    api_token = request.api_token
    adapter_source = adapter_source_enum_to_string(request.adapter_source)
    for adapter_id in adapter_parameters.adapter_ids:
        if adapter_id == BASE_MODEL_ADAPTER_ID:
            logger.info("No adapter to download for base model. Skipping.")
            continue

        adapter_bytes += download_adapter_weights(adapter_id, adapter_source, api_token)

    adapter_memory_size = model.adapter_memory_size()
    if adapter_memory_size > 0:
        logger.info(
            f"Downloaded adapter {adapter_id} memory size: {adapter_bytes} bytes "
            f"(reservation: {adapter_memory_size} bytes)"
        )
        adapter_memory_fraction = adapter_bytes / adapter_memory_size
        if adapter_memory_fraction > 1:
            raise ValueError(
                f"Adapter {adapter_id} is larger than adapter memory reservation: "
                f"{adapter_bytes} / {adapter_memory_size} bytes"
            )
    else:
        # Assume 0.0 memory fraction if adapter memory size is not set
        logger.info(f"Downloaded adapter {adapter_id} memory size: {adapter_bytes} bytes " f"(no reservation limit)")
        adapter_memory_fraction = 0.0

    return generate_pb2.DownloadAdapterResponse(downloaded=True, memory_fraction=adapter_memory_fraction)


def adapter_source_enum_to_string(adapter_source: int) -> str:
    # TODO(travis): refactor this to be less hacky
    if adapter_source == generate_pb2.AdapterSource.HUB:
        return HUB
    elif adapter_source == generate_pb2.AdapterSource.S3:
        return S3
    elif adapter_source == generate_pb2.AdapterSource.LOCAL:
        return LOCAL
    elif adapter_source == generate_pb2.AdapterSource.PBASE:
        return PBASE
    else:
        raise ValueError(f"Unknown adapter source {adapter_source}")


def enum_string_to_adapter_source(adapter_source: str) -> int:
    # TODO(travis): refactor this to be less hacky
    if adapter_source == HUB:
        return generate_pb2.AdapterSource.HUB
    elif adapter_source == S3:
        return generate_pb2.AdapterSource.S3
    elif adapter_source == LOCAL:
        return generate_pb2.AdapterSource.LOCAL
    elif adapter_source == PBASE:
        return generate_pb2.AdapterSource.PBASE
    else:
        raise ValueError(f"Unknown adapter source {adapter_source}")
