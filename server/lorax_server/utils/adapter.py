from dataclasses import dataclass
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Set, Tuple
import warnings

import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from loguru import logger
from peft import LoraConfig
from peft.utils import transpose
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
from filelock import FileLock

from lorax_server.pb import generate_pb2
from lorax_server.utils.sources import get_model_source, get_config_path, weight_files
from lorax_server.utils.merges.strategies import merge_adapters    


BASE_MODEL_ADAPTER_ID = "__base_model__"


ModuleMap = Dict[str, Dict[str, Tuple[torch.Tensor, str]]]


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
) -> Tuple[ModuleMap, LoraConfig, Set[str], PreTrainedTokenizer]:
    if len(adapter_parameters.adapter_ids) == 1:
        return load_module_map(
            model_id, adapter_parameters.adapter_ids[0], adapter_source, weight_names, api_token
        )

    adapter_params = AdapterParametersContainer(adapter_parameters, adapter_source, adapter_index)
    return _load_and_merge(model_id, adapter_params, weight_names, api_token)


@lru_cache(maxsize=32)
def _load_and_merge(
    model_id: str,
    adapter_params: AdapterParametersContainer,
    weight_names: Tuple[str],
    api_token: str,
) -> Tuple[ModuleMap, LoraConfig, Set[str], PreTrainedTokenizer]:
    params = adapter_params.adapter_parameters
    
    adapters_to_merge = []
    merged_weight_names = set()
    tokenizer = None
    for adapter_id in params.adapter_ids:
        if adapter_id == BASE_MODEL_ADAPTER_ID:
            raise ValueError("Base model adapter cannot be merged.")
        
        module_map, adapter_config, adapter_weight_names, adapter_tokenizer = load_module_map(
            model_id, adapter_id, adapter_params.adapter_source, weight_names, api_token,
        )
        
        adapters_to_merge.append((module_map, adapter_config))
        merged_weight_names = merged_weight_names.union(adapter_weight_names)
        if tokenizer is None:
            tokenizer = adapter_tokenizer

    if len(adapters_to_merge) == 0:
        raise ValueError("No adapters to merge.")
    
    module_map, adapter_config = merge_adapters(adapters_to_merge, params)
    return module_map, adapter_config, merged_weight_names, tokenizer


@lru_cache(maxsize=128)
def load_module_map(
    model_id: str,
    adapter_id: str,
    adapter_source: str,
    weight_names: Tuple[str],
    api_token: str,
) -> Tuple[ModuleMap, LoraConfig, Set[str], PreTrainedTokenizer]:
    # TODO(geoffrey): refactor this and merge parts of this function with
    # lorax_server/utils/adapter.py::create_merged_weight_files       
    source = get_model_source(adapter_source, adapter_id, extension=".safetensors", api_token=api_token)
    config_path = get_config_path(adapter_id, adapter_source)
    adapter_config = LoraConfig.from_pretrained(config_path, token=api_token)
    if adapter_config.base_model_name_or_path != model_id:
        expected_config = AutoConfig.from_pretrained(model_id)
        model_config = AutoConfig.from_pretrained(adapter_config.base_model_name_or_path)
        if model_config.architectures == expected_config.architectures:
            warnings.warn(
                f"Adapter '{adapter_id}' was not trained on base model '{model_id}'. "
                f"If you encounter issues, use --model-id '{adapter_config.base_model_name_or_path}' instead."
            )
        else:
            # TODO(travis): revisit this when we support clasification heads which will not use CausalLM
            raise ValueError(f"Adapter '{adapter_id}' is not compatible with model '{model_id}'. "
                             f"Architectures differ: {model_config.architectures} != {expected_config.architectures}. "
                             f"Use --model-id '{adapter_config.base_model_name_or_path}' instead.")

    try:
        adapter_tokenizer = AutoTokenizer.from_pretrained(config_path, token=api_token)
    except Exception:
        # Adapter does not have a tokenizer, so fallback to base model tokenizer
        adapter_tokenizer = None
    
    # load adapter weights from all shards (should have relatively small memory footprint)
    adapter_filenames = source.weight_files()
    adapter_weights = {}
    for filename in adapter_filenames:
        adapter_weights.update(load_file(filename))
        
    # map the model weights to the relevant adapter weights (LoRA A and B matrices)
    adapter_weight_names = set()
    module_map = {}
    for weight_name in weight_names:
        lora_a_name = f"base_model.model.{weight_name}.lora_A.weight"
        lora_b_name = f"base_model.model.{weight_name}.lora_B.weight"
        if lora_a_name not in adapter_weights or lora_b_name not in adapter_weights:
            continue
        
        module_map[weight_name] = {
            "lora_A": (adapter_weights[lora_a_name], lora_a_name),
            "lora_B": (adapter_weights[lora_b_name], lora_b_name),
        }
        adapter_weight_names.add(lora_a_name)
        adapter_weight_names.add(lora_b_name)
    return module_map, adapter_config, adapter_weight_names, adapter_tokenizer


def compute_delta_weight(
    lora_A: torch.Tensor, 
    lora_B: torch.Tensor, 
    fan_in_fan_out: bool, 
    alpha: float, 
    r: float,
    uses_rslora: bool = False
) -> torch.Tensor:
    """Computes the delta weight for a Linear layer given A and B LoRA matrices.

    TODO: add logic for other module types beyond Linear layers.

    Reference: https://github.com/huggingface/peft/blob/v0.4.0/src/peft/tuners/lora.py#L799-L806
    """
    scaling = get_scaling_factor(alpha, r, uses_rslora)
    delta_weight = transpose(lora_B @ lora_A, fan_in_fan_out) * scaling
    return delta_weight


def merge_adapter_weights(
    model_weights: Dict[str, torch.Tensor], 
    adapter_weights: Dict[str, torch.Tensor], 
    adapter_config: LoraConfig
) -> Tuple[Dict[str, torch.Tensor], Set[str]]:
    """
    Merges the adapter weights into the model weights.

    Args:
        model_weights (Dict[str, torch.Tensor]): The weights of the base model.
        adapter_weights (Dict[str, torch.Tensor]): The weights of the adapters.
        adapter_config (LoraConfig): The configuration for the LoRA adapter.

    Returns:
        Tuple[Dict[str, torch.Tensor], Set[str]]: A tuple containing the merged weights and the set of processed adapter weight names.
    """
    module_mapping = defaultdict(dict)
    processed_adapter_weight_names = set()

    # map the original tensor names to their adapter counterparts
    for weight_name in model_weights:
        end_idx = weight_name.rfind(".weight")
        key = weight_name[:end_idx]
        for adapter_weight_name in adapter_weights:
            if key in adapter_weight_name:
                # example value: 'base_model.model.model.layers.10.self_attn.v_proj.lora_B.weight'
                # matrix_type gets the second to last element in the module name, i.e. 'lora_B'
                matrix_type = adapter_weight_name.split(".")[-2]
                module_mapping[weight_name][matrix_type] = adapter_weight_name
                processed_adapter_weight_names.add(adapter_weight_name)

    # merge adapter weights into model weights
    merged_weights = {}
    for weight_name, adapter_weight_names in tqdm(
        module_mapping.items(), desc="Merging adapter weights", total=len(module_mapping)):

        # TODO: support adapter types beyond LoRA
        # TODO: put this on GPU if it is available. This should greatly speedup compute_delta_weight
        lora_A = adapter_weights[adapter_weight_names["lora_A"]]
        lora_B = adapter_weights[adapter_weight_names["lora_B"]]
        delta_weight = compute_delta_weight(
            lora_A,
            lora_B,
            adapter_config.fan_in_fan_out,
            adapter_config.lora_alpha,
            adapter_config.r,
            uses_rslora=uses_rslora(adapter_config),
        )

        # transpose delta weight if necessary
        # TODO(geoffrey): I believe this is required when using Conv1D layers (gpt2).
        # We can likely take this out once we've switched to using Linear layers.
        if (delta_weight.shape != model_weights[weight_name].shape and 
            delta_weight.T.shape == model_weights[weight_name].shape):
            delta_weight = delta_weight.T
        merged_weights[weight_name] = model_weights[weight_name] + delta_weight
    return merged_weights, processed_adapter_weight_names


def create_merged_weight_files(
    adapter_id: str, 
    model_id: str,
    model_weight_filenames: List[Path],
    adapter_source: str = "hub",
) -> List[Path]:
    """Creates merged weight files for the given adapter ID and filenames."""
    source = get_model_source(adapter_source, adapter_id)
    adapter_filenames = source.weight_files()

    adapter_path = get_config_path(adapter_id, adapter_source)
    adapter_config = LoraConfig.from_pretrained(adapter_path)
    if adapter_config.base_model_name_or_path != model_id:
        raise ValueError(f"Adapter '{adapter_id}' is not compatible with model '{model_id}'. "
                         f"Use --model-id '{adapter_config.base_model_name_or_path}' instead.")
    
    # load adapter weights from all shards (should have relatively small memory footprint)
    adapter_weights = {}
    for filename in adapter_filenames:
        adapter_weights.update(load_file(filename))
    remaining_adapter_weight_names = set(adapter_weights.keys())

    merged_weight_directory = Path(HUGGINGFACE_HUB_CACHE) / f"models--{adapter_id.replace('/', '--')}-merged"
    # just grab the existing files if they already exist and return immediately
    lock = FileLock(str(merged_weight_directory)+ ".lock")
    with lock:
        if merged_weight_directory.is_dir():
            logger.info(f"Merged weight directory {merged_weight_directory} exist, skipping merge computation.")
            return weight_files(merged_weight_directory)
        else:
            logger.info("Merged weight files do not exist, computing merge.")
            os.makedirs(merged_weight_directory)

        merged_weight_filenames = []
        for i, filename in enumerate(model_weight_filenames):
            logger.info(
                f"Merging adapter weights into model weights in "
                f"{filename} ({i+1} / {len(model_weight_filenames)})..."
            )
            model_weights = load_file(filename)
            merged_weights, processed_adapter_weight_names = merge_adapter_weights(
                model_weights, adapter_weights, adapter_config)
            
            merged_adapter_filename = Path(merged_weight_directory, os.path.basename(filename))
            save_file(merged_weights, merged_adapter_filename)
            logger.debug(f"Saved merged weights into {merged_adapter_filename}")

            merged_weight_filenames.append(merged_adapter_filename)
            remaining_adapter_weight_names = remaining_adapter_weight_names.difference(
                processed_adapter_weight_names)
        
        if len(remaining_adapter_weight_names) > 0:
            logger.warning("WARNING: The following lora weights were not merged into the model weights:")
            for lora_name in remaining_adapter_weight_names:
                logger.warning("\t" + lora_name)

        logger.info(
            f"Finished merging adapter weights. Merged weight files saved to: {merged_weight_directory}")
        return merged_weight_filenames


def uses_rslora(adapter_config: LoraConfig) -> bool:
    """ Returns True if the adapter uses RSLora for scaling the delta weights. """
    return adapter_config.use_rslora if hasattr(adapter_config, "use_rslora") else False


def get_scaling_factor(
    lora_alpha: int,
    r: int,
    uses_rslora: bool = False,
) -> float:
    """Computes the scaling factor for the lora weights."""
    if uses_rslora:
        return lora_alpha / (r ** 0.5)
    return lora_alpha / r


def main():
    adapter_id = "arnavgrg/codealpaca-qlora"
    adapter_config = LoraConfig.from_pretrained(adapter_id)
    model_id = adapter_config.base_model_name_or_path
    model_weight_filenames = weight_files(model_id, extension=".safetensors")

    merged_adapter_filenames = create_merged_weight_files(adapter_id, model_id, model_weight_filenames)
    print(merged_adapter_filenames)


if __name__ == '__main__':
    main()