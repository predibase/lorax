import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
from loguru import logger
from safetensors import safe_open

from lorax_server.utils.sources import PBASE, S3, map_pbase_model_id_to_s3
from lorax_server.utils.torch_utils import is_fp8


class AbstractWeights(ABC):
    @abstractmethod
    def get_slice(self, tensor_name: str) -> torch.Tensor:
        pass

    @abstractmethod
    def get_tensor(self, tensor_name: str) -> torch.Tensor:
        pass

    @abstractmethod
    def get_slice_shape(self, slice) -> torch.Size:
        pass

    @abstractmethod
    def has_tensor(self, tensor_name: str) -> bool:
        pass

    @property
    @abstractmethod
    def process_group(self):
        pass

    def get_shape(self, tensor_name: str) -> torch.Size:
        return self.get_slice_shape(self.get_slice(tensor_name))

    def get_partial_sharded(self, tensor_name: str, dim: int, range: Optional[Tuple[int, int]] = None):
        """Loads tensor with the given name and shards it along the given dimension.

        The optional range argument can be used to load and split on only a subset of the tensor.
        This is useful in cases where the tensor is stored as one contiguous block, but is logically
        split into different components that need to be sharded separately. For example, when storing
        QKV weights together as a single tensor on disk.

        Args:
            tensor_name (str): Name of the tensor to load.
            dim (int): Dimension to shard along.
            range (Optional[Tuple[int, int]]): Range of indices to load and shard as (offset, size).
        """
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        slice_ = self.get_slice(tensor_name)
        if range is not None:
            offset, size = range
        else:
            offset = 0
            size = self.get_slice_shape(slice_)[dim]
        start, stop = get_start_stop_idxs_for_rank(offset, size, rank, world_size)

        if dim == 0:
            tensor = slice_[start:stop]
        elif dim == 1:
            tensor = slice_[:, start:stop]
        else:
            raise NotImplementedError("Let's make that generic when needed")
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32
        if tensor.dtype not in [torch.int32, torch.int64, torch.float8_e4m3fn, torch.float8_e5m2]:
            tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor

    def get_sharded(self, tensor_name: str, dim: int, range: Optional[Tuple[int, int]] = None):
        slice_ = self.get_slice(tensor_name)
        world_size = self.process_group.size()
        size = self.get_slice_shape(slice_)[dim] if range is None else range[1]
        assert size % world_size == 0, f"The choosen size {size} is not compatible with sharding on {world_size} shards"
        return self.get_partial_sharded(tensor_name, dim, range=range)

    def get_sharded_prefix(self, module_name: str, prefix: Union[str, Tuple], dim: int):
        if isinstance(prefix, str):
            return self.get_sharded(f"{prefix}.{module_name}", dim=dim)
        else:
            assert isinstance(prefix, tuple)
            assert len(prefix) == 2
            return self.get_sharded(f"{prefix[0]}.{module_name}", dim=dim, range=prefix[1])

    def get_sharded_list(self, module_name: str, prefixes: List[Union[str, Tuple]], dim: int):
        return [self.get_sharded_prefix(module_name, p, dim=dim) for p in prefixes]

    def get_multi_weights_col(self, prefixes: List[Union[str, Tuple]], quantize: str, dim: int):
        if quantize in ["gptq", "awq"]:
            try:
                qweight = torch.cat(self.get_sharded_list("qweight", prefixes, dim=1), dim=1)
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `gptq` weight, make sure the model is already quantized, or quantize it with `lorax-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
                )

            qzeros = torch.cat(self.get_sharded_list("qzeros", prefixes, dim=1), dim=1)
            scales = torch.cat(self.get_sharded_list("scales", prefixes, dim=1), dim=1)
            if quantize == "gptq":
                # no tensor parallelism, so remove the range if provided
                prefixes = [p[0] if isinstance(p, tuple) else p for p in prefixes]
                w = [self.get_tensor(f"{p}.g_idx") for p in prefixes]
                for w2 in w[1:]:
                    torch.testing.assert_close(w2, w[0])
                g_idx = w[0]
            else:
                g_idx = None

            bits, groupsize = self._get_bits_and_groupsize()
            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, False)
        else:
            weight_list = self.get_sharded_list("weight", prefixes, dim=0)
            if is_fp8(quantize) and weight_list[0].dtype == torch.float8_e4m3fn:
                # Since there is no kernel for concatenating two tensors in PyTorch
                # for fp8 datatypes, we have to cast to fp16, concat, cast back to fp8
                fp16_weight_list = [w.to(torch.float16) for w in weight_list]
                weight = torch.cat(fp16_weight_list, dim=dim).to(torch.float8_e4m3fn)
                input_scale = None
                if self.has_tensor(f"{prefixes[0]}.input_scale"):
                    # if the layers are being fused, then they have the same inputs
                    # hence their input scales will have to be the same so we pick the first one
                    input_scale = self.get_tensor(f"{prefixes[0]}.input_scale", use_self_dtype=False)
                weight_scale_list = [self.get_tensor(f"{p}.weight_scale", use_self_dtype=False) for p in prefixes]
                if len(weight_scale_list[0].shape) > 1:
                    weight_scale_list = self.get_sharded_list("weight_scale", prefixes, dim=0)
                else:
                    weight_scale_list = [si.repeat(wi.shape[dim]) for si, wi in zip(weight_scale_list, weight_list)]
                # weight scales are in fp32 already so no problem with concatenating them
                weight_scale = torch.cat(weight_scale_list, dim=0)
                return weight, input_scale, weight_scale
            weight = torch.cat(weight_list, dim=dim)

        return weight

    def get_multi_weights_row(self, prefix: str, quantize: str):
        if quantize == "gptq":
            use_exllama = True
            bits, groupsize = self._get_bits_and_groupsize()

            if bits != 4:
                use_exllama = False

            if self.process_group.size() > 1:
                g_idx = self.get_tensor(f"{prefix}.g_idx")
                if g_idx is not None:
                    if (
                        not torch.equal(
                            g_idx.cpu(),
                            torch.tensor(
                                [i // groupsize for i in range(g_idx.shape[0])],
                                dtype=torch.int32,
                            ),
                        )
                        and not (g_idx == 0).all()
                    ):
                        # Exllama implementation does not support row tensor parallelism with act-order, as
                        # it would require to reorder input activations that are split unto several GPUs
                        use_exllama = False

            try:
                qweight = self.get_sharded(f"{prefix}.qweight", dim=0)
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `gptq` weight, make sure the model is already quantized, or quantize it with `lorax-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
                )

            from lorax_server.utils.layers import HAS_EXLLAMA

            if use_exllama:
                if not HAS_EXLLAMA:
                    logger.warning(
                        "Exllama GPTQ cuda kernels (which are faster) could have been used, but are not currently installed, try using BUILD_EXTENSIONS=True"
                    )
                    use_exllama = False
                else:
                    logger.info("Using exllama kernels")

            if use_exllama:
                if groupsize >= 0:
                    # Exllama reorders the weights in advance and the activations on the fly, thus
                    # the scales and zero-points do not need to be reordered.
                    qzeros = self.get_sharded(f"{prefix}.qzeros", dim=0)
                    scales = self.get_sharded(f"{prefix}.scales", dim=0)
                else:
                    qzeros = self.get_tensor(f"{prefix}.qzeros")
                    scales = self.get_tensor(f"{prefix}.scales")

                # For tp > 1, at this point we know we do not use act-order
                if self.process_group.size() == 1:
                    g_idx = self.get_tensor(f"{prefix}.g_idx")
                else:
                    g_idx = None
            else:
                # The triton kernel reorders the scales/zero points instead of the weight/activation.
                # Thus, each rank needs the full qzeros/scales.
                qzeros = self.get_tensor(f"{prefix}.qzeros")
                scales = self.get_tensor(f"{prefix}.scales")
                g_idx = self.get_sharded(f"{prefix}.g_idx", dim=0)

            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, use_exllama)
        elif quantize == "awq":
            bits, groupsize = self._get_bits_and_groupsize()

            try:
                qweight = self.get_sharded(f"{prefix}.qweight", dim=0)
            except RuntimeError:
                raise RuntimeError("Cannot load `awq` weight, make sure the model is already quantized")

            qzeros = self.get_sharded(f"{prefix}.qzeros", dim=0)
            scales = self.get_sharded(f"{prefix}.scales", dim=0)
            g_idx = None
            use_exllama = False
            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, use_exllama)
        else:
            weight = self.get_sharded(f"{prefix}.weight", dim=1)
            if is_fp8(quantize) and weight.dtype == torch.float8_e4m3fn:
                # weight_scale could be a tensor but if we're sharding row-wise then no
                # need to shard the weight_scale as its row dimension would be 1
                weight_scale = self.get_tensor(f"{prefix}.weight_scale", use_self_dtype=False)
                input_scale = None
                if self.has_tensor(f"{prefix}.input_scale"):
                    input_scale = self.get_tensor(f"{prefix}.input_scale", use_self_dtype=False)
                return weight, input_scale, weight_scale
        return weight

    def _get_bits_and_groupsize(self) -> Tuple[int, int]:
        try:
            bits = self.config.quantization_config["bits"]
            groupsize = self.config.quantization_config["group_size"]
        except KeyError:
            # be compatible with old hehavior for gptq
            try:
                bits = self.config.quantization_config["gptq_bits"]
                groupsize = self.config.quantization_config["gptq_groupsize"]
            except KeyError:
                try:
                    bits = self.get_tensor("gptq_bits").item()
                    groupsize = self.get_tensor("gptq_groupsize").item()
                except Exception as e:
                    raise e

        return bits, groupsize


class InMemoryWeights(AbstractWeights):
    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        process_group: torch.distributed.ProcessGroup,
    ):
        self.weights = weights
        self.device = device
        self.dtype = dtype
        self._process_group = process_group

    def get_slice(self, tensor_name: str) -> torch.Tensor:
        return self.get_tensor(tensor_name)

    def get_tensor(self, tensor_name: str) -> torch.Tensor:
        tensor = self.weights[tensor_name]
        return load_module_weight(tensor_name, tensor, self.device, self.dtype)

    def get_slice_shape(self, slice) -> torch.Size:
        return slice.shape

    def has_tensor(self, tensor_name: str) -> bool:
        return tensor_name in self.weights

    @property
    def process_group(self):
        return self._process_group


class Weights(AbstractWeights):
    """
    A class representing weights for a model.

    Args:
        filenames (List[Path]): List of file paths containing the weights.
        device: The device to load the weights onto.
        dtype: The data type to convert the weights to.
        process_group: The process group for distributed training.
        aliases (Optional[Dict[str, List[str]]]): Dictionary of aliases for weight names.
        merged_weight_filenames (Optional[List]): List of file paths containing merged weights.

    Attributes:
        aliases (Dict[str, List[str]]): Dictionary of aliases for weight names.
        routing (Dict[str, str]): Dictionary mapping weight names to file paths.
        device: The device to load the weights onto.
        dtype: The data type of the weights.
        process_group: The process group for distributed training.
        _handles (Dict[str, Any]): Dictionary of file handles for opened weight files.
    """

    def __init__(
        self,
        filenames: List[Path],
        device,
        dtype,
        process_group,
        aliases: Optional[Dict[str, List[str]]] = None,
        merged_weight_filenames: Optional[List] = None,
    ):
        # routes to adapter files take precedence over routes to main model files
        # to ensure that adapter weights are loaded instead of main model weights
        routing = {}
        if merged_weight_filenames is not None:
            for filename in merged_weight_filenames:
                with safe_open(filename, framework="pytorch") as f:
                    for k in f.keys():
                        if k in routing:
                            raise RuntimeError(
                                f"Key {k} was found in multiple adapter files: {filename} and {routing[k]}"
                            )
                        routing[k] = filename

        # set of keys that point to adapter files. Duplicates for these keys found
        # in main model files will be overridden.
        adapter_routes = set(routing.keys())

        for filename in filenames:
            with safe_open(filename, framework="pytorch") as f:
                for k in f.keys():
                    if k in adapter_routes:
                        logger.debug(f"Overriding main model weights with adapter weights for key: {k}")
                    elif k in routing:
                        raise RuntimeError(
                            f"Key {k} was found in multiple non-adapter files: {filename} and {routing[k]}"
                        )
                    else:
                        routing[k] = filename

        if aliases is None:
            aliases = {}
        self.aliases = aliases
        self.routing = routing
        self.device = device
        self.dtype = dtype
        self._process_group = process_group
        self._handles = {}

    @property
    def process_group(self):
        return self._process_group

    def _get_handle(self, filename):
        if filename not in self._handles:
            f = safe_open(filename, framework="pytorch")
            self._handles[filename] = f

        return self._handles[filename]

    def get_filename(self, tensor_name: str) -> (str, str):
        filename = self.routing.get(tensor_name, None)
        if filename is None:
            aliases = self.aliases.get(tensor_name, [])
            for alias in aliases:
                filename = self.routing.get(alias, None)
                if filename is not None:
                    return str(filename), alias
            raise RuntimeError(f"weight {tensor_name} does not exist")
        return str(filename), tensor_name

    def get_slice(self, tensor_name: str):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        return slice_

    def get_slice_shape(self, slice) -> torch.Size:
        return slice.get_shape()

    def get_tensor(self, tensor_name: str, use_self_dtype: bool = True):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        tensor = f.get_tensor(tensor_name)
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32
        if tensor.dtype not in [torch.int32, torch.int64, torch.float8_e4m3fn, torch.float8_e5m2]:
            if use_self_dtype:
                tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor

    def has_tensor(self, tensor_name: str) -> bool:
        filename = self.routing.get(tensor_name, None)
        if filename is None:
            aliases = self.aliases.get(tensor_name, [])
            for alias in aliases:
                filename = self.routing.get(alias, None)
                if filename is not None:
                    return True
            return False
        return True

    def _set_config(self, model_id, config):
        self.config = config

        if not hasattr(self.config, "quantization_config"):
            # fill from other config file
            filename = "quantize_config.json"
            try:
                if os.path.exists(os.path.join(model_id, filename)):
                    filename = os.path.join(model_id, filename)
                else:
                    filename = hf_hub_download(model_id, filename=filename)
                with open(filename, "r") as f:
                    data = json.load(f)
                self.config.quantization_config = data["quantization_config"]
            except Exception:
                filename = "quant_config.json"
                try:
                    if os.path.exists(os.path.join(model_id, filename)):
                        filename = os.path.join(model_id, filename)
                    else:
                        filename = hf_hub_download(model_id, filename=filename)
                    with open(filename, "r") as f:
                        data = json.load(f)
                    self.config.quantization_config = data["quantization_config"]
                except Exception:
                    pass


def get_start_stop_idxs_for_rank(offset, size, rank, world_size):
    block_size = size // world_size
    start = offset + rank * block_size
    stop = offset + (rank + 1) * block_size
    return start, stop


def shard_on_dim(t: torch.Tensor, dim: int, process_group: torch.distributed.ProcessGroup):
    world_size = process_group.size()
    rank = process_group.rank()

    size = t.shape[dim]
    start, stop = get_start_stop_idxs_for_rank(0, size, rank, world_size)

    if dim == 0:
        tensor = t[start:stop]
    elif dim == 1:
        tensor = t[:, start:stop]
    else:
        raise NotImplementedError("Let's make that generic when needed")

    return tensor


def download_weights(
    model_id: str,
    revision: Optional[str] = None,
    extension: str = ".safetensors",
    auto_convert: bool = True,
    source: str = "hub",
    api_token: Optional[str] = None,
    embedding_dim: Optional[int] = None,
):
    # Import here after the logger is added to log potential import exceptions
    from lorax_server import utils
    from lorax_server.utils import sources

    if source == PBASE:
        # TODO(travis): move this into `model_source` to handle behind the abstraction
        api_token = api_token or os.environ.get("PREDIBASE_API_TOKEN")
        model_id = map_pbase_model_id_to_s3(model_id, api_token)
        source = S3

    model_source = sources.get_model_source(source, model_id, revision, extension, api_token, embedding_dim)

    # Test if files were already download
    try:
        model_source.weight_files()
        logger.info("Files are already present on the host. " "Skipping download.")
        return
    # Local files not found
    except (LocalEntryNotFoundError, FileNotFoundError):
        pass

    is_local_model = (Path(model_id).exists() and Path(model_id).is_dir()) or os.getenv(
        "WEIGHTS_CACHE_OVERRIDE", None
    ) is not None

    if not is_local_model:
        # TODO: Combine into class that takes the source as input
        # Try to download weights from the hub
        try:
            model_source.download_model_assets()
            return
        # No weights found on the hub with this extension
        except EntryNotFoundError as e:
            # Check if we want to automatically convert to safetensors or if we can use .bin weights instead
            if not extension == ".safetensors" or not auto_convert:
                raise e

    # Try to see if there are local pytorch weights
    try:
        # Get weights for a local model, a hub cached model and inside the WEIGHTS_CACHE_OVERRIDE
        local_pt_files = model_source.weight_files(extension=".bin")

    # No local pytorch weights
    except LocalEntryNotFoundError:
        if extension == ".safetensors":
            logger.warning(
                f"No safetensors weights found for model {model_id} at revision {revision}. "
                f"Downloading PyTorch weights."
            )

        # Try to see if there are pytorch weights on the hub
        pt_filenames = model_source.remote_weight_files(extension=".bin")
        # Download pytorch weights
        local_pt_files = model_source.download_weights(pt_filenames)

    if auto_convert:
        logger.warning(
            f"No safetensors weights found for model {model_id} at revision {revision}. "
            f"Converting PyTorch weights to safetensors."
        )

        # Safetensors final filenames
        local_st_files = [p.parent / f"{p.stem.lstrip('pytorch_')}.safetensors" for p in local_pt_files]
        try:
            import transformers
            from transformers import AutoConfig

            config_path = sources.get_config_path(model_id, source)
            config = AutoConfig.from_pretrained(
                config_path,
                revision=revision,
            )
            architecture = config.architectures[0]

            class_ = getattr(transformers, architecture)

            # Name for this varible depends on transformers version.
            discard_names = getattr(class_, "_tied_weights_keys", [])
            discard_names.extend(getattr(class_, "_keys_to_ignore_on_load_missing", []))

        except Exception:
            discard_names = []
        # Convert pytorch weights to safetensors
        utils.convert_files(local_pt_files, local_st_files, discard_names)


def load_module_weight(name: str, module: Union[torch.Tensor, str], device, dtype):
    if isinstance(module, torch.Tensor):
        return module.to(device, dtype)

    if isinstance(device, torch.device):
        if device.type == "cuda":
            device = device.index
        elif device.type == "cpu":
            device = "cpu"

    # module would be just the filename if lazy loading happened before
    with safe_open(module, framework="pt", device=device) as f:
        return f.get_tensor(name).to(dtype)
