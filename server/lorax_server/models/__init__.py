import os
import torch

from loguru import logger
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import modeling_auto
from typing import Optional

from lorax_server.models.model import Model
from lorax_server.models.causal_lm import CausalLM
from lorax_server.models.flash_causal_lm import FlashCausalLM
from lorax_server.models.bloom import BLOOMSharded
from lorax_server.models.mpt import MPTSharded
from lorax_server.models.seq2seq_lm import Seq2SeqLM
from lorax_server.models.rw import RW
from lorax_server.models.opt import OPTSharded
from lorax_server.models.galactica import GalacticaSharded
from lorax_server.models.santacoder import SantaCoder
from lorax_server.models.t5 import T5Sharded
from lorax_server.models.gpt_neox import GPTNeoxSharded
from lorax_server.utils.sources import get_s3_model_local_dir

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# Disable gradients
torch.set_grad_enabled(False)

__all__ = [
    "Model",
    "BLOOMSharded",
    "CausalLM",
    "FlashCausalLM",
    "GalacticaSharded",
    "Seq2SeqLM",
    "SantaCoder",
    "OPTSharded",
    "T5Sharded",
    "get_model",
]

FLASH_ATT_ERROR_MESSAGE = "{} requires Flash Attention enabled models."

FLASH_ATTENTION = True
try:
    from lorax_server.models.flash_rw import FlashRWSharded
    from lorax_server.models.flash_neox import FlashNeoXSharded
    from lorax_server.models.flash_llama import FlashLlama
    from lorax_server.models.flash_gpt2 import FlashGPT2
    from lorax_server.models.flash_qwen import FlashQwen
    from lorax_server.models.flash_phi import FlashPhi
    from lorax_server.models.flash_santacoder import (
        FlashSantacoderSharded,
    )

except ImportError as e:
    logger.warning(f"Could not import Flash Attention enabled models: {e}")
    FLASH_ATTENTION = False

if FLASH_ATTENTION:
    __all__.append(FlashNeoXSharded)
    __all__.append(FlashRWSharded)
    __all__.append(FlashSantacoderSharded)
    __all__.append(FlashLlama)
    __all__.append(FlashGPT2)
    __all__.append(FlashQwen)
    __all__.append(FlashPhi)
    
MISTRAL = True
try:
    from lorax_server.models.flash_mistral import FlashMistral
except ImportError as e:
    logger.warning(f"Could not import Mistral model: {e}")
    MISTRAL = False

MIXTRAL = True
try:
    from lorax_server.models.flash_mixtral import FlashMixtral
except ImportError as e:
    logger.warning(f"Could not import Mixtral model: {e}")
    MIXTRAL = False

if MISTRAL:
    __all__.append(FlashMistral)

if MIXTRAL:
    __all__.append(FlashMixtral)


def get_model(
    model_id: str,
    adapter_id: str,
    revision: Optional[str],
    sharded: bool,
    quantize: Optional[str],
    compile: bool,
    dtype: Optional[str],
    trust_remote_code: bool,
    source: str,
    adapter_source: str,
) -> Model:
    config_dict = None
    if source == "s3":
        # change the model id to be the local path to the folder so
        # we can load the config_dict locally
        logger.info(f"Using the local files since we are coming from s3")
        model_path = get_s3_model_local_dir(model_id)
        logger.info(f"model_path: {model_path}")
        config_dict, _ = PretrainedConfig.get_config_dict(
            model_path, revision=revision, trust_remote_code=trust_remote_code
        )
        logger.info(f"config_dict: {config_dict}")
        model_id = model_path
    elif source == "hub":
        config_dict, _ = PretrainedConfig.get_config_dict(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
    else: 
        raise ValueError(f"Unknown source {source}")
    
    model_type = config_dict["model_type"]

    if dtype is None:
        dtype = torch.float16
    elif dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise RuntimeError(f"Unknown dtype {dtype}")

    if "facebook/galactica" in model_id:
        return GalacticaSharded(
            model_id,
            revision,
            quantize=quantize,
            compile=compile,
            dtype=dtype,
            dtypetrust_remote_code=trust_remote_code,
        )

    if model_id.startswith("bigcode/"):
        if FLASH_ATTENTION:
            return FlashSantacoderSharded(
                model_id,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            raise NotImplementedError(
                FLASH_ATT_ERROR_MESSAGE.format("Sharded Santacoder")
            )
        else:
            return SantaCoder(
                model_id,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type == "gpt_bigcode":
        if FLASH_ATTENTION:
            return FlashSantacoderSharded(
                model_id,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            raise NotImplementedError(
                FLASH_ATT_ERROR_MESSAGE.format("Sharded Santacoder")
            )
        else:
            return SantaCoder(
                model_id,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type == "bloom":
        return BLOOMSharded(
            model_id,
            revision,
            quantize=quantize,
            compile=compile,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    if model_type == "mpt":
        return MPTSharded(
            model_id, revision, quantize=quantize, compile=compile, trust_remote_code=trust_remote_code
        )

    if model_type == "gpt_neox":
        if FLASH_ATTENTION:
            return FlashNeoXSharded(
                model_id,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            return GPTNeoxSharded(
                model_id,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type == "llama":
        if FLASH_ATTENTION:
            return FlashLlama(
                model_id,
                adapter_id,
                adapter_source,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        elif sharded:
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("Sharded Llama"))
        else:
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    if model_type == "gpt2":
        if FLASH_ATTENTION:
            return FlashGPT2(
                model_id,
                adapter_id,
                adapter_source,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format("GPT-2"))

    if model_type in ["RefinedWeb", "RefinedWebModel", "falcon"]:
        if sharded:
            if FLASH_ATTENTION:
                if config_dict.get("alibi", False):
                    raise NotImplementedError("sharded is not supported for this model")
                return FlashRWSharded(
                    model_id,
                    revision,
                    quantize=quantize,
                    compile=compile,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                )
            raise NotImplementedError(FLASH_ATT_ERROR_MESSAGE.format(f"Sharded Falcon"))
        else:
            if FLASH_ATTENTION and not config_dict.get("alibi", False):
                return FlashRWSharded(
                    model_id,
                    revision,
                    quantize=quantize,
                    compile=compile,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                )
            else:
                return RW(
                    model_id,
                    revision,
                    quantize=quantize,
                    compile=compile,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                )

    if model_type == "mistral":
        if MISTRAL:
            return FlashMistral(
                model_id,
                adapter_id,
                adapter_source,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        raise NotImplementedError("Mistral model requires flash attention v2")
    
    if model_type == "mixtral":
        if MIXTRAL:
            return FlashMixtral(
                model_id,
                adapter_id,
                adapter_source,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        raise NotImplementedError("Mixtral models requires flash attention v2, stk and megablocks")
    
    if model_type == "qwen":
        if FLASH_ATTENTION:
            return FlashQwen(
                model_id,
                adapter_id,
                adapter_source,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        raise NotImplementedError("Qwen model requires flash attention v2")
    
    if model_type in ["phi-msft", "phi"]:
        if FLASH_ATTENTION:
            return FlashPhi(
                model_id,
                adapter_id,
                adapter_source,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        raise NotImplementedError("Phi model requires flash attention v2")

    if model_type == "opt":
        return OPTSharded(
            model_id,
            revision,
            quantize=quantize,
            compile=compile,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if model_type == "t5":
        return T5Sharded(
            model_id,
            revision,
            quantize=quantize,
            compile=compile,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if sharded:
        raise ValueError("sharded is not supported for AutoModel")
    if quantize == "gptq":
        raise ValueError(
            "gptq quantization is not supported for AutoModel, you can try to quantize it with `lorax-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
        )
    elif (quantize == "bitsandbytes-fp4") or (quantize == "bitsandbytes-nf4"):
        raise ValueError(
            "4bit quantization is not supported for AutoModel"
        )
    if quantize == "awq":
        raise ValueError(
            "awq quantization is not supported for AutoModel"
        )

    if model_type in modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return CausalLM(
            model_id,
            revision,
            quantize=quantize,
            compile=compile,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    if model_type in modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
        return Seq2SeqLM(
            model_id,
            revision,
            quantize=quantize,
            compile=compile,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    auto_map = config_dict.get("auto_map", None)
    if trust_remote_code and auto_map is not None:
        if "AutoModelForCausalLM" in auto_map.keys():
            return CausalLM(
                model_id,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        if "AutoModelForSeq2SeqLM" in auto_map.keys():
            return Seq2SeqLM(
                model_id,
                revision,
                quantize=quantize,
                compile=compile,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

    raise ValueError(f"Unsupported model type {model_type}")
