from typing import Optional

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

from lorax_server.models.bloom import BLOOMSharded
from lorax_server.models.causal_lm import CausalLM
from lorax_server.models.flash_causal_lm import FlashCausalLM
from lorax_server.models.galactica import GalacticaSharded
from lorax_server.models.model import Model
from lorax_server.models.mpt import MPTSharded
from lorax_server.models.opt import OPTSharded
from lorax_server.models.santacoder import SantaCoder
from lorax_server.models.seq2seq_lm import Seq2SeqLM
from lorax_server.models.t5 import T5Sharded
from lorax_server.utils.sources import get_s3_model_local_dir
from lorax_server.utils.torch_utils import is_bf16_supported

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
    merge_adapter_weights: bool,
    embedding_dim: Optional[int] = None,
) -> Model:
    config_dict = None
    if source == "s3":
        # change the model id to be the local path to the folder so
        # we can load the config_dict locally
        logger.info("Using the local files since we are coming from s3")
        model_path = get_s3_model_local_dir(model_id)
        logger.info(f"model_path: {model_path}")
        config_dict, _ = PretrainedConfig.get_config_dict(
            model_path, revision=revision, trust_remote_code=trust_remote_code
        )
        logger.info(f"config_dict: {config_dict}")
        model_id = str(model_path)
    elif source == "hub":
        config_dict, _ = PretrainedConfig.get_config_dict(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
    else:
        raise ValueError(f"Unknown source {source}")

    model_type = config_dict["model_type"]
    is_dtype_provided = dtype is not None
    dtype = dtype or config_dict.get("torch_dtype", "float16")

    if dtype in {"float16", "float32"}:
        dtype = torch.float16
    elif dtype == "bfloat16":
        if not is_bf16_supported():
            if is_dtype_provided:
                raise RuntimeError("bfloat16 is not supported on this device, set --dtype float16.")
            logger.warning("bfloat16 is not supported on this device, falling back to float16")
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
    else:
        try:
            dtype = getattr(torch, dtype)
        except AttributeError:
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

    if model_type == "bert":
        from lorax_server.models.flash_bert import FlashBert

        if config_dict["architectures"][0] == "BertForTokenClassification":
            return FlashBert(model_id, revision=revision, dtype=dtype, classifcation_head=True)
        return FlashBert(model_id, revision=revision, dtype=dtype)

    if model_type == "distilbert":
        from lorax_server.models.flash_distilbert import FlashDistilBert

        if config_dict["architectures"][0] == "DistilBertForMaskedLM":
            return FlashDistilBert(model_id, revision=revision, dtype=dtype)

        if config_dict["architectures"][0] == "DistilBertForTokenClassification":
            return FlashDistilBert(model_id, revision=revision, dtype=dtype, classifcation_head=True)

    flash_causal_lm_kwargs = dict(
        quantize=quantize,
        compile=compile,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        merge_adapter_weights=merge_adapter_weights,
    )

    if model_id.startswith("bigcode/") or model_type == "gpt_bigcode":
        from lorax_server.models.flash_santacoder import FlashSantacoderSharded

        return FlashSantacoderSharded(
            model_id,
            revision,
            **flash_causal_lm_kwargs,
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
            model_id,
            revision,
            quantize=quantize,
            compile=compile,
            trust_remote_code=trust_remote_code,
        )

    if model_type == "gpt_neox":
        from lorax_server.models.flash_neox import FlashNeoXSharded

        return FlashNeoXSharded(
            model_id,
            revision,
            quantize=quantize,
            compile=compile,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    if model_type == "llama":
        from lorax_server.models.flash_llama import FlashLlama

        return FlashLlama(
            model_id,
            adapter_id,
            adapter_source,
            revision,
            **flash_causal_lm_kwargs,
        )

    if model_type == "gpt2":
        from lorax_server.models.flash_gpt2 import FlashGPT2

        return FlashGPT2(
            model_id,
            adapter_id,
            adapter_source,
            revision,
            **flash_causal_lm_kwargs,
        )

    if model_type in ["RefinedWeb", "RefinedWebModel", "falcon"]:
        from lorax_server.models.flash_rw import FlashRWSharded

        return FlashRWSharded(
            model_id,
            revision,
            **flash_causal_lm_kwargs,
        )

    if model_type == "mistral":
        from lorax_server.models.flash_mistral import FlashMistral

        return FlashMistral(
            model_id,
            adapter_id,
            adapter_source,
            revision,
            **flash_causal_lm_kwargs,
        )

    if model_type == "mixtral":
        from lorax_server.models.flash_mixtral import FlashMixtral

        return FlashMixtral(
            model_id,
            adapter_id,
            adapter_source,
            revision,
            **flash_causal_lm_kwargs,
        )

    if model_type == "qwen":
        from lorax_server.models.flash_qwen import FlashQwen

        return FlashQwen(
            model_id,
            adapter_id,
            adapter_source,
            revision,
            **flash_causal_lm_kwargs,
        )

    if model_type == "qwen2":
        from lorax_server.models.flash_qwen2 import FlashQwen2

        return FlashQwen2(
            model_id,
            adapter_id,
            adapter_source,
            revision,
            embedding_dim=embedding_dim,
            **flash_causal_lm_kwargs,
        )

    if model_type in ["phi-msft", "phi"]:
        from lorax_server.models.flash_phi import FlashPhi

        return FlashPhi(
            model_id,
            adapter_id,
            adapter_source,
            revision,
            **flash_causal_lm_kwargs,
        )

    if model_type == "phi3":
        from lorax_server.models.flash_phi3 import FlashPhi3

        return FlashPhi3(
            model_id,
            adapter_id,
            adapter_source,
            revision,
            **flash_causal_lm_kwargs,
        )

    if model_type == "gemma":
        from lorax_server.models.flash_gemma import FlashGemma

        return FlashGemma(
            model_id,
            adapter_id,
            adapter_source,
            revision,
            **flash_causal_lm_kwargs,
        )

    if model_type == "gemma2":
        from lorax_server.models.flash_gemma2 import FlashGemma2

        return FlashGemma2(
            model_id,
            adapter_id,
            adapter_source,
            revision,
            **flash_causal_lm_kwargs,
        )

    if model_type == "cohere":
        from lorax_server.models.flash_cohere import FlashCohere

        return FlashCohere(
            model_id,
            adapter_id,
            adapter_source,
            revision,
            **flash_causal_lm_kwargs,
        )

    if model_type == "dbrx":
        from lorax_server.models.flash_dbrx import FlashDbrx

        return FlashDbrx(
            model_id,
            adapter_id,
            adapter_source,
            revision,
            **flash_causal_lm_kwargs,
        )

    if model_type == "llava_next" or model_type == "llava":
        from lorax_server.models.custom_modeling.llava_next import LlavaNextForConditionalGeneration
        from lorax_server.models.vlm_causal_lm import VlmCausalLM

        return VlmCausalLM(
            model_class=LlavaNextForConditionalGeneration,
            model_id=model_id,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            revision=revision,
            **flash_causal_lm_kwargs,
        )

    if model_type == "mllama":
        from lorax_server.models.custom_modeling.mllama import MllamaForConditionalGeneration
        from lorax_server.models.mllama import MllamaCausalLM, MllamaCausalLMBatch

        return MllamaCausalLM(
            model_id=model_id,
            model_class=MllamaForConditionalGeneration,
            batch_class=MllamaCausalLMBatch,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            revision=revision,
            **flash_causal_lm_kwargs,
        )

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

    raise ValueError(f"Unsupported model type {model_type}")
