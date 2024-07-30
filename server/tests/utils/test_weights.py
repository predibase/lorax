import pytest
import torch
from transformers.models.qwen2 import Qwen2Config

from lorax_server.utils.dist import initialize_torch_distributed
from lorax_server.utils.sources.hub import (
    download_weights,
    weight_hub_files,
)
from lorax_server.utils.weights import Weights


@pytest.mark.parametrize(
    "model_id", [
        'neuralmagic/Qwen2-0.5B-Instruct-FP8',
        'Qwen/Qwen2-0.5B-Instruct'
    ],
)
def test_get_multi_weights_col(model_id):
    process_group, _, _ = initialize_torch_distributed()
    filenames = weight_hub_files(model_id, "main", ".safetensors")
    local_filenames = download_weights(filenames, model_id, "main")
    config = Qwen2Config.from_pretrained(model_id, revision="main", trust_remote_code=False)
    quantize = None
    if hasattr(config, 'quantization_config'):
        quantize = config.quantization_config['quant_method']
    
    weights = Weights(local_filenames, 'cpu', torch.bfloat16, process_group=process_group)
    prefix = 'model.layers.0.self_attn'
    weight = weights.get_multi_weights_col(
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        quantize=quantize,
        dim=0,
    )
    if quantize is not None:
        assert type(weight) is tuple
        weight, input_scale, weight_scale = weight
        assert weight.dtype == torch.float8_e4m3fn
        assert input_scale.dtype == torch.float
        assert weight_scale.dtype == torch.float
        assert weight.shape[0] == weight_scale.shape[0]
    else:
        assert weight.dtype == torch.bfloat16
