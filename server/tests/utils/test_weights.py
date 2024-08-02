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
    'model_id', [
        'neuralmagic/Qwen2-0.5B-Instruct-FP8',
        'Qwen/Qwen2-0.5B-Instruct'
    ]
)
@pytest.mark.parametrize(
    'prefixes', [
        ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
        ['mlp.gate_proj', 'mlp.up_proj']
    ]
)
def test_get_multi_weights_col(model_id, prefixes):
    process_group, _, _ = initialize_torch_distributed()
    filenames = weight_hub_files(model_id, 'main', '.safetensors')
    local_filenames = download_weights(filenames, model_id, 'main')
    config = Qwen2Config.from_pretrained(model_id, revision='main', trust_remote_code=False)
    quantize = None
    if hasattr(config, 'quantization_config'):
        quantize = config.quantization_config['quant_method']

    weights = Weights(local_filenames, 'cpu', torch.bfloat16, process_group=process_group)
    prefix = 'model.layers.0'
    prefixes = [f'{prefix}.{k}' for k in prefixes]
    weight = weights.get_multi_weights_col(
        prefixes=prefixes,
        quantize=quantize,
        dim=0,
    )
    if quantize is not None:
        assert type(weight) is tuple
        weight, input_scale, weight_scale = weight
        assert weight.dtype == torch.float8_e4m3fn
        assert input_scale.dtype == torch.float
        assert weight_scale.dtype == torch.float
    else:
        assert weight.dtype == torch.bfloat16

@pytest.mark.parametrize(
    'model_id', [
        'neuralmagic/Qwen2-0.5B-Instruct-FP8',
        'Qwen/Qwen2-0.5B-Instruct'
    ]
)
@pytest.mark.parametrize(
    'prefix', ['self_attn.o_proj', 'mlp.down_proj'],
)
def test_get_multi_weights_row(model_id, prefix):
    process_group, _, _ = initialize_torch_distributed()
    filenames = weight_hub_files(model_id, 'main', '.safetensors')
    local_filenames = download_weights(filenames, model_id, 'main')
    config = Qwen2Config.from_pretrained(model_id, revision='main', trust_remote_code=False)
    quantize = None
    if hasattr(config, 'quantization_config'):
        quantize = config.quantization_config['quant_method']

    weights = Weights(local_filenames, 'cpu', torch.bfloat16, process_group=process_group)
    weight = weights.get_multi_weights_row(f'model.layers.0.{prefix}', quantize=quantize)
    if quantize is not None:
        assert type(weight) is tuple
        weight, input_scale, weight_scale = weight
        assert weight.dtype == torch.float8_e4m3fn
        assert input_scale.dtype == torch.float
        assert weight_scale.dtype == torch.float
    else:
        assert weight.dtype == torch.bfloat16
