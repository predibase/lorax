import torch


def is_bf16_supported() -> bool:
    """Check if the current GPU supports bfloat16.

    Returns:
        True if supported, False otherwise.
    """
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def is_fp8_quantized(config, layer_name):
    # check if quantization is fp8 and either of the fused layers is not ignored
    # typically, either all qkv will be quantized or none so just check for one
    if config.quantize == 'fp8' and hasattr(config, 'quantization_config'):
        ignored_layers = set(config.quantization_config.get('ignored_layers', []))
        if layer_name not in ignored_layers:
            return 'fp8'
    return None
