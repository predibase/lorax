import torch


def is_bf16_supported() -> bool:
    """Check if the current GPU supports bfloat16.

    Returns:
        True if supported, False otherwise.
    """
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def is_quantized(quantize):
    return quantize and quantize in ["gptq", "awq", "fp8", "fp8-kv"]


def is_fp8_supported():
    return torch.cuda.is_available() and \
        (torch.cuda.get_device_capability()[0] >= 9) or \
        (torch.cuda.get_device_capability()[0] == 8 and torch.cuda.get_device_capability()[1] >= 9)


def is_fp8_kv(quantize):
    return quantize and quantize == 'fp8-kv'


def is_fp8(quantize):
    return quantize and quantize.startswith('fp8')
