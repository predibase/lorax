import torch


def is_bf16_supported() -> bool:
    """Check if the current GPU supports bfloat16.

    Returns:
        True if supported, False otherwise.
    """
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
