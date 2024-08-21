import os
from contextlib import contextmanager

from loguru import logger

WARMUP = False
SPECULATIVE_TOKENS = 0

FLASH_INFER = bool(os.environ.get("FLASH_INFER", ""))
if FLASH_INFER:
    logger.info("Using flashinfer")


def set_warmup(value: bool):
    global WARMUP
    WARMUP = value


def is_warmup() -> bool:
    return WARMUP


@contextmanager
def warmup_mode():
    try:
        set_warmup(True)
        yield
    finally:
        set_warmup(False)


def set_speculative_tokens(value: int):
    global SPECULATIVE_TOKENS
    SPECULATIVE_TOKENS = value


def get_speculative_tokens() -> int:
    return SPECULATIVE_TOKENS
