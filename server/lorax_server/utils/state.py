import os
from contextlib import contextmanager
from typing import Optional

from loguru import logger

WARMUP = False
SPECULATIVE_TOKENS = 0


PREFIX_CACHING = bool(os.environ.get("PREFIX_CACHING", ""))
CHUNKED_PREFILL = bool(os.environ.get("CHUNKED_PREFILL", ""))

# Always use flashinfer when prefix caching is enabled
FLASH_INFER = bool(os.environ.get("FLASH_INFER", "")) or PREFIX_CACHING
if FLASH_INFER:
    logger.info("Backend = flashinfer")
else:
    logger.info("Backend = fa2")

logger.info(f"Prefix caching = {PREFIX_CACHING}")
logger.info(f"Chunked prefill = {CHUNKED_PREFILL}")

SUPPORTS_CHUNKING: Optional[bool] = None
MAX_PREFILL_TOKENS: Optional[int] = None


BLOCK_SIZE: int
if FLASH_INFER:
    BLOCK_SIZE = 1
else:
    BLOCK_SIZE = 16


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


def set_supports_chunking(supports_chunking: bool):
    global SUPPORTS_CHUNKING
    SUPPORTS_CHUNKING = supports_chunking


def get_supports_chunking() -> bool:
    global SUPPORTS_CHUNKING
    return SUPPORTS_CHUNKING


def set_max_prefill_tokens(max_prefill_tokens: int):
    global MAX_PREFILL_TOKENS
    MAX_PREFILL_TOKENS = max_prefill_tokens


def get_max_prefill_tokens() -> int:
    global MAX_PREFILL_TOKENS
    return MAX_PREFILL_TOKENS
