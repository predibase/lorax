import os
from contextlib import contextmanager
from typing import Optional

from loguru import logger

WARMUP = False
SPECULATIVE_TOKENS = 0
NGRAM = False


LORAX_PROFILER_DIR = os.environ.get("LORAX_PROFILER_DIR", None)
PREFIX_CACHING = bool(os.environ.get("PREFIX_CACHING", ""))
CHUNKED_PREFILL = bool(os.environ.get("CHUNKED_PREFILL", ""))
LORAX_SPECULATION_MAX_BATCH_SIZE = int(os.environ.get("LORAX_SPECULATION_MAX_BATCH_SIZE", 32))

# Always use flashinfer when prefix caching is enabled
FLASH_INFER = bool(os.environ.get("FLASH_INFER", "")) or PREFIX_CACHING
if FLASH_INFER:
    logger.info("Backend = flashinfer")
else:
    logger.info("Backend = fa2")

logger.info(f"Prefix caching = {PREFIX_CACHING}")
logger.info(f"Chunked prefill = {CHUNKED_PREFILL}")

if LORAX_PROFILER_DIR:
    logger.info(f"Torch profiling enabled, output dir = {LORAX_PROFILER_DIR}")

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


def set_speculative_tokens(value: int, use_ngram: bool):
    global SPECULATIVE_TOKENS
    global NGRAM
    SPECULATIVE_TOKENS = value
    NGRAM = use_ngram


def get_speculative_tokens() -> int:
    return SPECULATIVE_TOKENS


def use_ngram() -> bool:
    return NGRAM


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
