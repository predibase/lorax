import os
import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger

from lorax_server.utils.weights import download_weights as _download_weights

app = typer.Typer()


class Quantization(str, Enum):
    bitsandbytes = "bitsandbytes"
    bitsandbytes_nf4 = "bitsandbytes-nf4"
    bitsandbytes_fp4 = "bitsandbytes-fp4"
    gptq = "gptq"
    awq = "awq"
    eetq = "eetq"
    hqq_4bit = "hqq-4bit"
    hqq_3bit = "hqq-3bit"
    hqq_2bit = "hqq-2bit"
    fp8 = "fp8"


class Dtype(str, Enum):
    float16 = "float16"
    bloat16 = "bfloat16"


@app.command()
def serve(
    model_id: str,
    adapter_id: str = "",
    revision: Optional[str] = None,
    sharded: bool = False,
    quantize: Optional[Quantization] = None,
    compile: bool = False,
    dtype: Optional[Dtype] = None,
    trust_remote_code: bool = False,
    uds_path: Path = "/tmp/lorax-server",
    logger_level: str = "INFO",
    json_output: bool = False,
    otlp_endpoint: Optional[str] = None,
    source: str = "hub",
    adapter_source: str = "hub",
    speculative_tokens: int = 0,
    preloaded_adapter_ids: Optional[List[str]] = typer.Option(None),
    preloaded_adapter_source: Optional[str] = None,
    embedding_dim: Optional[int] = None,
):
    preloaded_adapter_ids = preloaded_adapter_ids or []
    preloaded_adapter_source = preloaded_adapter_source or adapter_source

    if sharded:
        assert os.getenv("RANK", None) is not None, "RANK must be set when sharded is True"
        assert os.getenv("WORLD_SIZE", None) is not None, "WORLD_SIZE must be set when sharded is True"
        assert os.getenv("MASTER_ADDR", None) is not None, "MASTER_ADDR must be set when sharded is True"
        assert os.getenv("MASTER_PORT", None) is not None, "MASTER_PORT must be set when sharded is True"

    # Remove default handler
    logger.remove()
    logger.add(
        sys.stdout,
        format="{file}:{line} {message}",
        filter="lorax_server",
        level=logger_level,
        serialize=json_output,
        backtrace=True,
        diagnose=False,
    )

    # Import here after the logger is added to log potential import exceptions
    from lorax_server import server
    from lorax_server.tracing import setup_tracing

    # Setup OpenTelemetry distributed tracing
    if otlp_endpoint is not None:
        setup_tracing(shard=os.getenv("RANK", 0), otlp_endpoint=otlp_endpoint)

    # Downgrade enum into str for easier management later on
    quantize = None if quantize is None else quantize.value
    dtype = None if dtype is None else dtype.value
    if dtype is not None and quantize is not None:
        raise RuntimeError(
            "Only 1 can be set between `dtype` and `quantize`, as they both decide how the final model is initialized."
        )
    server.serve(
        model_id,
        adapter_id,
        revision,
        sharded,
        quantize,
        compile,
        dtype,
        trust_remote_code,
        uds_path,
        source,
        adapter_source,
        speculative_tokens,
        preloaded_adapter_ids,
        preloaded_adapter_source,
        embedding_dim,
    )


@app.command()
def download_weights(
    model_id: str,
    revision: Optional[str] = None,
    extension: str = ".safetensors",
    auto_convert: bool = True,
    logger_level: str = "INFO",
    json_output: bool = False,
    source: str = "hub",
    adapter_id: str = "",
    adapter_source: str = "hub",
    api_token: Optional[str] = None,
    embedding_dim: Optional[int] = None,
):
    # Remove default handler
    logger.remove()
    logger.add(
        sys.stdout,
        format="{file}:{line} {message}",
        filter="lorax_server",
        level=logger_level,
        serialize=json_output,
        backtrace=True,
        diagnose=False,
    )
    _download_weights(model_id, revision, extension, auto_convert, source, api_token, embedding_dim)
    if adapter_id:
        _download_weights(adapter_id, revision, extension, auto_convert, adapter_source, api_token)


@app.command()
def quantize(
    model_id: str,
    output_dir: str,
    revision: Optional[str] = None,
    logger_level: str = "INFO",
    json_output: bool = False,
    trust_remote_code: bool = False,
    upload_to_model_id: Optional[str] = None,
    percdamp: float = 0.01,
    act_order: bool = False,
):
    if revision is None:
        revision = "main"
    download_weights(
        model_id=model_id,
        revision=revision,
        logger_level=logger_level,
        json_output=json_output,
    )
    from lorax_server.utils.gptq.quantize import quantize

    quantize(
        model_id=model_id,
        bits=4,
        groupsize=128,
        output_dir=output_dir,
        revision=revision,
        trust_remote_code=trust_remote_code,
        upload_to_model_id=upload_to_model_id,
        percdamp=percdamp,
        act_order=act_order,
    )


if __name__ == "__main__":
    app()
