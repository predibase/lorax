import asyncio
import os
import shutil
import torch
from peft import PeftConfig

from grpc import aio
from loguru import logger

from grpc_reflection.v1alpha import reflection
from pathlib import Path
from typing import List, Optional

from lorax_server.cache import Cache
from lorax_server.cli import download_weights
from lorax_server.interceptor import ExceptionInterceptor
from lorax_server.models import Model, get_model
from lorax_server.pb import generate_pb2_grpc, generate_pb2
from lorax_server.tracing import UDSOpenTelemetryAioServerInterceptor
from lorax_server.utils import HUB, LOCAL, S3, PBASE, get_config_path, get_local_dir, map_pbase_model_id_to_s3
from lorax_server.utils.adapter import BASE_MODEL_ADAPTER_ID


class LoraxService(generate_pb2_grpc.LoraxServiceServicer):
    """
    Implementation of the LoraxService gRPC service.

    Args:
        model (Model): The model used for inference.
        cache (Cache): The cache used for storing and retrieving batches.
        server_urls (List[str]): List of server URLs for service discovery.
    """

    def __init__(self, model: Model, cache: Cache, server_urls: List[str]):
        self.cache = cache
        self.model = model
        self.server_urls = server_urls
        # For some reason, inference_mode does not work well with GLOO which we use on CPU
        if model.device.type == "cuda":
            # Force inference mode for the lifetime of LoraxService
            self._inference_mode_raii_guard = torch._C._InferenceMode(True)

    async def Info(self, request, context):
        return self.model.info

    async def Health(self, request, context):
        if self.model.device.type == "cuda":
            torch.zeros((2, 2)).cuda()
        return generate_pb2.HealthResponse()

    async def ServiceDiscovery(self, request, context):
        return generate_pb2.ServiceDiscoveryResponse(urls=self.server_urls)

    async def ClearCache(self, request, context):
        if request.HasField("id"):
            self.cache.delete(request.id)
        else:
            self.cache.clear()
        return generate_pb2.ClearCacheResponse()

    async def FilterBatch(self, request, context):
        batch = self.cache.pop(request.batch_id)
        if batch is None:
            raise ValueError(f"Batch ID {request.batch_id} not found in cache.")
        filtered_batch = batch.filter(request.request_ids)
        self.cache.set(filtered_batch)

        return generate_pb2.FilterBatchResponse(batch=filtered_batch.to_pb())

    async def Warmup(self, request, context):
        batch = self.model.batch_type.from_pb(
            request.batch, self.model.tokenizer, self.model.dtype, self.model.device
        )
        max_supported_total_tokens = self.model.warmup(batch)

        return generate_pb2.WarmupResponse(
            max_supported_total_tokens=max_supported_total_tokens
        )

    async def Prefill(self, request, context):
        batch = self.model.batch_type.from_pb(
            request.batch, self.model.tokenizer, self.model.dtype, self.model.device
        )

        generations, next_batch = self.model.generate_token(batch)
        self.cache.set(next_batch)

        return generate_pb2.PrefillResponse(
            generations=[generation.to_pb() for generation in generations],
            batch=next_batch.to_pb() if next_batch else None,
        )

    async def Decode(self, request, context):
        if len(request.batches) == 0:
            raise ValueError("Must provide at least one batch")

        batches = []
        for batch_pb in request.batches:
            batch = self.cache.pop(batch_pb.id)
            if batch is None:
                raise ValueError(f"Batch ID {batch_pb.id} not found in cache.")
            batches.append(batch)

        if len(batches) == 0:
            raise ValueError("All batches are empty")

        if len(batches) > 1:
            batch = self.model.batch_type.concatenate(batches)
        else:
            batch = batches[0]

        generations, next_batch = self.model.generate_token(batch)
        self.cache.set(next_batch)

        return generate_pb2.DecodeResponse(
            generations=[generation.to_pb() for generation in generations],
            batch=next_batch.to_pb() if next_batch else None,
        )
        
    async def DownloadAdapter(self, request, context):
        adapter_id = request.adapter_id
        if adapter_id == BASE_MODEL_ADAPTER_ID:
            logger.info("No adapter to download for base model. Skipping.")
            return generate_pb2.DownloadAdapterResponse(
                adapter_id=adapter_id,
                adapter_source=request.adapter_source,
            )

        adapter_source = _adapter_source_enum_to_string(request.adapter_source)
        if adapter_source == PBASE:
            adapter_id = map_pbase_model_id_to_s3(adapter_id, request.api_token)
            adapter_source = S3
        try:
            # fail fast if ID is not an adapter (i.e. it is a full model)
            # TODO(geoffrey): do this for S3– can't do it this way because the
            # files are not yet downloaded locally at this point.
            if adapter_source == HUB:
                config_path = get_config_path(adapter_id, adapter_source)
                PeftConfig.from_pretrained(config_path)

            download_weights(adapter_id, source=adapter_source)
            return generate_pb2.DownloadAdapterResponse(
                adapter_id=adapter_id,
                adapter_source=request.adapter_source,
            )
        except Exception:
            logger.exception("Error when downloading adapter")

            if adapter_source != LOCAL:
                # delete safetensors files if there is an issue downloading or converting 
                # the weights to prevent cache hits by subsequent calls
                try:
                    local_path = get_local_dir(adapter_id, adapter_source)
                    shutil.rmtree(local_path)
                except Exception as e:
                    logger.warning(f"Error cleaning up safetensors files after "
                                f"download error: {e}\nIgnoring.")
            raise

    async def LoadAdapter(self, request, context):
        try:
            adapter_id = request.adapter_id
            adapter_source = _adapter_source_enum_to_string(request.adapter_source)
            adapter_index = request.adapter_index
            if adapter_source == PBASE:
                adapter_id = map_pbase_model_id_to_s3(adapter_id, request.api_token)
                adapter_source = S3
            self.model.load_adapter(adapter_id, adapter_source, adapter_index)
            
            return generate_pb2.LoadAdapterResponse(
                adapter_id=adapter_id,
                adapter_source=request.adapter_source,
                adapter_index=adapter_index,
            )
        except Exception:
            logger.exception("Error when loading adapter")
            raise

    async def OffloadAdapter(self, request, context):
        try:
            adapter_id = request.adapter_id
            adapter_source = _adapter_source_enum_to_string(request.adapter_source)
            adapter_index = request.adapter_index
            self.model.offload_adapter(adapter_id, adapter_source, adapter_index)
            
            return generate_pb2.OffloadAdapterResponse(
                adapter_id=adapter_id,
                adapter_source=request.adapter_source,
                adapter_index=adapter_index,
            )
        except Exception:
            logger.exception("Error when offloading adapter")
            raise


def serve(
    model_id: str,
    adapter_id: str,
    revision: Optional[str],
    sharded: bool,
    quantize: Optional[str],
    compile: bool,
    dtype: Optional[str],
    trust_remote_code: bool,
    uds_path: Path,
    source: str,
    adapter_source: str,
):
    async def serve_inner(
        model_id: str,
        adapter_id: str,
        revision: Optional[str],
        sharded: bool = False,
        quantize: Optional[str] = None,
        compile: bool = False,
        dtype: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        unix_socket_template = "unix://{}-{}"
        if sharded:
            server_urls = [
                unix_socket_template.format(uds_path, rank)
                for rank in range(int(os.environ["WORLD_SIZE"]))
            ]
            local_url = server_urls[int(os.environ["RANK"])]
        else:
            local_url = unix_socket_template.format(uds_path, 0)
            server_urls = [local_url]

        try:
            model = get_model(
                model_id, adapter_id, revision, sharded, quantize, compile, dtype, trust_remote_code, source, adapter_source
            )
        except Exception:
            logger.exception("Error when initializing model")
            raise

        if quantize == "gptq":
            try:
                # When using GPTQ, Exllama kernels need some global kernels
                # For which we have the finale shapes only after the model has loaded
                # This will allocate those buffers.
                from lorax_server.utils.gptq.exllamav2 import (
                    create_exllama_buffers,
                    set_device,
                )

                set_device(model.device)
                create_exllama_buffers()
            except ImportError:
                pass

        server = aio.server(
            interceptors=[
                ExceptionInterceptor(),
                UDSOpenTelemetryAioServerInterceptor(),
            ]
        )
        generate_pb2_grpc.add_LoraxServiceServicer_to_server(
            LoraxService(model, Cache(), server_urls), server
        )
        SERVICE_NAMES = (
            generate_pb2.DESCRIPTOR.services_by_name["LoraxService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        server.add_insecure_port(local_url)

        await server.start()

        logger.info("Server started at {}".format(local_url))

        try:
            await server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Signal received. Shutting down")
            await server.stop(0)

    asyncio.run(
        serve_inner(model_id, adapter_id, revision, sharded, quantize, compile, dtype, trust_remote_code)
    )


def _adapter_source_enum_to_string(adapter_source: int) -> str:
    # TODO(travis): refactor this to be less hacky
    if adapter_source == generate_pb2.AdapterSource.HUB:
        return HUB
    elif adapter_source == generate_pb2.AdapterSource.S3:
        return S3
    elif adapter_source == generate_pb2.AdapterSource.LOCAL:
        return LOCAL
    elif adapter_source == generate_pb2.AdapterSource.PBASE:
        return PBASE
    else:
        raise ValueError(f"Unknown adapter source {adapter_source}")