import asyncio
import os
from pathlib import Path
from typing import List, Optional

import torch
from grpc import aio
from grpc_reflection.v1alpha import reflection
from loguru import logger

from lorax_server.adapters.utils import download_adapter
from lorax_server.cache import Cache
from lorax_server.interceptor import ExceptionInterceptor
from lorax_server.models import Model, get_model
from lorax_server.pb import generate_pb2, generate_pb2_grpc
from lorax_server.tracing import UDSOpenTelemetryAioServerInterceptor
from lorax_server.utils import HUB, LOCAL, PBASE, S3, map_pbase_model_id_to_s3
from lorax_server.utils.adapter import BASE_MODEL_ADAPTER_ID, is_base_model
from lorax_server.utils.sgmv import has_sgmv
from lorax_server.utils.state import set_speculative_tokens


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

    async def Warmup(self, request: generate_pb2.WarmupRequest, context):
        batch = self.model.batch_type.from_pb(
            request.batch,
            self.model.tokenizer,
            self.model.tokenizers,
            self.model.dtype,
            self.model.device,
        )
        max_supported_total_tokens = self.model.warmup(batch, request.max_new_tokens)

        return generate_pb2.WarmupResponse(max_supported_total_tokens=max_supported_total_tokens)

    async def Prefill(self, request: generate_pb2.PrefillRequest, context):
        batch = self.model.batch_type.from_pb(
            request.batch,
            self.model.tokenizer,
            self.model.tokenizers,
            self.model.dtype,
            self.model.device,
        )

        generations, next_batch = self.model.generate_token(batch)
        self.cache.set(next_batch)

        return generate_pb2.PrefillResponse(
            generations=[generation.to_pb() for generation in generations],
            batch=next_batch.to_pb() if next_batch else None,
        )

    async def Embed(self, request: generate_pb2.EmbedRequest, context):
        if not self.model.supports_embeddings:
            logger.error("Model does not support embeddings")
            return generate_pb2.EmbedResponse(
                embeddings=generate_pb2.Embedding(), errorMsg="Model does not support embeddings"
            )
        batch = request.inputs
        tokenised_batch = self.model.tokenize_to_batch(batch)
        embeddings = self.model.embed(tokenised_batch)
        return generate_pb2.EmbedResponse(embeddings=embeddings)

    async def Decode(self, request: generate_pb2.DecodeRequest, context):
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

    async def DownloadAdapter(self, request: generate_pb2.DownloadAdapterRequest, context):
        adapter_parameters = request.adapter_parameters
        if is_base_model(adapter_parameters):
            logger.info("No adapter to download for base model. Skipping.")
            return generate_pb2.DownloadAdapterResponse(downloaded=False)

        adapter_bytes = 0
        api_token = request.api_token
        adapter_source = _adapter_source_enum_to_string(request.adapter_source)
        for adapter_id in adapter_parameters.adapter_ids:
            if adapter_id == BASE_MODEL_ADAPTER_ID:
                logger.info("No adapter to download for base model. Skipping.")
                continue

            adapter_bytes += download_adapter(adapter_id, adapter_source, api_token)

        adapter_memory_size = self.model.adapter_memory_size()
        if adapter_memory_size > 0:
            logger.info(
                f"Downloaded adapter {adapter_id} memory size: {adapter_bytes} bytes "
                f"(reservation: {adapter_memory_size} bytes)"
            )
            adapter_memory_fraction = adapter_bytes / adapter_memory_size
            if adapter_memory_fraction > 1:
                raise ValueError(
                    f"Adapter {adapter_id} is larger than adapter memory reservation: "
                    f"{adapter_bytes} / {adapter_memory_size} bytes"
                )
        else:
            # Assume 0.0 memory fraction if adapter memory size is not set
            logger.info(
                f"Downloaded adapter {adapter_id} memory size: {adapter_bytes} bytes " f"(no reservation limit)"
            )
            adapter_memory_fraction = 0.0

        return generate_pb2.DownloadAdapterResponse(downloaded=True, memory_fraction=adapter_memory_fraction)

    async def LoadAdapter(self, request: generate_pb2.LoadAdapterRequest, context):
        adapter_parameters = request.adapter_parameters
        if is_base_model(adapter_parameters):
            logger.info("No adapter to load for base model. Skipping.")
            return generate_pb2.LoadAdapterResponse(loaded=False)

        try:
            adapter_source = _adapter_source_enum_to_string(request.adapter_source)
            adapter_index = request.adapter_index
            api_token = request.api_token

            if adapter_source == PBASE:
                for i in range(len(adapter_parameters.adapter_ids)):
                    adapter_id = adapter_parameters.adapter_ids[i]
                    adapter_id = map_pbase_model_id_to_s3(adapter_id, api_token)
                    adapter_parameters.adapter_ids[i] = adapter_id
                adapter_source = S3

            self.model.load_adapter(adapter_parameters, adapter_source, adapter_index, api_token)

            return generate_pb2.LoadAdapterResponse(loaded=True)
        except Exception:
            logger.exception("Error when loading adapter")
            raise

    async def OffloadAdapter(self, request: generate_pb2.OffloadAdapterRequest, context):
        adapter_parameters = request.adapter_parameters
        if is_base_model(adapter_parameters):
            logger.info("No adapter to offload for base model. Skipping.")
            return generate_pb2.OffloadAdapterResponse(offloaded=False)

        try:
            adapter_idx = request.adapter_index
            adapter_source = _adapter_source_enum_to_string(request.adapter_source)
            adapter_index = request.adapter_index
            self.model.offload_adapter(adapter_idx, adapter_source, adapter_index)

            # Ensure there is enough memory for the next adapter
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.model.device)

            return generate_pb2.OffloadAdapterResponse(offloaded=True)
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
    speculative_tokens: int,
):
    async def serve_inner(
        model_id: str,
        adapter_id: str,
        revision: Optional[str],
        sharded: bool,
        quantize: Optional[str],
        compile: bool,
        dtype: Optional[str],
        trust_remote_code: bool,
        speculative_tokens: int,
    ):
        unix_socket_template = "unix://{}-{}"
        if sharded:
            server_urls = [unix_socket_template.format(uds_path, rank) for rank in range(int(os.environ["WORLD_SIZE"]))]
            local_url = server_urls[int(os.environ["RANK"])]
        else:
            local_url = unix_socket_template.format(uds_path, 0)
            server_urls = [local_url]

        try:
            model = get_model(
                model_id,
                adapter_id,
                revision,
                sharded,
                quantize,
                compile,
                dtype,
                trust_remote_code,
                source,
                adapter_source,
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

        # set speculative decoding tokens
        speculative_tokens = max(model.max_speculative_tokens, speculative_tokens)
        if speculative_tokens > 0:
            set_speculative_tokens(speculative_tokens)

        server = aio.server(
            interceptors=[
                ExceptionInterceptor(),
                UDSOpenTelemetryAioServerInterceptor(),
            ]
        )
        generate_pb2_grpc.add_LoraxServiceServicer_to_server(LoraxService(model, Cache(), server_urls), server)
        SERVICE_NAMES = (
            generate_pb2.DESCRIPTOR.services_by_name["LoraxService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        server.add_insecure_port(local_url)

        await server.start()

        # Log SGMV kernel status
        if has_sgmv():
            logger.info("SGMV kernel is enabled, multi-LoRA inference will be fast!")
        else:
            logger.info("SGMV kernel is disabled, multi-LoRA inference may be slow")

        logger.info("Server started at {}".format(local_url))

        try:
            await server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Signal received. Shutting down")
            await server.stop(0)

    asyncio.run(
        serve_inner(
            model_id,
            adapter_id,
            revision,
            sharded,
            quantize,
            compile,
            dtype,
            trust_remote_code,
            speculative_tokens,
        )
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
