import asyncio
import concurrent.futures
import os
import time
from pathlib import Path
from typing import List, Optional

import torch
from grpc import aio
from grpc_reflection.v1alpha import reflection
from loguru import logger
from tqdm import tqdm

from lorax_server.cache import Cache
from lorax_server.interceptor import ExceptionInterceptor
from lorax_server.models import Model, get_model
from lorax_server.pb import generate_pb2, generate_pb2_grpc
from lorax_server.tracing import UDSOpenTelemetryAioServerInterceptor
from lorax_server.utils import PBASE, S3, map_pbase_model_id_to_s3
from lorax_server.utils.adapter import (
    adapter_source_enum_to_string,
    download_adapter,
    enum_string_to_adapter_source,
    is_base_model,
)
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
            self.model.processor,
            self.model.model.config,
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
            self.model.processor,
            self.model.model.config,
            self.model.dtype,
            self.model.device,
        )

        generations, next_batch = self.model.generate_token(batch)
        self.cache.set(next_batch)

        return generate_pb2.PrefillResponse(
            generations=[generation.to_pb() for generation in generations],
            batch=next_batch.to_pb() if next_batch else None,
        )

    async def Classify(self, request: generate_pb2.ClassifyRequest, context):
        if not self.model.supports_classification:
            raise ValueError("Model does not support classification")

        batch = self.model.batch_type.from_pb(
            request.batch,
            self.model.tokenizer,
            self.model.tokenizers,
            self.model.processor,
            self.model.model.config,
            self.model.dtype,
            self.model.device,
        )
        predicated_token_class, confidence_scores = self.model.classify(batch)
        ner_results = self.model.batch_type.to_pb_classify(batch, predicated_token_class, confidence_scores)
        return ner_results

    async def Embed(self, request: generate_pb2.EmbedRequest, context):
        if not self.model.supports_embeddings:
            raise ValueError("Model does not support embeddings")

        batch = self.model.batch_type.from_pb_embed(
            request.batch,
            self.model.tokenizer,
            self.model.tokenizers,
            self.model.processor,
            self.model.model.config,
            self.model.dtype,
            self.model.device,
        )
        embeddings = self.model.embed(batch)
        embeddings_pb = self.model.batch_type.to_pb_embed(batch, embeddings)
        return embeddings_pb

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
        if (
            len(request.adapter_parameters.adapter_ids) == 1
            and request.adapter_parameters.adapter_ids[0] in self.model.preloaded_adapter_memory_fractions
        ):
            logger.info("Adapter is already preloaded. Skipping.")
            return generate_pb2.DownloadAdapterResponse(
                downloaded=True,
                memory_fraction=self.model.preloaded_adapter_memory_fractions[
                    request.adapter_parameters.adapter_ids[0]
                ],
            )

        return download_adapter(request, self.model)

    async def LoadAdapter(self, request: generate_pb2.LoadAdapterRequest, context):
        adapter_parameters = request.adapter_parameters
        if is_base_model(adapter_parameters):
            logger.info("No adapter to load for base model. Skipping.")
            return generate_pb2.LoadAdapterResponse(loaded=False)

        if request.adapter_index in self.model.loaded_adapters:
            logger.info(f"Adapter {request.adapter_index} is already loaded. Skipping.")
            return generate_pb2.LoadAdapterResponse(loaded=True)

        try:
            adapter_source = adapter_source_enum_to_string(request.adapter_source)
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
            adapter_source = adapter_source_enum_to_string(request.adapter_source)
            adapter_index = request.adapter_index

            offloaded = self.model.offload_adapter(adapter_idx, adapter_source, adapter_index)
            if offloaded:
                # Ensure there is enough memory for the next adapter
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self.model.device)

            return generate_pb2.OffloadAdapterResponse(offloaded=offloaded)
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
    preloaded_adapter_ids: List[str],
    preloaded_adapter_source: str,
    embedding_dim: Optional[int] = None,
):
    async def serve_inner(
        model_id: str,
        adapter_id: str,
        adapter_source: str,
        revision: Optional[str],
        sharded: bool,
        quantize: Optional[str],
        compile: bool,
        dtype: Optional[str],
        trust_remote_code: bool,
        speculative_tokens: int,
        preloaded_adapter_ids: List[str],
        preloaded_adapter_source: str,
        embedding_dim: Optional[int] = None,
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
                embedding_dim,
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

        if preloaded_adapter_ids:
            logger.info(f"Preloading {len(preloaded_adapter_ids)} adapters")

            _adapter_source = enum_string_to_adapter_source(preloaded_adapter_source)
            adapter_preload_api_token = None
            if _adapter_source == generate_pb2.AdapterSource.PBASE:
                # Derive the predibase token from an env variable if we are using predibase adapters.
                adapter_preload_api_token = os.getenv("PREDIBASE_API_TOKEN")

            preloaded_adapters = [
                generate_pb2.PreloadedAdapter(
                    adapter_parameters=generate_pb2.AdapterParameters(adapter_ids=[adapter_id]),
                    adapter_source=_adapter_source,
                    adapter_index=i + 1,
                )
                for i, adapter_id in enumerate(preloaded_adapter_ids)
            ]

            download_requests = [
                generate_pb2.DownloadAdapterRequest(
                    adapter_parameters=adapter_info.adapter_parameters,
                    adapter_source=adapter_info.adapter_source,
                    api_token=adapter_preload_api_token,
                )
                for adapter_info in preloaded_adapters
            ]
            models = [model] * len(download_requests)

            # Download adapters
            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                download_responses = list(
                    tqdm(executor.map(download_adapter, download_requests, models), total=len(download_requests))
                )
            logger.info(f"Downloaded {len(download_requests)} adapters in {time.time() - t0:.2f}s")

            if not all(download_responses):
                raise RuntimeError("Failed to download all adapters")

            def load_adapter(adapter_info: generate_pb2.PreloadedAdapter) -> bool:
                _adapter_source = adapter_source_enum_to_string(adapter_info.adapter_source)
                _adapter_id = adapter_info.adapter_parameters.adapter_ids[0]
                if _adapter_source == PBASE:
                    _adapter_id = map_pbase_model_id_to_s3(_adapter_id, api_token=adapter_preload_api_token)
                    _adapter_source = S3

                model.load_adapter(
                    generate_pb2.AdapterParameters(adapter_ids=[_adapter_id]),
                    _adapter_source,
                    adapter_index=adapter_info.adapter_index,
                    api_token=None,
                    dynamic=True,
                )
                return True

            # Load adapters
            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                responses = list(tqdm(executor.map(load_adapter, preloaded_adapters), total=len(preloaded_adapters)))

            if not all(responses):
                raise RuntimeError("Failed to preload all adapters")

            logger.info(f"Preloaded {len(preloaded_adapters)} adapters in {time.time() - t0:.2f}s")

            adapter_memory_fractions = [r.memory_fraction for r in download_responses]
            model.register_preloaded_adapters(preloaded_adapters, adapter_memory_fractions)

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
            adapter_source,
            revision,
            sharded,
            quantize,
            compile,
            dtype,
            trust_remote_code,
            speculative_tokens,
            preloaded_adapter_ids,
            preloaded_adapter_source,
            embedding_dim,
        )
    )
