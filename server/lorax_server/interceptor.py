from typing import Any, Callable

import grpc
import torch
from google.rpc import code_pb2, status_pb2
from grpc_interceptor.server import AsyncServerInterceptor
from grpc_status import rpc_status
from loguru import logger


class ExceptionInterceptor(AsyncServerInterceptor):
    """Intercepts and handles exceptions that occur during gRPC method execution."""

    async def intercept(
        self,
        method: Callable,
        request_or_iterator: Any,
        context: grpc.ServicerContext,
        method_name: str,
    ) -> Any:
        """
        Intercepts the gRPC method execution and handles any exceptions that occur.

        Args:
            method (Callable): The gRPC method to be executed.
            request_or_iterator (Any): The request object or iterator.
            context (grpc.ServicerContext): The gRPC servicer context.
            method_name (str): The name of the gRPC method.

        Returns:
            Any: The response of the gRPC method.

        Raises:
            Exception: If an error occurs during the execution of the gRPC method.
        """
        try:
            response = method(request_or_iterator, context)
            return await response
        except Exception as err:
            method_name = method_name.split("/")[-1]
            logger.exception(f"Method {method_name} encountered an error.")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            await context.abort_with_status(
                rpc_status.to_status(status_pb2.Status(code=code_pb2.INTERNAL, message=str(err)))
            )
