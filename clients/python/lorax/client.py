import json
import logging
import requests
from requests.adapters import HTTPAdapter, Retry

from aiohttp import ClientSession, ClientTimeout
from pydantic import ValidationError
from typing import Any, Dict, Optional, List, AsyncIterator, Iterator, Union

from lorax.types import (
    BatchRequest,
    StreamResponse,
    Response,
    Request,
    Parameters,
    MergedAdapters,
    ResponseFormat,
    EmbedResponse,
    ClassifyResponse
)
from lorax.errors import parse_error
import os 

LORAX_DEBUG_MODE = os.getenv("LORAX_DEBUG_MODE", None) is not None
if LORAX_DEBUG_MODE:
    # https://stackoverflow.com/a/16630836/1869739
    # These two lines enable debugging at httplib level (requests->urllib3->http.client)
    # You will see the REQUEST, including HEADERS and DATA, and RESPONSE with HEADERS but without DATA.
    # The only thing missing will be the response.body which is not logged.
    import http.client as http_client
    http_client.HTTPConnection.debuglevel = 1

    # You must initialize logging, otherwise you'll not see debug output.
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True


class Client:
    """Client to make calls to a LoRAX instance

     Example:

     ```python
     >>> from lorax import Client

     >>> client = Client("http://127.0.0.1:8080")
     >>> client.generate("Why is the sky blue?", adapter_id="some/adapter").generated_text
     ' Rayleigh scattering'

     >>> result = ""
     >>> for response in client.generate_stream("Why is the sky blue?", adapter_id="some/adapter"):
     >>>     if not response.token.special:
     >>>         result += response.token.text
     >>> result
    ' Rayleigh scattering'
     ```
    """

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 60,
        session: Optional[requests.Session] = None,
        max_session_retries: int = 2,
    ):
        """
        Args:
            base_url (`str`):
                LoRAX instance base url
            headers (`Optional[Dict[str, str]]`):
                Additional headers
            cookies (`Optional[Dict[str, str]]`):
                Cookies to include in the requests
            timeout (`int`):
                Timeout in seconds
            session (`Optional[requests.Session]`):
                HTTP requests session object to reuse
            max_session_retries (`int`):
                Maximum retries for session refreshing on errors
        """
        self.base_url = base_url
        self.embed_endpoint = f"{base_url}/embed"
        self.classify_endpoint = f"{base_url}/classify"
        self.classify_batch_endpoint = f"{base_url}/classify_batch"
        self.headers = headers
        self.cookies = cookies
        self.timeout = timeout
        self.session = session
        self.max_session_retries = max_session_retries

    def _create_session(self):
        """
        Create a new session object to make HTTP calls.
        """
        self.session = requests.Session()

        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[104, 429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def _post(self, json: dict, stream: bool = False) -> requests.Response:
        """
        Given inputs, make an HTTP POST call

        Args:
            json (`dict`):
                HTTP POST request JSON body

            stream (`bool`):
                Whether to stream the HTTP response or not
        
        Returns: 
            requests.Response: HTTP response object
        """
        # Instantiate session if currently None
        if not self.session:
            self._create_session()

        # Retry if the session is stale and hits a ConnectionError
        current_retry_attempt = 0
        
        # Make the HTTP POST request
        while True:
            try:
                resp = self.session.post(
                    self.base_url,
                    json=json,
                    headers=self.headers,
                    cookies=self.cookies,
                    timeout=self.timeout,
                    stream=stream
                )
                return resp
            except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as e:
                # Refresh session if there is a ConnectionError
                self.session = None
                self._create_session()
                
                # Raise error if retries have been exhausted
                if current_retry_attempt >= self.max_session_retries:
                    raise e
                
                current_retry_attempt += 1
            except Exception as e:
                # Raise any other exception
                raise e

    def generate(
        self,
        prompt: str,
        adapter_id: Optional[str] = None,
        adapter_source: Optional[str] = None,
        merged_adapters: Optional[MergedAdapters] = None,
        api_token: Optional[str] = None,
        do_sample: bool = False,
        max_new_tokens: Optional[int] = None,
        ignore_eos_token: bool = False,
        best_of: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        response_format: Optional[Union[Dict[str, Any], ResponseFormat]] = None,
        decoder_input_details: bool = False,
        return_k_alternatives: Optional[int] = None,
        details: bool = True,
    ) -> Response:
        """
        Given a prompt, generate the following text

        Args:
            prompt (`str`):
                Input text
            adapter_id (`Optional[str]`):
                Adapter ID to apply to the base model for the request
            adapter_source (`Optional[str]`):
                Source of the adapter ("hub", "local", "s3", "pbase")
            merged_adapters (`Optional[MergedAdapters]`):
                Merged adapters to apply to the base model for the request
            api_token (`Optional[str]`):
                API token for accessing private adapters
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`Optional[int]`):
                Maximum number of generated tokens
            ignore_eos_token (`bool`):
                Whether to ignore EOS tokens during generation
            best_of (`int`):
                Generate best_of sequences and return the one if the highest token logprobs
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            response_format (`Optional[Union[Dict[str, Any], ResponseFormat]]`):
                Optional specification of a format to impose upon the generated text, e.g.,:
                ```
                {
                    "type": "json_object",
                    "schema": {
                        "type": "string",
                        "title": "response"
                    }
                }
                ```
            decoder_input_details (`bool`):
                Return the decoder input token logprobs and ids
            return_k_alternatives (`int`):
                The number of highest probability vocabulary tokens to return as alternative tokens in the generation result
            details (`bool`):
                Return the token logprobs and ids for generated tokens

        Returns:
            Response: generated response
        """
        # Validate parameters
        parameters = Parameters(
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            merged_adapters=merged_adapters,
            api_token=api_token,
            best_of=best_of,
            details=details,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            ignore_eos_token=ignore_eos_token,
            repetition_penalty=repetition_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop_sequences if stop_sequences is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=truncate,
            typical_p=typical_p,
            watermark=watermark,
            response_format=response_format,
            decoder_input_details=decoder_input_details,
            return_k_alternatives=return_k_alternatives
        )

        # Instantiate the request object
        request = Request(inputs=prompt, stream=False, parameters=parameters)

        resp = self._post(
            json=request.dict(by_alias=True),
        )

        try:
            payload = resp.json()
        except requests.JSONDecodeError as e:
            # If the status code is success-like, reset it to 500 since the server is sending an invalid response.
            if 200 <= resp.status_code < 400:
                resp.status_code = 500

            payload = {"message": e.msg}

        if resp.status_code != 200:
            raise parse_error(resp.status_code, payload, resp.headers if LORAX_DEBUG_MODE else None)

        return Response(**payload[0])

    def generate_stream(
        self,
        prompt: str,
        adapter_id: Optional[str] = None,
        adapter_source: Optional[str] = None,
        merged_adapters: Optional[MergedAdapters] = None,
        api_token: Optional[str] = None,
        do_sample: bool = False,
        max_new_tokens: Optional[int] = None,
        ignore_eos_token: bool = False,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        response_format: Optional[Union[Dict[str, Any], ResponseFormat]] = None,
        details: bool = True,
    ) -> Iterator[StreamResponse]:
        """
        Given a prompt, generate the following stream of tokens

        Args:
            prompt (`str`):
                Input text
            adapter_id (`Optional[str]`):
                Adapter ID to apply to the base model for the request
            adapter_source (`Optional[str]`):
                Source of the adapter ("hub", "local", "s3", "pbase")
            merged_adapters (`Optional[MergedAdapters]`):
                Merged adapters to apply to the base model for the request
            api_token (`Optional[str]`):
                API token for accessing private adapters
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`Optional[int]`):
                Maximum number of generated tokens
            ignore_eos_token (`bool`):
                Whether to ignore EOS tokens during generation
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            response_format (`Optional[Union[Dict[str, Any], ResponseFormat]]`):
                Optional specification of a format to impose upon the generated text, e.g.,:
                ```
                {
                    "type": "json_object",
                    "schema": {
                        "type": "string",
                        "title": "response"
                    }
                }
                ```
            details (`bool`):
                Return the token logprobs and ids for generated tokens

        Returns:
            Iterator[StreamResponse]: stream of generated tokens
        """
        # Validate parameters
        parameters = Parameters(
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            merged_adapters=merged_adapters,
            api_token=api_token,
            best_of=None,
            details=details,
            decoder_input_details=False,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            ignore_eos_token=ignore_eos_token,
            repetition_penalty=repetition_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop_sequences if stop_sequences is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=truncate,
            typical_p=typical_p,
            watermark=watermark,
            response_format=response_format,
        )
        # Instantiate the request and session objects
        request = Request(inputs=prompt, stream=True, parameters=parameters)

        resp = self._post(
            json=request.dict(by_alias=True),
            stream=True,
        )

        if resp.status_code != 200:
            raise parse_error(resp.status_code, resp.json(), resp.headers if LORAX_DEBUG_MODE else None)

        # Parse ServerSentEvents
        for byte_payload in resp.iter_lines():
            # Skip line
            if byte_payload == b"\n":
                continue

            payload = byte_payload.decode("utf-8")

            # Event data
            if payload.startswith("data:"):
                # Decode payload
                json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
                # Parse payload
                try:
                    response = StreamResponse(**json_payload)
                except ValidationError:
                    # If we failed to parse the payload, then it is an error payload
                    raise parse_error(resp.status_code, json_payload, resp.headers if LORAX_DEBUG_MODE else None)
                yield response

    
    def embed(self, inputs: str) -> EmbedResponse:
        """
        Given inputs, embed the text using the model

        Args:
            inputs (`str`):
                Input text
        
        Returns: 
            Embeddings: computed embeddings
        """
        request = Request(inputs=inputs)

        resp = requests.post(
            self.embed_endpoint,
            json=request.dict(by_alias=True),
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
        )

        payload = resp.json()
        if resp.status_code != 200:
            raise parse_error(resp.status_code, resp.json(), resp.headers if LORAX_DEBUG_MODE else None)
        
        return EmbedResponse(**payload)


    def classify(self, inputs: str) -> ClassifyResponse:
        """
        Given inputs, run token classification on the text using the model

        Args:
            inputs (`str`):
                Input text
        
        Returns: 
            Entities: Entities found in the input text
        """
        request = Request(inputs=inputs)

        resp = requests.post(
            self.classify_endpoint,
            json=request.dict(by_alias=True),
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
        )

        payload = resp.json()
        if resp.status_code != 200:
            raise parse_error(resp.status_code, resp.json(), resp.headers if LORAX_DEBUG_MODE else None)
        
        return ClassifyResponse(entities=payload)
    
    def classify_batch(self, inputs: List[str]) -> List[ClassifyResponse]:
        """
        Given a list of inputs, run token classification on the text using the model

        Args:
            inputs (`List[str]`):
                List of input texts
        
        Returns: 
            List[Entities]: Entities found in the input text
        """
        request = BatchRequest(inputs=inputs)

        resp = requests.post(
            self.classify_batch_endpoint,
            json=request.dict(by_alias=True),
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
        )

        payload = resp.json()
        if resp.status_code != 200:
            raise parse_error(resp.status_code, resp.json(), resp.headers if LORAX_DEBUG_MODE else None)
        
        return [ClassifyResponse(entities=e) for e in payload]


class AsyncClient:
    """Asynchronous Client to make calls to a LoRAX instance

     Example:

     ```python
     >>> from lorax import AsyncClient

     >>> client = AsyncClient("https://api-inference.huggingface.co/models/bigscience/bloomz")
     >>> response = await client.generate("Why is the sky blue?", adapter_id="some/adapter")
     >>> response.generated_text
     ' Rayleigh scattering'

     >>> result = ""
     >>> async for response in client.generate_stream("Why is the sky blue?", adapter_id="some/adapter"):
     >>>     if not response.token.special:
     >>>         result += response.token.text
     >>> result
    ' Rayleigh scattering'
     ```
    """

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 60,
    ):
        """
        Args:
            base_url (`str`):
                LoRAX instance base url
            headers (`Optional[Dict[str, str]]`):
                Additional headers
            cookies (`Optional[Dict[str, str]]`):
                Cookies to include in the requests
            timeout (`int`):
                Timeout in seconds
        """
        self.base_url = base_url
        self.embed_endpoint = f"{base_url}/embed"
        self.classify_endpoint = f"{base_url}/classify"
        self.headers = headers
        self.cookies = cookies
        self.timeout = ClientTimeout(timeout * 60)

    async def generate(
        self,
        prompt: str,
        adapter_id: Optional[str] = None,
        adapter_source: Optional[str] = None,
        merged_adapters: Optional[MergedAdapters] = None,
        api_token: Optional[str] = None,
        do_sample: bool = False,
        max_new_tokens: Optional[int] = None,
        ignore_eos_token: bool = False,
        best_of: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        response_format: Optional[Union[Dict[str, Any], ResponseFormat]] = None,
        decoder_input_details: bool = False,
        return_k_alternatives: Optional[int] = None,
        details: bool = True,
    ) -> Response:
        """
        Given a prompt, generate the following text asynchronously

        Args:
            prompt (`str`):
                Input text
            adapter_id (`Optional[str]`):
                Adapter ID to apply to the base model for the request
            adapter_source (`Optional[str]`):
                Source of the adapter (hub, local, s3)
            merged_adapters (`Optional[MergedAdapters]`):
                Merged adapters to apply to the base model for the request
            api_token (`Optional[str]`):
                API token for accessing private adapters
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`Optional[int]`):
                Maximum number of generated tokens
            ignore_eos_token (`bool`):
                Whether to ignore EOS tokens during generation
            best_of (`int`):
                Generate best_of sequences and return the one if the highest token logprobs
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            response_format (`Optional[Union[Dict[str, Any], ResponseFormat]]`):
                Optional specification of a format to impose upon the generated text, e.g.,:
                ```
                {
                    "type": "json_object",
                    "schema": {
                        "type": "string",
                        "title": "response"
                    }
                }
                ```
            decoder_input_details (`bool`):
                Return the decoder input token logprobs and ids
            return_k_alternatives (`int`):
                The number of highest probability vocabulary tokens to return as alternative tokens in the generation result
            details (`bool`):
                Return the token logprobs and ids for generated tokens

        Returns:
            Response: generated response
        """
        # Validate parameters
        parameters = Parameters(
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            merged_adapters=merged_adapters,
            api_token=api_token,
            best_of=best_of,
            details=details,
            decoder_input_details=decoder_input_details,
            return_k_alternatives=return_k_alternatives,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            ignore_eos_token=ignore_eos_token,
            repetition_penalty=repetition_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop_sequences if stop_sequences is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=truncate,
            typical_p=typical_p,
            watermark=watermark,
            response_format=response_format,
        )
        request = Request(inputs=prompt, stream=False, parameters=parameters)

        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            async with session.post(self.base_url, json=request.dict(by_alias=True)) as resp:
                payload = await resp.json()

                if resp.status != 200:
                    raise parse_error(resp.status, payload, resp.headers if LORAX_DEBUG_MODE else None)
                return Response(**payload[0])

    async def generate_stream(
        self,
        prompt: str,
        adapter_id: Optional[str] = None,
        adapter_source: Optional[str] = None,
        merged_adapters: Optional[MergedAdapters] = None,
        api_token: Optional[str] = None,
        do_sample: bool = False,
        max_new_tokens: Optional[int] = None,
        ignore_eos_token: bool = False,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        response_format: Optional[Union[Dict[str, Any], ResponseFormat]] = None,
        details: bool = True,
        return_k_alternatives: Optional[int] = None,

    ) -> AsyncIterator[StreamResponse]:
        """
        Given a prompt, generate the following stream of tokens asynchronously

        Args:
            prompt (`str`):
                Input text
            adapter_id (`Optional[str]`):
                Adapter ID to apply to the base model for the request
            adapter_source (`Optional[str]`):
                Source of the adapter (hub, local, s3)
            merged_adapters (`Optional[MergedAdapters]`):
                Merged adapters to apply to the base model for the request
            api_token (`Optional[str]`):
                API token for accessing private adapters
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`Optional[int]`):
                Maximum number of generated tokens
            ignore_eos_token (`bool`):
                Whether to ignore EOS tokens during generation
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            response_format (`Optional[Union[Dict[str, Any], ResponseFormat]]`):
                Optional specification of a format to impose upon the generated text, e.g.,:
                ```
                {
                    "type": "json_object",
                    "schema": {
                        "type": "string",
                        "title": "response"
                    }
                }
                ```
            details (`bool`):
                Return the token logprobs and ids for generated tokens
            return_k_alternatives (`int`):
                The number of highest probability vocabulary tokens to return as alternative tokens in the generation result

        Returns:
            AsyncIterator[StreamResponse]: stream of generated tokens
        """
        # Validate parameters
        parameters = Parameters(
            adapter_id=adapter_id,
            adapter_source=adapter_source,
            merged_adapters=merged_adapters,
            api_token=api_token,
            best_of=None,
            details=details,
            decoder_input_details=False,
            return_k_alternatives=return_k_alternatives,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            ignore_eos_token=ignore_eos_token,
            repetition_penalty=repetition_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop_sequences if stop_sequences is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=truncate,
            typical_p=typical_p,
            watermark=watermark,
            response_format=response_format,
        )
        request = Request(inputs=prompt, stream=True, parameters=parameters)

        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            async with session.post(self.base_url, json=request.dict(by_alias=True)) as resp:

                if resp.status != 200:
                    raise parse_error(resp.status, await resp.json(), resp.headers if LORAX_DEBUG_MODE else None)

                # Parse ServerSentEvents
                async for byte_payload in resp.content:
                    # Skip line
                    if byte_payload == b"\n":
                        continue

                    payload = byte_payload.decode("utf-8")

                    # Event data
                    if payload.startswith("data:"):
                        # Decode payload
                        json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
                        # Parse payload
                        try:
                            response = StreamResponse(**json_payload)
                        except ValidationError:
                            # If we failed to parse the payload, then it is an error payload
                            raise parse_error(resp.status, json_payload, resp.headers if LORAX_DEBUG_MODE else None)
                        yield response
    

    async def embed(self, inputs: str) -> EmbedResponse:
        """
        Given inputs, embed the text using the model

        Args:
            inputs (`str`):
                Input text
        
        Returns: 
            Embeddings: computed embeddings
        """
        request = Request(inputs=inputs)
        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            async with session.post(self.embed_endpoint, json=request.dict(by_alias=True)) as resp:
                payload = await resp.json()

                if resp.status != 200:
                    raise parse_error(resp.status, payload, resp.headers if LORAX_DEBUG_MODE else None)
                return EmbedResponse(**payload)


    async def classify(self, inputs: str) -> ClassifyResponse:
        """
        Given inputs, run token classification on the text using the model

        Args:
            inputs (`str`):
                Input text
        
        Returns: 
            Entities: Entities found in the input text
        """
        request = Request(inputs=inputs)
        async with ClientSession(
            headers=self.headers, cookies=self.cookies, timeout=self.timeout
        ) as session:
            async with session.post(self.classify_endpoint, json=request.dict(by_alias=True)) as resp:
                payload = await resp.json()

                if resp.status != 200:
                    raise parse_error(resp.status, payload, resp.headers if LORAX_DEBUG_MODE else None)
                return ClassifyResponse(**payload)
