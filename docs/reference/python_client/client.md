# Table of Contents

* [lorax.client](#lorax.client)
  * [Client](#lorax.client.Client)
    * [\_\_init\_\_](#lorax.client.Client.__init__)
    * [generate](#lorax.client.Client.generate)
    * [generate\_stream](#lorax.client.Client.generate_stream)
  * [AsyncClient](#lorax.client.AsyncClient)
    * [\_\_init\_\_](#lorax.client.AsyncClient.__init__)
    * [generate](#lorax.client.AsyncClient.generate)
    * [generate\_stream](#lorax.client.AsyncClient.generate_stream)

<a id="lorax.client"></a>

# lorax.client

<a id="lorax.client.Client"></a>

## Client Objects

```python
class Client()
```

Client to make calls to a LoRAX instance

**Example**:

  
```python
from lorax import Client

client = Client("http://127.0.0.1:8080")
client.generate("Why is the sky blue?", adapter_id="some/adapter").generated_text
 ' Rayleigh scattering'

result = ""
for response in client.generate_stream("Why is the sky blue?", adapter_id="some/adapter"):
    if not response.token.special:
        result += response.token.text
result
' Rayleigh scattering'
```

<a id="lorax.client.Client.__init__"></a>

#### \_\_init\_\_

```python
def __init__(base_url: str,
             headers: Optional[Dict[str, str]] = None,
             cookies: Optional[Dict[str, str]] = None,
             timeout: int = 60)
```

**Arguments**:

  - base_url (`str`):
  LoRAX instance base url
  - headers (`Optional[Dict[str, str]]`):
  Additional headers
  - cookies (`Optional[Dict[str, str]]`):
  Cookies to include in the requests
  - timeout (`int`):
  Timeout in seconds

<a id="lorax.client.Client.generate"></a>

#### generate

```python
def generate(prompt: str,
             adapter_id: Optional[str] = None,
             adapter_source: Optional[str] = None,
             merged_adapters: Optional[MergedAdapters] = None,
             api_token: Optional[str] = None,
             do_sample: bool = False,
             max_new_tokens: int = 20,
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
             response_format: Optional[Union[Dict[str, Any],
                                             ResponseFormat]] = None,
             decoder_input_details: bool = False,
             details: bool = True) -> Response
```

Given a prompt, generate the following text

**Arguments**:

  - prompt (`str`):
  Input text
  - adapter_id (`Optional[str]`):
  Adapter ID to apply to the base model for the request
  - adapter_source (`Optional[str]`):
  Source of the adapter ("hub", "local", "s3", "pbase")
  - merged_adapters (`Optional[MergedAdapters]`):
  Merged adapters to apply to the base model for the request
  - api_token (`Optional[str]`):
  API token for accessing private adapters
  - do_sample (`bool`):
  Activate logits sampling
  - max_new_tokens (`int`):
  Maximum number of generated tokens
  - best_of (`int`):
  Generate best_of sequences and return the one if the highest token logprobs
  - repetition_penalty (`float`):
  The parameter for repetition penalty. 1.0 means no penalty. See [this
  paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
  return_full_text (`bool`):
  Whether to prepend the prompt to the generated text
  - seed (`int`):
  Random sampling seed
  - stop_sequences (`List[str]`):
  Stop generating tokens if a member of `stop_sequences` is generated
  - temperature (`float`):
  The value used to module the logits distribution.
  - top_k (`int`):
  The number of highest probability vocabulary tokens to keep for top-k-filtering.
  - top_p (`float`):
  If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
  higher are kept for generation.
  - truncate (`int`):
  Truncate inputs tokens to the given size
  - typical_p (`float`):
  Typical Decoding mass
  See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
  - watermark (`bool`):
  Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
  - response_format (`Optional[Union[Dict[str, Any], ResponseFormat]]`):
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
  - decoder_input_details (`bool`):
  Return the decoder input token logprobs and ids
  - details (`bool`):
  Return the token logprobs and ids for generated tokens
  

**Returns**:

- `Response` - generated response

<a id="lorax.client.Client.generate_stream"></a>

#### generate\_stream

```python
def generate_stream(prompt: str,
                    adapter_id: Optional[str] = None,
                    adapter_source: Optional[str] = None,
                    merged_adapters: Optional[MergedAdapters] = None,
                    api_token: Optional[str] = None,
                    do_sample: bool = False,
                    max_new_tokens: int = 20,
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
                    response_format: Optional[Union[Dict[str, Any],
                                                    ResponseFormat]] = None,
                    details: bool = True) -> Iterator[StreamResponse]
```

Given a prompt, generate the following stream of tokens

**Arguments**:

  - prompt (`str`):
  Input text
  - adapter_id (`Optional[str]`):
  Adapter ID to apply to the base model for the request
  - adapter_source (`Optional[str]`):
  Source of the adapter (hub, local, s3)
  - merged_adapters (`Optional[MergedAdapters]`):
  Merged adapters to apply to the base model for the request
  - api_token (`Optional[str]`):
  API token for accessing private adapters
  - do_sample (`bool`):
  Activate logits sampling
  - max_new_tokens (`int`):
  Maximum number of generated tokens
  - repetition_penalty (`float`):
  The parameter for repetition penalty. 1.0 means no penalty. See [this
  paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
  return_full_text (`bool`):
  Whether to prepend the prompt to the generated text
  - seed (`int`):
  Random sampling seed
  - stop_sequences (`List[str]`):
  Stop generating tokens if a member of `stop_sequences` is generated
  - temperature (`float`):
  The value used to module the logits distribution.
  - top_k (`int`):
  The number of highest probability vocabulary tokens to keep for top-k-filtering.
  - top_p (`float`):
  If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
  higher are kept for generation.
  - truncate (`int`):
  Truncate inputs tokens to the given size
  - typical_p (`float`):
  Typical Decoding mass
  See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
  - watermark (`bool`):
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
  - details (`bool`):
  Return the token logprobs and ids for generated tokens
  

**Returns**:

- `Iterator[StreamResponse]` - stream of generated tokens

<a id="lorax.client.AsyncClient"></a>

## AsyncClient Objects

```python
class AsyncClient()
```

Asynchronous Client to make calls to a LoRAX instance

**Example**:

  
```python
from lorax import AsyncClient

client = AsyncClient("https://api-inference.huggingface.co/models/bigscience/bloomz")
response = await client.generate("Why is the sky blue?", adapter_id="some/adapter")
response.generated_text
' Rayleigh scattering'

result = ""
async for response in client.generate_stream("Why is the sky blue?", adapter_id="some/adapter"):
    if not response.token.special:
        result += response.token.text
result
' Rayleigh scattering'
```

<a id="lorax.client.AsyncClient.__init__"></a>

#### \_\_init\_\_

```python
def __init__(base_url: str,
             headers: Optional[Dict[str, str]] = None,
             cookies: Optional[Dict[str, str]] = None,
             timeout: int = 60)
```

**Arguments**:

  - base_url (`str`):
  LoRAX instance base url
  - headers (`Optional[Dict[str, str]]`):
  Additional headers
  - cookies (`Optional[Dict[str, str]]`):
  Cookies to include in the requests
  - timeout (`int`):
  Timeout in seconds

<a id="lorax.client.AsyncClient.generate"></a>

#### generate

```python
async def generate(prompt: str,
                   adapter_id: Optional[str] = None,
                   adapter_source: Optional[str] = None,
                   merged_adapters: Optional[MergedAdapters] = None,
                   api_token: Optional[str] = None,
                   do_sample: bool = False,
                   max_new_tokens: int = 20,
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
                   response_format: Optional[Union[Dict[str, Any],
                                                   ResponseFormat]] = None,
                   decoder_input_details: bool = False,
                   details: bool = True) -> Response
```

Given a prompt, generate the following text asynchronously

**Arguments**:

  - prompt (`str`):
  Input text
  - adapter_id (`Optional[str]`):
  Adapter ID to apply to the base model for the request
  - adapter_source (`Optional[str]`):
  Source of the adapter (hub, local, s3)
  - merged_adapters (`Optional[MergedAdapters]`):
  Merged adapters to apply to the base model for the request
  - api_token (`Optional[str]`):
  API token for accessing private adapters
  - do_sample (`bool`):
  Activate logits sampling
  - max_new_tokens (`int`):
  Maximum number of generated tokens
  - best_of (`int`):
  Generate best_of sequences and return the one if the highest token logprobs
  repetition_penalty (`float`):
  The parameter for repetition penalty. 1.0 means no penalty. See [this
  paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
  return_full_text (`bool`):
  Whether to prepend the prompt to the generated text
  - seed (`int`):
  Random sampling seed
  - stop_sequences (`List[str]`):
  Stop generating tokens if a member of `stop_sequences` is generated
  - temperature (`float`):
  The value used to module the logits distribution.
  - top_k (`int`):
  The number of highest probability vocabulary tokens to keep for top-k-filtering.
  - top_p (`float`):
  If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
  higher are kept for generation.
  - truncate (`int`):
  Truncate inputs tokens to the given size
  - typical_p (`float`):
  Typical Decoding mass
  See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
  - watermark (`bool`):
  Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
  - response_format (`Optional[Union[Dict[str, Any], ResponseFormat]]`):
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
  - decoder_input_details (`bool`):
  Return the decoder input token logprobs and ids
  - details (`bool`):
  Return the token logprobs and ids for generated tokens
  

**Returns**:

- `Response` - generated response

<a id="lorax.client.AsyncClient.generate_stream"></a>

#### generate\_stream

```python
async def generate_stream(
        prompt: str,
        adapter_id: Optional[str] = None,
        adapter_source: Optional[str] = None,
        merged_adapters: Optional[MergedAdapters] = None,
        api_token: Optional[str] = None,
        do_sample: bool = False,
        max_new_tokens: int = 20,
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
        response_format: Optional[Union[Dict[str, Any],
                                        ResponseFormat]] = None,
        details: bool = True) -> AsyncIterator[StreamResponse]
```

Given a prompt, generate the following stream of tokens asynchronously

**Arguments**:

  - prompt (`str`):
  Input text
  - adapter_id (`Optional[str]`):
  Adapter ID to apply to the base model for the request
  - adapter_source (`Optional[str]`):
  Source of the adapter (hub, local, s3)
  - merged_adapters (`Optional[MergedAdapters]`):
  Merged adapters to apply to the base model for the request
  - api_token (`Optional[str]`):
  API token for accessing private adapters
  - do_sample (`bool`):
  Activate logits sampling
  - max_new_tokens (`int`):
  Maximum number of generated tokens
  - repetition_penalty (`float`):
  The parameter for repetition penalty. 1.0 means no penalty. See [this
  paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
  return_full_text (`bool`):
  Whether to prepend the prompt to the generated text
  - seed (`int`):
  Random sampling seed
  - stop_sequences (`List[str]`):
  Stop generating tokens if a member of `stop_sequences` is generated
  - temperature (`float`):
  The value used to module the logits distribution.
  - top_k (`int`):
  The number of highest probability vocabulary tokens to keep for top-k-filtering.
  - top_p (`float`):
  If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
  higher are kept for generation.
  - truncate (`int`):
  Truncate inputs tokens to the given size
  - typical_p (`float`):
  Typical Decoding mass
  See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
  - watermark (`bool`):
  Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
  - response_format (`Optional[Union[Dict[str, Any], ResponseFormat]]`):
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
  - details (`bool`):
  Return the token logprobs and ids for generated tokens
  

**Returns**:

- `AsyncIterator[StreamResponse]` - stream of generated tokens

