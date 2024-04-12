# Python Client

LoRAX Python client provides a convenient way of interfacing with a
`lorax` instance running in your environment.

## Install

```shell
pip install lorax-client
```

## Usage

```python
from lorax import Client

endpoint_url = "http://127.0.0.1:8080"

client = Client(endpoint_url)
text = client.generate("Why is the sky blue?", adapter_id="some/adapter").generated_text
print(text)
# ' Rayleigh scattering'

# Token Streaming
text = ""
for response in client.generate_stream("Why is the sky blue?", adapter_id="some/adapter"):
    if not response.token.special:
        text += response.token.text

print(text)
# ' Rayleigh scattering'
```

or with the asynchronous client:

```python
from lorax import AsyncClient

endpoint_url = "http://127.0.0.1:8080"

client = AsyncClient(endpoint_url)
response = await client.generate("Why is the sky blue?", adapter_id="some/adapter")
print(response.generated_text)
# ' Rayleigh scattering'

# Token Streaming
text = ""
async for response in client.generate_stream("Why is the sky blue?", adapter_id="some/adapter"):
    if not response.token.special:
        text += response.token.text

print(text)
# ' Rayleigh scattering'
```

See [API reference](./client.md) for full details.

### Batch Inference

In some cases you may have a list of prompts that you wish to process in bulk ("batch processing").

Rather than process each prompt one at a time, you can take advantage of the `AsyncClient` and LoRAX's native
parallelism to submit your prompts at once and await the results:

```python
import asyncio
import time
from lorax import AsyncClient

# Batch of prompts to submit
prompts = [
    "The quick brown fox",
    "The rain in Spain",
    "What comes up",
]

# Initialize the async client
endpoint_url = "http://127.0.0.1:8080"
async_client = AsyncClient(endpoint_url)

# Submit all prompts and do not block on the response
t0 = time.time()
futures = []
for prompt in prompts:
    resp = async_client.generate(prompt, max_new_tokens=64)
    futures.append(resp)

# Await the completion of all the prompt requests
responses = await asyncio.gather(*futures)

# Print responses
# Responses will always come back in the same order as the original list
for resp in responses:
    print(resp.generated_text)

# Print duration to process all requests in batch
print("duration (s):", time.time() - t0)
```

Output:

```txt
duration (s): 2.9093329906463623
```

Compare this against the duration of submitting one at a time. You should find that for 3 prompts the duration of
async is about 2.5 - 3x faster than serial processing:

```python
from lorax import Client

client = Client(endpoint_url)

t0 = time.time()
responses = []
for prompt in prompts:
    resp = client.generate(prompt, max_new_tokens=64)
    responses.append(resp)

for resp in responses:
    print(resp.generated_text)

print("duration (s):", time.time() - t0)
```

Output:

```txt
duration (s): 8.385080099105835
```

### Predibase Inference Endpoints

The LoRAX client can also be used to connect to [Predibase](https://predibase.com/) managed LoRAX endpoints (including Predibase's [serverless endpoints](https://docs.predibase.com/user-guide/inference/serverless_deployments)).

You need only make the following changes to the above examples:

1. Change the `endpoint_url` to match the endpoint of your Predibase LLM of choice.
2. Provide your Predibase API token in the `headers` provided to the client.

Example:

```python
from lorax import Client

# You can get your Predibase API token by going to Settings > My Profile > Generate API Token
# You can get your Predibase Tenant short code by going to Settings > My Profile > Overview > Tenant ID
endpoint_url = f"https://serving.app.predibase.com/{predibase_tenant_short_code}/deployments/v2/llms/{llm_deployment_name}"
headers = {
    "Authorization": f"Bearer {api_token}"
}

client = Client(endpoint_url, headers=headers)

# same as above from here ...
response = client.generate("Why is the sky blue?", adapter_id=f"{model_repo}/{model_version}")
```

Note that by default Predibase will use its internal model repos as the default `adapter_source`. To use an adapter from Huggingface:

```python
response = client.generate("Why is the sky blue?", adapter_id="some/adapter", adapter_source="hub")
```
