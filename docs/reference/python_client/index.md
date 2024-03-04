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
