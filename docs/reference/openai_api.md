LoRAX supports [OpenAI Chat Completions v1](https://platform.openai.com/docs/api-reference/completions/create) compatible endpoints that serve as a drop-in replacement for the OpenAI SDK. It supports multi-turn
chat conversations while retaining support for dynamic adapter loading.

## Chat Completions v1

Using the existing OpenAI Python SDK, replace the `base_url` with your LoRAX endpoint with `/v1` appended. The `api_key` can be anything, as it is unused.

The `model` parameter can be set to the empty string `""` to use the base model, or any adapter ID on the HuggingFace hub.

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8080/v1",
)

resp = client.chat.completions.create(
    model="alignment-handbook/zephyr-7b-dpo-lora",
    messages=[
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    ],
    max_tokens=100,
)
print("Response:", resp.choices[0].message.content)
```

### Streaming

The streaming API is supported with the `stream=True` parameter:

```python
messages = client.chat.completions.create(
    model="alignment-handbook/zephyr-7b-dpo-lora",
    messages=[
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    ],
    max_tokens=100,
    stream=True,
)

for message in messages:
    print(message)
```

### REST API

The REST API can be used directly in addition to the Python SDK:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "alignment-handbook/zephyr-7b-dpo-lora",
  "messages": [
  {
      "role": "system",
      "content": "You are a friendly chatbot who always responds in the style of a pirate"
  },
  {
      "role": "user",
      "content": "How many helicopters can a human eat in one sitting?"
  }
  ],
  "max_tokens": 100
}'
```

### Chat Templates

Multi-turn chat conversations are supported through [HuggingFace chat templates](https://huggingface.co/docs/transformers/chat_templating).

If the adapter selected with the `model` parameter has its own tokenizer and chat template, LoRAX will apply the adapter's chat template
to the request during inference. If, however, the adapter does not have its own chat template, LoRAX will fallback to using the base model
chat template. If this does not exist, an error will be raised, as chat templates are required for multi-turn conversations.

### Structured Output (JSON)

See [here](../guides/structured_output.md#example-openai-compatible-api) for an example.

## Completions v1

The legacy completions v1 API can be used as well. This is useful in cases where the model does not have a chat template or you do not wish to
interact with the model in a multi-turn conversation.

Note, however, that you will need to provide any template boilerplate as part of the `prompt` as unlike the `v1/chat/completions` API, it will not
be inserted automatically.

!!! note

    Structured Output (JSON mode) is not supported in the legacy completions API. Please use the Chat Completions API above instead.

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8080/v1",
)

# synchronous completions
completion = client.completions.create(
    model=adapter_id,
    prompt=prompt,
)
print("Completion result:", completion.choices[0].text)

# streaming completions
completion_stream = client.completions.create(
    model=adapter_id,
    prompt=prompt,
    stream=True,
)

for message in completion_stream:
    print("Completion message:", message)
```

### REST API

```bash
curl http://127.0.0.1:8080/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "",
"prompt": "Instruct: Write a detailed analogy between mathematics and a lighthouse.\nOutput:",
"max_tokens": 100
}'
```
