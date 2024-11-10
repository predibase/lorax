<p align="center">
  <a href="https://github.com/predibase/lorax">
    <img src="docs/images/lorax_guy.png" alt="LoRAX Logo" style="width:200px;" />
  </a>
</p>

<div align="center">

_LoRAX: Multi-LoRA inference server that scales to 1000s of fine-tuned LLMs_

[![](https://dcbadge.vercel.app/api/server/CBgdrGnZjy?style=flat&theme=discord-inverted)](https://discord.gg/CBgdrGnZjy)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/predibase/lorax/blob/master/LICENSE)
[![Artifact Hub](https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/lorax)](https://artifacthub.io/packages/search?repo=lorax)

</div>

LoRAX (LoRA eXchange) is a framework that allows users to serve thousands of fine-tuned models on a single GPU, dramatically reducing the cost of serving without compromising on throughput or latency.

## 📖 Table of contents

- [📖 Table of contents](#-table-of-contents)
- [🌳 Features](#-features)
- [🏠 Models](#-models)
- [🏃‍♂️ Getting Started](#️-getting-started)
  - [Requirements](#requirements)
  - [Launch LoRAX Server](#launch-lorax-server)
  - [Prompt via REST API](#prompt-via-rest-api)
  - [Prompt via Python Client](#prompt-via-python-client)
  - [Chat via OpenAI API](#chat-via-openai-api)
  - [Next steps](#next-steps)
- [🙇 Acknowledgements](#-acknowledgements)
- [🗺️ Roadmap](#️-roadmap)

## 🌳 Features

- 🚅 **Dynamic Adapter Loading:** include any fine-tuned LoRA adapter from [HuggingFace](https://predibase.github.io/lorax/models/adapters/#huggingface-hub), [Predibase](https://predibase.github.io/lorax/models/adapters/#predibase), or [any filesystem](https://predibase.github.io/lorax/models/adapters/#local) in your request, it will be loaded just-in-time without blocking concurrent requests. [Merge adapters](https://predibase.github.io/lorax/guides/merging_adapters/) per request to instantly create powerful ensembles.
- 🏋️‍♀️ **Heterogeneous Continuous Batching:** packs requests for different adapters together into the same batch, keeping latency and throughput nearly constant with the number of concurrent adapters.
- 🧁 **Adapter Exchange Scheduling:** asynchronously prefetches and offloads adapters between GPU and CPU memory, schedules request batching to optimize the aggregate throughput of the system.
- 👬 **Optimized Inference:**  high throughput and low latency optimizations including tensor parallelism, pre-compiled CUDA kernels ([flash-attention](https://arxiv.org/abs/2307.08691), [paged attention](https://arxiv.org/abs/2309.06180), [SGMV](https://arxiv.org/abs/2310.18547)), quantization, token streaming.
- 🚢  **Ready for Production** prebuilt Docker images, Helm charts for Kubernetes, Prometheus metrics, and distributed tracing with Open Telemetry. OpenAI compatible API supporting multi-turn chat conversations. Private adapters through per-request tenant isolation. [Structured Output](https://predibase.github.io/lorax/guides/structured_output) (JSON mode).
- 🤯 **Free for Commercial Use:** Apache 2.0 License. Enough said 😎.


<p align="center">
  <img src="https://github.com/predibase/lorax/assets/29719151/f88aa16c-66de-45ad-ad40-01a7874ed8a9" />
</p>


## 🏠 Models

Serving a fine-tuned model with LoRAX consists of two components:

- [Base Model](https://predibase.github.io/lorax/models/base_models): pretrained large model shared across all adapters.
- [Adapter](https://predibase.github.io/lorax/models/adapters): task-specific adapter weights dynamically loaded per request.

LoRAX supports a number of Large Language Models as the base model including [Llama](https://huggingface.co/meta-llama) (including [CodeLlama](https://huggingface.co/codellama)), [Mistral](https://huggingface.co/mistralai) (including [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)), and [Qwen](https://huggingface.co/Qwen). See [Supported Architectures](https://predibase.github.io/lorax/models/base_models/#supported-architectures) for a complete list of supported base models. 

Base models can be loaded in fp16 or quantized with `bitsandbytes`, [GPT-Q](https://arxiv.org/abs/2210.17323), or [AWQ](https://arxiv.org/abs/2306.00978).

Supported adapters include LoRA adapters trained using the [PEFT](https://github.com/huggingface/peft) and [Ludwig](https://ludwig.ai/) libraries. Any of the linear layers in the model can be adapted via LoRA and loaded in LoRAX.

## 🏃‍♂️ Getting Started

We recommend starting with our pre-built Docker image to avoid compiling custom CUDA kernels and other dependencies.

### Requirements

The minimum system requirements need to run LoRAX include:

- Nvidia GPU (Ampere generation or above)
- CUDA 11.8 compatible device drivers and above
- Linux OS
- Docker (for this guide)

### Launch LoRAX Server

#### Prerequisites
Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
Then 
 - `sudo systemctl daemon-reload`
 - `sudo systemctl restart docker`

```shell
model=mistralai/Mistral-7B-Instruct-v0.1
volume=$PWD/data

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data \
    ghcr.io/predibase/lorax:main --model-id $model
```

For a full tutorial including token streaming and the Python client, see [Getting Started - Docker](https://predibase.github.io/lorax/getting_started/docker).

### Prompt via REST API

Prompt base LLM:

```shell
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{
        "inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]",
        "parameters": {
            "max_new_tokens": 64
        }
    }' \
    -H 'Content-Type: application/json'
```

Prompt a LoRA adapter:

```shell
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{
        "inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]",
        "parameters": {
            "max_new_tokens": 64,
            "adapter_id": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"
        }
    }' \
    -H 'Content-Type: application/json'
```

See [Reference - REST API](https://predibase.github.io/lorax/reference/rest_api) for full details.

### Prompt via Python Client

Install:

```shell
pip install lorax-client
```

Run:

```python
from lorax import Client

client = Client("http://127.0.0.1:8080")

# Prompt the base LLM
prompt = "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]"
print(client.generate(prompt, max_new_tokens=64).generated_text)

# Prompt a LoRA adapter
adapter_id = "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"
print(client.generate(prompt, max_new_tokens=64, adapter_id=adapter_id).generated_text)
```

See [Reference - Python Client](https://predibase.github.io/lorax/reference/python_client) for full details.

For other ways to run LoRAX, see [Getting Started - Kubernetes](https://predibase.github.io/lorax/getting_started/kubernetes), [Getting Started - SkyPilot](https://predibase.github.io/lorax/getting_started/skypilot), and [Getting Started - Local](https://predibase.github.io/lorax/getting_started/local).

### Chat via OpenAI API

LoRAX supports multi-turn chat conversations combined with dynamic adapter loading through an OpenAI compatible API. Just specify any adapter as the `model` parameter.

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

See [OpenAI Compatible API](https://predibase.github.io/lorax/reference/openai_api) for details.

### Next steps

Here are some other interesting Mistral-7B fine-tuned models to try out:

- [alignment-handbook/zephyr-7b-dpo-lora](https://huggingface.co/alignment-handbook/zephyr-7b-dpo-lora): Mistral-7b fine-tuned on Zephyr-7B dataset with DPO.
- [IlyaGusev/saiga_mistral_7b_lora](https://huggingface.co/IlyaGusev/saiga_mistral_7b_lora): Russian chatbot based on `Open-Orca/Mistral-7B-OpenOrca`.
- [Undi95/Mistral-7B-roleplay_alpaca-lora](https://huggingface.co/Undi95/Mistral-7B-roleplay_alpaca-lora): Fine-tuned using role-play prompts.

You can find more LoRA adapters [here](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending&search=-lora), or try fine-tuning your own with [PEFT](https://github.com/huggingface/peft) or [Ludwig](https://ludwig.ai).

## 🙇 Acknowledgements

LoRAX is built on top of HuggingFace's [text-generation-inference](https://github.com/huggingface/text-generation-inference), forked from v0.9.4 (Apache 2.0).

We'd also like to acknowledge [Punica](https://github.com/punica-ai/punica) for their work on the SGMV kernel, which is used to speed up multi-adapter inference under heavy load.

## 🗺️ Roadmap

Our roadmap is tracked [here](https://github.com/predibase/lorax/issues/57).
