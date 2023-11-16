<div align="center">

![image](https://images.ctfassets.net/ft0odixqevnv/3cWNkdDkt08y0Tz7Sx8ZZQ/794ced27db7253025790c248595499ac/LoraxBlog-SocialCard.png?w=1104&h=585&q=100&fm=webp&bg=transparent)

# LoRA Exchange (LoRAX)

<a href="https://github.com/predibase/lorax">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/predibase/lorax?style=social">
</a>
<a href="https://github.com/predibase/lorax/blob/main/LICENSE">
  <img alt="License" src="https://img.shields.io/github/license/predibase/lorax">
</a>
</div>

The LLM inference server that speaks for the GPUs!

LoRAX (LoRA eXchange) is a framework that allows users to serve over a hundred fine-tuned models on a single GPU, dramatically reducing the cost of serving without compromising on throughput or latency.

## 📖 Table of contents

- [LoRAX (LoRA eXchange)](#lora-exchange-lorax)
  - [📖 Table of contents](#-table-of-contents)
  - [🔥 Features](#-features)
  - [🏠 Supported Models and Adapters](#-supported-models-and-adapters)
  - [🏃‍♂️ Get started](#️-get-started)
    - [Docker](#docker)
    - [📓 API documentation](#-api-documentation)
    - [🛠️ Local install](#️-local-install)
    - [🙇 Acknowledgements](#-acknowledgements)
    - [🗺️ Roadmap](#-roadmap)

## 🔥 Features

- 🚅 **Dynamic Adapter Loading:** allowing each set of fine-tuned LoRA weights to be loaded from storage just-in-time as requests come in at runtime, without blocking concurrent requests.
- 🏋️‍♀️ **Tiered Weight Caching:** to support fast exchanging of LoRA adapters between requests, and offloading of adapter weights to CPU and disk to avoid out-of-memory errors.
- 🧁 **Continuous Multi-Adapter Batching:** a fair scheduling policy for optimizing aggregate throughput of the system that extends the popular continuous batching strategy to work across multiple sets of LoRA adapters in parallel.
- 👬 **Optimized Inference:**  high throughput and low latency optimizations including tensor parallelism, [continuous batching](https://github.com/predibase/lorax/tree/main/router) across different adapters, [flash-attention](https://github.com/HazyResearch/flash-attention), [paged attention](https://github.com/vllm-project/vllm), quantization with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and [GPT-Q](https://arxiv.org/abs/2210.17323), token streaming, weight prefetching and offloading.
- 🚢  **Ready for Production** prebuilt Docker images, Helm charts for Kubernetes, Prometheus metrics, and distributed tracing with Open Telemetry.
- 🤯 **Free for Commercial Use:** Apache 2.0 License. Enough said 😎.


<p align="center">
  <img src="https://github.com/predibase/lorax/assets/29719151/6f4f78fc-c1e9-4a01-8675-dbafa74a2534" />
</p>


## 🏠 Supported Models and Adapters

### Models

- 🦙 [Llama](https://huggingface.co/meta-llama)
- 🌬️[Mistral](https://huggingface.co/mistralai)

Other architectures are supported on a best effort basis, but do not support dynamical adapter loading.

### Adapters

LoRAX currently supports LoRA adapters, which can be trained using frameworks like [PEFT](https://github.com/huggingface/peft) and [Ludwig](https://ludwig.ai/).

The following modules can be targeted:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

## 🏃‍♂️ Getting started

### Docker

The easiest way of getting started is using the official Docker container:

```shell
model=mistralai/Mistral-7B-Instruct-v0.1
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/predibase/lorax:latest --model-id $model
```
**Note:** To use GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). We also recommend using NVIDIA drivers with CUDA version 11.8 or higher.

To see all options to serve your models (in the [code](https://github.com/predibase/lorax/blob/main/launcher/src/main.rs) or in the cli:
```
lorax-launcher --help
```

You can then query the model using either the `/generate` or `/generate_stream` routes:

```shell
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "parameters": {"adapter_id": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"}}' \
    -H 'Content-Type: application/json'
```

```shell
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "parameters": {"adapter_id": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"}}' \
    -H 'Content-Type: application/json'
```

or from Python:

```shell
pip install lorax-client
```

```python
from lorax import Client

client = Client("http://127.0.0.1:8080")
prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

print(client.generate(prompt, adapter_id="vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k").generated_text)

text = ""
for response in client.generate_stream(prompt, adapter_id="vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"):
    if not response.token.special:
        text += response.token.text
print(text)
```

### 📓 API documentation

You can consult the OpenAPI documentation of the `lorax` REST API using the `/docs` route.

### 🛠️ Local Development

```
# window 1 (server)
make server-dev

# window 2 (router)
make router-dev
```

### 🙇 Acknowledgements

LoRAX is built on top of HuggingFace's [text-generation-inference](https://github.com/huggingface/text-generation-inference), forked from v0.9.4 (Apache 2.0).

### 🗺️ Roadmap

[ ] Serve pretrained embedding models
[ ] Serve embedding model MLP adapters
[ ] Serve LLM MLP adapters for classification
[ ] Blend multiple adapters per request
[ ] SGMV kernel for adapters with different ranks
