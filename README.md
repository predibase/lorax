<div align="center">

![image](https://images.ctfassets.net/ft0odixqevnv/3cWNkdDkt08y0Tz7Sx8ZZQ/794ced27db7253025790c248595499ac/LoraxBlog-SocialCard.png?w=1104&h=585&q=100&fm=webp&bg=transparent)

# LoRA Exchange (LoRAX)

<a href="https://github.com/predibase/lorax">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/huggingface/lorax-inference?style=social">
</a>
<a href="https://github.com/predibase/lorax/blob/main/LICENSE">
  <img alt="License" src="https://img.shields.io/github/license/predibase/lorax">
</a>
</div>

The LLM inference server that speaks for the GPUs!

Lorax is a framework that allows users to serve over a hundred fine-tuned models on a single GPU, dramatically reducing the cost of serving without compromising on throughput or latency.

## 📖 Table of contents

- [LoRA Exchange (LoRAX)](#lora-exchange-lorax)
  - [📖 Table of contents](#-table-of-contents)
  - [🔥 Features](#-features)
  - [🏠 Optimized architectures](#-optimized-architectures)
  - [🏃‍♂️ Get started](#️-get-started)
    - [Docker](#docker)
    - [📓 API documentation](#-api-documentation)
    - [🛠️ Local install](#️-local-install)

## 🔥 Features

- 🚅 **Dynamic Adapter Loading:** allowing each set of fine-tuned LoRA weights to be loaded from storage just-in-time as requests come in at runtime, without blocking concurrent requests.
- 🏋️‍♀️ **Tiered Weight Caching:** to support fast exchanging of LoRA adapters between requests, and offloading of adapter weights to CPU and disk to avoid out-of-memory errors.
- 🧁 **Continuous Multi-Adapter Batching:** a fair scheduling policy for optimizing aggregate throughput of the system that extends the popular continuous batching strategy to work across multiple sets of LoRA adapters in parallel.
- 👬 **Optimized Inference:**  [flash-attention](https://github.com/HazyResearch/flash-attention), [paged attention](https://github.com/vllm-project/vllm), quantization with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and [GPT-Q](https://arxiv.org/abs/2210.17323), tensor parallelism, token streaming, and [continuous batching](https://github.com/huggingface/lorax-inference/tree/main/router) work together to optimize our inference speeds.
- ✅ **Production Readiness** reliably stable, Lorax supports  Prometheus metrics and distributed tracing with Open Telemetry
- 🤯 **Free Commercial Use:** Apache 2.0 License. Enough said 😎.

## 🏠 Optimized architectures

- 🦙 [Llama V2](https://huggingface.co/meta-llama)
- 🌬️[Mistral](https://huggingface.co/mistralai)

Other architectures are supported on a best effort basis using:

`AutoModelForCausalLM.from_pretrained(<model>, device_map="auto")`

or

`AutoModelForSeq2SeqLM.from_pretrained(<model>, device_map="auto")`

## 🏃‍♂️ Get started

### Docker

The easiest way of getting started is using the official Docker container:

```shell
model=mistralai/Mistral-7B-v0.1
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/lorax-inference:0.9.4 --model-id $model
```
**Note:** To use GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). We also recommend using NVIDIA drivers with CUDA version 11.8 or higher.

To see all options to serve your models (in the [code](https://github.com/huggingface/lorax-inference/blob/main/launcher/src/main.rs) or in the cli:
```
lorax-launcher --help
```

You can then query the model using either the `/generate` or `/generate_stream` routes:

```shell
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

```shell
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

or from Python:

```shell
pip install lorax
```

```python
from lorax import Client

client = Client("http://127.0.0.1:8080")
print(client.generate("What is Deep Learning?", max_new_tokens=20).generated_text)

text = ""
for response in client.generate_stream("What is Deep Learning?", max_new_tokens=20):
    if not response.token.special:
        text += response.token.text
print(text)
```

### 📓 API documentation

You can consult the OpenAPI documentation of the `lorax-inference` REST API using the `/docs` route.
The Swagger UI is also available at: [https://huggingface.github.io/lorax-inference](https://huggingface.github.io/lorax-inference).

### 🛠️ Local install

MAGDY AND WAEL TODO