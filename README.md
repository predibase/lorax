<p align="center">
  <a href="https://github.com/predibase/lorax">
    <img src="https://images.ctfassets.net/ft0odixqevnv/3cWNkdDkt08y0Tz7Sx8ZZQ/794ced27db7253025790c248595499ac/LoraxBlog-SocialCard.png?w=1104&h=585&q=100&fm=webp&bg=transparent">
  </a>
</p>

<div align="center">

_The LLM inference server that speaks for the GPUs!_

[![Discord](https://img.shields.io/discord/1174495433565945916)](https://discord.gg/zMFfDrWT)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/predibase/lorax/blob/master/LICENSE)
[![Artifact Hub](https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/lorax)](https://artifacthub.io/packages/search?repo=lorax)


</div>

LoRAX (LoRA eXchange) is a framework that allows users to serve over a hundred fine-tuned models on a single GPU, dramatically reducing the cost of serving without compromising on throughput or latency.

## 📖 Table of contents

- [LoRA eXchange (LoRAX)](#lora-exchange-lorax)
  - [📖 Table of contents](#-table-of-contents)
  - [🔥 Features](#-features)
  - [🏠 Supported Models and Adapters](#-supported-models-and-adapters)
    - [Models](#models)
    - [Adapters](#adapters)
  - [🏃‍♂️ Getting started](#️-getting-started)
    - [Docker](#docker)
    - [📓 API documentation](#-api-documentation)
    - [🛠️ Local Development](#️-local-development)
    - [CUDA Kernels](#cuda-kernels)
  - [Run Mistral](#run-mistral)
    - [Run](#run)
  - [🙇 Acknowledgements](#-acknowledgements)
  - [🗺️ Roadmap](#️-roadmap)

## 🔥 Features

- 🚅 **Dynamic Adapter Loading:** allows each set of fine-tuned LoRA weights to be loaded from storage just-in-time as requests come in at runtime, without blocking concurrent requests.
- 🏋️‍♀️ **Tiered Weight Caching:** supports fast exchanging of LoRA adapters between requests, and offloading of adapter weights to CPU and disk to avoid out-of-memory errors.
- 🧁 **Continuous Multi-Adapter Batching:** a fair scheduling policy for optimizing aggregate throughput of the system that extends the popular continuous batching strategy to work across multiple sets of LoRA adapters in parallel.
- 👬 **Optimized Inference:**  high throughput and low latency optimizations including tensor parallelism, [continuous batching](https://github.com/predibase/lorax/tree/main/router) across different adapters, pre-compiled CUDA kernels ([flash-attention](https://arxiv.org/abs/2307.08691), [paged attention](https://arxiv.org/abs/2309.06180), [SGMV](https://arxiv.org/abs/2310.18547)), quantization with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and [GPT-Q](https://arxiv.org/abs/2210.17323), token streaming, weight prefetching and offloading.
- 🚢  **Ready for Production** prebuilt Docker images, Helm charts for Kubernetes, Prometheus metrics, and distributed tracing with Open Telemetry.
- 🤯 **Free for Commercial Use:** Apache 2.0 License. Enough said 😎.


<p align="center">
  <img src="https://github.com/predibase/lorax/assets/29719151/f88aa16c-66de-45ad-ad40-01a7874ed8a9" />
</p>


## 🏠 Supported Models and Adapters

### Models

- 🦙 [Llama](https://huggingface.co/meta-llama)
- 🌬️[Mistral](https://huggingface.co/mistralai)

Other architectures are supported on a best effort basis, but do not support dynamical adapter loading.

Check the [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) to find supported base models.

### Adapters

LoRAX currently supports LoRA adapters, which can be trained using frameworks like [PEFT](https://github.com/huggingface/peft) and [Ludwig](https://ludwig.ai/).

The following modules can be targeted:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

You can provide an adapter from the HuggingFace Hub, a local file path, or S3. 

Just make sure that the adapter was trained on the same base model used in the deployment. LoRAX only supports one base model at a time, but any number of adapters derived from it!

## 🏃‍♂️ Getting started

### Docker

We recommend starting with our pre-build Docker image to avoid compiling custom CUDA kernels and other dependencies.

#### 1. Start Docker container with base LLM

In this example, we'll use Mistral-7B-Instruct as the base model, but you can use any Mistral or Llama model from HuggingFace.

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

#### 2. Prompt the base model

LoRAX supports the same `/generate` and `/generate_stream` REST API from [text-generation-inference](https://github.com/huggingface/text-generation-inference) for prompting the base model.

REST:

```shell
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"}' \
    -H 'Content-Type: application/json'
```

```shell
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"}' \
    -H 'Content-Type: application/json'
```

Python:

```shell
pip install lorax-client
```

```python
from lorax import Client

client = Client("http://127.0.0.1:8080")
prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

print(client.generate(prompt).generated_text)

text = ""
for response in client.generate_stream(prompt):
    if not response.token.special:
        text += response.token.text
print(text)
```

#### 3. Prompt with a LoRA Adapter

You probably noticed that the response from the base model wasn't what you would expect. So let's now prompt our LLM again with a LoRA adapter
trained to answer this type of question.

REST:

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

Python:

```python
adapter_id = "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"

print(client.generate(prompt, adapter_id=adapter_id).generated_text)

text = ""
for response in client.generate_stream(prompt, adapter_id=adapter_id):
    if not response.token.special:
        text += response.token.text
print(text)
```

### Kubernetes (Helm)

LoRAX includes Helm charts that make it easy to start using LoRAX in production with high availability and load balancing on Kubernetes.

To spin up a LoRAX deployment with Helm, you only need to be connected to a Kubernetes cluster through `kubectl``. We provide a default values.yaml file that can be used to deploy a Mistral 7B base model to your Kubernetes cluster:

```shell
helm install mistral-7b-release charts/lorax
```

The default [values.yaml](charts/lorax/values.yaml) configuration deploys a single replica of the Mistral 7B model. You can tailor configuration parameters to deploy any Llama or Mistral model by creating a new values file from the template and updating variables. Once a new values file is created, you can run the following command to deploy your LLM with LoRAX:

```shell
helm install -f your-values-file.yaml your-model-release charts/lorax
```

To delete the resources:

```shell
helm uninstall your-model-release
```

### 📓 API documentation

You can consult the OpenAPI documentation of the `lorax` REST API using the `/docs` route.

### 🛠️ Local Development

You can also opt to install `lorax` locally.

First [install Rust](https://rustup.rs/) and create a Python virtual environment with at least
Python 3.9, e.g. using `conda`:

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

conda create -n lorax python=3.9 
conda activate lorax
```

You may also need to install Protoc.

On Linux:

```shell
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```

On MacOS, using Homebrew:

```shell
brew install protobuf
```

Then run:

```shell
BUILD_EXTENSIONS=True make install # Install repository and HF/transformer fork with CUDA kernels
make run-mistral-7b-instruct
```

**Note:** on some machines, you may also need the OpenSSL libraries and gcc. On Linux machines, run:

```shell
sudo apt-get install libssl-dev gcc -y
```

### CUDA Kernels

The custom CUDA kernels are only tested on NVIDIA A100s. If you have any installation or runtime issues, you can remove 
the kernels by using the `DISABLE_CUSTOM_KERNELS=True` environment variable.

Be aware that the official Docker image has them enabled by default.

## Run Mistral

### Run

```shell
make run-mistral-7b-instruct
```

## 🙇 Acknowledgements

LoRAX is built on top of HuggingFace's [text-generation-inference](https://github.com/huggingface/text-generation-inference), forked from v0.9.4 (Apache 2.0).

## 🗺️ Roadmap

- [ ] Serve pretrained embedding models
- [ ] Serve embedding model MLP adapters
- [ ] Serve LLM MLP adapters for classification
- [ ] Blend multiple adapters per request
- [ ] SGMV kernel for adapters with different ranks
