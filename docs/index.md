# LoRAX

<p align="center">
    <img src="images/lorax_guy.png" alt="LoRAX Logo" style="width:200px;" />
</p>
<p align="center">
    <em>The LLM inference server that speaks for the GPUs</em>
</p>
<p align="center">

<a href="https://discord.gg/CBgdrGnZjy" target="_blank" style="text-decoration: none;">
    <img src="https://dcbadge.vercel.app/api/server/CBgdrGnZjy?style=flat&theme=discord-inverted" alt="discord">
</a>
<a href="https://github.com/predibase/lorax/blob/master/LICENSE" target="_blank" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="license">
</a>
<a href="https://artifacthub.io/packages/search?repo=lorax" target="_blank" style="text-decoration: none;">
    <img src="https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/lorax" alt="artifacthub">
</a>
</p>

---

## 📖 What is LoRAX?

LoRAX (LoRA eXchange) is a framework that allows users to serve over a hundred fine-tuned models on a single GPU, dramatically reducing the cost of serving without compromising on throughput or latency.

## 🌳 Features

- 🚅 **Dynamic Adapter Loading:** include any fine-tuned LoRA adapter in your request, it will be loaded just-in-time without blocking concurrent requests.
- 🏋️‍♀️ **Heterogeneous Continuous Batching:** packs requests for different adapters together into the same batch, keeping latency and throughput nearly constant with the number of concurrent adapters.
- 🧁 **Adapter Exchange Scheduling:** asynchronously prefetches and offloads adapters between GPU and CPU memory, schedules request batching to optimize the aggregate throughput of the system.
- 👬 **Optimized Inference:**  high throughput and low latency optimizations including tensor parallelism, pre-compiled CUDA kernels ([flash-attention](https://arxiv.org/abs/2307.08691), [paged attention](https://arxiv.org/abs/2309.06180), [SGMV](https://arxiv.org/abs/2310.18547)), quantization with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and [GPT-Q](https://arxiv.org/abs/2210.17323), token streaming.
- 🚢  **Ready for Production** prebuilt Docker images, Helm charts for Kubernetes, Prometheus metrics, and distributed tracing with Open Telemetry.
- 🤯 **Free for Commercial Use:** Apache 2.0 License. Enough said 😎.


<p align="center">
  <img src="https://github.com/predibase/lorax/assets/29719151/f88aa16c-66de-45ad-ad40-01a7874ed8a9" />
</p>


## 🏠 Supported Models and Adapters

### Models

- 🦙 [Llama](https://huggingface.co/meta-llama)
    - [CodeLlama](https://huggingface.co/codellama)
- 🌬️[Mistral](https://huggingface.co/mistralai)
    - [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
- 🔮 [Qwen](https://huggingface.co/Qwen)

Other architectures are supported on a best effort basis, but do not support dynamical adapter loading.

Check the [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) to find supported base models.

#### Quantization

Base models can be loaded in fp16 (default) or with quantization using either the `bitsandbytes` or [GPT-Q](https://arxiv.org/abs/2210.17323) format. When using quantization, it is not necessary that
the adapter was fine-tuned using the quantized version of the base model, but be aware that enabling quantization can have an effect on the response.

### Adapters

LoRAX currently supports LoRA adapters, which can be trained using frameworks like [PEFT](https://github.com/huggingface/peft) and [Ludwig](https://ludwig.ai/).

Any combination of linear layers can be targeted in the adapters, including:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`
- `lm_head`

You can provide an adapter from the HuggingFace Hub, a local file path, or S3. 

Just make sure that the adapter was trained on the same base model used in the deployment. LoRAX only supports one base model at a time, but any number of adapters derived from it!

## 🏃‍♂️ Getting started

We recommend starting with our pre-build Docker image to avoid compiling custom CUDA kernels and other dependencies.

### 1. Start Docker container with base LLM

In this example, we'll use Mistral-7B-Instruct as the base model, but you can use any Mistral or Llama model from HuggingFace.

```shell
model=mistralai/Mistral-7B-Instruct-v0.1
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/predibase/lorax:latest --model-id $model
```
**Note:** To use GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). We also recommend using NVIDIA drivers with CUDA version 11.8 or higher.

To see all options to serve your models:

```
lorax-launcher --help
```

### 2. Prompt the base model

LoRAX supports the same `/generate` and `/generate_stream` REST API from [text-generation-inference](https://github.com/huggingface/text-generation-inference) for prompting the base model.

REST:

```shell
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]", "parameters": {"max_new_tokens": 64}}' \
    -H 'Content-Type: application/json'
```

```shell
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]", "parameters": {"max_new_tokens": 64}}' \
    -H 'Content-Type: application/json'
```

Python:

```shell
pip install lorax-client
```

```python
from lorax import Client

client = Client("http://127.0.0.1:8080")
prompt = "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]"

print(client.generate(prompt, max_new_tokens=64).generated_text)

text = ""
for response in client.generate_stream(prompt, max_new_tokens=64):
    if not response.token.special:
        text += response.token.text
print(text)
```

### 3. Prompt with a LoRA Adapter

You probably noticed that the response from the base model wasn't what you would expect. So let's now prompt our LLM again with a LoRA adapter
trained to answer this type of question.

REST:

```shell
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]", "parameters": {"max_new_tokens": 64, "adapter_id": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"}}' \
    -H 'Content-Type: application/json'
```

```shell
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]", "parameters": {"max_new_tokens": 64, "adapter_id": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"}}' \
    -H 'Content-Type: application/json'
```

Python:

```python
adapter_id = "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"

print(client.generate(prompt, max_new_tokens=64, adapter_id=adapter_id).generated_text)

text = ""
for response in client.generate_stream(prompt, max_new_tokens=64, adapter_id=adapter_id):
    if not response.token.special:
        text += response.token.text
print(text)
```

## 🙇 Acknowledgements

LoRAX is built on top of HuggingFace's [text-generation-inference](https://github.com/huggingface/text-generation-inference), forked from v0.9.4 (Apache 2.0).

We'd also like to acknowledge [Punica](https://github.com/punica-ai/punica) for their work on the SGMV kernel, which is used to speed up multi-adapter inference under heavy load.

## 🗺️ Roadmap

Our roadmap is tracked [here](https://github.com/predibase/lorax/issues/57).
