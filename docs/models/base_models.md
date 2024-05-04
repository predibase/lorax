# Base Models

## Supported Architectures

- 🦙 [Llama](https://huggingface.co/meta-llama)
  - [CodeLlama](https://huggingface.co/codellama)
- 🌬️[Mistral](https://huggingface.co/mistralai)
  - [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
- 🔄 [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
- 💎 [Gemma](https://blog.google/technology/developers/gemma-open-models/)
- 🏛️ [Phi](https://huggingface.co/microsoft/phi-2)
- 🔮 [Qwen](https://huggingface.co/Qwen)
- 🤖 [GPT2](https://huggingface.co/gpt2)
- 🔆 [Solar](https://huggingface.co/upstage/SOLAR-10.7B-v1.0)
- 🌸 [Bloom](https://huggingface.co/bigscience/bloom)

Other architectures are supported on a best effort basis, but do not support dynamic adapter loading.

## Selecting a Base Model

Check the [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) to find supported base models.

Usage:

```shell
lorax-launcher --model-id mistralai/Mistral-7B-v0.1 ...
```

## Quantization

LoRAX supports loading the base model with quantization to reduce memory overhead, while loading adapters in
full (fp32) or half precision (fp16, bf16), similar to the approach described in [QLoRA](https://arxiv.org/abs/2305.14314).

See [Quantization](../guides/quantization.md) for details on the various quantization strategies provided by LoRAX.
