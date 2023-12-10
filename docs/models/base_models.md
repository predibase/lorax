# Base Models

## Supported Architectures

- 🦙 [Llama](https://huggingface.co/meta-llama)
    - [CodeLlama](https://huggingface.co/codellama)
- 🌬️[Mistral](https://huggingface.co/mistralai)
    - [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
- 🔮 [Qwen](https://huggingface.co/Qwen)
- 🤖 [GPT2](https://huggingface.co/gpt2)

Other architectures are supported on a best effort basis, but do not support dynamic adapter loading.

## Selecting a Base Model

Check the [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) to find supported base models.

Usage:

```shell
lorax-launcher --model-id mistralai/Mistral-7B-v0.1 ...
```

## Quantization

Base models can be loaded in fp16 (default) or with quantization using any of `bitsandbytes`, [GPT-Q](https://arxiv.org/abs/2210.17323), or [AWQ](https://arxiv.org/abs/2306.00978) format. When using quantization, it is not necessary that
the adapter was fine-tuned using the quantized version of the base model, but be aware that enabling quantization can have an effect on the response.

### bitsandbytes

`bitsandbytes` quantization can be applied to any base model saved in fp16 or bf16 format. It performs quantization at runtime in a model and dataset agnostic manner. As such, it is more flexible but potentially less performant (both in terms of quality and latency) than other quantization options.

There are three flavors of `bitsandbytes` quantization:

- `bitsandbytes` (8-bit integer)
- `bitsandbytes-fp4` (4-bit float)
- `bitsandbytes-nf4` (4-bit normal float)

Usage:

```shell
lorax-launcher --model-id mistralai/Mistral-7B-v0.1 --quantize bitsandbytes-nf4 ...
```

### GPT-Q

[GPT-Q](https://arxiv.org/abs/2210.17323) is a static quantization method, meaning that the quantization needs to be done outside of LoRAX and the weights persisted in order for it to be used with a base model. Thanks to the ExLlama-v2 CUDA kernels, GPT-Q offers very strong inference performance compared with `bitsandbytes`, but may not generalize as well to unseen tasks (as the quantization is done with respect to a particular dataset). 

Usage:

```shell
lorax-launcher --model-id TheBloke/Mistral-7B-v0.1-GPTQ --quantize gptq ...
```

### AWQ

[AWQ](https://arxiv.org/abs/2306.00978) is similar to GPT-Q in that the weights are quantized in advance of inference (statically). AWQ is generally faster than GPT-Q for inference, and achieves similarly high levels of quality.

Usage:

```shell
lorax-launcher --model-id TheBloke/Mistral-7B-v0.1-AWQ --quantize awq ...
```
