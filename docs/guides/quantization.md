# Quantization

LoRAX supports loading the base model with quantization to reduce memory overhead, while loading adapters in
full (fp32) or half precision (fp16, bf16), similar to the approach described in [QLoRA](https://arxiv.org/abs/2305.14314).

When using quantization, it is not necessary that the adapter was fine-tuned using the quantized version of the base model, but be aware that enabling quantization can have an effect on the response.

## bitsandbytes

`bitsandbytes` quantization can be applied to any base model saved in fp16 or bf16 format. It performs quantization just-in-time at runtime in a model and dataset agnostic manner. As such, it is more flexible but potentially less performant (both in terms of quality and latency) than other quantization options.

There are three flavors of `bitsandbytes` quantization:

- `bitsandbytes` (8-bit integer)
- `bitsandbytes-fp4` (4-bit float)
- `bitsandbytes-nf4` (4-bit normal float)

Usage:

```shell
lorax-launcher --model-id mistralai/Mistral-7B-v0.1 --quantize bitsandbytes-nf4 ...
```

## AWQ

[AWQ](https://arxiv.org/abs/2306.00978) is a static quantization method applied outside of LoRAX using a framework such as [AutoAWQ](https://github.com/casper-hansen/AutoAWQ). Compared with other quantization methods, AWQ is very fast and very closely matches the quality of the original model, despite using int4 quantization.

AWQ supports 4-bit quantization.

Usage:

```shell
lorax-launcher --model-id TheBloke/Mistral-7B-v0.1-AWQ --quantize awq ...
```

## GPT-Q

[GPT-Q](https://arxiv.org/abs/2210.17323) is a static quantization method, meaning that the quantization needs to be done outside of LoRAX and the weights persisted in order for it to be used with a base model. GPT-Q offers faster inference performance compared with `bitsandbytes` but is noticeably slower than AWQ.

Apart from inference speed, the major difference between AWQ and GPT-Q is the way quantization bins are determined. While AWQ looks at the distribution of activations, GPT-Q looks at the distribution of weights.

GPT-Q supports 8 and 4-bit quantization, which is determined during quantized model creation outside of LoRAX.

Usage:

```shell
lorax-launcher --model-id TheBloke/Mistral-7B-v0.1-GPTQ --quantize gptq ...
```

## EETQ

[EETQ](https://github.com/NetEase-FuXi/EETQ) is an efficient just-in-time int8 quantization method that boasts very fast inference speed when compared against bitsandbytes.

EETQ supports 8-bit quantization.

Usage:

```shell
lorax-launcher --model-id mistralai/Mistral-7B-v0.1 --quantize eetq ...
```

## HQQ

[HQQ](https://mobiusml.github.io/hqq_blog/) is a fast just-in-time quantization method. Empircally, it is faster to load than other just-in-time methods
like bitsandbytes or EETQ, but results in some amount of degradation in performance.

HQQ supports 4, 3, and 2-bit quantization, making it particularly well suited to low VRAM GPUs.

- `hqq-4bit` (4-bit)
- `hqq-3bit` (3-bit)
- `hqq-2bit` (2-bit)

Usage:

```shell
lorax-launcher --model-id mistralai/Mistral-7B-v0.1 --quantize hqq-2bit ...
```