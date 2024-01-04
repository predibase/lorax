LoRAX supports compiling the model into a static CUDA Graph to speedup inference by upwards of 2x. See [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) for more details on CUDA graphs and how they can reduce latency.

## Usage

To enable this (experimental) feature:

```
lorax-launcher ... --compile
```

## When should I use this?

CUDA graph compilation is a simple way to decrease latency for smaller LLMs (O(1b params)) that are compute bound rather than memory bound.

There is a tradeoff to be aware of when using CUDA graphs, namely that it increases memory overhead by 3-10GB depending on model size. However, the observed decrease in latency can be as much as 50%, so if you don't need to run with very large batch sizes and are more latency constrained than throughput, this is a very compelling feature to enable.

In practice, CUDA graphs are most useful in cases where there are excess GPU flops available, such as during decoding. As such, we do not use the compiled version of the model during prefill, only during the decoding steps. Which means in practice that the benefits of enabling compilation will be most pronounced when generating longer sequences (for which more time is spent during decoding).

## Limitations

Current limitations:

- Batch size < 256
- Context length (input + output) < 8192
- LoRA rank >= 8 and <= 64
- Only one LoRA rank in the batch
- 1 GPU (no sharding)

If any of these conditions are not met, then LoRAX will fallback to using eager execution for the batch.

## Benchmarks

gpt2-medium, 1x A100, time to generate 100 tokens:

no adapter:

- baseline: 1.044 s
- cuda graph: 0.422 s

1 adapter (rank 16):

- baseline: 1.503 s
- cuda graph: 0.583 s