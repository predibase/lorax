# Speculative Decoding

Speculative decoding describes a set of the methods for speeding up next token generation for autoregressive language models
by attempting to "guess" the next N tokens of the base model. These guesses can be generated in a number of different ways
including:

- An addtional smaller "draft" model (e.g., Llama-70b and Llama-7b)
- An adapter that extends the sequence dimension of the logits (e.g., Medusa)
- A heuristic (e.g., looking for recurring sequences in the prompt)

LoRAX implements some of these approaches, with a particular emphasis on supporting adapter-based methods like Medusa
that can be applied per request for task-level speedups.

## Process

Most all of the above speculative decoding methods consist of the same two phases: a "draft" phase that generates
candidate tokens and a "verification" phase that accepts some subset of the candidates to add to the response.

### Draft

For methods other than assisted generation via a draft model, the *draft step* happens at the end the normal next token
selection phase after generating the logits. Given the logits for the next token and all the tokens that have been
processed previously (input or output) a number of speculative tokens are generated and added to the batch state
for verification in the next inference step.

### Verification

Once the speculative logits have been generated, a separate *verification step* is performed whereby the most likely next `S` tokens
are passed through the model again (as part of the normal decoding process) to check for correctness. If any prefix of the `S` tokens
are deemed *correct*, then they can be appended to the response directly. The remaining incorrect speculative tokens are discarded.

Note that this process adds some compute overhead to the normal decoding step. As such, it will only confer benefits when:

1. The decoding step is *memory bound* (generally true for most LLMs on modern GPUs).
2. The speculation process is able to consistently predict future tokens correctly.

## Options

### Medusa

See the [Medusa](../models/adapters/medusa.md) guide for details on how this method works and how to use it.

### Prompt Lookup Decoding

[Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding?tab=readme-ov-file) is a simple
herustic method that uses string matching on the input + previously generated tokens to find candidate n-grams. This
method is particularly useful if your generation task will reuse many similar phrases from the input (e.g., in 
retrieval augmented generation where citing the input is important). If there is no need to repeat anything from the
input, there will be no speedup and performance may decrease.

#### Usage

Initialize LoRAX with the `--speculative-tokens` param. This controls the length of the sequence LoRAX will attempt
to match against in the input and suggest as the continuation of the current token:

```bash
docker run --gpus all --shm-size 1g -p 8080:80 -v $PWD:/data \
    ghcr.io/predibase/lorax:main \
    --model-id mistralai/Mistral-7B-Instruct-v0.2 \
    --speculative-tokens 3
```

Increasing this value will yield greater speedups when there are long common sequences, but slow things down if there
is little overlap.

Note that this method is not compatible with Medusa adapters per request.
