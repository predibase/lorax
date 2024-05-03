# Medusa

[Medusa](https://arxiv.org/abs/2401.10774) is a [speculative decoding](../../guides/speculative_decoding.md) method 
that trains new projection layers (similar to LoRA layers) for the purpose of predicting future tokens and speedng up 
the text generation process.

## How it works

``` mermaid
graph BT
  X[H] --> S((Stack));
  X --> M1[Medusa 1];
  X --> M2[Medusa 2];
  X --> M3[Medusa 3];
  M1 --> S;
  M2 --> S;
  M3 --> S;
  S --> LM[LM Head]
```

The goal of Medusa is to speed up text generation. Unlike LoRA, Medusa does not aim to improve response quality, and in
fact enabling Medusa will have no effect at all on the model output itself. Instead, Medusa works by adding additional
projections (called "medusa heads") that the last hidden state `H` of the LLM is passed through that attempt to predict
the next N tokens (rather than just the next 1 token).

The result is that the output logit shape of the model at each decoding step is no longer `[B, 1, V]` for batch size `B` and vocabulary
size `V`, but instead `[B, S, V]` where `S` is the number of Medusa speculative heads `N` plus `1` for the original model
head.

See the [Speculative Decoding](../../guides/speculative_decoding.md#verification) guide for more information on the verification
step that follows.

## How to train