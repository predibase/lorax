# LoRA

[Low Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685) is a popular adapter method for fine-tuning response quality. 

LoRAX supports LoRA adapters trained using frameworks like [PEFT](https://github.com/huggingface/peft) and [Ludwig](https://ludwig.ai/).

## How it works

``` mermaid
graph BT
  I{{X}} --> W;
  I --> A[/LoRA A\];
  A --> B[\LoRA B/];
  W --> P((+));
  B--> P;
  P --> O{{Y}}
```

LoRA works by targeting specific layers of the base model and inserting a new low-rank pair of weights `LoRA A` and `LoRA B` alongside each base model
param `W`. The input `X` is passed through both the original weights and the LoRA weights, and then the activations are summed together
to produce the final layer output `Y`.

## Usage

### Supported Target Modules

When training a LoRA adapter, you can specify which of these layers (or "modules") you wish to target for adaptation. Typically
these are the projection layers in the attention blocks (`q` and `v`, sometimes `k` and `o` as well for LLaMA like models), but can
usually be any linear layer.

Here is a list of supported target modules for each architecture in LoRAX. Note that in cases where your adapter contains target
modules that LoRAX does not support, LoRAX will ignore those layers and emit a warning on the backend.

#### Llama

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`
- `lm_head`

#### Mistral

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`
- `lm_head`

#### Mixtral

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `lm_head`

#### Gemma

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

#### Phi-3

- `qkv_proj`
- `o_proj`
- `gate_up_proj`
- `down_proj`
- `lm_head`

#### Phi-2

- `q_proj`
- `k_proj`
- `v_proj`
- `dense`
- `fc1`
- `fc2`
- `lm_head`

#### Qwen2

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`
- `lm_head`

#### Qwen

- `c_attn`
- `c_proj`
- `w1`
- `w2`
- `lm_head`

#### Command-R

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`
- `lm_head`

#### DBRX

- `Wqkv`
- `out_proj`
- `lm_head`

#### GPT2

- `c_attn`
- `c_proj`
- `c_fc`

#### Bloom

- `query_key_value`
- `dense`
- `dense_h_to_4h`
- `dense_4h_to_h`
- `lm_head`

## How to train

LoRA is a very popular fine-tuning method for LLMs, and as such there are a number of options for creating them
from your data, including the following (non-exhaustive) options.

### Open Source

- [PEFT](https://github.com/huggingface/peft)
- [Ludwig](https://ludwig.ai/)

### Commercial

- [Predibase](https://predibase.com/)
