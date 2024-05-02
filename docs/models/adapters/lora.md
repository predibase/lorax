# LoRA

LoRAX currently supports LoRA adapters, which can be trained using frameworks like [PEFT](https://github.com/huggingface/peft) and [Ludwig](https://ludwig.ai/).

## Target Modules

Any combination of linear layers can be targeted in the adapters, which corresponds to the following target modules for each base model.

### Llama

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`
- `lm_head`

### Mistral

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`
- `lm_head`

### Mixtral

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `lm_head`

### Gemma

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

### Phi

- `q_proj`
- `k_proj`
- `v_proj`
- `dense`
- `fc1`
- `fc2`
- `lm_head`

### Qwen

- `c_attn`
- `c_proj`
- `w1`
- `w2`
- `lm_head`

### GPT2

- `c_attn`
- `c_proj`
- `c_fc`

### Bloom

- `query_key_value`
- `dense`
- `dense_h_to_4h`
- `dense_4h_to_h`
- `lm_head`