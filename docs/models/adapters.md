# Adapters

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

## Source

You can provide an adapter from the HuggingFace Hub, a local file path, or S3. 

Just make sure that the adapter was trained on the same base model used in the deployment. LoRAX only supports one base model at a time, but any number of adapters derived from it!

### Huggingface Hub

### Local

### S3