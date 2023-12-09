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

By default, LoRAX will load adapters from the Huggingface Hub.

Usage:

```json
"parameters": {
    "adapter_id": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k",
    "adapter_source": "hub",
}
```

### Local

When specifying an adapter in a local path, the `adapter_id` should correspond to the root directory of the adapter containing the following files:

```shell
root_adapter_path/
    adapter_config.json
    adapter_model.bin
    adapter_model.safetensors
```

The weights must be in one of either a `adapter_model.bin` (pickle) or `adapter_model.safetensors` (safetensors) format. If both are provided, safestensors will be used.

See the [PEFT](https://github.com/huggingface/peft) library for detailed examples showing how to save adapters in this format.

Usage:

```json
"parameters": {
    "adapter_id": "/data/adapters/vineetsharma--qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k",
    "adapter_source": "local",
}
```

### S3

Similar to a local path, an S3 path can be provided. Just make sure you have the appropriate environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` set so you can authenticate to AWS.

Usage:

```json
"parameters": {
    "adapter_id": "s3://adapters_bucket/vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k",
    "adapter_source": "s3",
}
```