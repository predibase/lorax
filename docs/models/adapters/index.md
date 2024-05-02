# Adapters

## Source

You can provide an adapter from the HuggingFace Hub, a local file path, or S3. 

Just make sure that the adapter was trained on the same base model used in the deployment. LoRAX only supports one base model at a time, but any number of adapters derived from it!

### Huggingface Hub

By default, LoRAX will load adapters from the Huggingface Hub.

Usage:

```json
"parameters": {
    "adapter_id": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k",
    "adapter_source": "hub"
}
```

### Predibase

Any adapter hosted in [Predibase](https://predibase.com/) can be used in LoRAX by setting `adapter_source="pbase"`.

When using Predibase hosted adapters, the `adapter_id` format is `<model_repo>/<model_version>`. If the `model_version` is
omitted, the latest version in the [Model Repoistory](https://docs.predibase.com/ui-guide/Supervised-ML/models/model-repos)
will be used.

Usage:

```json
"parameters": {
    "adapter_id": "model_repo/model_version",
    "adapter_source": "pbase"
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
    "adapter_source": "local"
}
```

### S3

Similar to a local path, an S3 path can be provided. Just make sure you have the appropriate environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` set so you can authenticate to AWS.

Usage:

```json
"parameters": {
    "adapter_id": "s3://adapters_bucket/vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k",
    "adapter_source": "s3"
}
```

## Merging Adapters

Multiple adapters can be mixed / merged together per request to create powerful ensembles of different specialized adapters.

This is particularly useful when you want your LLM to be capable of handling multiple types of tasks based on the user's prompt without
requiring them to specify the type of task they wish to perform.

See [Merging Adapters](../guides/merging_adapters.md) for details.

## Private Adapter Repositories

For hosted adapter repositories like HuggingFace Hub and [Predibase](https://predibase.com/), you can perform inference using private adapters per request.

Usage:

```json
"parameters": {
    "adapter_id": "my-repo/private-adapter",
    "api_token": "<auth_token>"
}
```

The authorization check is performed per-request in the background (prior to batching to prevent slowing down inference) every time, so even if the
adapter is cachd locally or the authorization token has been invalidated, the check will be performed and handled appropriately.

For details on generating API tokens, see:

- [HuggingFace docs](https://huggingface.co/docs/hub/security-tokens)
- [Predibase docs](https://docs.predibase.com/)
