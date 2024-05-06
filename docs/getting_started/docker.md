# Docker

We recommend starting with our [pre-built Docker image](https://ghcr.io/predibase/lorax) to avoid compiling custom CUDA kernels and other dependencies.

## Run container with base LLM

In this example, we'll use Mistral-7B-Instruct as the base model, but you can use any [supported model](../models/base_models.md) from HuggingFace.

```shell
model=mistralai/Mistral-7B-Instruct-v0.1
volume=$PWD/data  # share a volume with the container as a weight cache

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data \
    ghcr.io/predibase/lorax:main --model-id $model
```

!!! note
    
    The `main` tag will use the image built from the HEAD of the main branch of the repo. For the latest stable image (built from a 
    tagged version) use the `latest` tag.

!!! note
    
    The LoRAX server in the pre-built Docker image is configured to listen on port 80 (instead of on the default port number, which is 3000).

!!! note
    
    To use GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). We also recommend using NVIDIA drivers with CUDA version 11.8 or higher.

See the references docs for the [Launcher](../reference/launcher.md) to view all available options, or run the following from within your container:

```
lorax-launcher --help
```

## Prompt the base LLM

=== "REST"

    ```shell
    curl 127.0.0.1:8080/generate \
        -X POST \
        -d '{"inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]", "parameters": {"max_new_tokens": 64}}' \
        -H 'Content-Type: application/json'
    ```

=== "REST (Streaming)"

    ```shell
    curl 127.0.0.1:8080/generate_stream \
        -X POST \
        -d '{"inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]", "parameters": {"max_new_tokens": 64}}' \
        -H 'Content-Type: application/json'
    ```

=== "Python"

    ```shell
    pip install lorax-client
    ```

    ```python
    from lorax import Client

    client = Client("http://127.0.0.1:8080")
    prompt = "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]"

    print(client.generate(prompt, max_new_tokens=64).generated_text)
    ```

=== "Python (Streaming)"

    ```shell
    pip install lorax-client
    ```

    ```python
    from lorax import Client

    client = Client("http://127.0.0.1:8080")
    prompt = "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]"

    text = ""
    for response in client.generate_stream(prompt, max_new_tokens=64):
        if not response.token.special:
            text += response.token.text
    print(text)
    ```

## Prompt a LoRA adapter

=== "REST"

    ```shell
    curl 127.0.0.1:8080/generate \
        -X POST \
        -d '{"inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]", "parameters": {"max_new_tokens": 64, "adapter_id": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"}}' \
        -H 'Content-Type: application/json'
    ```

=== "REST (Streaming)"

    ```shell
    curl 127.0.0.1:8080/generate_stream \
        -X POST \
        -d '{"inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]", "parameters": {"max_new_tokens": 64, "adapter_id": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"}}' \
        -H 'Content-Type: application/json'
    ```

=== "Python"

    ```shell
    pip install lorax-client
    ```

    ```python
    from lorax import Client

    client = Client("http://127.0.0.1:8080")
    prompt = "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]"
    adapter_id = "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"

    print(client.generate(prompt, max_new_tokens=64, adapter_id=adapter_id).generated_text)
    ```

=== "Python (Streaming)"

    ```shell
    pip install lorax-client
    ```

    ```python
    from lorax import Client

    client = Client("http://127.0.0.1:8080")
    prompt = "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]"
    adapter_id = "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"

    text = ""
    for response in client.generate_stream(prompt, max_new_tokens=64, adapter_id=adapter_id):
        if not response.token.special:
            text += response.token.text
    print(text)
    ```
