We recommend starting with our [pre-built Docker image](https://ghcr.io/predibase/lorax) to avoid compiling custom CUDA kernels and other dependencies.

#### 1. Start Docker container with base LLM

In this example, we'll use Mistral-7B-Instruct as the base model, but you can use any [supported model](../models/base_models.md) from HuggingFace.

```shell
model=mistralai/Mistral-7B-Instruct-v0.1
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/predibase/lorax:latest --model-id $model
```
**Note:** To use GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). We also recommend using NVIDIA drivers with CUDA version 11.8 or higher.

See the references docs for the [Launcher](../reference/launcher.md) to view all available options, or run the following from within your container:

```
lorax-launcher --help
```