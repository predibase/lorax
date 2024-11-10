# Development Environment

This guide will walk through how you can setup a LoRAX devlopment environment within Docker.

## Prerequisites

- Docker
- Nvidia GPU (Ampere or newer)
- CUDA 11.8 drivers or above

## Launch development container

Pull and run the latest LoRAX docker image, mounting the directory containing your local `lorax` repo as a volume within the container:

```shell
# we will assume the lorax repo is found at ~/data/lorax
volume=~/data

docker pull ghcr.io/predibase/lorax:main
docker run \
    --cap-add=SYS_PTRACE \
    --gpus all --shm-size 1g \
    -v $volume:/data \
    -itd --entrypoint /bin/bash ghcr.io/predibase/lorax:main
```

!!! note

    `SYS_PTRACE` is set so we can obtain stack traces from running processes for debugging.

Next, find the name of the container using `docker ps` and then SSH in:

```shell
docker exec -it <container_id> /bin/bash
```

Using two additional terminal windows, repeat the SSH process into the container.

We'll be working out of three different terminals during development, each serving a different purpose:

1. **Server** window running the Python LoRAX server.
2. **Router** window running the Rust LoRAX router.
3. **Client** window for executing requests against the running LoRAX instance.

## Server window setup

Install development dependencies:

```shell
DEBIAN_FRONTEND=noninteractive apt install pkg-config rsync tmux rust-gdb git -y && \
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP && \
hash -r
```

Download weights from HF:

```shell
lorax-server download-weights mistralai/Mistral-7B-Instruct-v0.1
```

Create `tmux` session so we don't lose our state when we close our laptop:

```shell
tmux new -s server
```

From within the `tmux` session, move into the LoRAX `server` directory within the repo (assumed to be in `/data/lorax`) and install dependencies:

```shell
cd /data/lorax/server && pip install -e .
make gen-server
```

Launch the server:

```shell
SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run \
    --nproc_per_node=1 lorax_server/cli.py \
    serve mistralai/Mistral-7B-Instruct-v0.1
```

## Router window setup

As we did for the server, let's create a `tmux` session so we don't lose our state when we close our laptop:

```shell
tmux new -s router
```

Now move into the `router` directory within the repo and install dependencies:

```shell
cd /data/lorax/router && \
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
export PATH=$PATH:$HOME/.cargo/bin && \
touch ../proto/generate.proto
```

Launch the router:

```shell
RUST_BACKTRACE=1 cargo run -- --port 8080
```

## Client window setup

From the third window, install the Python client from source:

```shell
cd /data/lorax/clients/python
pip install -e .
```

Now you can send requests to our LoRAX instance from either REST or Python locally:

```shell
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{
        "inputs": "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]",
        "parameters": {
            "max_new_tokens": 64
        }
    }' \
    -H 'Content-Type: application/json'
```

## (Optional) Compile CUDA kernels

If you need compile CUDA kernels (for example, when updating to a newer version of a CUDA kernel), you'll need to install a few additional dependencies in your container.

First, it recommended to run the kernel building from within a separate `tmux` session:

```shell
tmux new -s builder
```

Next, install the toolchain for compile CUDA kernels:

```shell
export PATH=/opt/conda/bin:$PATH
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

conda update --force conda
/opt/conda/bin/conda install -c "nvidia"  cuda==12.4 cudnn && \
    /opt/conda/bin/conda clean -ya
```

You should now have everything you need to build and install LoRAX's CUDA kernels:

```shell
cd /data/lorax/server/punica_kernels
rm -rf build && python setup.py build
cp build/lib.linux-x86_64-cpython-310/punica_kernels.cpython-310-x86_64-linux-gnu.so /opt/conda/lib/python3.10/site-packages/.
```
