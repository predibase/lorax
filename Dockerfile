# Rust builder
FROM lukemathwalker/cargo-chef:latest-rust-1.70 AS chef
WORKDIR /usr/src

ARG CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

FROM chef as planner
COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY router router
COPY launcher launcher
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

ARG GIT_SHA
ARG DOCKER_LABEL

RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY --from=planner /usr/src/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY router router
COPY launcher launcher
RUN cargo build --release

# Python builder
# Adapted from: https://github.com/pytorch/pytorch/blob/master/Dockerfile
FROM debian:bullseye-slim as pytorch-install

ARG PYTORCH_VERSION=2.1.1
ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=11.8
ARG MAMBA_VERSION=23.1.0-1
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
# Automatically set by buildx
ARG TARGETPLATFORM

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    sudo \
    curl \
    git && \
    rm -rf /var/lib/apt/lists/*

# Install conda
# translating Docker's TARGETPLATFORM into mamba arches
RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  MAMBA_ARCH=aarch64  ;; \
    *)              MAMBA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -v -o ~/mambaforge.sh -O  "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Mambaforge-${MAMBA_VERSION}-Linux-${MAMBA_ARCH}.sh"
RUN chmod +x ~/mambaforge.sh && \
    bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh

# Install pytorch
# On arm64 we exit with an error code
RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  exit 1 ;; \
    *)              /opt/conda/bin/conda update -y conda &&  \
    /opt/conda/bin/conda install -y "python=3.10" && \
    /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y "python=${PYTHON_VERSION}" pytorch==$PYTORCH_VERSION "pytorch-cuda=$(echo $CUDA_VERSION | cut -d'.' -f 1-2)"  ;; \
    esac && \
    /opt/conda/bin/conda clean -ya

# CUDA kernels builder image
FROM pytorch-install as kernel-builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install -c "nvidia/label/cuda-11.8.0"  cuda==11.8 && \
    /opt/conda/bin/conda clean -ya

# Build Flash Attention CUDA kernels
FROM kernel-builder as flash-att-builder
WORKDIR /usr/src
COPY server/Makefile-flash-att Makefile

# Build specific version of flash attention
RUN make build-flash-attention
# Build Flash Attention v2 CUDA kernels
FROM kernel-builder as flash-att-v2-builder
WORKDIR /usr/src
COPY server/Makefile-flash-att-v2 Makefile
# Build specific version of flash attention v2
RUN make build-flash-attention-v2

# Build Transformers exllamav2 kernels
FROM kernel-builder as exllama-kernels-builder
WORKDIR /usr/src
COPY server/exllamav2_kernels/ .
# Build specific version of transformers
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" python setup.py build

# Build awq kernels
FROM kernel-builder as awq-kernels-builder
WORKDIR /usr/src
COPY server/awq_kernels/ .
# Build specific version of transformers
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" python setup.py build

# Build Transformers CUDA kernels
FROM kernel-builder as custom-kernels-builder
WORKDIR /usr/src
COPY server/custom_kernels/ .
# Build specific version of transformers
RUN python setup.py build

# Build vllm CUDA kernels
FROM kernel-builder as vllm-builder
RUN /opt/conda/bin/conda install packaging
WORKDIR /usr/src
COPY server/Makefile-vllm Makefile
# Build specific version of vllm
RUN make build-vllm

# Build megablocks kernels
FROM kernel-builder as megablocks-kernels-builder
WORKDIR /usr/src
COPY server/Makefile-megablocks Makefile
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
RUN make build-megablocks

# Build punica CUDA kernels
FROM kernel-builder as punica-builder
WORKDIR /usr/src
COPY server/punica_kernels/ .
# Build specific version of punica
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
RUN python setup.py build

# LoRAX base image
FROM nvidia/cuda:11.8.0-base-ubuntu20.04 as base

# Conda env
ENV PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda

# LoRAX base env
ENV HUGGINGFACE_HUB_CACHE=/data \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80

WORKDIR /usr/src

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libssl-dev \
    ca-certificates \
    make \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Copy conda with PyTorch installed
COPY --from=pytorch-install /opt/conda /opt/conda

# Copy build artifacts from flash attention builder
COPY --from=flash-att-builder /usr/src/flash-attention/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/layer_norm/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/rotary/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages

# Copy build artifacts from flash attention v2 builder
COPY --from=flash-att-v2-builder /usr/src/flash-attention-v2/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages

# Copy build artifacts from custom kernels builder
COPY --from=custom-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
# Copy build artifacts from exllama kernels builder
COPY --from=exllama-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages
# Copy build artifacts from awq kernels builder
COPY --from=awq-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages

# Copy builds artifacts from vllm builder
COPY --from=vllm-builder /usr/src/vllm/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages

# Copy builds artifacts from punica builder
COPY --from=punica-builder /usr/src/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages

# Copy build artifacts from megablocks builder
COPY --from=megablocks-kernels-builder /usr/src/megablocks/build/lib.linux-x86_64-cpython-310 /opt/conda/lib/python3.10/site-packages

# Install flash-attention dependencies
RUN pip install einops --no-cache-dir

# Install the pip requirements 
COPY server/requirements.txt .
RUN pip install -r requirements.txt

# Install server
COPY proto proto
COPY server server
COPY server/Makefile server/Makefile

RUN cd server && \
    make gen-server && \
    pip install ".[bnb, accelerate, quantize, peft]" --no-cache-dir

# Install router
COPY --from=builder /usr/src/target/release/lorax-router /usr/local/bin/lorax-router
# Install launcher
COPY --from=builder /usr/src/target/release/lorax-launcher /usr/local/bin/lorax-launcher

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*


# Final image
FROM base

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends sudo curl unzip parallel time

COPY container-entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh
COPY sync.sh sync.sh
RUN chmod +x sync.sh


RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    sudo ./aws/install && \
    rm -rf aws awscliv2.zip

# ENTRYPOINT ["./entrypoint.sh"]
ENTRYPOINT ["lorax-launcher"]
CMD ["--json-output"]
