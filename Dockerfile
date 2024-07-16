# Python builder
# Adapted from: https://github.com/pytorch/pytorch/blob/master/Dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 as pytorch-install

ARG PYTORCH_VERSION=2.3.0
ARG PYTHON_VERSION=3.10
# Keep in sync with `server/pyproject.toml
ARG CUDA_VERSION=12.4
ARG MAMBA_VERSION=24.3.0-0
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
# Automatically set by buildx
ARG TARGETPLATFORM

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
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

RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  exit 1 ;; \
    *)              /opt/conda/bin/conda update -y conda &&  \
    /opt/conda/bin/conda install -y python=3.10 pytorch pytorch-cuda=12.4 -c pytorch-nightly -c nvidia  ;; \
    esac && \
    /opt/conda/bin/conda clean -ya

# CUDA kernels builder image
FROM pytorch-install as kernel-builder

ARG MAX_JOBS=2

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ninja-build wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget 'https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0-linux-x86_64.tar.gz'
RUN tar xzvf 'cmake-3.30.0-linux-x86_64.tar.gz'
RUN ln -s "$(pwd)/cmake-3.30.0-linux-x86_64/bin/cmake" /usr/local/bin/cmake

# Build vllm CUDA kernels
FROM kernel-builder as vllm-builder
WORKDIR /usr/src
COPY vllm  .
RUN python setup.py build

