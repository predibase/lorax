# Rust builder
FROM lukemathwalker/cargo-chef:latest-rust-1.75 AS chef
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

FROM nvcr.io/nvidia/pytorch:23.07-py3 as pytorch-install
FROM pytorch-install as kernel-builder

ARG MAX_JOBS=2

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ninja-build cmake \
    && rm -rf /var/lib/apt/lists/*

# Build Flash Attention CUDA kernels
FROM kernel-builder as flash-att-builder
WORKDIR /usr/src
COPY server/Makefile-flash-att Makefile
RUN make build-flash-attention

# Build Flash Attention v2 CUDA kernels
FROM kernel-builder as flash-att-v2-builder
WORKDIR /usr/src
COPY server/Makefile-flash-att-v2 Makefile
RUN make build-flash-attention-v2-cuda

# Build punica CUDA kernels
FROM kernel-builder as punica-builder
WORKDIR /usr/src
COPY server/punica_kernels/ .
# Build specific version of punica
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
RUN python setup.py build

# LoRAX base image
FROM nvcr.io/nvidia/pytorch:23.07-py3 as base

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

# Copy build artifacts from flash attention builder
COPY --from=flash-att-builder /usr/src/flash-attention/build/lib.linux-x86_64-cpython-310 /usr/local/lib/python3.10/dist-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/layer_norm/build/lib.linux-x86_64-cpython-310 /usr/local/lib/python3.10/dist-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/rotary/build/lib.linux-x86_64-cpython-310 /usr/local/lib/python3.10/dist-packages

# Copy build artifacts from flash attention v2 builder
COPY --from=flash-att-v2-builder /usr/src/flash-attention-v2/build/lib.linux-x86_64-cpython-310 /usr/local/lib/python3.10/dist-packages

# Copy builds artifacts from punica builder
COPY --from=punica-builder /usr/src/build/lib.linux-x86_64-cpython-310 /usr/local/lib/python3.10/dist-packages

# Install flash-attention dependencies
RUN pip install einops --no-cache-dir

# Install server
COPY proto proto
COPY server server
COPY server/Makefile server/Makefile

RUN cd server && \
    make gen-server && \
    pip install -r requirements.txt && \
    pip install ".[bnb, accelerate, quantize, peft, outlines]" --no-cache-dir

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
