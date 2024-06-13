# Base image
FROM ghcr.io/predibase/lorax:0.10.0
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /

# Update and upgrade the system packages (Worker Template)
COPY builder/setup.sh /setup.sh
RUN /bin/bash /setup.sh && \
    rm /setup.sh

# Install Python dependencies (Worker Template)
# COPY builder/requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install runpod

# Add src files (Worker Template)
ADD src .

# Whether to download the model into /runpod-volume or not.
ARG DOWNLOAD_MODEL=
ENV DOWNLOAD_MODEL=$DOWNLOAD_MODEL

# Set environment variables
ARG HF_MODEL_ID=
ENV HF_MODEL_ID=$HF_MODEL_ID

ARG HF_MODEL_REVISION=
ENV HF_MODEL_REVISION=$HF_MODEL_REVISION

ARG SM_NUM_GPUS=
ENV SM_NUM_GPUS=$SM_NUM_GPUS

ARG HF_MODEL_QUANTIZE=
ENV HF_MODEL_QUANTIZE=$HF_MODEL_QUANTIZE

ARG HF_MODEL_TRUST_REMOTE_CODE=
ENV HF_MODEL_TRUST_REMOTE_CODE=$HF_MODEL_TRUST_REMOTE_CODE

ARG MODEL_BASE_PATH="/runpod-volume/"
ENV MODEL_BASE_PATH=$MODEL_BASE_PATH

ARG HUGGING_FACE_HUB_TOKEN=
ENV HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN

ARG HF_MAX_TOTAL_TOKENS=
ENV HF_MAX_TOTAL_TOKENS=$HF_MAX_TOTAL_TOKENS

ARG HF_MAX_INPUT_LENGTH=
ENV HF_MAX_INPUT_LENGTH=$HF_MAX_INPUT_LENGTH

ARG HF_MAX_BATCH_TOTAL_TOKENS=
ENV HF_MAX_BATCH_TOTAL_TOKENS=$HF_MAX_BATCH_TOTAL_TOKENS

ARG HF_MAX_BATCH_PREFILL_TOKENS=
ENV HF_MAX_BATCH_PREFILL_TOKENS=$HF_MAX_BATCH_PREFILL_TOKENS

# Prepare the hugging face directories for caching datasets, models, and more.
ENV HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

# Conditionally download the model weights based on DOWNLOAD_MODEL
RUN if [ "$DOWNLOAD_MODEL" = "1" ]; then \
    lorax-server download-weights $HF_MODEL_ID; \
  fi

# Quick temporary updates
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install git+https://github.com/runpod/runpod-python@a1#egg=runpod --compile
RUN python3.10 -m pip install lorax-client

ENTRYPOINT ["./entrypoint.sh"]
