#!/bin/bash

# configure S3 parameters.
aws configure set default.s3.preferred_transfer_client crt
aws configure set default.s3.payload_signing_enabled false
aws configure set default.s3.target_bandwidth 50Gb/s

echo "HuggingFace Model ID: $MODEL_ID"
echo "HuggingFace local cache directory: $HUGGINGFACE_HUB_CACHE"

MODEL_DIRECTORY="models--${MODEL_ID//\//--}"
S3_PATH="s3://${HF_CACHE_BUCKET}/${MODEL_DIRECTORY}/"
LOCAL_MODEL_DIR="${HUGGINGFACE_HUB_CACHE}/${MODEL_DIRECTORY}"

if [[ $(aws s3 ls "${S3_PATH}" | wc -l) -eq 0 ]]; then
    echo "$MODEL_ID not found in S3 cache."
    exit 0
fi

echo "Syncing $MODEL_ID from S3 cache to local cache."
time sudo aws s3 sync "${S3_PATH}" "${LOCAL_MODEL_DIR}"
