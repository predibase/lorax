#!/bin/bash

# configure S3 parameters.
aws configure set default.s3.preferred_transfer_client crt

if [[ "$AWS_ACCESS_KEY_ID" == "minio" ]]; then
    echo "Weights cache not available in Minio."
    exit 0
fi

echo "HuggingFace Model ID: $MODEL_ID"
echo "HuggingFace local cache directory: $HUGGINGFACE_HUB_CACHE"

MODEL_DIRECTORY="models--${MODEL_ID//\//--}"
S3_PATH="s3://${HF_CACHE_BUCKET}/${MODEL_DIRECTORY}/"
LOCAL_MODEL_DIR="${HUGGINGFACE_HUB_CACHE}/${MODEL_DIRECTORY}"

MODEL_CONTENTS=$(aws s3api list-objects-v2 --bucket "${HF_CACHE_BUCKET}" --prefix "${MODEL_DIRECTORY}" --query 'Contents[]')
EXIT_STATUS=$?

if [[ $EXIT_STATUS -ne 0 || "$MODEL_CONTENTS" == "null" ]]; then
    echo "$MODEL_ID not found in S3 cache."
    exit 0
fi

echo "Syncing $MODEL_ID from S3 cache to local cache."
time sudo aws s3 sync "${S3_PATH}" "${LOCAL_MODEL_DIR}"
