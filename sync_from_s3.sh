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

# Make sure that we can successfully list objects in the S3 bucket. (e.g. make sure we don't have any permission issues)
aws s3api list-objects-v2 --bucket "${HF_CACHE_BUCKET}" --prefix "${MODEL_DIRECTORY}" --query 'Contents[]' 2>&1 > /dev/null
if [[ $? -ne 0 ]]; then
    echo "Failed to list objects in S3 bucket."
    exit 0
fi

MODEL_CONTENTS=$(aws s3api list-objects-v2 --bucket "${HF_CACHE_BUCKET}" --prefix "${MODEL_DIRECTORY}" --query 'Contents[]' 2>&1)
if [[ "$MODEL_CONTENTS" == "null" ]]; then
    echo "$MODEL_ID not found in S3 cache."
    exit 0
fi

# TODO: If we want to speed up download times we could consider switching to s5cmd
echo "Syncing $MODEL_ID from S3 cache to local cache."
time aws s3 sync "${S3_PATH}" "${LOCAL_MODEL_DIR}"
