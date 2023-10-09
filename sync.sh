#!/bin/bash

# configure S3 parameters.
aws configure set default.s3.preferred_transfer_client crt
aws configure set default.s3.payload_signing_enabled false
aws configure set default.s3.target_bandwidth 50Gb/s

echo "HuggingFace Model ID: $MODEL_ID"
echo "HuggingFace local cache directory: $HUGGINGFACE_HUB_CACHE"

OBJECT_ID="${MODEL_ID//\//--}"
S3_BASE_DIRECTORY="models--$OBJECT_ID"
S3_PATH="s3://${HF_CACHE_BUCKET}/${S3_BASE_DIRECTORY}/"
LOCAL_MODEL_DIR="${HUGGINGFACE_HUB_CACHE}/${S3_BASE_DIRECTORY}"

# Function to check if text-generation-launcher is running
is_launcher_running() {
    local launcher_pid="$1"
    # this checks whether the process is alive or not. Redirects the output of kill -0 to devnull.
    kill -0 "$launcher_pid" >/dev/null 2>&1
}

sudo mkdir -p $LOCAL_MODEL_DIR

if [ -n "$(ls -A $LOCAL_MODEL_DIR)" ]; then
    echo "Files have already been downloaded to ${LOCAL_MODEL_DIR}"
    ls -Rlah $LOCAL_MODEL_DIR
    exit 0
fi

# if a cache doesn't exist for the model, exit gracefully.
if [ -z "$(aws s3 ls ${S3_PATH})" ]; then
  echo "No files found in the cache ${S3_PATH}. Downloading from HuggingFace Hub."
  echo "Received arguments: $@"
  
  # Trap SIGTERM signals and call the cleanup function
  trap '' SIGTERM SIGKILL EXIT

  # text-generation-server download-weights $MODEL_ID 
  text-generation-launcher "$@" &

  # Capture the PID of the process we just launched
  launcher_pid="$!"

  # Loop to continuously check if text-generation-launcher is running
  while is_launcher_running "$launcher_pid"; do
      sleep 1
  done

  echo "Uploading to cache."
  echo "running aws s3 sync /data/${S3_BASE_DIRECTORY}/ s3://${HF_CACHE_BUCKET}/${S3_BASE_DIRECTORY}/ --exclude blobs/* --exclude *.bin"
  aws s3 sync "/data/${S3_BASE_DIRECTORY}/" "s3://${HF_CACHE_BUCKET}/${S3_BASE_DIRECTORY}/" --exclude "blobs/*" --exclude "*.bin"

  exit 0
else
  echo "Downloading weights from ${S3_PATH}"
fi

echo "Files found for model ${MODEL_ID}"
aws s3 ls "${S3_PATH}" --recursive | awk '{print $4}'

# List S3 objects and create local directories.
aws s3 ls "${S3_PATH}" --recursive | awk '{print $4}' | xargs -I {} bash -c 'sudo mkdir -p "${HUGGINGFACE_HUB_CACHE}/$(dirname "{}")"'

copy_file() {
    file="$1"
    sudo -E aws s3 cp "s3://${HF_CACHE_BUCKET}/${file}" "${HUGGINGFACE_HUB_CACHE}/${file}"
}
export -f copy_file

# Download files concurrently while preserving directory structure.
FILES=$(aws s3 ls "${S3_PATH}" --recursive | awk '{print $4}')
parallel --env HUGGINGFACE_HUB_CACHE,HF_CACHE_BUCKET,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY --jobs 5 copy_file ::: "${FILES[@]}"
