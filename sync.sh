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
LOCKFILE="${HUGGINGFACE_HUB_CACHE}/cache.lock"
CACHE_FILE="${HUGGINGFACE_HUB_CACHE}/cache.txt"

DEFAULT_CACHE_SIZE=4
CACHE_SIZE=${CACHE_SIZE:-$DEFAULT_CACHE_SIZE}

sudo mkdir -p $LOCAL_MODEL_DIR

# Function to check if lorax-launcher is running
is_launcher_running() {
    local launcher_pid="$1"
    # this checks whether the process is alive or not. Redirects the output of kill -0 to devnull.
    kill -0 "$launcher_pid" >/dev/null 2>&1
}

clean_up_cache() {
    local temp_file=$(mktemp)
    local removed_lines=""
    local key=$1
    local file=$2

    # Remove the key if it exists
    grep -v "^$key\$" "$file" > "$temp_file"

    # Add the key to the bottom of the file
    echo "$key" >> "$temp_file"

    # Count total lines in temp file
    local total_lines=$(wc -l < "$temp_file")

    # Calculate number of lines to be removed, if any
    local lines_to_remove=$((total_lines - CACHE_SIZE))

    if [ "$lines_to_remove" -gt 0 ]; then
        # Store removed lines in a variable
        removed_lines=$(head -n "$lines_to_remove" "$temp_file")
        echo "Deleting $removed_lines from cache"
    fi

    # Ensure only the last CACHE_SIZE items are retained
    tail -n $CACHE_SIZE "$temp_file" > "$file"

    # Clean up the temporary file
    rm "$temp_file"

    for line in $removed_lines; do
        model_to_remove="${HUGGINGFACE_HUB_CACHE}/${line}"
        echo "Removing $model_to_remove"     
        rm -rf $model_to_remove
    done
}

(
    # Wait for lock on $LOCKFILE (fd 200)
    flock -x 200

    echo "Lock acquired."

    if [ -f "$CACHE_FILE" ]; then
        echo "Cache file exists."
        while read -r line; do
            echo "Line read: $line"
            if [ "$line" = "$S3_BASE_DIRECTORY" ]; then
                echo "Model found in cache."
            fi
        done < "$CACHE_FILE"
    else
        echo "Cache file does not exist."
    fi
    clean_up_cache "$S3_BASE_DIRECTORY" "$CACHE_FILE"
) 200>$LOCKFILE


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

  # lorax-server download-weights $MODEL_ID 
  lorax-launcher "$@" &

  # Capture the PID of the process we just launched
  launcher_pid="$!"

  # Loop to continuously check if lorax-launcher is running
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
    # The files are a list of files without the bucket prefix. 
    # In the event that the env variable HF_CACHE_BUCKET is not just a bucket, but a bucket plus a subfolder, 
    # the subfolder will already be included into the file variable. 
    # In this case, strip all subpaths from the env variable HF_CACHE_BUCKET before attempting to download weights. 
    file="$1"
    true_bucket=`echo $HF_CACHE_BUCKET | cut -d / -f 1`
    sudo -E aws s3 cp "s3://${true_bucket}/${file}" "${HUGGINGFACE_HUB_CACHE}/${file}"
}
export -f copy_file

# Download files concurrently while preserving directory structure.
FILES=$(aws s3 ls "${S3_PATH}" --recursive | awk '{print $4}')
parallel --env HUGGINGFACE_HUB_CACHE,HF_CACHE_BUCKET,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY --jobs 5 copy_file ::: "${FILES[@]}"
