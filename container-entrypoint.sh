#!/bin/bash

# upload weights to cache
upload() {
    echo "Received a SIGTERM or SIGKILL signal. Uploading to cache."
    object_id="${MODEL_ID//\//--}"
    S3_DIRECTORY="models--$object_id"
    aws configure set default.s3.preferred_transfer_client crt
    aws configure set default.s3.payload_signing_enabled false
    aws configure set default.s3.target_bandwidth 50Gb/s
    echo "running aws s3 sync /data/${S3_DIRECTORY}/ s3://${HF_CACHE_BUCKET}/${S3_DIRECTORY}/ --exclude blobs/* --exclude *.bin"
    aws s3 sync "/data/${S3_DIRECTORY}/" "s3://${HF_CACHE_BUCKET}/${S3_DIRECTORY}/" --exclude "blobs/*" --exclude "*.bin"
    exit 0
}

# Trap SIGTERM signals and call the cleanup function
trap upload SIGTERM SIGKILL

# print AWS CLI version
aws --version

# download files
time ./sync.sh

# Function to check if lorax-launcher is running
is_launcher_running() {
    local launcher_pid="$1"
    # this checks whether the process is alive or not. Redirects the output of kill -0 to devnull.
    kill -0 "$launcher_pid" >/dev/null 2>&1
}

# launch TG launcher in the background
lorax-launcher "$@" &

# Capture the PID of the process we just launched
launcher_pid="$!"

# Loop to continuously check if lorax-launcher is running
while is_launcher_running "$launcher_pid"; do
    sleep 1
done

# Once lorax-launcher has stopped, the loop exits, and upload is called
upload
