#!/bin/bash

# Exit if any command fails
set -ex

# Check if there are any uncommitted changes
if [[ -n $(git status -s) ]]; then
    DIRTY="-dirty"
else
    DIRTY=""
fi

# Get the latest commit SHA
COMMIT_SHA=$(git rev-parse --short HEAD)

# Combine the SHA and dirty status to form the complete tag
TAG="${COMMIT_SHA}${DIRTY}"

# Name of the Docker image
IMAGE_NAME="kubellm"

# ECR Repository URL (replace with your actual ECR repository URL)
ECR_REPO="474375891613.dkr.ecr.us-west-2.amazonaws.com"

echo "Building ${IMAGE_NAME}:${TAG}"

# Build the Docker image
docker build -t ${IMAGE_NAME}:${TAG} .
docker tag ${IMAGE_NAME}:${TAG} ${IMAGE_NAME}:latest

# Tag the Docker image for ECR repository
docker tag ${IMAGE_NAME}:${TAG} ${ECR_REPO}/${IMAGE_NAME}:${TAG}

# Log in to the ECR registry (assumes AWS CLI and permissions are set up)
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin ${ECR_REPO}

# Push to ECR
docker push ${ECR_REPO}/${IMAGE_NAME}:${TAG}


latest_flag=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --latest)
            latest_flag=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if the --latest flag has been passed
if $latest_flag; then
    # Tag and push as 'latest'
    docker tag ${IMAGE_NAME}:${TAG} ${ECR_REPO}/${IMAGE_NAME}:latest
    docker push ${ECR_REPO}/${IMAGE_NAME}:latest
else
    echo "The --latest flag has not been passed. Skipping push to ECR as latest."
fi
