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
ECR_REPO="474375891613.dkr.ecr.us-west-2.amazonaws.com/kubellm"

echo "Building ${IMAGE_NAME}:${TAG}"

# Build the Docker image
docker build -t ${IMAGE_NAME}:${TAG} .

# Tag the Docker image for ECR repository
docker tag ${IMAGE_NAME}:${TAG} ${ECR_REPO}:${TAG}

# Push to ECR
docker push ${ECR_REPO}:${TAG}

# Optional: Tag and push as 'latest'
docker tag ${IMAGE_NAME}:${TAG} ${ECR_REPO}:latest
docker push ${ECR_REPO}:latest


