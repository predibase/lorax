#!/bin/bash

# Default values
DOCKERFILE="Dockerfile.amd"
BASE_IMAGE_NAME="lorax-amd"
REPO="ghcr.io/predibase/lorax"
PUSH=false
TAGS=()

# Function to display usage
usage() {
    echo "Usage: $0 [-p|--push] [-t|--tag tag1,tag2,...] [-f|--file dockerfile] [-h|--help]"
    echo "  -p, --push        Push images after building"
    echo "  -t, --tag         Comma-separated list of tags (e.g., 'latest,v1.0,dev')"
    echo "  -f, --file        Specify Dockerfile (default: $DOCKERFILE)"
    echo "  -h, --help        Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--push)
            PUSH=true
            shift
            ;;
        -t|--tag)
            IFS=',' read -ra TAGS <<< "$2"
            shift 2
            ;;
        -f|--file)
            DOCKERFILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Ensure at least one tag if none provided
if [ ${#TAGS[@]} -eq 0 ]; then
    TAGS=("amd-latest")
fi

# Build the image
echo "Building Docker image from $DOCKERFILE..."
if ! docker build -f "$DOCKERFILE" . -t "$BASE_IMAGE_NAME"; then
    echo "Error: Docker build failed"
    exit 1
fi

# Tag the image
for tag in "${TAGS[@]}"; do
    echo "Tagging image with: $tag"
    if ! docker tag "$BASE_IMAGE_NAME" "$REPO:$tag"; then
        echo "Error: Failed to tag image with $tag"
        exit 1
    fi
done

# Push if requested
if [ "$PUSH" = true ]; then
    echo "Pushing images to repository..."
    for tag in "${TAGS[@]}"; do
        echo "Pushing $REPO:$tag"
        if ! docker push "$REPO:$tag"; then
            echo "Error: Failed to push image with tag $tag"
            exit 1
        fi
    done
fi

echo "Build script completed successfully"
