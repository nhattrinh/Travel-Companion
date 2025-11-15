#!/bin/bash
# Build script for Menu Translation Backend Docker images

set -e

# Default values
ENVIRONMENT="development"
TAG="latest"
PUSH=false
REGISTRY=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -e, --env ENVIRONMENT    Target environment (development, staging, production)"
            echo "  -t, --tag TAG           Docker image tag (default: latest)"
            echo "  -p, --push              Push image to registry after build"
            echo "  -r, --registry REGISTRY Registry URL (required if --push is used)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate environment
case $ENVIRONMENT in
    development|staging|production|testing)
        ;;
    *)
        echo "Error: Invalid environment '$ENVIRONMENT'. Must be one of: development, staging, production, testing"
        exit 1
        ;;
esac

# Set image name
IMAGE_NAME="menu-translation-backend"
if [[ -n "$REGISTRY" ]]; then
    FULL_IMAGE_NAME="$REGISTRY/$IMAGE_NAME:$TAG"
else
    FULL_IMAGE_NAME="$IMAGE_NAME:$TAG"
fi

echo "Building Docker image for environment: $ENVIRONMENT"
echo "Image name: $FULL_IMAGE_NAME"

# Determine build target based on environment
case $ENVIRONMENT in
    development)
        BUILD_TARGET="development"
        ;;
    testing)
        BUILD_TARGET="testing"
        ;;
    staging|production)
        BUILD_TARGET="runtime"
        ;;
esac

# Build the image
echo "Building with target: $BUILD_TARGET"
docker build \
    --target "$BUILD_TARGET" \
    --build-arg ENVIRONMENT="$ENVIRONMENT" \
    --build-arg APP_VERSION="$(date +%Y%m%d-%H%M%S)" \
    -t "$FULL_IMAGE_NAME" \
    .

echo "✓ Build completed successfully"

# Push to registry if requested
if [[ "$PUSH" == "true" ]]; then
    if [[ -z "$REGISTRY" ]]; then
        echo "Error: Registry URL is required when pushing"
        exit 1
    fi
    
    echo "Pushing image to registry..."
    docker push "$FULL_IMAGE_NAME"
    echo "✓ Push completed successfully"
fi

echo "Docker image ready: $FULL_IMAGE_NAME"