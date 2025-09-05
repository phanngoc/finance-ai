#!/bin/bash

# LSTM Stock Prediction - Build Script
# This script builds the Docker image for the LSTM Stock Prediction application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="lstm-stock-prediction"
VERSION=${1:-"latest"}
DOCKER_REGISTRY=${2:-""}

echo -e "${BLUE}🚀 Building LSTM Stock Prediction Application${NC}"
echo -e "${BLUE}==============================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}❌ Dockerfile not found. Please run this script from the project root directory.${NC}"
    exit 1
fi

# Clean up any existing containers and images
echo -e "${YELLOW}🧹 Cleaning up existing containers and images...${NC}"
docker-compose down --remove-orphans 2>/dev/null || true
docker rmi ${APP_NAME}:${VERSION} 2>/dev/null || true

# Build the application
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
if [ -n "$DOCKER_REGISTRY" ]; then
    IMAGE_NAME="${DOCKER_REGISTRY}/${APP_NAME}:${VERSION}"
else
    IMAGE_NAME="${APP_NAME}:${VERSION}"
fi

docker build -t ${IMAGE_NAME} .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Successfully built image: ${IMAGE_NAME}${NC}"
else
    echo -e "${RED}❌ Failed to build image${NC}"
    exit 1
fi

# Test the image
echo -e "${YELLOW}🧪 Testing the built image...${NC}"
docker run --rm -d --name test-${APP_NAME} -p 8501:8501 ${IMAGE_NAME}

# Wait for the container to start
sleep 10

# Check if the container is running
if docker ps | grep -q test-${APP_NAME}; then
    echo -e "${GREEN}✅ Container is running successfully${NC}"
    
    # Test health endpoint
    if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Health check passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Health check failed, but container is running${NC}"
    fi
    
    # Stop test container
    docker stop test-${APP_NAME} > /dev/null 2>&1
    echo -e "${GREEN}✅ Test completed successfully${NC}"
else
    echo -e "${RED}❌ Container failed to start${NC}"
    docker logs test-${APP_NAME}
    docker stop test-${APP_NAME} > /dev/null 2>&1
    exit 1
fi

# Show image information
echo -e "${BLUE}📊 Image Information:${NC}"
docker images | grep ${APP_NAME}

echo -e "${GREEN}🎉 Build completed successfully!${NC}"
echo -e "${BLUE}To run the application:${NC}"
echo -e "  docker-compose up -d"
echo -e "${BLUE}To run with custom image:${NC}"
echo -e "  docker run -d -p 8501:8501 --name ${APP_NAME} ${IMAGE_NAME}"
echo -e "${BLUE}To push to registry:${NC}"
echo -e "  docker push ${IMAGE_NAME}"
