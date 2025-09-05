#!/bin/bash

# LSTM Stock Prediction - Quick Deploy Script
# This script provides a quick deployment option with minimal configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Quick Deploy - LSTM Stock Prediction${NC}"
echo -e "${BLUE}=======================================${NC}"

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}❌ docker-compose.yml not found. Please run this script from the project root directory.${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Stop existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker-compose down --remove-orphans 2>/dev/null || true

# Build and start the application
echo -e "${YELLOW}🔨 Building and starting application...${NC}"
docker-compose up -d --build

# Wait for services to start
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 30

# Check if services are running
if docker-compose ps | grep -q 'Up'; then
    echo -e "${GREEN}✅ Application deployed successfully!${NC}"
else
    echo -e "${RED}❌ Application deployment failed${NC}"
    echo -e "${YELLOW}📋 Container logs:${NC}"
    docker-compose logs
    exit 1
fi

# Display status
echo -e "${BLUE}📊 Deployment Status:${NC}"
docker-compose ps

echo -e "${BLUE}🌐 Application URLs:${NC}"
echo -e "  Local: http://localhost:8501"
echo -e "  Nginx: http://localhost"

echo -e "${BLUE}📋 Useful Commands:${NC}"
echo -e "  View logs: docker-compose logs -f"
echo -e "  Restart: docker-compose restart"
echo -e "  Stop: docker-compose down"
echo -e "  Status: docker-compose ps"

echo -e "${GREEN}🎉 Quick deployment completed!${NC}"
