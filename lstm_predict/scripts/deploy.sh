#!/bin/bash

# LSTM Stock Prediction - Deploy Script for AWS EC2
# This script deploys the LSTM Stock Prediction application to AWS EC2

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
EC2_HOST=${2:-""}
EC2_USER=${3:-"ubuntu"}
SSH_KEY=${4:-"~/.ssh/id_rsa"}
DOCKER_REGISTRY=${5:-""}

echo -e "${BLUE}🚀 Deploying LSTM Stock Prediction to AWS EC2${NC}"
echo -e "${BLUE}==============================================${NC}"

# Validate parameters
if [ -z "$EC2_HOST" ]; then
    echo -e "${RED}❌ EC2_HOST is required. Usage: $0 <version> <ec2_host> [ec2_user] [ssh_key] [docker_registry]${NC}"
    echo -e "${YELLOW}Example: $0 latest ec2-xx-xx-xx-xx.compute-1.amazonaws.com ubuntu ~/.ssh/id_rsa${NC}"
    exit 1
fi

# Check if SSH key exists
if [ ! -f "${SSH_KEY}" ]; then
    echo -e "${RED}❌ SSH key not found: ${SSH_KEY}${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}❌ docker-compose.yml not found. Please run this script from the project root directory.${NC}"
    exit 1
fi

echo -e "${YELLOW}📋 Deployment Configuration:${NC}"
echo -e "  App Name: ${APP_NAME}"
echo -e "  Version: ${VERSION}"
echo -e "  EC2 Host: ${EC2_HOST}"
echo -e "  EC2 User: ${EC2_USER}"
echo -e "  SSH Key: ${SSH_KEY}"
echo -e "  Docker Registry: ${DOCKER_REGISTRY:-"Local"}"
echo ""

# Test SSH connection
echo -e "${YELLOW}🔌 Testing SSH connection...${NC}"
if ! ssh -i "${SSH_KEY}" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "${EC2_USER}@${EC2_HOST}" "echo 'SSH connection successful'" > /dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to EC2 instance. Please check:${NC}"
    echo -e "  - EC2 instance is running"
    echo -e "  - Security group allows SSH (port 22)"
    echo -e "  - SSH key is correct"
    echo -e "  - EC2 user has Docker permissions"
    exit 1
fi
echo -e "${GREEN}✅ SSH connection successful${NC}"

# Create deployment directory on EC2
echo -e "${YELLOW}📁 Creating deployment directory on EC2...${NC}"
ssh -i "${SSH_KEY}" "${EC2_USER}@${EC2_HOST}" "mkdir -p /home/${EC2_USER}/${APP_NAME}"

# Copy application files
echo -e "${YELLOW}📤 Copying application files...${NC}"
rsync -avz --delete \
    -e "ssh -i ${SSH_KEY}" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='data/' \
    . "${EC2_USER}@${EC2_HOST}:/home/${EC2_USER}/${APP_NAME}/"

# Install Docker and Docker Compose on EC2 (if not already installed)
echo -e "${YELLOW}🐳 Installing Docker and Docker Compose on EC2...${NC}"
ssh -i "${SSH_KEY}" "${EC2_USER}@${EC2_HOST}" "
    # Update package list
    sudo apt-get update
    
    # Install Docker if not installed
    if ! command -v docker &> /dev/null; then
        echo 'Installing Docker...'
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker ${EC2_USER}
        rm get-docker.sh
    else
        echo 'Docker is already installed'
    fi
    
    # Install Docker Compose if not installed
    if ! command -v docker-compose &> /dev/null; then
        echo 'Installing Docker Compose...'
        sudo curl -L \"https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)\" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    else
        echo 'Docker Compose is already installed'
    fi
    
    # Start Docker service
    sudo systemctl start docker
    sudo systemctl enable docker
"

# Build and deploy on EC2
echo -e "${YELLOW}🔨 Building and deploying application on EC2...${NC}"
ssh -i "${SSH_KEY}" "${EC2_USER}@${EC2_HOST}" "
    cd /home/${EC2_USER}/${APP_NAME}
    
    # Stop existing containers
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Build and start the application
    docker-compose up -d --build
    
    # Wait for services to start
    sleep 30
    
    # Check if services are running
    if docker-compose ps | grep -q 'Up'; then
        echo '✅ Application deployed successfully'
    else
        echo '❌ Application deployment failed'
        docker-compose logs
        exit 1
    fi
"

# Configure firewall (if needed)
echo -e "${YELLOW}🔥 Configuring firewall...${NC}"
ssh -i "${SSH_KEY}" "${EC2_USER}@${EC2_HOST}" "
    # Allow HTTP and HTTPS traffic
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    sudo ufw allow 8501/tcp
    
    # Enable firewall if not already enabled
    sudo ufw --force enable
"

# Test the deployment
echo -e "${YELLOW}🧪 Testing deployment...${NC}"
sleep 10

# Test HTTP endpoint
if curl -f "http://${EC2_HOST}" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Application is accessible via HTTP${NC}"
else
    echo -e "${YELLOW}⚠️  Application may not be accessible via HTTP (check security groups)${NC}"
fi

# Test Streamlit endpoint
if curl -f "http://${EC2_HOST}:8501/_stcore/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Streamlit application is running${NC}"
else
    echo -e "${YELLOW}⚠️  Streamlit application may not be accessible (check security groups)${NC}"
fi

# Show deployment status
echo -e "${YELLOW}📊 Deployment Status:${NC}"
ssh -i "${SSH_KEY}" "${EC2_USER}@${EC2_HOST}" "
    cd /home/${EC2_USER}/${APP_NAME}
    echo 'Docker containers:'
    docker-compose ps
    echo ''
    echo 'Docker images:'
    docker images | grep ${APP_NAME}
    echo ''
    echo 'Disk usage:'
    df -h
    echo ''
    echo 'Memory usage:'
    free -h
"

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "${BLUE}Application URLs:${NC}"
echo -e "  HTTP: http://${EC2_HOST}"
echo -e "  Streamlit: http://${EC2_HOST}:8501"
echo -e "${BLUE}To check logs:${NC}"
echo -e "  ssh -i ${SSH_KEY} ${EC2_USER}@${EC2_HOST} 'cd /home/${EC2_USER}/${APP_NAME} && docker-compose logs -f'"
echo -e "${BLUE}To restart application:${NC}"
echo -e "  ssh -i ${SSH_KEY} ${EC2_USER}@${EC2_HOST} 'cd /home/${EC2_USER}/${APP_NAME} && docker-compose restart'"
echo -e "${BLUE}To stop application:${NC}"
echo -e "  ssh -i ${SSH_KEY} ${EC2_USER}@${EC2_HOST} 'cd /home/${EC2_USER}/${APP_NAME} && docker-compose down'"
