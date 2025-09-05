#!/bin/bash

# LSTM Stock Prediction - EC2 Setup Script
# This script sets up an EC2 instance for running the LSTM Stock Prediction application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up EC2 instance for LSTM Stock Prediction${NC}"
echo -e "${BLUE}===================================================${NC}"

# Update system packages
echo -e "${YELLOW}📦 Updating system packages...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# Install essential packages
echo -e "${YELLOW}🔧 Installing essential packages...${NC}"
sudo apt-get install -y \
    curl \
    wget \
    git \
    unzip \
    htop \
    vim \
    ufw \
    fail2ban \
    nginx \
    certbot \
    python3-certbot-nginx

# Install Docker
echo -e "${YELLOW}🐳 Installing Docker...${NC}"
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo -e "${GREEN}✅ Docker installed successfully${NC}"
else
    echo -e "${GREEN}✅ Docker is already installed${NC}"
fi

# Install Docker Compose
echo -e "${YELLOW}🐳 Installing Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}✅ Docker Compose installed successfully${NC}"
else
    echo -e "${GREEN}✅ Docker Compose is already installed${NC}"
fi

# Configure firewall
echo -e "${YELLOW}🔥 Configuring firewall...${NC}"
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8501/tcp
echo -e "${GREEN}✅ Firewall configured${NC}"

# Configure fail2ban
echo -e "${YELLOW}🛡️  Configuring fail2ban...${NC}"
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
echo -e "${GREEN}✅ Fail2ban configured${NC}"

# Create application directory
echo -e "${YELLOW}📁 Creating application directory...${NC}"
mkdir -p /home/$USER/lstm-stock-prediction
cd /home/$USER/lstm-stock-prediction

# Create systemd service for auto-start
echo -e "${YELLOW}⚙️  Creating systemd service...${NC}"
sudo tee /etc/systemd/system/lstm-stock-prediction.service > /dev/null <<EOF
[Unit]
Description=LSTM Stock Prediction Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/$USER/lstm-stock-prediction
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable lstm-stock-prediction.service
echo -e "${GREEN}✅ Systemd service created${NC}"

# Create monitoring script
echo -e "${YELLOW}📊 Creating monitoring script...${NC}"
cat > /home/$USER/monitor.sh << 'EOF'
#!/bin/bash

# LSTM Stock Prediction - Monitoring Script
echo "=== LSTM Stock Prediction Status ==="
echo "Date: $(date)"
echo ""

echo "=== Docker Containers ==="
docker-compose -f /home/$USER/lstm-stock-prediction/docker-compose.yml ps

echo ""
echo "=== System Resources ==="
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}'

echo "Memory Usage:"
free -h

echo "Disk Usage:"
df -h

echo ""
echo "=== Application Logs (last 10 lines) ==="
docker-compose -f /home/$USER/lstm-stock-prediction/docker-compose.yml logs --tail=10

echo ""
echo "=== Network Status ==="
netstat -tlnp | grep -E ':(80|443|8501)'
EOF

chmod +x /home/$USER/monitor.sh
echo -e "${GREEN}✅ Monitoring script created${NC}"

# Create backup script
echo -e "${YELLOW}💾 Creating backup script...${NC}"
cat > /home/$USER/backup.sh << 'EOF'
#!/bin/bash

# LSTM Stock Prediction - Backup Script
BACKUP_DIR="/home/$USER/backups"
APP_DIR="/home/$USER/lstm-stock-prediction"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

echo "Creating backup: lstm-stock-prediction_$DATE.tar.gz"
tar -czf "$BACKUP_DIR/lstm-stock-prediction_$DATE.tar.gz" -C $APP_DIR .

# Keep only last 7 backups
cd $BACKUP_DIR
ls -t lstm-stock-prediction_*.tar.gz | tail -n +8 | xargs -r rm

echo "Backup completed: $BACKUP_DIR/lstm-stock-prediction_$DATE.tar.gz"
EOF

chmod +x /home/$USER/backup.sh
echo -e "${GREEN}✅ Backup script created${NC}"

# Set up log rotation
echo -e "${YELLOW}📝 Setting up log rotation...${NC}"
sudo tee /etc/logrotate.d/lstm-stock-prediction > /dev/null <<EOF
/home/$USER/lstm-stock-prediction/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
}
EOF
echo -e "${GREEN}✅ Log rotation configured${NC}"

# Create environment file template
echo -e "${YELLOW}📄 Creating environment file template...${NC}"
cat > /home/$USER/lstm-stock-prediction/.env.example << 'EOF'
# LSTM Stock Prediction Environment Variables

# Application settings
APP_NAME=lstm-stock-prediction
VERSION=latest

# Streamlit settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Docker settings
COMPOSE_PROJECT_NAME=lstm-stock-prediction

# Optional: Docker Registry
# DOCKER_REGISTRY=your-registry.com
EOF
echo -e "${GREEN}✅ Environment file template created${NC}"

# Display system information
echo -e "${BLUE}📊 System Information:${NC}"
echo -e "  OS: $(lsb_release -d | cut -f2)"
echo -e "  Kernel: $(uname -r)"
echo -e "  Architecture: $(uname -m)"
echo -e "  CPU: $(nproc) cores"
echo -e "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo -e "  Disk: $(df -h / | tail -1 | awk '{print $2}')"

# Display next steps
echo -e "${GREEN}🎉 EC2 setup completed successfully!${NC}"
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Copy your application files to /home/$USER/lstm-stock-prediction/"
echo -e "  2. Run: cd /home/$USER/lstm-stock-prediction && docker-compose up -d"
echo -e "  3. Check status: /home/$USER/monitor.sh"
echo -e "  4. Create backup: /home/$USER/backup.sh"
echo -e ""
echo -e "${BLUE}Useful commands:${NC}"
echo -e "  - Monitor: /home/$USER/monitor.sh"
echo -e "  - Backup: /home/$USER/backup.sh"
echo -e "  - View logs: docker-compose logs -f"
echo -e "  - Restart: docker-compose restart"
echo -e "  - Stop: docker-compose down"
echo -e ""
echo -e "${YELLOW}⚠️  Don't forget to:${NC}"
echo -e "  - Configure security groups for ports 80, 443, and 8501"
echo -e "  - Set up SSL certificate with certbot if needed"
echo -e "  - Configure monitoring and alerting"
