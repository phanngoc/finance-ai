# LSTM Stock Prediction - Standalone Version

Ứng dụng dự đoán giá cổ phiếu Việt Nam sử dụng mạng neural LSTM với giao diện Streamlit.

## 🚀 Tính năng chính

- **Dự đoán LSTM**: Dự đoán giá cổ phiếu sử dụng mạng neural LSTM
- **Dự đoán tương lai**: Dự đoán giá cho 10 ngày kinh doanh tiếp theo
- **Phân tích xu hướng**: Xác định hướng và cường độ xu hướng giá
- **Độ chính xác mô hình**: Đánh giá hiệu suất dự đoán
- **Giao diện thân thiện**: Streamlit web app dễ sử dụng

## 📁 Cấu trúc Project

```
lstm_predict/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose configuration
├── nginx/                   # Nginx configuration
│   ├── nginx.conf
│   └── conf.d/default.conf
├── scripts/                 # Deployment scripts
│   ├── build.sh            # Build script
│   ├── deploy.sh           # Deploy to AWS EC2
│   ├── setup-ec2.sh        # EC2 setup script
│   └── quick-deploy.sh     # Quick local deployment
└── utils/                   # Utility modules
    ├── __init__.py
    ├── data_processing.py   # Data loading and processing
    ├── lstm_model.py        # LSTM model utilities
    └── plotting.py          # Chart and visualization
```

## 🛠️ Cài đặt và Chạy

### 1. Cài đặt Local

```bash
# Clone repository
git clone <repository-url>
cd lstm_predict

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy ứng dụng
streamlit run app.py
```

### 2. Chạy với Docker

```bash
# Quick deployment
./scripts/quick-deploy.sh

# Hoặc manual
docker-compose up -d
```

### 3. Deploy lên AWS EC2

```bash
# Setup EC2 instance
./scripts/setup-ec2.sh

# Deploy application
./scripts/deploy.sh latest ec2-xx-xx-xx-xx.compute-1.amazonaws.com ubuntu ~/.ssh/id_rsa
```

## 📊 Sử dụng

1. **Nhập mã cổ phiếu** trong sidebar (VD: ACB, VCB, VHM, FPT)
2. **Chọn khoảng thời gian** phân tích
3. **Cấu hình tham số LSTM** (lookback days, epochs)
4. **Nhấn "Bắt đầu phân tích"** để xem kết quả

## 🔧 Cấu hình

### Environment Variables

Tạo file `.env` (tùy chọn):

```bash
# Application settings
APP_NAME=lstm-stock-prediction
VERSION=latest

# Streamlit settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Docker settings
COMPOSE_PROJECT_NAME=lstm-stock-prediction
```

### Docker Configuration

- **Port**: 8501 (Streamlit), 80 (Nginx)
- **Memory**: Tối thiểu 2GB RAM
- **Storage**: Tối thiểu 5GB disk space

## 🚀 Deployment Options

### 1. Local Development

```bash
# Chạy trực tiếp
streamlit run app.py

# Hoặc với Docker
docker-compose up -d
```

### 2. Production (AWS EC2)

```bash
# Setup EC2
./scripts/setup-ec2.sh

# Deploy
./scripts/deploy.sh latest your-ec2-host ubuntu ~/.ssh/key
```

### 3. Docker Registry

```bash
# Build và push
./scripts/build.sh latest your-registry.com

# Deploy từ registry
./scripts/deploy.sh latest your-ec2-host ubuntu ~/.ssh/key your-registry.com
```

## 📈 Performance Optimization

### 1. Model Optimization

- **Lookback days**: 60 (cân bằng giữa accuracy và performance)
- **Epochs**: 20 (đủ để training mà không overfitting)
- **Batch size**: 32 (tối ưu cho memory)

### 2. Docker Optimization

- **Multi-stage build**: Giảm image size
- **Health checks**: Đảm bảo service stability
- **Resource limits**: Tránh memory leak

### 3. Nginx Optimization

- **Gzip compression**: Giảm bandwidth
- **Caching**: Tăng tốc độ response
- **Rate limiting**: Bảo vệ khỏi abuse

## 🔍 Monitoring

### Health Checks

```bash
# Check application status
curl http://localhost:8501/_stcore/health

# Check Docker containers
docker-compose ps

# View logs
docker-compose logs -f
```

### Monitoring Script

```bash
# Run monitoring script (on EC2)
./monitor.sh
```

## 🛡️ Security

### 1. Docker Security

- **Non-root user**: Chạy với user không có quyền root
- **Read-only filesystem**: Giới hạn quyền ghi
- **Resource limits**: Giới hạn CPU và memory

### 2. Nginx Security

- **Security headers**: XSS, CSRF protection
- **Rate limiting**: Chống DDoS
- **SSL/TLS**: Mã hóa traffic

### 3. EC2 Security

- **Firewall**: Chỉ mở ports cần thiết
- **Fail2ban**: Chống brute force
- **Regular updates**: Cập nhật security patches

## 📋 Troubleshooting

### Common Issues

1. **Out of memory**
   ```bash
   # Increase Docker memory limit
   docker-compose down
   docker system prune -a
   docker-compose up -d
   ```

2. **Port already in use**
   ```bash
   # Kill process using port 8501
   sudo lsof -ti:8501 | xargs kill -9
   ```

3. **Docker build fails**
   ```bash
   # Clear Docker cache
   docker system prune -a
   docker-compose build --no-cache
   ```

### Logs

```bash
# Application logs
docker-compose logs -f lstm-app

# Nginx logs
docker-compose logs -f nginx

# System logs (on EC2)
sudo journalctl -u lstm-stock-prediction.service -f
```

## 🔄 Backup và Recovery

### Backup

```bash
# Manual backup
./backup.sh

# Automated backup (cron)
0 2 * * * /home/ubuntu/backup.sh
```

### Recovery

```bash
# Restore from backup
cd /home/ubuntu/lstm-stock-prediction
tar -xzf /home/ubuntu/backups/lstm-stock-prediction_YYYYMMDD_HHMMSS.tar.gz
docker-compose up -d
```

## 📝 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## ⚠️ Lưu ý quan trọng

- **Không phải lời khuyên đầu tư**: Ứng dụng chỉ mang tính chất tham khảo
- **Có rủi ro**: Thị trường chứng khoán luôn có rủi ro cao
- **Tự nghiên cứu**: Luôn tự nghiên cứu kỹ trước khi đầu tư
- **Tham khảo chuyên gia**: Nên tham khảo ý kiến chuyên gia tài chính

## 📞 Support

Nếu gặp vấn đề, vui lòng tạo issue trên GitHub hoặc liên hệ qua email.
