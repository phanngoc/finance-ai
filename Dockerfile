# Sử dụng Python 3.9 slim image
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Node.js (cần thiết cho Playwright)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Copy requirements và cài đặt Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cài đặt Playwright browsers
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy source code
COPY . .

# Expose port
EXPOSE 8501

# Thiết lập environment variables
ENV BROWSERLESS_URL=ws://browserless:3000
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Command để chạy ứng dụng
CMD ["streamlit", "run", "fireant_browserless.py", "--server.port=8501", "--server.address=0.0.0.0"] 