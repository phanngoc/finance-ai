# LSTM Stock Prediction - Makefile
# Provides convenient commands for development and deployment

.PHONY: help install run build deploy clean test lint format

# Default target
help:
	@echo "LSTM Stock Prediction - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install     Install dependencies"
	@echo "  run         Run the application locally"
	@echo "  test        Run tests"
	@echo "  lint        Run linting"
	@echo "  format      Format code"
	@echo ""
	@echo "Docker:"
	@echo "  build       Build Docker image"
	@echo "  run-docker  Run with Docker Compose"
	@echo "  stop        Stop Docker containers"
	@echo "  clean       Clean Docker resources"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy      Deploy to EC2"
	@echo "  setup-ec2   Setup EC2 instance"
	@echo "  quick       Quick local deployment"
	@echo ""
	@echo "Monitoring:"
	@echo "  logs        View application logs"
	@echo "  status      Check application status"
	@echo "  monitor     Run monitoring script"

# Development commands
install:
	pip install -r requirements.txt

run:
	streamlit run app.py

test:
	python -m pytest tests/ -v

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black .
	isort .

# Docker commands
build:
	./scripts/build.sh

run-docker:
	docker-compose up -d

stop:
	docker-compose down

clean:
	docker-compose down --remove-orphans
	docker system prune -f
	docker volume prune -f

# Deployment commands
deploy:
	@read -p "Enter EC2 host: " host; \
	read -p "Enter EC2 user (default: ubuntu): " user; \
	user=$${user:-ubuntu}; \
	read -p "Enter SSH key path (default: ~/.ssh/id_rsa): " key; \
	key=$${key:-~/.ssh/id_rsa}; \
	./scripts/deploy.sh latest $$host $$user $$key

setup-ec2:
	./scripts/setup-ec2.sh

quick:
	./scripts/quick-deploy.sh

# Monitoring commands
logs:
	docker-compose logs -f

status:
	docker-compose ps

monitor:
	./monitor.sh

# Utility commands
backup:
	./backup.sh

restart:
	docker-compose restart

update:
	git pull
	docker-compose down
	docker-compose up -d --build

# Development setup
setup:
	pip install -r requirements.txt
	pre-commit install
	@echo "Development environment setup complete!"

# Production setup
prod-setup:
	./scripts/setup-ec2.sh
	@echo "Production environment setup complete!"

# Health check
health:
	@curl -f http://localhost:8501/_stcore/health && echo "✅ Application is healthy" || echo "❌ Application is not responding"

# Full deployment pipeline
deploy-full: build deploy
	@echo "Full deployment completed!"

# Clean deployment
deploy-clean: clean build deploy
	@echo "Clean deployment completed!"
