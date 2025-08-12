.PHONY: help install test lint format clean docker-build docker-run deploy

help:
	@echo "Available commands:"
	@echo "  install       Install dependencies"
	@echo "  test          Run tests"
	@echo "  test-integration Run integration tests"
	@echo "  lint          Run linting"
	@echo "  format        Format code"
	@echo "  clean         Clean build artifacts"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run Docker container locally"
	@echo "  deploy        Deploy to DigitalOcean"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v -m "not integration"

test-integration:
	INTEGRATION_TEST=1 pytest tests/ -v -m "integration"

test-all:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/

docker-build:
	docker build -t singlestore-mcp-server:latest .

docker-run:
	docker run --rm -p 8080:8080 \
		--env-file .env \
		singlestore-mcp-server:latest

deploy:
	doctl apps create --spec .do/app.yaml

# Development shortcuts
dev:
	python -m singlestore_mcp.server

dev-remote:
	DEPLOYMENT_MODE=remote python -m singlestore_mcp.server --remote

# Generate requirements.txt from pyproject.toml
requirements:
	pip-compile pyproject.toml -o requirements.txt --upgrade