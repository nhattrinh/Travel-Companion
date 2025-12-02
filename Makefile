# Makefile for Menu Translation Backend - Local Development

.PHONY: help build up down restart logs ps clean test dev prod staging dev-backend lint migrate perf-smoke

# Default environment
ENV ?= development

# Docker compose files
COMPOSE_DEV = docker-compose.yml
COMPOSE_STAGING = docker-compose.yml -f docker-compose.staging.yml
COMPOSE_PROD = docker-compose.yml -f docker-compose.production.yml

# Set compose command based on environment
ifeq ($(ENV),staging)
    COMPOSE_CMD = docker-compose -f $(COMPOSE_STAGING)
else ifeq ($(ENV),production)
    COMPOSE_CMD = docker-compose -f $(COMPOSE_PROD)
else
    COMPOSE_CMD = docker-compose -f $(COMPOSE_DEV)
endif

help: ## Show this help message
	@echo "Menu Translation Backend - Local Development"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
travel: ## Start Travel Companion stack (api + postgres + redis)
	@echo "Starting Travel Companion stack using docker-compose.travel.yml"
	@export $$(grep -v '^#' .env.development | xargs) && docker compose -f docker-compose.travel.yml up -d
	@docker compose -f docker-compose.travel.yml ps

travel-down: ## Stop Travel Companion stack
	@docker compose -f docker-compose.travel.yml down

travel-logs: ## Tail API logs
	@docker compose -f docker-compose.travel.yml logs -f api

travel-restart: ## Restart API container
	@docker compose -f docker-compose.travel.yml restart api


build: ## Build Docker images
	@echo "Building development images..."
	@docker-compose build

up: ## Start services
	@echo "Starting services..."
	@docker-compose up -d
	@echo "Services started. Checking health..."
	@sleep 5
	@docker-compose ps

down: ## Stop and remove services
	@echo "Stopping services..."
	@docker-compose down

restart: ## Restart services
	@echo "Restarting services..."
	@docker-compose restart

logs: ## Show service logs (follow mode)
	@docker-compose logs -f

ps: ## Show service status
	@docker-compose ps

clean: ## Clean up containers, networks, and volumes
	@echo "Cleaning up Docker resources..."
	@docker-compose down -v --remove-orphans
	@docker system prune -f

test: ## Run tests in Docker container
	@echo "Running tests..."
	@docker-compose run --rm menu-translation-backend python -m pytest -v

dev-backend: ## Run backend locally without Docker (uvicorn)
	@echo "Starting local backend (uvicorn)..."
	@python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

lint: ## Run code linters & format check
	@echo "Running lint & type checks..."
	@black --check app || echo "Black formatting differences detected"
	@flake8 app
	@mypy app

migrate: ## Run Alembic migrations (upgrade head)
	@echo "Running Alembic migrations..."
	@alembic upgrade head || echo "Alembic not configured yet"

perf-smoke: ## Run performance smoke tests
	@echo "Running performance smoke tests..."
	@pytest -q tests/perf || echo "Perf tests not present yet"

dev: ## Start development environment
	@$(MAKE) ENV=development up

staging: ## Start staging environment
	@$(MAKE) ENV=staging up

prod: ## Start production environment
	@$(MAKE) ENV=production up

# Development specific targets
dev-build: ## Build development image
	@$(MAKE) ENV=development build

dev-logs: ## Show development logs
	@$(MAKE) ENV=development logs

dev-shell: ## Access development container shell
	@docker-compose exec menu-translation-backend bash

redis-cli: ## Access Redis CLI
	@docker-compose exec redis redis-cli

health: ## Check service health
	@echo "Checking service health..."
	@curl -f http://localhost:8000/health || echo "Service not responding"

stats: ## Show container resource usage
	@docker stats --no-stream

# Combined commands
dev: build up ## Build and start development environment

reset: clean build up ## Reset environment (clean + build + up)