# Makefile for Menu Translation Backend Docker operations

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
	@echo "Menu Translation Backend - Docker Operations"
	@echo ""
	@echo "Usage: make [target] [ENV=environment]"
	@echo ""
	@echo "Environments:"
	@echo "  development (default) - Development environment with hot reload"
	@echo "  staging              - Staging environment"
	@echo "  production           - Production environment with Nginx"
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
	@echo "Building images for $(ENV) environment..."
	@ENVIRONMENT=$(ENV) $(COMPOSE_CMD) build

up: ## Start services
	@echo "Starting services for $(ENV) environment..."
	@ENVIRONMENT=$(ENV) $(COMPOSE_CMD) up -d
	@echo "Services started. Checking health..."
	@sleep 10
	@$(COMPOSE_CMD) ps

down: ## Stop and remove services
	@echo "Stopping services for $(ENV) environment..."
	@$(COMPOSE_CMD) down

restart: ## Restart services
	@echo "Restarting services for $(ENV) environment..."
	@$(COMPOSE_CMD) restart

logs: ## Show service logs
	@$(COMPOSE_CMD) logs -f

ps: ## Show service status
	@$(COMPOSE_CMD) ps

clean: ## Clean up containers, networks, and volumes
	@echo "Cleaning up Docker resources..."
	@$(COMPOSE_CMD) down -v --remove-orphans
	@docker system prune -f

test: ## Run tests in Docker container
	@echo "Running tests..."
	@docker-compose -f docker-compose.yml run --rm menu-translation-backend python -m pytest -v

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

dev-redis: ## Access Redis CLI in development
	@docker-compose exec redis redis-cli

# Production specific targets
prod-build: ## Build production image
	@$(MAKE) ENV=production build

prod-deploy: ## Deploy to production (build + up)
	@$(MAKE) ENV=production build
	@$(MAKE) ENV=production up

prod-logs: ## Show production logs
	@$(MAKE) ENV=production logs

# Utility targets
health: ## Check service health
	@echo "Checking service health..."
	@curl -f http://localhost:8000/health || echo "Service not responding"

backup-redis: ## Backup Redis data
	@echo "Backing up Redis data..."
	@docker run --rm -v menu-translation_redis_data:/data -v $(PWD):/backup alpine tar czf /backup/redis-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /data .
	@echo "Backup completed: redis-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz"

restore-redis: ## Restore Redis data (requires BACKUP_FILE variable)
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Error: BACKUP_FILE variable required"; exit 1; fi
	@echo "Restoring Redis data from $(BACKUP_FILE)..."
	@$(COMPOSE_CMD) stop redis
	@docker run --rm -v menu-translation_redis_data:/data -v $(PWD):/backup alpine tar xzf /backup/$(BACKUP_FILE) -C /data
	@$(COMPOSE_CMD) start redis
	@echo "Restore completed"

stats: ## Show container resource usage
	@docker stats --no-stream

# Environment setup targets
setup-dev: ## Setup development environment
	@echo "Setting up development environment..."
	@cp .env.development .env
	@$(MAKE) dev-build
	@$(MAKE) dev

setup-staging: ## Setup staging environment
	@echo "Setting up staging environment..."
	@cp .env.staging .env
	@$(MAKE) ENV=staging build
	@$(MAKE) staging

setup-prod: ## Setup production environment
	@echo "Setting up production environment..."
	@cp .env.production .env
	@$(MAKE) ENV=production build
	@$(MAKE) prod

# Maintenance targets
update: ## Update and rebuild services
	@echo "Updating services for $(ENV) environment..."
	@$(COMPOSE_CMD) pull
	@$(COMPOSE_CMD) build --no-cache
	@$(COMPOSE_CMD) up -d

reset: ## Reset environment (clean + build + up)
	@echo "Resetting $(ENV) environment..."
	@$(MAKE) ENV=$(ENV) clean
	@$(MAKE) ENV=$(ENV) build
	@$(MAKE) ENV=$(ENV) up