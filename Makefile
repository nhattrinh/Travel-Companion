# Makefile for Menu Translation Backend - Local Development

.PHONY: help build up down restart logs ps clean test shell redis-cli health stats

help: ## Show this help message
	@echo "Menu Translation Backend - Local Development"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

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

shell: ## Access application container shell
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