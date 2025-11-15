#!/bin/bash
# Deployment script for Menu Translation Backend

set -e

# Default values
ENVIRONMENT="development"
ACTION="up"
DETACH=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        --no-detach)
            DETACH=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -e, --env ENVIRONMENT    Target environment (development, staging, production)"
            echo "  -a, --action ACTION      Docker compose action (up, down, restart, logs)"
            echo "  --no-detach              Don't run in detached mode"
            echo "  -h, --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate environment
case $ENVIRONMENT in
    development|staging|production)
        ;;
    *)
        echo "Error: Invalid environment '$ENVIRONMENT'. Must be one of: development, staging, production"
        exit 1
        ;;
esac

# Set compose files based on environment
COMPOSE_FILES="-f docker-compose.yml"

case $ENVIRONMENT in
    development)
        # Use override file for development (loaded automatically)
        ;;
    staging)
        COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.staging.yml"
        ;;
    production)
        COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.production.yml"
        ;;
esac

# Set environment variables
export ENVIRONMENT="$ENVIRONMENT"
export COMPOSE_PROJECT_NAME="menu-translation-${ENVIRONMENT}"

echo "Deploying Menu Translation Backend"
echo "Environment: $ENVIRONMENT"
echo "Action: $ACTION"
echo "Compose files: $COMPOSE_FILES"

# Execute the requested action
case $ACTION in
    up)
        echo "Starting services..."
        if [[ "$DETACH" == "true" ]]; then
            docker-compose $COMPOSE_FILES up -d
        else
            docker-compose $COMPOSE_FILES up
        fi
        
        if [[ "$DETACH" == "true" ]]; then
            echo "✓ Services started successfully"
            echo "Checking service health..."
            sleep 10
            docker-compose $COMPOSE_FILES ps
        fi
        ;;
    down)
        echo "Stopping services..."
        docker-compose $COMPOSE_FILES down
        echo "✓ Services stopped successfully"
        ;;
    restart)
        echo "Restarting services..."
        docker-compose $COMPOSE_FILES restart
        echo "✓ Services restarted successfully"
        ;;
    logs)
        echo "Showing logs..."
        docker-compose $COMPOSE_FILES logs -f
        ;;
    ps)
        echo "Service status:"
        docker-compose $COMPOSE_FILES ps
        ;;
    build)
        echo "Building services..."
        docker-compose $COMPOSE_FILES build
        echo "✓ Build completed successfully"
        ;;
    *)
        echo "Error: Invalid action '$ACTION'. Must be one of: up, down, restart, logs, ps, build"
        exit 1
        ;;
esac