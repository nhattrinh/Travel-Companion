# Travel Companion - Menu Translation Backend

A FastAPI-based microservice that provides intelligent menu translation capabilities with image processing, OCR, and multi-language support through a pluggable AI model architecture.

## Project Overview

This backend service processes food menu images, extracts text using OCR, and provides intelligent translations. It features a robust processing pipeline with caching, rate limiting, batch processing, and comprehensive health monitoring.

## Project Structure

```
app/
├── __init__.py
├── main.py                          # FastAPI application setup with lifespan management
├── api/                             # API endpoints and routers
│   ├── __init__.py
│   ├── health_endpoints.py          # Health check and monitoring endpoints
│   ├── menu_endpoints.py            # Menu translation endpoints
│   └── batch_endpoints.py           # Batch processing endpoints
├── config/                          # Configuration management
│   ├── __init__.py
│   ├── settings.py                  # Pydantic settings with environment support
│   └── loader.py                    # Configuration loader utilities
├── core/                            # Core infrastructure and base classes
│   ├── __init__.py
│   ├── models.py                    # Base model interfaces and ModelManager
│   ├── dependencies.py              # Dependency injection setup
│   ├── exceptions.py                # Custom exception classes
│   ├── error_handlers.py            # Global error handling
│   ├── cache_client.py              # Redis caching client
│   ├── concurrency_manager.py       # Async concurrency control
│   ├── health_monitor.py            # System health monitoring
│   ├── lifecycle_manager.py         # Service lifecycle management
│   └── processing_pipeline.py       # Image processing pipeline
├── middleware/                      # HTTP middleware
│   ├── __init__.py
│   ├── auth.py                      # Authentication middleware
│   └── rate_limit.py                # Rate limiting middleware
├── models/                          # Data models and schemas
│   ├── __init__.py
│   ├── api_models.py                # API request/response models
│   └── internal_models.py           # Internal service models
└── services/                        # Business logic services
    ├── __init__.py
    ├── food_image_service.py        # Food image analysis service
    ├── image_processor.py           # Image preprocessing service
    ├── ocr_service.py               # OCR text extraction service
    └── translation_service.py       # Translation service
```

## Features

- **🔌 Pluggable Model Architecture**: Easy integration and hot-swapping of AI models
- **⚡ FastAPI Framework**: Modern, fast web framework with automatic API documentation
- **🖼️ Image Processing Pipeline**: Intelligent menu image preprocessing and enhancement
- **📝 OCR Integration**: Advanced text extraction from food menu images
- **🌐 Multi-Language Translation**: Support for multiple languages with translation service
- **⚙️ Batch Processing**: Efficient handling of multiple menu items simultaneously
- **🔄 Redis Caching**: High-performance caching layer for improved response times
- **🚦 Rate Limiting**: Request throttling to prevent abuse
- **🔐 Authentication**: API key-based authentication middleware
- **📊 Health Monitoring**: Comprehensive health checks and system monitoring
- **🐳 Docker Support**: Full containerization with Docker Compose for all environments
- **🔧 Dependency Injection**: Clean separation of concerns and testable code
- **📝 Comprehensive Logging**: Structured logging with request tracking
- **⚠️ Error Handling**: Global exception handling with structured error responses

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- Redis (for caching)

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env.development` file:
   ```env
   ENVIRONMENT=development
   DEBUG=true
   LOG_LEVEL=INFO
   MAX_CONCURRENT_REQUESTS=10
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_DB=0
   ```

3. **Run with Python:**
   ```bash
   python run.py
   ```

4. **Access the API documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker Deployment

#### Development Environment
```bash
# Build and start services
make dev

# View logs
make logs

# Stop services
make down
```

#### Staging Environment
```bash
# Deploy to staging
make staging

# View staging logs
make logs ENV=staging
```

#### Production Environment
```bash
# Deploy to production with Nginx
make prod

# View production logs
make logs ENV=production
```

### Available Make Commands

```bash
make help          # Show all available commands
make build         # Build Docker images
make up            # Start all services
make down          # Stop all services
make restart       # Restart all services
make logs          # View service logs
make ps            # List running containers
make clean         # Clean up containers and volumes
make test          # Run tests
```

## API Endpoints

### Health & Monitoring
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed system health status

### Menu Translation
- `POST /menu/translate` - Translate a single menu item
- `POST /menu/image` - Process and translate menu from image

### Batch Processing
- `POST /batch/translate` - Process multiple menu items in batch

## Configuration

The application supports multiple environment configurations:

- **Development** (`.env.development`): Hot reload, verbose logging
- **Staging** (`.env.staging`): Production-like setup for testing
- **Production** (`.env.production`): Optimized for performance and security

### Key Configuration Options

```env
# Application
ENVIRONMENT=development
DEBUG=true
APP_NAME=Menu Translation Backend
APP_VERSION=1.0.0

# Server
HOST_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Redis Cache
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Concurrency
MAX_CONCURRENT_REQUESTS=10
```

## Testing

Run the test suite:

```bash
# Unit tests
pytest test_basic_setup.py

# Integration tests
pytest test_integration_pipeline.py

# Model infrastructure tests
pytest test_model_infrastructure.py

# All tests
pytest
```

## Model Management

The system supports pluggable AI models through the `ModelManager` class:

- **Model Registration**: Register models dynamically with the system
- **Hot-Swapping**: Replace models without downtime
- **Health Monitoring**: Monitor model status and handle failures gracefully
- **Lifecycle Management**: Automatic initialization and cleanup
- **Caching**: Intelligent result caching for improved performance

## Architecture Highlights

### Processing Pipeline
The image processing pipeline includes:
1. Image validation and preprocessing
2. OCR text extraction
3. Language detection
4. Translation processing
5. Result caching and delivery

### Concurrency Management
- Async/await for non-blocking I/O
- Semaphore-based concurrency control
- Request throttling and rate limiting

### Error Handling
- Custom exception hierarchy
- Global error handlers
- Structured error responses
- Request tracking with unique IDs

## Monitoring & Logging

The application includes comprehensive monitoring:

- **Health Checks**: Kubernetes-ready liveness and readiness probes
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **System Metrics**: CPU, memory, and service status monitoring
- **Redis Health**: Cache connectivity and performance monitoring

## Docker Infrastructure

The project includes complete Docker infrastructure:

- **Dockerfile**: Multi-stage builds for development and production
- **docker-compose.yml**: Base configuration with Redis
- **docker-compose.staging.yml**: Staging-specific overrides
- **docker-compose.production.yml**: Production setup with Nginx reverse proxy
- **Nginx Configuration**: Optimized reverse proxy and load balancing

## Development Guidelines

1. **Code Style**: Follow PEP 8 guidelines
2. **Type Hints**: Use type annotations for better IDE support
3. **Testing**: Write tests for new features
4. **Documentation**: Update docstrings and README as needed
5. **Error Handling**: Use custom exceptions from `core.exceptions`
6. **Logging**: Use structured logging with appropriate log levels

## Contributing

1. Create a feature branch
2. Make your changes
3. Write/update tests
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions, please open an issue in the repository.