# Menu Translation Backend

A FastAPI-based microservice that provides intelligent menu translation capabilities through a pluggable AI model architecture.

## Project Structure

```
app/
├── __init__.py
├── main.py                 # FastAPI application setup
├── api/                    # API endpoints and routers
│   └── __init__.py
├── config/                 # Configuration management
│   ├── __init__.py
│   └── settings.py         # Pydantic settings
├── core/                   # Core interfaces and base classes
│   ├── __init__.py
│   ├── models.py           # Base model interfaces and ModelManager
│   └── dependencies.py     # Dependency injection setup
├── models/                 # Data models and schemas
│   └── __init__.py
└── services/               # Business logic services
    └── __init__.py
```

## Features

- **Pluggable Model Architecture**: Easy integration and hot-swapping of AI models
- **FastAPI Framework**: Modern, fast web framework with automatic API documentation
- **Dependency Injection**: Clean separation of concerns and testable code
- **Configuration Management**: Environment-based configuration with Pydantic
- **Comprehensive Logging**: Request/response logging with unique request IDs
- **Error Handling**: Global exception handling with structured error responses

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

3. Access the API documentation at: http://localhost:8000/docs

## Configuration

The application uses environment variables for configuration. Create a `.env` file with your settings:

```env
DEBUG=true
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=10
REDIS_URL=redis://localhost:6379
```

## Model Management

The system supports pluggable AI models through the `ModelManager` class:

- **Model Registration**: Register models with the system
- **Hot-Swapping**: Replace models without downtime
- **Health Monitoring**: Monitor model status and handle failures
- **Lifecycle Management**: Automatic initialization and cleanup