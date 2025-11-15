"""
FastAPI application setup with dependency injection.
Implements Requirement 1.1 - FastAPI application initialization.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import uuid
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

from config import get_settings

# Get application settings
settings = get_settings()
from app.core.models import ModelManager
from app.core.dependencies import get_model_manager
from app.middleware import AuthenticationMiddleware, RateLimitMiddleware
from app.core.error_handlers import setup_error_handlers


# Configure logging based on settings
log_config = {
    "level": getattr(logging, settings.log_level.value),
    "format": settings.log_format,
}

if settings.log_file:
    log_config["filename"] = settings.log_file

logging.basicConfig(**log_config)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management with service container.
    Handles startup and shutdown events with proper service lifecycle.
    Implements Requirements 1.1, 1.3 - service lifecycle management.
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    try:
        # Initialize service container
        from app.core.dependencies import service_container
        await service_container.initialize_services()
        app.state.service_container = service_container
        
        logger.info("Application startup complete")
        
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}", exc_info=True)
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down application")
        
        try:
            # Cleanup service container
            if hasattr(app.state, 'service_container'):
                await app.state.service_container.cleanup_services()
            
            logger.info("Application shutdown complete")
            
        except Exception as e:
            logger.error(f"Application shutdown failed: {e}", exc_info=True)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan
    )
    
    # Add CORS middleware with configuration
    app.add_middleware(
        CORSMiddleware,
        **settings.get_cors_config()
    )
    
    # Add rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        max_concurrent_per_client=5,
        max_requests_per_minute=settings.security.rate_limit_requests_per_minute,
        global_max_concurrent=settings.concurrency.max_concurrent_requests
    )
    
    # Add authentication middleware
    app.add_middleware(AuthenticationMiddleware)
    
    # Set up comprehensive error handlers
    setup_error_handlers(app)
    
    # Add request logging middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        """Log requests and responses with timing information."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request {request_id} started: {request.method} {request.url.path}",
            extra={
                'request_id': request_id,
                'method': request.method,
                'path': request.url.path,
                'client_ip': request.client.host if request.client else 'unknown'
            }
        )
        
        response = await call_next(request)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log response
        logger.info(
            f"Request {request_id} completed: {response.status_code} ({processing_time:.2f}ms)",
            extra={
                'request_id': request_id,
                'status_code': response.status_code,
                'processing_time_ms': processing_time
            }
        )
        
        return response
    

    
    # Include API routers
    from app.api.menu_endpoints import router as menu_router
    from app.api.health_endpoints import router as health_router
    from app.api.batch_endpoints import router as batch_router
    app.include_router(menu_router)
    app.include_router(health_router)
    app.include_router(batch_router)
    
    return app


# Create application instance
app = create_app()


@app.get("/")
async def root():
    """Root endpoint for basic health check."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with comprehensive system status."""
    from app.core.error_handlers import error_handler
    
    # Get service container
    service_container = getattr(app.state, 'service_container', None)
    
    if not service_container:
        return {
            "status": "unhealthy",
            "message": "Service container not initialized",
            "version": settings.app_version,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        # Get comprehensive system status from lifecycle manager
        lifecycle_manager = service_container.get_lifecycle_manager()
        system_status = lifecycle_manager.get_system_status()
        
        # Get model manager status
        model_manager = service_container.get_model_manager()
        models_status = {}
        
        for model_type in model_manager._models:
            model = model_manager._models[model_type]
            if model:
                try:
                    models_status[model_type.value] = await model.health_check()
                except Exception:
                    models_status[model_type.value] = False
            else:
                models_status[model_type.value] = False
        
        # Determine overall status
        if not models_status:
            overall_status = "healthy"  # No models registered yet
        elif all(models_status.values()):
            overall_status = "healthy"
        elif any(models_status.values()):
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        # Get error statistics
        error_stats = error_handler.get_error_statistics()
        
        return {
            "status": overall_status,
            "models_status": models_status,
            "system_status": system_status,
            "version": settings.app_version,
            "error_statistics": error_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}",
            "version": settings.app_version,
            "timestamp": datetime.utcnow().isoformat()
        }