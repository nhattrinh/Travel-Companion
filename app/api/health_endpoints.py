"""
Health check and system status API endpoints.

Implements comprehensive health monitoring including:
- GET /health: Basic health check with model status
- GET /status: Detailed system information and metrics
- Model availability monitoring
- System resource tracking

Requirements: 6.1, 6.2
"""

from fastapi import APIRouter, Depends, Request
from typing import Dict, Any
import logging
import time
import asyncio
from datetime import datetime, timedelta

# Optional psutil import for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from app.models import (
    HealthCheckResponse,
    SystemHealth,
    ModelType,
    ModelStatus
)
from app.core.dependencies import (
    get_model_manager, 
    get_request_id,
    get_health_monitor,
    get_lifecycle_manager
)
from app.core.models import ModelManager
from app.core.health_monitor import ModelHealthMonitor
from app.core.lifecycle_manager import ModelLifecycleManager
from app.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["health"])

# Application start time for uptime calculation
_app_start_time = time.time()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Basic health check",
    description="Returns basic application health status with model availability"
)
async def health_check(
    request: Request,
    model_manager: ModelManager = Depends(get_model_manager),
    health_monitor: ModelHealthMonitor = Depends(get_health_monitor),
    request_id: str = Depends(get_request_id)
) -> HealthCheckResponse:
    """
    Basic health check endpoint with model status monitoring.
    
    This endpoint provides essential health information including:
    - Overall application status
    - Individual model availability and health
    - Basic system metrics
    - Application version and uptime
    
    Requirements:
    - 6.1: Proper error logging and system health monitoring
    - 6.2: Request/response logging for monitoring
    
    Returns:
        HealthCheckResponse: Basic health status information
    """
    logger.info(
        f"Health check requested",
        extra={
            'request_id': request_id,
            'endpoint': '/health'
        }
    )
    
    try:
        # Check model status
        models_status = {}
        overall_healthy = True
        
        # Check each registered model type
        for model_type in ModelType:
            try:
                model = await model_manager.get_model(model_type)
                if model:
                    # Perform health check on the model
                    is_healthy = await model.health_check()
                    models_status[model_type.value] = is_healthy
                    if not is_healthy:
                        overall_healthy = False
                else:
                    # Model not registered
                    models_status[model_type.value] = False
                    overall_healthy = False
                    
            except Exception as e:
                logger.warning(
                    f"Health check failed for model {model_type.value}: {str(e)}",
                    extra={'request_id': request_id, 'model_type': model_type.value}
                )
                models_status[model_type.value] = False
                overall_healthy = False
        
        # Calculate uptime
        uptime_seconds = int(time.time() - _app_start_time)
        
        # Get basic system metrics
        try:
            if PSUTIL_AVAILABLE:
                memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
            else:
                memory_usage_mb = 0.0
        except Exception:
            memory_usage_mb = 0.0
        
        # Determine overall status
        if overall_healthy:
            status = "healthy"
        elif any(models_status.values()):
            status = "degraded"
        else:
            status = "unhealthy"
        
        # Get concurrent request count (approximate)
        concurrent_requests = getattr(request.app.state, 'active_requests', 0)
        
        response = HealthCheckResponse(
            status=status,
            models_status=models_status,
            uptime_seconds=uptime_seconds,
            version=settings.app_version,
            concurrent_requests=concurrent_requests,
            memory_usage_mb=round(memory_usage_mb, 2)
        )
        
        logger.info(
            f"Health check completed: {status}",
            extra={
                'request_id': request_id,
                'status': status,
                'models_healthy': sum(models_status.values()),
                'total_models': len(models_status),
                'uptime_seconds': uptime_seconds,
                'memory_usage_mb': memory_usage_mb
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(
            f"Health check endpoint failed: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        
        # Return unhealthy status on any error
        return HealthCheckResponse(
            status="unhealthy",
            models_status={},
            uptime_seconds=int(time.time() - _app_start_time),
            version=settings.app_version,
            concurrent_requests=0,
            memory_usage_mb=0.0
        )


@router.get(
    "/status",
    response_model=SystemHealth,
    summary="Detailed system status",
    description="Returns comprehensive system information and performance metrics"
)
async def system_status(
    request: Request,
    model_manager: ModelManager = Depends(get_model_manager),
    health_monitor: ModelHealthMonitor = Depends(get_health_monitor),
    lifecycle_manager: ModelLifecycleManager = Depends(get_lifecycle_manager),
    request_id: str = Depends(get_request_id)
) -> SystemHealth:
    """
    Detailed system status endpoint with comprehensive metrics.
    
    This endpoint provides detailed system information including:
    - Model status with detailed health information
    - System resource usage (CPU, memory, disk)
    - Performance metrics and statistics
    - Configuration information
    - Error statistics and recent issues
    
    Requirements:
    - 6.1: Comprehensive system monitoring and error tracking
    - 6.2: Detailed logging and system information
    
    Returns:
        SystemHealth: Comprehensive system status and metrics
    """
    logger.info(
        f"System status requested",
        extra={
            'request_id': request_id,
            'endpoint': '/status'
        }
    )
    
    try:
        # Collect detailed model information
        models_info = {}
        
        for model_type in ModelType:
            try:
                model = await model_manager.get_model(model_type)
                model_status = await model_manager.get_model_status(model_type)
                
                model_info = {
                    "status": model_status.value,
                    "available": model is not None,
                    "healthy": False,
                    "last_health_check": None,
                    "error_count": 0,
                    "last_error": None
                }
                
                if model:
                    try:
                        model_info["healthy"] = await model.health_check()
                        model_info["last_health_check"] = datetime.utcnow().isoformat()
                    except Exception as e:
                        model_info["last_error"] = str(e)
                        model_info["error_count"] = 1
                
                models_info[model_type.value] = model_info
                
            except Exception as e:
                logger.warning(
                    f"Failed to get status for model {model_type.value}: {str(e)}",
                    extra={'request_id': request_id, 'model_type': model_type.value}
                )
                models_info[model_type.value] = {
                    "status": "failed",
                    "available": False,
                    "healthy": False,
                    "last_health_check": None,
                    "error_count": 1,
                    "last_error": str(e)
                }
        
        # Collect system metrics
        try:
            if PSUTIL_AVAILABLE:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_info = {
                    "total_mb": round(memory.total / 1024 / 1024, 2),
                    "available_mb": round(memory.available / 1024 / 1024, 2),
                    "used_mb": round(memory.used / 1024 / 1024, 2),
                    "percent": memory.percent
                }
                
                # Process-specific memory
                process = psutil.Process()
                process_memory_mb = process.memory_info().rss / 1024 / 1024
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_info = {
                    "total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
                    "free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
                    "used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
                    "percent": round((disk.used / disk.total) * 100, 2)
                }
            else:
                # Fallback values when psutil is not available
                cpu_percent = 0.0
                memory_info = {"total_mb": 0, "available_mb": 0, "used_mb": 0, "percent": 0}
                process_memory_mb = 0.0
                disk_info = {"total_gb": 0, "free_gb": 0, "used_gb": 0, "percent": 0}
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {str(e)}")
            cpu_percent = 0.0
            memory_info = {"total_mb": 0, "available_mb": 0, "used_mb": 0, "percent": 0}
            process_memory_mb = 0.0
            disk_info = {"total_gb": 0, "free_gb": 0, "used_gb": 0, "percent": 0}
        
        # Calculate uptime
        uptime_seconds = int(time.time() - _app_start_time)
        uptime_str = str(timedelta(seconds=uptime_seconds))
        
        # Get comprehensive system status from lifecycle manager
        try:
            system_status_info = lifecycle_manager.get_system_status()
            health_summary = health_monitor.get_health_summary()
            system_health_info = health_monitor.get_system_health()
        except Exception as e:
            logger.warning(f"Failed to get comprehensive system status: {str(e)}")
            system_status_info = {}
            health_summary = {}
            system_health_info = {}
        
        # Get error statistics (from error handler if available)
        error_stats = {}
        try:
            from app.core.error_handlers import error_handler
            error_stats = error_handler.get_error_statistics()
        except Exception:
            error_stats = {
                "total_errors": 0,
                "errors_last_hour": 0,
                "most_common_errors": []
            }
        
        # Performance metrics
        performance_metrics = {
            "average_response_time_ms": 0.0,  # Would be calculated from request logs
            "requests_per_minute": 0.0,      # Would be calculated from request logs
            "active_connections": getattr(request.app.state, 'active_requests', 0),
            "total_requests_processed": 0     # Would be tracked in middleware
        }
        
        # Configuration info (non-sensitive)
        config_info = {
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "debug_mode": settings.debug,
            "log_level": settings.log_level,
            "max_concurrent_requests": settings.max_concurrent_requests,
            "api_prefix": getattr(settings, 'api_prefix', '/api/v1')
        }
        
        # Determine overall system health
        healthy_models = sum(1 for info in models_info.values() if info["healthy"])
        total_models = len(models_info)
        
        if total_models == 0:
            overall_status = "initializing"
        elif healthy_models == total_models:
            overall_status = "healthy"
        elif healthy_models > 0:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        # Create comprehensive status response
        system_health = SystemHealth(
            status=overall_status,
            timestamp=datetime.utcnow(),
            uptime_seconds=uptime_seconds,
            uptime_string=uptime_str,
            version=settings.app_version,
            
            # Model information
            models=models_info,
            healthy_models=healthy_models,
            total_models=total_models,
            
            # System resources
            cpu_usage_percent=cpu_percent,
            memory=memory_info,
            process_memory_mb=round(process_memory_mb, 2),
            disk=disk_info,
            
            # Performance metrics
            performance=performance_metrics,
            
            # Error tracking
            errors=error_stats,
            
            # Configuration
            configuration=config_info
        )
        
        logger.info(
            f"System status completed: {overall_status}",
            extra={
                'request_id': request_id,
                'status': overall_status,
                'healthy_models': healthy_models,
                'total_models': total_models,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_info["percent"],
                'uptime_seconds': uptime_seconds
            }
        )
        
        return system_health
        
    except Exception as e:
        logger.error(
            f"System status endpoint failed: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        
        # Return minimal status on error
        return SystemHealth(
            status="error",
            timestamp=datetime.utcnow(),
            uptime_seconds=int(time.time() - _app_start_time),
            uptime_string="unknown",
            version=settings.app_version,
            models={},
            healthy_models=0,
            total_models=0,
            cpu_usage_percent=0.0,
            memory={"total_mb": 0, "available_mb": 0, "used_mb": 0, "percent": 0},
            process_memory_mb=0.0,
            disk={"total_gb": 0, "free_gb": 0, "used_gb": 0, "percent": 0},
            performance={
                "average_response_time_ms": 0.0,
                "requests_per_minute": 0.0,
                "active_connections": 0,
                "total_requests_processed": 0
            },
            errors={
                "total_errors": 0,
                "errors_last_hour": 0,
                "most_common_errors": []
            },
            configuration={
                "app_name": settings.app_name,
                "app_version": settings.app_version,
                "debug_mode": settings.debug
            }
        )