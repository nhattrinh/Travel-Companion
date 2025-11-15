"""
Core interfaces and base classes for the menu translation backend.
Provides model management, health monitoring, lifecycle management, and caching.
"""

from .models import BaseModel, ModelManager
from .health_monitor import ModelHealthMonitor, HealthCheckResult, HealthMetrics
from .lifecycle_manager import ModelLifecycleManager, MockModelImplementation
from .cache_client import CacheClient, get_cache_client, close_cache_client, CacheClientError

__all__ = [
    "BaseModel",
    "ModelManager", 
    "ModelHealthMonitor",
    "HealthCheckResult",
    "HealthMetrics",
    "ModelLifecycleManager",
    "MockModelImplementation",
    "CacheClient",
    "get_cache_client",
    "close_cache_client",
    "CacheClientError"
]