"""
Metrics endpoint for observability and monitoring.
Implements T119 - System metrics for observability.
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
import time
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

from app.core.dependencies import get_current_user
from app.models.internal_models import User
from app.core.metrics_translation import (
    snapshot_latency_stats as translation_latency,
)
from app.core.metrics_navigation_phrase import (
    snapshot_metrics as nav_phrase_metrics,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/metrics", tags=["metrics"])


@dataclass
class EndpointMetrics:
    """Metrics tracking for a single endpoint."""
    request_count: int = 0
    error_count: int = 0
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_request(self, latency: float, is_error: bool = False):
        """Record a request with its latency."""
        self.request_count += 1
        if is_error:
            self.error_count += 1
        self.latencies.append(latency)
    
    def get_stats(self) -> Dict[str, Any]:
        """Calculate statistics from recorded latencies."""
        if not self.latencies:
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": 0.0,
                "latency_p50": 0.0,
                "latency_p95": 0.0,
                "latency_p99": 0.0,
                "latency_mean": 0.0
            }
        
        latencies_list = list(self.latencies)
        error_rate = (
            self.error_count / self.request_count * 100
            if self.request_count > 0
            else 0.0
        )
        
        latency_p50 = statistics.median(latencies_list) * 1000
        if len(latencies_list) >= 20:
            latency_p95 = statistics.quantiles(latencies_list, n=20)[18] * 1000
        else:
            latency_p95 = max(latencies_list) * 1000
        if len(latencies_list) >= 100:
            latency_p99 = statistics.quantiles(latencies_list, n=100)[98] * 1000
        else:
            latency_p99 = max(latencies_list) * 1000
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": round(error_rate, 2),
            "latency_p50": round(latency_p50, 2),  # ms
            "latency_p95": round(latency_p95, 2),
            "latency_p99": round(latency_p99, 2),
            "latency_mean": round(statistics.mean(latencies_list) * 1000, 2),
        }


@dataclass
class CacheMetrics:
    """Cache hit/miss tracking."""
    hits: int = 0
    misses: int = 0
    
    def record_hit(self):
        """Record a cache hit."""
        self.hits += 1
    
    def record_miss(self):
        """Record a cache miss."""
        self.misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Calculate cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": round(hit_rate, 2)
        }


class MetricsCollector:
    """Global metrics collector singleton."""
    
    def __init__(self):
        self.endpoint_metrics: Dict[str, EndpointMetrics] = defaultdict(
            EndpointMetrics
        )
        self.cache_metrics: Dict[str, CacheMetrics] = defaultdict(CacheMetrics)
        self.start_time = time.time()
    
    def record_request(
        self, endpoint: str, latency: float, is_error: bool = False
    ):
        """Record a request for an endpoint."""
        self.endpoint_metrics[endpoint].add_request(latency, is_error)
    
    def record_cache_hit(self, cache_name: str):
        """Record a cache hit."""
        self.cache_metrics[cache_name].record_hit()
    
    def record_cache_miss(self, cache_name: str):
        """Record a cache miss."""
        self.cache_metrics[cache_name].record_miss()
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        uptime_seconds = int(time.time() - self.start_time)
        
        # Endpoint metrics
        endpoints = {}
        for endpoint, metrics in self.endpoint_metrics.items():
            endpoints[endpoint] = metrics.get_stats()
        
        # Cache metrics
        caches = {}
        for cache_name, metrics in self.cache_metrics.items():
            caches[cache_name] = metrics.get_stats()
        
        # Overall statistics
        total_requests = sum(
            m.request_count for m in self.endpoint_metrics.values()
        )
        total_errors = sum(
            m.error_count for m in self.endpoint_metrics.values()
        )
        overall_error_rate = (
            total_errors / total_requests * 100
            if total_requests > 0
            else 0.0
        )
        
        return {
            "system": {
                "uptime_seconds": uptime_seconds,
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": round(overall_error_rate, 2)
            },
            "endpoints": endpoints,
            "cache": caches
        }
    
    def reset(self):
        """Reset all metrics (for testing)."""
        self.endpoint_metrics.clear()
        self.cache_metrics.clear()
        self.start_time = time.time()


# Global metrics collector instance
metrics_collector = MetricsCollector()


@router.get("")
async def get_metrics(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get comprehensive system metrics for observability.
    
    Returns metrics including:
    - Request counts and error rates per endpoint
    - Latency percentiles (p50, p95, p99) per endpoint
    - Cache hit/miss ratios
    - System uptime
    
    Requires authentication.
    
    Returns:
        {
            "status": "ok",
            "data": {
                "system": {...},
                "endpoints": {...},
                "cache": {...}
            },
            "error": null
        }
    """
    try:
        metrics_data = metrics_collector.get_all_metrics()
        # Append custom translation & nav/phrase metrics
        metrics_data["translation_pipeline"] = translation_latency()
        metrics_data["navigation_phrase"] = nav_phrase_metrics()
        
        return {
            "status": "ok",
            "data": metrics_data,
            "error": None
        }
    
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}", exc_info=True)
        return {
            "status": "error",
            "data": None,
            "error": "Failed to retrieve metrics"
        }
