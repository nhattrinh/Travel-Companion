"""
Health monitoring service for model lifecycle management.
Implements Requirements 6.3, 6.4 for health monitoring and lifecycle management.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

from ..models.internal_models import ModelType, ModelStatus, SystemHealth


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    model_type: ModelType
    is_healthy: bool
    response_time_ms: float
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HealthMetrics:
    """Health metrics for a model."""
    model_type: ModelType
    total_checks: int = 0
    successful_checks: int = 0
    failed_checks: int = 0
    average_response_time_ms: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    
    def add_check_result(self, result: HealthCheckResult) -> None:
        """Add a health check result to metrics."""
        self.total_checks += 1
        
        if result.is_healthy:
            self.successful_checks += 1
            self.last_success = result.timestamp
            self.consecutive_failures = 0
        else:
            self.failed_checks += 1
            self.last_failure = result.timestamp
            self.consecutive_failures += 1
        
        # Update average response time
        if self.total_checks > 0:
            self.average_response_time_ms = (
                (self.average_response_time_ms * (self.total_checks - 1) + result.response_time_ms) 
                / self.total_checks
            )
    
    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_checks == 0:
            return 0.0
        return (self.successful_checks / self.total_checks) * 100.0
    
    def is_degraded(self, threshold_failures: int = 3) -> bool:
        """Check if model is in degraded state."""
        return self.consecutive_failures >= threshold_failures


class ModelHealthMonitor:
    """
    Comprehensive health monitoring service for AI models.
    Implements Requirements 6.3, 6.4 for health monitoring and lifecycle management.
    """
    
    def __init__(self, check_interval_seconds: int = 60, failure_threshold: int = 3):
        self.check_interval_seconds = check_interval_seconds
        self.failure_threshold = failure_threshold
        self._metrics: Dict[ModelType, HealthMetrics] = {}
        self._health_callbacks: List[Callable[[HealthCheckResult], None]] = []
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self._logger = logging.getLogger(__name__)
        self._system_start_time = datetime.utcnow()
    
    def add_health_callback(self, callback: Callable[[HealthCheckResult], None]) -> None:
        """
        Add callback to be notified of health check results.
        Implements Requirement 6.4 - health monitoring notifications.
        
        Args:
            callback: Function to call with health check results
        """
        self._health_callbacks.append(callback)
    
    def remove_health_callback(self, callback: Callable[[HealthCheckResult], None]) -> None:
        """
        Remove health callback.
        
        Args:
            callback: Function to remove from callbacks
        """
        if callback in self._health_callbacks:
            self._health_callbacks.remove(callback)
    
    async def start_monitoring(self, model_manager) -> None:
        """
        Start health monitoring for all models.
        Implements Requirement 6.4 - periodic status updates.
        
        Args:
            model_manager: ModelManager instance to monitor
        """
        if self._running:
            self._logger.warning("Health monitoring already running")
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitoring_loop(model_manager)
        )
        self._logger.info(f"Started health monitoring with {self.check_interval_seconds}s interval")
    
    async def stop_monitoring(self) -> None:
        """
        Stop health monitoring.
        Implements Requirement 6.3 - cleanup procedures.
        """
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self._logger.info("Stopped health monitoring")
    
    async def _monitoring_loop(self, model_manager) -> None:
        """
        Main monitoring loop.
        Implements Requirement 6.4 - periodic health checks.
        
        Args:
            model_manager: ModelManager instance to monitor
        """
        while self._running:
            try:
                await self._perform_health_checks(model_manager)
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Health monitoring loop error: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval_seconds)
    
    async def _perform_health_checks(self, model_manager) -> None:
        """
        Perform health checks on all registered models.
        Implements Requirement 6.4 - health monitoring.
        
        Args:
            model_manager: ModelManager instance to check
        """
        check_tasks = []
        
        for model_type in ModelType:
            model = await model_manager.get_model(model_type)
            if model:
                task = asyncio.create_task(
                    self._check_model_health(model_type, model)
                )
                check_tasks.append(task)
        
        if check_tasks:
            results = await asyncio.gather(*check_tasks, return_exceptions=True)
            
            # Process results and notify callbacks
            for result in results:
                if isinstance(result, HealthCheckResult):
                    await self._process_health_result(result)
                elif isinstance(result, Exception):
                    self._logger.error(f"Health check task failed: {result}", exc_info=True)
    
    async def _check_model_health(self, model_type: ModelType, model) -> HealthCheckResult:
        """
        Check health of a specific model.
        Implements Requirement 6.4 - individual model health checks.
        
        Args:
            model_type: Type of model to check
            model: Model instance to check
            
        Returns:
            HealthCheckResult: Result of the health check
        """
        start_time = time.time()
        
        try:
            is_healthy = await model._health_check_with_monitoring()
            response_time_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                model_type=model_type,
                is_healthy=is_healthy,
                response_time_ms=response_time_ms,
                error_message=None if is_healthy else model.last_error
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                model_type=model_type,
                is_healthy=False,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
    
    async def _process_health_result(self, result: HealthCheckResult) -> None:
        """
        Process health check result and update metrics.
        Implements Requirement 6.4 - health monitoring.
        
        Args:
            result: Health check result to process
        """
        # Update metrics
        if result.model_type not in self._metrics:
            self._metrics[result.model_type] = HealthMetrics(model_type=result.model_type)
        
        self._metrics[result.model_type].add_check_result(result)
        
        # Log significant events
        if not result.is_healthy:
            consecutive_failures = self._metrics[result.model_type].consecutive_failures
            self._logger.warning(
                f"Model {result.model_type.value} health check failed "
                f"(consecutive failures: {consecutive_failures}): {result.error_message}"
            )
            
            if consecutive_failures >= self.failure_threshold:
                self._logger.error(
                    f"Model {result.model_type.value} exceeded failure threshold "
                    f"({self.failure_threshold})"
                )
        else:
            # Log recovery from failures
            if self._metrics[result.model_type].consecutive_failures > 0:
                self._logger.info(f"Model {result.model_type.value} recovered from failures")
        
        # Notify callbacks
        for callback in self._health_callbacks:
            try:
                callback(result)
            except Exception as e:
                self._logger.error(f"Health callback error: {e}", exc_info=True)
    
    def get_model_metrics(self, model_type: ModelType) -> Optional[HealthMetrics]:
        """
        Get health metrics for a specific model.
        
        Args:
            model_type: Type of model to get metrics for
            
        Returns:
            Optional[HealthMetrics]: Metrics if available
        """
        return self._metrics.get(model_type)
    
    def get_all_metrics(self) -> Dict[ModelType, HealthMetrics]:
        """
        Get health metrics for all monitored models.
        
        Returns:
            Dict[ModelType, HealthMetrics]: All available metrics
        """
        return self._metrics.copy()
    
    def get_system_health(self) -> SystemHealth:
        """
        Get overall system health status.
        Implements Requirement 6.4 - system health monitoring.
        
        Returns:
            SystemHealth: Current system health status
        """
        model_statuses = {}
        healthy_models = 0
        degraded_models = 0
        failed_models = 0
        
        for model_type, metrics in self._metrics.items():
            if metrics.consecutive_failures == 0:
                model_statuses[model_type.value] = ModelStatus.READY
                healthy_models += 1
            elif metrics.is_degraded(self.failure_threshold):
                model_statuses[model_type.value] = ModelStatus.FAILED
                failed_models += 1
            else:
                model_statuses[model_type.value] = ModelStatus.FAILED
                degraded_models += 1
        
        # Determine overall status
        if failed_models > 0 or degraded_models > len(self._metrics) // 2:
            overall_status = "unhealthy"
        elif degraded_models > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        uptime = int((datetime.utcnow() - self._system_start_time).total_seconds())
        
        return SystemHealth(
            overall_status=overall_status,
            model_statuses=model_statuses,
            uptime_seconds=uptime,
            last_updated=datetime.utcnow()
        )
    
    def reset_metrics(self, model_type: Optional[ModelType] = None) -> None:
        """
        Reset health metrics.
        
        Args:
            model_type: Specific model type to reset, or None for all
        """
        if model_type:
            if model_type in self._metrics:
                del self._metrics[model_type]
                self._logger.info(f"Reset metrics for {model_type.value}")
        else:
            self._metrics.clear()
            self._logger.info("Reset all health metrics")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive health summary.
        Implements Requirement 6.4 - health reporting.
        
        Returns:
            Dict[str, Any]: Health summary with key metrics
        """
        summary = {
            "monitoring_active": self._running,
            "check_interval_seconds": self.check_interval_seconds,
            "failure_threshold": self.failure_threshold,
            "uptime_seconds": int((datetime.utcnow() - self._system_start_time).total_seconds()),
            "models": {}
        }
        
        for model_type, metrics in self._metrics.items():
            summary["models"][model_type.value] = {
                "total_checks": metrics.total_checks,
                "success_rate": round(metrics.get_success_rate(), 2),
                "consecutive_failures": metrics.consecutive_failures,
                "average_response_time_ms": round(metrics.average_response_time_ms, 2),
                "is_degraded": metrics.is_degraded(self.failure_threshold),
                "last_success": metrics.last_success.isoformat() if metrics.last_success else None,
                "last_failure": metrics.last_failure.isoformat() if metrics.last_failure else None
            }
        
        return summary