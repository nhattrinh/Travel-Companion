"""
Concurrency manager for handling concurrent request processing.

Manages concurrent request processing with:
- Request queuing and resource management
- Timeout handling for long-running operations
- Memory monitoring and cleanup
- Semaphore-based concurrency control

Requirements: 7.1, 7.3
"""

import asyncio
import logging
import time
from typing import Callable, Any, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# Optional psutil import for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from app.models import ConcurrencyConfig, QueuedRequest

logger = logging.getLogger(__name__)


class ConcurrencyManager:
    """
    Manages concurrent request processing with resource control.
    
    Provides:
    - Semaphore-based concurrency limiting
    - Request queuing when resources are limited
    - Timeout handling for processing operations
    - Memory monitoring and cleanup
    - Request tracking and metrics
    """
    
    def __init__(self, config: ConcurrencyConfig):
        self.config = config
        self.active_requests: Dict[str, asyncio.Task] = {}
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.memory_monitor = MemoryMonitor(config.memory_limit_mb)
        self.request_metrics: Dict[str, RequestMetrics] = {}
        self.logger = logging.getLogger(__name__)
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def process_request(
        self,
        request_id: str,
        processing_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Process request with concurrency control and timeout handling.
        
        Args:
            request_id: Unique identifier for the request
            processing_func: Function to execute for processing
            *args: Arguments for processing function
            **kwargs: Keyword arguments for processing function
            
        Returns:
            Result from processing function
            
        Raises:
            asyncio.TimeoutError: If processing exceeds timeout
            MemoryError: If memory limit is exceeded
        """
        start_time = time.time()
        
        self.logger.info(
            f"Starting request processing: {request_id}",
            extra={'request_id': request_id, 'active_requests': len(self.active_requests)}
        )
        
        # Check memory before processing
        if not await self.memory_monitor.check_memory_available():
            self.logger.warning(
                f"Memory limit exceeded, queuing request: {request_id}",
                extra={'request_id': request_id}
            )
            
            # Queue the request for later processing
            queued_request = QueuedRequest(
                request_id=request_id,
                processing_func=processing_func,
                args=args,
                kwargs=kwargs,
                priority=0,
                queued_at=datetime.utcnow()
            )
            
            await self.queue_request(queued_request)
            return await self._wait_for_queued_request(request_id)
        
        # Acquire semaphore for concurrency control
        async with self.semaphore:
            try:
                # Create and track the processing task
                processing_task = asyncio.create_task(
                    self._execute_with_timeout(
                        request_id,
                        processing_func,
                        *args,
                        **kwargs
                    )
                )
                
                self.active_requests[request_id] = processing_task
                
                # Track request metrics
                self.request_metrics[request_id] = RequestMetrics(
                    request_id=request_id,
                    start_time=start_time,
                    memory_used_mb=self.memory_monitor.get_current_usage()
                )
                
                # Execute the processing
                result = await processing_task
                
                # Update metrics on success
                end_time = time.time()
                processing_time = int((end_time - start_time) * 1000)
                
                if request_id in self.request_metrics:
                    self.request_metrics[request_id].end_time = end_time
                    self.request_metrics[request_id].processing_time_ms = processing_time
                
                self.logger.info(
                    f"Request processing completed: {request_id} ({processing_time}ms)",
                    extra={
                        'request_id': request_id,
                        'processing_time_ms': processing_time,
                        'success': True
                    }
                )
                
                return result
                
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Request processing timed out: {request_id}",
                    extra={'request_id': request_id, 'timeout_seconds': self.config.processing_timeout_seconds}
                )
                raise
                
            except Exception as e:
                self.logger.error(
                    f"Request processing failed: {request_id} - {str(e)}",
                    extra={'request_id': request_id},
                    exc_info=True
                )
                
                # Update metrics on failure
                if request_id in self.request_metrics:
                    self.request_metrics[request_id].add_error(str(e))
                
                raise
                
            finally:
                # Clean up request tracking
                await self.cleanup_resources(request_id)
    
    async def _execute_with_timeout(
        self,
        request_id: str,
        processing_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute processing function with timeout."""
        try:
            return await asyncio.wait_for(
                processing_func(*args, **kwargs),
                timeout=self.config.processing_timeout_seconds
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                f"Processing function timed out for request: {request_id}",
                extra={'request_id': request_id}
            )
            await self.handle_timeout(request_id)
            raise
    
    async def queue_request(self, request: QueuedRequest) -> None:
        """
        Queue request when resources are limited.
        
        Args:
            request: Request to queue for later processing
        """
        try:
            await asyncio.wait_for(
                self.request_queue.put(request),
                timeout=self.config.queue_timeout_seconds
            )
            
            self.logger.info(
                f"Request queued: {request.request_id}",
                extra={'request_id': request.request_id, 'queue_size': self.request_queue.qsize()}
            )
            
        except asyncio.TimeoutError:
            self.logger.error(
                f"Failed to queue request (queue full): {request.request_id}",
                extra={'request_id': request.request_id}
            )
            raise asyncio.TimeoutError("Request queue is full")
    
    async def _wait_for_queued_request(self, request_id: str) -> Any:
        """Wait for queued request to be processed."""
        # This is a simplified implementation
        # In a real system, you'd want to implement proper queue processing
        await asyncio.sleep(1)  # Wait briefly
        raise asyncio.TimeoutError(f"Queued request {request_id} timed out")
    
    async def handle_timeout(self, request_id: str) -> None:
        """
        Handle processing timeouts.
        
        Args:
            request_id: ID of the request that timed out
        """
        self.logger.warning(
            f"Handling timeout for request: {request_id}",
            extra={'request_id': request_id}
        )
        
        # Cancel the task if it's still running
        if request_id in self.active_requests:
            task = self.active_requests[request_id]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Update metrics
        if request_id in self.request_metrics:
            self.request_metrics[request_id].add_error("Processing timeout")
    
    async def cleanup_resources(self, request_id: str) -> None:
        """
        Clean up resources after processing.
        
        Args:
            request_id: ID of the request to clean up
        """
        # Remove from active requests
        if request_id in self.active_requests:
            del self.active_requests[request_id]
        
        # Clean up old metrics (keep recent ones for monitoring)
        if request_id in self.request_metrics:
            metrics = self.request_metrics[request_id]
            if metrics.end_time and (datetime.utcnow() - metrics.end_time).total_seconds() > 3600:
                del self.request_metrics[request_id]
        
        self.logger.debug(
            f"Cleaned up resources for request: {request_id}",
            extra={'request_id': request_id, 'active_requests': len(self.active_requests)}
        )
    
    def get_active_request_count(self) -> int:
        """Get number of currently active requests."""
        return len(self.active_requests)
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.request_queue.qsize()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of processing metrics."""
        total_requests = len(self.request_metrics)
        completed_requests = sum(1 for m in self.request_metrics.values() if m.end_time)
        failed_requests = sum(1 for m in self.request_metrics.values() if m.errors_encountered)
        
        avg_processing_time = 0.0
        if completed_requests > 0:
            total_time = sum(
                m.processing_time_ms or 0 
                for m in self.request_metrics.values() 
                if m.processing_time_ms
            )
            avg_processing_time = total_time / completed_requests
        
        return {
            "total_requests": total_requests,
            "completed_requests": completed_requests,
            "failed_requests": failed_requests,
            "active_requests": len(self.active_requests),
            "queued_requests": self.request_queue.qsize(),
            "average_processing_time_ms": avg_processing_time,
            "memory_usage_mb": self.memory_monitor.get_current_usage()
        }
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of old resources and metrics."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                
                # Clean up old metrics
                current_time = datetime.utcnow()
                old_metrics = [
                    request_id for request_id, metrics in self.request_metrics.items()
                    if metrics.end_time and (current_time - metrics.end_time).total_seconds() > 3600
                ]
                
                for request_id in old_metrics:
                    del self.request_metrics[request_id]
                
                if old_metrics:
                    self.logger.debug(f"Cleaned up {len(old_metrics)} old metric entries")
                
                # Force garbage collection if memory usage is high
                if self.memory_monitor.get_current_usage() > self.config.memory_limit_mb * 0.8:
                    import gc
                    gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error during periodic cleanup: {str(e)}", exc_info=True)
    
    async def shutdown(self) -> None:
        """Shutdown the concurrency manager and clean up resources."""
        self.logger.info("Shutting down concurrency manager")
        
        # Cancel cleanup task
        if hasattr(self, '_cleanup_task'):
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active requests
        for request_id, task in self.active_requests.items():
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete or be cancelled
        if self.active_requests:
            await asyncio.gather(*self.active_requests.values(), return_exceptions=True)
        
        self.logger.info("Concurrency manager shutdown complete")


class MemoryMonitor:
    """Monitor system memory usage for resource management."""
    
    def __init__(self, memory_limit_mb: int):
        self.memory_limit_mb = memory_limit_mb
        self.logger = logging.getLogger(__name__)
    
    async def check_memory_available(self) -> bool:
        """Check if sufficient memory is available for processing."""
        try:
            current_usage = self.get_current_usage()
            available = current_usage < self.memory_limit_mb
            
            if not available:
                self.logger.warning(
                    f"Memory limit exceeded: {current_usage:.2f}MB / {self.memory_limit_mb}MB"
                )
            
            return available
            
        except Exception as e:
            self.logger.error(f"Failed to check memory usage: {str(e)}")
            return True  # Assume available on error
    
    def get_current_usage(self) -> float:
        """Get current process memory usage in MB."""
        if not PSUTIL_AVAILABLE:
            return 0.0
        
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0


@dataclass
class RequestMetrics:
    """Metrics for individual request processing."""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    processing_time_ms: Optional[int] = None
    memory_used_mb: float = 0.0
    errors_encountered: list = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Add error to metrics."""
        self.errors_encountered.append(error)