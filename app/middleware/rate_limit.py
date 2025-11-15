"""
Rate limiting middleware for concurrent request management.
Implements Requirements 7.2 - Rate limiting for concurrent requests.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
import time
import logging
from typing import Dict, Optional
from collections import defaultdict, deque
from dataclasses import dataclass

from app.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class RateLimitInfo:
    """Information about rate limiting for a client."""
    requests: deque
    concurrent_count: int
    last_request_time: float


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with support for:
    - Concurrent request limits per client
    - Request rate limits (requests per time window)
    - Global concurrent request limits
    """
    
    def __init__(
        self, 
        app,
        max_concurrent_per_client: int = 5,
        max_requests_per_minute: int = 60,
        global_max_concurrent: Optional[int] = None
    ):
        super().__init__(app)
        self.max_concurrent_per_client = max_concurrent_per_client
        self.max_requests_per_minute = max_requests_per_minute
        self.global_max_concurrent = global_max_concurrent or settings.max_concurrent_requests
        
        # Track client rate limits
        self.client_limits: Dict[str, RateLimitInfo] = defaultdict(
            lambda: RateLimitInfo(
                requests=deque(),
                concurrent_count=0,
                last_request_time=0.0
            )
        )
        
        # Global concurrent request counter
        self.global_concurrent_count = 0
        self.global_lock = asyncio.Lock()
        
        # Cleanup task for expired rate limit data
        self._cleanup_task = None
    
    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier for rate limiting.
        
        Uses API key if available, otherwise falls back to IP address.
        """
        # Use API key if authenticated
        if hasattr(request.state, 'api_key'):
            return f"api_key:{request.state.api_key}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else 'unknown'
        return f"ip:{client_ip}"
    
    def _cleanup_expired_requests(self, client_info: RateLimitInfo) -> None:
        """Remove expired requests from the tracking deque."""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        while client_info.requests and client_info.requests[0] < window_start:
            client_info.requests.popleft()
    
    def _check_rate_limit(self, client_id: str) -> Optional[JSONResponse]:
        """
        Check if client has exceeded rate limits.
        
        Returns:
            JSONResponse with 429 status if rate limited, None otherwise
        """
        client_info = self.client_limits[client_id]
        current_time = time.time()
        
        # Clean up expired requests
        self._cleanup_expired_requests(client_info)
        
        # Check requests per minute limit
        if len(client_info.requests) >= self.max_requests_per_minute:
            oldest_request = client_info.requests[0]
            retry_after = int(60 - (current_time - oldest_request)) + 1
            
            logger.warning(
                f"Rate limit exceeded for client {client_id}: {len(client_info.requests)} requests in last minute"
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded: {self.max_requests_per_minute} requests per minute",
                    "retry_after_seconds": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        # Check concurrent requests per client
        if client_info.concurrent_count >= self.max_concurrent_per_client:
            logger.warning(
                f"Concurrent limit exceeded for client {client_id}: {client_info.concurrent_count} concurrent requests"
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error_code": "CONCURRENT_LIMIT_EXCEEDED",
                    "message": f"Too many concurrent requests: {self.max_concurrent_per_client} max per client",
                    "retry_after_seconds": 5
                },
                headers={"Retry-After": "5"}
            )
        
        return None
    
    async def _check_global_limit(self) -> Optional[JSONResponse]:
        """
        Check global concurrent request limit.
        
        Returns:
            JSONResponse with 503 status if globally rate limited, None otherwise
        """
        async with self.global_lock:
            if self.global_concurrent_count >= self.global_max_concurrent:
                logger.warning(
                    f"Global concurrent limit exceeded: {self.global_concurrent_count} requests"
                )
                
                return JSONResponse(
                    status_code=503,
                    content={
                        "error_code": "SERVICE_OVERLOADED",
                        "message": "Service temporarily overloaded, please retry later",
                        "retry_after_seconds": 10
                    },
                    headers={"Retry-After": "10"}
                )
        
        return None
    
    async def dispatch(self, request: Request, call_next):
        """
        Process rate limiting for incoming requests.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response object
        """
        # Skip rate limiting for health checks and docs
        if request.url.path in {"/", "/health", "/status", "/docs", "/redoc", "/openapi.json"}:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        current_time = time.time()
        
        # Check client-specific rate limits
        rate_limit_response = self._check_rate_limit(client_id)
        if rate_limit_response:
            return rate_limit_response
        
        # Check global concurrent limit
        global_limit_response = await self._check_global_limit()
        if global_limit_response:
            return global_limit_response
        
        # Update counters before processing
        client_info = self.client_limits[client_id]
        client_info.requests.append(current_time)
        client_info.concurrent_count += 1
        client_info.last_request_time = current_time
        
        async with self.global_lock:
            self.global_concurrent_count += 1
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Add rate limit headers to response
            response.headers["X-RateLimit-Limit"] = str(self.max_requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, self.max_requests_per_minute - len(client_info.requests))
            )
            response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
            
            return response
            
        finally:
            # Decrement counters after processing
            client_info.concurrent_count -= 1
            
            async with self.global_lock:
                self.global_concurrent_count -= 1
    
    async def cleanup_expired_data(self):
        """Periodic cleanup of expired rate limit data."""
        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
                current_time = time.time()
                expired_clients = []
                
                for client_id, client_info in self.client_limits.items():
                    # Clean up expired requests
                    self._cleanup_expired_requests(client_info)
                    
                    # Remove clients with no recent activity (1 hour)
                    if current_time - client_info.last_request_time > 3600:
                        expired_clients.append(client_id)
                
                # Remove expired clients
                for client_id in expired_clients:
                    del self.client_limits[client_id]
                
                if expired_clients:
                    logger.debug(f"Cleaned up rate limit data for {len(expired_clients)} expired clients")
                    
            except Exception as e:
                logger.error(f"Error in rate limit cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying