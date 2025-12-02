"""
Authentication middleware for API key-based authentication.
Implements Requirements 7.1 - API key-based authentication.
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import os
from typing import Optional, Set

from app.config.settings import settings

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    API key-based authentication middleware.
    
    Validates API keys for protected endpoints and allows
    public access to health check and documentation endpoints.
    """
    
    def __init__(self, app, valid_api_keys: Optional[Set[str]] = None):
        super().__init__(app)
        self.valid_api_keys = valid_api_keys or self._load_api_keys()
        self.public_paths = {
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/status",
            "/translation/text",
            "/auth/login",
            "/auth/register",
            "/auth/refresh"
        }
    
    def _load_api_keys(self) -> Set[str]:
        """Load valid API keys from environment or configuration."""
        api_keys = set()
        
        # Load from environment variable (comma-separated)
        env_keys = os.getenv("VALID_API_KEYS", "")
        if env_keys:
            api_keys.update(key.strip() for key in env_keys.split(",") if key.strip())
        
        # Add default development key if in debug mode
        if settings.debug and not api_keys:
            api_keys.add("dev-key-12345")
            logger.warning("Using default development API key. Set VALID_API_KEYS in production.")
        
        return api_keys
    
    async def dispatch(self, request: Request, call_next):
        """
        Process authentication for incoming requests.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response object
        """
        # Skip authentication for public paths
        if request.url.path in self.public_paths:
            return await call_next(request)
        
        # Skip authentication if not required
        if not settings.require_api_key:
            return await call_next(request)
        
        # Extract API key from header
        api_key = request.headers.get(settings.api_key_header)
        
        if not api_key:
            logger.warning(
                f"Missing API key for request {getattr(request.state, 'request_id', 'unknown')}",
                extra={
                    'request_id': getattr(request.state, 'request_id', 'unknown'),
                    'path': request.url.path,
                    'client_ip': request.client.host if request.client else 'unknown'
                }
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error_code": "MISSING_API_KEY",
                    "message": f"API key required in {settings.api_key_header} header",
                    "request_id": getattr(request.state, 'request_id', 'unknown')
                }
            )
        
        # Validate API key
        if api_key not in self.valid_api_keys:
            logger.warning(
                f"Invalid API key for request {getattr(request.state, 'request_id', 'unknown')}",
                extra={
                    'request_id': getattr(request.state, 'request_id', 'unknown'),
                    'path': request.url.path,
                    'client_ip': request.client.host if request.client else 'unknown',
                    'api_key_prefix': api_key[:8] + "..." if len(api_key) > 8 else api_key
                }
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error_code": "INVALID_API_KEY",
                    "message": "Invalid API key provided",
                    "request_id": getattr(request.state, 'request_id', 'unknown')
                }
            )
        
        # Store authenticated API key in request state
        request.state.api_key = api_key
        
        logger.debug(
            f"Request {getattr(request.state, 'request_id', 'unknown')} authenticated successfully"
        )
        
        return await call_next(request)