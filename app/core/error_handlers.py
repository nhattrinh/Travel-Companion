"""
Comprehensive error handlers for FastAPI application.
Implements Requirements 6.1, 6.2, 6.3, 6.4 - Error handling and graceful degradation.
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import traceback
import time
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional

from app.core.exceptions import (
    MenuTranslationException,
    ErrorCode,
    GracefulDegradationError,
    ModelUnavailableError,
    ProcessingTimeoutError,
    ServiceUnavailableError
)
from app.models.api_models import StandardErrorResponse

logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Comprehensive error handling system with logging and graceful degradation.
    """
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.last_error_time: Dict[str, float] = {}
    
    async def handle_menu_translation_exception(
        self, 
        request: Request, 
        exc: MenuTranslationException
    ) -> JSONResponse:
        """
        Handle custom MenuTranslationException with detailed logging.
        
        Args:
            request: FastAPI request object
            exc: MenuTranslationException instance
            
        Returns:
            JSONResponse with structured error information
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log error with context
        logger.error(
            f"MenuTranslationException in request {request_id}: {exc.message}",
            extra={
                'request_id': request_id,
                'error_code': exc.error_code.value,
                'status_code': exc.status_code,
                'details': exc.details,
                'request_path': request.url.path,
                'request_method': request.method,
                'client_ip': request.client.host if request.client else 'unknown'
            }
        )
        
        # Track error frequency for monitoring
        self._track_error(exc.error_code.value)
        
        return self._create_error_response(
            error_code=exc.error_code.value,
            message=exc.message,
            details=exc.details,
            request_id=request_id,
            status_code=exc.status_code
        )
    
    async def handle_graceful_degradation_exception(
        self, 
        request: Request, 
        exc: GracefulDegradationError
    ) -> JSONResponse:
        """
        Handle graceful degradation scenarios with partial results.
        
        Args:
            request: FastAPI request object
            exc: GracefulDegradationError instance
            
        Returns:
            JSONResponse with partial results and degradation information
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log degradation event
        logger.warning(
            f"Graceful degradation in request {request_id}: {exc.message}",
            extra={
                'request_id': request_id,
                'successful_components': exc.details.get('successful_components', []),
                'failed_components': exc.details.get('failed_components', []),
                'partial_results_available': exc.details.get('partial_results_available', False)
            }
        )
        
        return self._create_error_response(
            error_code=exc.error_code.value,
            message=exc.message,
            details=exc.details,
            request_id=request_id,
            status_code=exc.status_code
        )
    
    async def handle_validation_error(
        self, 
        request: Request, 
        exc: RequestValidationError
    ) -> JSONResponse:
        """
        Handle Pydantic validation errors with detailed field information.
        
        Args:
            request: FastAPI request object
            exc: RequestValidationError instance
            
        Returns:
            JSONResponse with validation error details
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Extract validation error details
        validation_errors = []
        for error in exc.errors():
            validation_errors.append({
                'field': '.'.join(str(loc) for loc in error['loc']),
                'message': error['msg'],
                'type': error['type'],
                'input': error.get('input')
            })
        
        logger.warning(
            f"Validation error in request {request_id}: {len(validation_errors)} field errors",
            extra={
                'request_id': request_id,
                'validation_errors': validation_errors,
                'request_path': request.url.path
            }
        )
        
        return self._create_error_response(
            error_code=ErrorCode.VALIDATION_ERROR.value,
            message="Request validation failed",
            details={'validation_errors': validation_errors},
            request_id=request_id,
            status_code=422
        )
    
    async def handle_http_exception(
        self, 
        request: Request, 
        exc: HTTPException
    ) -> JSONResponse:
        """
        Handle FastAPI HTTPException with proper logging.
        
        Args:
            request: FastAPI request object
            exc: HTTPException instance
            
        Returns:
            JSONResponse with HTTP error information
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Map common HTTP status codes to error codes
        error_code_map = {
            400: ErrorCode.VALIDATION_ERROR,
            401: ErrorCode.MISSING_API_KEY,
            403: ErrorCode.INVALID_API_KEY,
            404: "NOT_FOUND",
            408: ErrorCode.PROCESSING_TIMEOUT,
            413: ErrorCode.IMAGE_TOO_LARGE,
            429: ErrorCode.RATE_LIMIT_EXCEEDED,
            503: ErrorCode.SERVICE_UNAVAILABLE,
            507: ErrorCode.MEMORY_LIMIT_EXCEEDED
        }
        
        error_code = error_code_map.get(exc.status_code, ErrorCode.INTERNAL_SERVER_ERROR)
        
        logger.warning(
            f"HTTP exception in request {request_id}: {exc.status_code} - {exc.detail}",
            extra={
                'request_id': request_id,
                'status_code': exc.status_code,
                'detail': exc.detail,
                'request_path': request.url.path
            }
        )
        
        return self._create_error_response(
            error_code=error_code.value if hasattr(error_code, 'value') else error_code,
            message=str(exc.detail),
            request_id=request_id,
            status_code=exc.status_code
        )
    
    async def handle_timeout_error(
        self, 
        request: Request, 
        exc: asyncio.TimeoutError
    ) -> JSONResponse:
        """
        Handle asyncio timeout errors with proper context.
        
        Args:
            request: FastAPI request object
            exc: TimeoutError instance
            
        Returns:
            JSONResponse with timeout error information
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.error(
            f"Request timeout in request {request_id}",
            extra={
                'request_id': request_id,
                'request_path': request.url.path,
                'request_method': request.method
            }
        )
        
        return self._create_error_response(
            error_code=ErrorCode.PROCESSING_TIMEOUT.value,
            message="Request processing timed out",
            request_id=request_id,
            status_code=408
        )
    
    async def handle_generic_exception(
        self, 
        request: Request, 
        exc: Exception
    ) -> JSONResponse:
        """
        Handle unexpected exceptions with full error logging.
        
        Args:
            request: FastAPI request object
            exc: Exception instance
            
        Returns:
            JSONResponse with generic error information
        """
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log full exception with traceback
        logger.error(
            f"Unhandled exception in request {request_id}: {type(exc).__name__}: {str(exc)}",
            exc_info=True,
            extra={
                'request_id': request_id,
                'exception_type': type(exc).__name__,
                'exception_message': str(exc),
                'request_path': request.url.path,
                'request_method': request.method,
                'client_ip': request.client.host if request.client else 'unknown'
            }
        )
        
        # Track critical errors
        self._track_error('INTERNAL_SERVER_ERROR')
        
        return self._create_error_response(
            error_code=ErrorCode.INTERNAL_SERVER_ERROR.value,
            message="An internal server error occurred",
            request_id=request_id,
            status_code=500
        )
    
    def _create_error_response(
        self,
        error_code: str,
        message: str,
        request_id: str,
        status_code: int,
        details: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """
        Create standardized error response.
        
        Args:
            error_code: Standardized error code
            message: Human-readable error message
            request_id: Request identifier
            status_code: HTTP status code
            details: Additional error details
            
        Returns:
            JSONResponse with standardized error format
        """
        error_response = StandardErrorResponse(
            error_code=error_code,
            message=message,
            details=details,
            request_id=request_id,
            timestamp=datetime.utcnow()
        )
        
        # Use json.loads(error_response.json()) to properly serialize datetime
        return JSONResponse(
            status_code=status_code,
            content=json.loads(error_response.json())
        )
    
    def _track_error(self, error_code: str) -> None:
        """
        Track error frequency for monitoring and alerting.
        
        Args:
            error_code: Error code to track
        """
        current_time = time.time()
        
        # Increment error count
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
        self.last_error_time[error_code] = current_time
        
        # Log high-frequency errors
        if self.error_counts[error_code] % 10 == 0:
            logger.warning(
                f"High frequency error detected: {error_code} occurred {self.error_counts[error_code]} times"
            )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics for monitoring.
        
        Returns:
            Dictionary with error statistics
        """
        current_time = time.time()
        
        return {
            'error_counts': dict(self.error_counts),
            'recent_errors': {
                code: count for code, count in self.error_counts.items()
                if current_time - self.last_error_time.get(code, 0) < 3600  # Last hour
            },
            'total_errors': sum(self.error_counts.values())
        }


# Global error handler instance
error_handler = ErrorHandler()


def setup_error_handlers(app):
    """
    Set up all error handlers for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    
    @app.exception_handler(MenuTranslationException)
    async def menu_translation_exception_handler(request: Request, exc: MenuTranslationException):
        return await error_handler.handle_menu_translation_exception(request, exc)
    
    @app.exception_handler(GracefulDegradationError)
    async def graceful_degradation_exception_handler(request: Request, exc: GracefulDegradationError):
        return await error_handler.handle_graceful_degradation_exception(request, exc)
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return await error_handler.handle_validation_error(request, exc)
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return await error_handler.handle_http_exception(request, exc)
    
    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
        # Convert Starlette HTTPException to FastAPI HTTPException
        fastapi_exc = HTTPException(status_code=exc.status_code, detail=exc.detail)
        return await error_handler.handle_http_exception(request, fastapi_exc)
    
    @app.exception_handler(asyncio.TimeoutError)
    async def timeout_exception_handler(request: Request, exc: asyncio.TimeoutError):
        return await error_handler.handle_timeout_error(request, exc)
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return await error_handler.handle_generic_exception(request, exc)