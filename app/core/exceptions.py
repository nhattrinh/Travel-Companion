"""
Custom exceptions for the menu translation backend.
Implements Requirements 6.1, 6.2, 6.3, 6.4 - Error handling and graceful degradation.
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(str, Enum):
    """Standardized error codes for the application."""
    
    # Image validation errors
    INVALID_IMAGE_FORMAT = "INVALID_IMAGE_FORMAT"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    IMAGE_CORRUPTED = "IMAGE_CORRUPTED"
    
    # Processing errors
    OCR_PROCESSING_FAILED = "OCR_PROCESSING_FAILED"
    TRANSLATION_FAILED = "TRANSLATION_FAILED"
    FOOD_IMAGE_RETRIEVAL_FAILED = "FOOD_IMAGE_RETRIEVAL_FAILED"
    
    # Model errors
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_INITIALIZATION_FAILED = "MODEL_INITIALIZATION_FAILED"
    MODEL_PROCESSING_FAILED = "MODEL_PROCESSING_FAILED"
    
    # Language errors
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
    LANGUAGE_DETECTION_FAILED = "LANGUAGE_DETECTION_FAILED"
    
    # System errors
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"
    CONCURRENT_LIMIT_EXCEEDED = "CONCURRENT_LIMIT_EXCEEDED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    
    # Authentication errors
    MISSING_API_KEY = "MISSING_API_KEY"
    INVALID_API_KEY = "INVALID_API_KEY"
    
    # Rate limiting errors
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    SERVICE_OVERLOADED = "SERVICE_OVERLOADED"
    
    # Generic errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"


class MenuTranslationException(Exception):
    """Base exception for menu translation backend."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code


class ImageValidationError(MenuTranslationException):
    """Raised when image validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_IMAGE_FORMAT,
            details=details,
            status_code=400
        )


class ImageTooLargeError(MenuTranslationException):
    """Raised when uploaded image exceeds size limits."""
    
    def __init__(self, size_mb: float, max_size_mb: int):
        super().__init__(
            message=f"Image size {size_mb:.1f}MB exceeds maximum allowed size of {max_size_mb}MB",
            error_code=ErrorCode.IMAGE_TOO_LARGE,
            details={"size_mb": size_mb, "max_size_mb": max_size_mb},
            status_code=413
        )


class ImageCorruptedError(MenuTranslationException):
    """Raised when image is corrupted or unreadable."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Image is corrupted or unreadable",
            error_code=ErrorCode.IMAGE_CORRUPTED,
            details=details,
            status_code=400
        )


class OCRProcessingError(MenuTranslationException):
    """Raised when OCR processing fails."""
    
    def __init__(self, message: str = "OCR processing failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.OCR_PROCESSING_FAILED,
            details=details,
            status_code=422
        )


class TranslationError(MenuTranslationException):
    """Raised when translation fails."""
    
    def __init__(self, message: str = "Translation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.TRANSLATION_FAILED,
            details=details,
            status_code=422
        )


class UnsupportedLanguageError(MenuTranslationException):
    """Raised when requested language is not supported."""
    
    def __init__(self, language: str, supported_languages: Optional[list] = None):
        details = {"requested_language": language}
        if supported_languages:
            details["supported_languages"] = supported_languages
            
        super().__init__(
            message=f"Language '{language}' is not supported",
            error_code=ErrorCode.UNSUPPORTED_LANGUAGE,
            details=details,
            status_code=400
        )


class ModelUnavailableError(MenuTranslationException):
    """Raised when required model is unavailable."""
    
    def __init__(self, model_type: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Model '{model_type}' is currently unavailable",
            error_code=ErrorCode.MODEL_UNAVAILABLE,
            details=details or {"model_type": model_type},
            status_code=503
        )


class ProcessingTimeoutError(MenuTranslationException):
    """Raised when processing exceeds timeout limits."""
    
    def __init__(self, timeout_seconds: int, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Processing timed out after {timeout_seconds} seconds",
            error_code=ErrorCode.PROCESSING_TIMEOUT,
            details=details or {"timeout_seconds": timeout_seconds},
            status_code=408
        )


class MemoryLimitExceededError(MenuTranslationException):
    """Raised when memory usage exceeds limits."""
    
    def __init__(self, current_mb: float, limit_mb: float):
        super().__init__(
            message=f"Memory usage {current_mb:.1f}MB exceeds limit of {limit_mb:.1f}MB",
            error_code=ErrorCode.MEMORY_LIMIT_EXCEEDED,
            details={"current_mb": current_mb, "limit_mb": limit_mb},
            status_code=507
        )


class ServiceUnavailableError(MenuTranslationException):
    """Raised when service is temporarily unavailable."""
    
    def __init__(self, service_name: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Service '{service_name}' is temporarily unavailable",
            error_code=ErrorCode.SERVICE_UNAVAILABLE,
            details=details or {"service_name": service_name},
            status_code=503
        )


class GracefulDegradationError(MenuTranslationException):
    """
    Raised when partial processing succeeds but some components fail.
    This allows for graceful degradation of service.
    """
    
    def __init__(
        self,
        message: str,
        successful_components: list,
        failed_components: list,
        partial_results: Optional[Dict[str, Any]] = None
    ):
        details = {
            "successful_components": successful_components,
            "failed_components": failed_components,
            "partial_results_available": partial_results is not None
        }
        if partial_results:
            details["partial_results"] = partial_results
            
        super().__init__(
            message=message,
            error_code=ErrorCode.SERVICE_UNAVAILABLE,
            details=details,
            status_code=206  # Partial Content
        )