"""
Models package for the menu translation backend.

This package contains all data models used throughout the application,
including API models for requests/responses and internal models for
processing context and system state.
"""

# API Models
from .api_models import (
    SupportedLanguage,
    MenuProcessingRequest,
    ExtractedMenuItem,
    MenuProcessingResponse,
    BatchProcessingRequest,
    BatchProcessingResponse,
    HealthCheckResponse,
    StandardErrorResponse,
)

# Internal Models
from .internal_models import (
    ModelType,
    ModelStatus,
    ErrorCode,
    ProcessingStage,
    OCRResult,
    TranslationResult,
    FoodImage,
    ProcessingContext,
    ModelConfig,
    ConcurrencyConfig,
    RequestMetrics,
    QueuedRequest,
    SystemHealth,
)

__all__ = [
    # API Models
    "SupportedLanguage",
    "MenuProcessingRequest",
    "ExtractedMenuItem", 
    "MenuProcessingResponse",
    "BatchProcessingRequest",
    "BatchProcessingResponse",
    "HealthCheckResponse",
    "StandardErrorResponse",
    
    # Internal Models
    "ModelType",
    "ModelStatus",
    "ErrorCode",
    "ProcessingStage",
    "OCRResult",
    "TranslationResult",
    "FoodImage",
    "ProcessingContext",
    "ModelConfig",
    "ConcurrencyConfig",
    "RequestMetrics",
    "QueuedRequest",
    "SystemHealth",
]