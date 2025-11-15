"""
Internal data models and enums for the menu translation backend.

This module contains internal data structures used for processing context,
model configuration, error handling, and system state management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import asyncio


class ModelType(Enum):
    """Types of AI models supported by the system (Requirements 6.1, 6.2)"""
    OCR = "ocr"
    TRANSLATION = "translation"
    NAVIGATION = "navigation"


class ModelStatus(Enum):
    """Status states for AI models (Requirements 6.1, 6.2)"""
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    SWAPPING = "swapping"
    MAINTENANCE = "maintenance"


class ErrorCode(str, Enum):
    """Standardized error codes for the system (Requirements 6.1, 6.2)"""
    # Image validation errors
    INVALID_IMAGE_FORMAT = "INVALID_IMAGE_FORMAT"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    IMAGE_CORRUPTED = "IMAGE_CORRUPTED"
    
    # Processing errors
    OCR_PROCESSING_FAILED = "OCR_PROCESSING_FAILED"
    TRANSLATION_FAILED = "TRANSLATION_FAILED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    
    # Language errors
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
    LANGUAGE_DETECTION_FAILED = "LANGUAGE_DETECTION_FAILED"
    
    # System errors
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"
    CONCURRENT_LIMIT_EXCEEDED = "CONCURRENT_LIMIT_EXCEEDED"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    
    # Service errors
    FOOD_IMAGE_SERVICE_UNAVAILABLE = "FOOD_IMAGE_SERVICE_UNAVAILABLE"
    CACHE_SERVICE_UNAVAILABLE = "CACHE_SERVICE_UNAVAILABLE"
    
    # Authentication and authorization
    INVALID_API_KEY = "INVALID_API_KEY"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"


class ProcessingStage(str, Enum):
    """Processing pipeline stages for tracking (Requirements 6.1, 6.2)"""
    INITIALIZED = "initialized"
    IMAGE_VALIDATION = "image_validation"
    OCR_PROCESSING = "ocr_processing"
    LANGUAGE_DETECTION = "language_detection"
    TRANSLATION = "translation"
    FOOD_IMAGE_RETRIEVAL = "food_image_retrieval"
    RESPONSE_FORMATTING = "response_formatting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OCRResult:
    """Internal model for OCR processing results (Requirements 6.1, 6.2)"""
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    group_id: Optional[str] = None
    processing_time_ms: Optional[int] = None
    
    def __post_init__(self):
        """Validate OCR result data"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if len(self.bbox) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")


@dataclass
class TranslationResult:
    """Internal model for translation results (Requirements 6.1, 6.2)"""
    translated_text: str
    source_language: str
    confidence: float
    error_indicator: Optional[str] = None
    processing_time_ms: Optional[int] = None
    
    def __post_init__(self):
        """Validate translation result data"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class FoodImage:
    """Internal model for food image data (Requirements 6.1, 6.2)"""
    url: str
    description: str
    relevance_score: float
    is_placeholder: bool = False
    cache_key: Optional[str] = None
    
    def __post_init__(self):
        """Validate food image data"""
        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError("Relevance score must be between 0.0 and 1.0")


@dataclass
class ProcessingContext:
    """Context object for tracking processing state (Requirements 6.1, 6.2)"""
    request_id: str
    start_time: float
    ocr_results: List[OCRResult] = field(default_factory=list)
    translation_results: List[TranslationResult] = field(default_factory=list)
    food_images: Dict[str, List[FoodImage]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    partial_failures: List[str] = field(default_factory=list)
    processing_stage: ProcessingStage = ProcessingStage.INITIALIZED
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, stage: ProcessingStage, error: str) -> None:
        """Add error with stage information for debugging (Requirement 6.1)"""
        error_msg = f"{stage.value}: {error}"
        self.errors.append(error_msg)
    
    def add_partial_failure(self, item: str, reason: str) -> None:
        """Track partial failures for graceful degradation (Requirement 6.2)"""
        failure_msg = f"{item}: {reason}"
        self.partial_failures.append(failure_msg)
    
    def update_stage(self, stage: ProcessingStage) -> None:
        """Update current processing stage"""
        self.processing_stage = stage
    
    def get_processing_time_ms(self) -> int:
        """Calculate total processing time in milliseconds"""
        import time
        return int((time.time() - self.start_time) * 1000)
    
    def has_errors(self) -> bool:
        """Check if any errors occurred during processing"""
        return len(self.errors) > 0
    
    def has_partial_failures(self) -> bool:
        """Check if any partial failures occurred"""
        return len(self.partial_failures) > 0


@dataclass
class ModelConfig:
    """Configuration for AI models (Requirements 6.1, 6.2)"""
    model_type: ModelType
    config_path: str
    weights_path: Optional[str] = None
    initialized: bool = False
    last_health_check: Optional[datetime] = None
    failure_count: int = 0
    max_failures: int = 3
    initialization_timeout_seconds: int = 300
    health_check_interval_seconds: int = 60
    
    def __post_init__(self):
        """Validate model configuration"""
        if self.max_failures < 1:
            raise ValueError("max_failures must be at least 1")
        if self.initialization_timeout_seconds < 1:
            raise ValueError("initialization_timeout_seconds must be at least 1")
        if self.health_check_interval_seconds < 1:
            raise ValueError("health_check_interval_seconds must be at least 1")
    
    def increment_failure_count(self) -> None:
        """Increment failure count for the model"""
        self.failure_count += 1
    
    def reset_failure_count(self) -> None:
        """Reset failure count after successful operation"""
        self.failure_count = 0
    
    def is_failure_threshold_exceeded(self) -> bool:
        """Check if failure threshold has been exceeded"""
        return self.failure_count >= self.max_failures
    
    def update_health_check(self) -> None:
        """Update last health check timestamp"""
        self.last_health_check = datetime.utcnow()


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrent request handling (Requirements 6.1, 6.2)"""
    max_concurrent_requests: int = 10
    queue_timeout_seconds: int = 30
    processing_timeout_seconds: int = 120
    memory_limit_mb: int = 1024
    cleanup_interval_seconds: int = 300
    
    def __post_init__(self):
        """Validate concurrency configuration"""
        if self.max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be at least 1")
        if self.queue_timeout_seconds < 1:
            raise ValueError("queue_timeout_seconds must be at least 1")
        if self.processing_timeout_seconds < 1:
            raise ValueError("processing_timeout_seconds must be at least 1")
        if self.memory_limit_mb < 1:
            raise ValueError("memory_limit_mb must be at least 1")


@dataclass
class RequestMetrics:
    """Metrics for monitoring and logging (Requirements 6.1, 6.2)"""
    request_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    processing_time_ms: Optional[int] = None
    memory_used_mb: float = 0.0
    models_used: List[str] = field(default_factory=list)
    errors_encountered: List[str] = field(default_factory=list)
    processing_stages: List[ProcessingStage] = field(default_factory=list)
    
    def add_model_used(self, model_type: ModelType) -> None:
        """Track which models were used in processing"""
        if model_type.value not in self.models_used:
            self.models_used.append(model_type.value)
    
    def add_error(self, error: str) -> None:
        """Add error to metrics tracking"""
        self.errors_encountered.append(error)
    
    def add_processing_stage(self, stage: ProcessingStage) -> None:
        """Track processing stages for performance analysis"""
        self.processing_stages.append(stage)
    
    def complete_processing(self) -> None:
        """Mark processing as complete and calculate final metrics"""
        self.end_time = datetime.utcnow()
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            self.processing_time_ms = int(delta.total_seconds() * 1000)


@dataclass
class QueuedRequest:
    """Model for queued requests in concurrency management (Requirements 6.1, 6.2)"""
    request_id: str
    processing_func: Any  # Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    queued_at: datetime = field(default_factory=datetime.utcnow)
    timeout_seconds: int = 120
    
    def __post_init__(self):
        """Validate queued request data"""
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")
    
    def is_expired(self) -> bool:
        """Check if the queued request has expired"""
        elapsed = datetime.utcnow() - self.queued_at
        return elapsed.total_seconds() > self.timeout_seconds


@dataclass
class SystemHealth:
    """Model for system health monitoring (Requirements 6.1, 6.2)"""
    overall_status: str  # healthy, degraded, unhealthy
    model_statuses: Dict[str, ModelStatus] = field(default_factory=dict)
    active_requests: int = 0
    queued_requests: int = 0
    memory_usage_mb: float = 0.0
    uptime_seconds: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate system health data"""
        valid_statuses = {"healthy", "degraded", "unhealthy"}
        if self.overall_status not in valid_statuses:
            raise ValueError(f"overall_status must be one of: {valid_statuses}")
    
    def update_model_status(self, model_type: ModelType, status: ModelStatus) -> None:
        """Update status for a specific model"""
        self.model_statuses[model_type.value] = status
        self.last_updated = datetime.utcnow()
    
    def get_healthy_models_count(self) -> int:
        """Get count of healthy models"""
        return sum(1 for status in self.model_statuses.values() 
                  if status == ModelStatus.READY)
    
    def get_failed_models_count(self) -> int:
        """Get count of failed models"""
        return sum(1 for status in self.model_statuses.values() 
                  if status == ModelStatus.FAILED)


@dataclass
class SystemHealth:
    """Comprehensive system health information for detailed status endpoint"""
    status: str
    timestamp: datetime
    uptime_seconds: int
    uptime_string: str
    version: str
    
    # Model information
    models: Dict[str, Any]
    healthy_models: int
    total_models: int
    
    # System resources
    cpu_usage_percent: float
    memory: Dict[str, Any]
    process_memory_mb: float
    disk: Dict[str, Any]
    
    # Performance metrics
    performance: Dict[str, Any]
    
    # Error tracking
    errors: Dict[str, Any]
    
    # Configuration
    configuration: Dict[str, Any]