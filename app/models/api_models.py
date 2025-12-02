"""
API request and response models for the menu translation backend.

This module contains Pydantic models for API requests and responses,
including validation rules for supported languages, image limits, and request parameters.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


class SupportedLanguage(str, Enum):
    """Supported languages for translation (Requirements 8.1, 8.2)"""
    ENGLISH = "en"
    VIETNAMESE = "vn"
    KOREAN = "ko"


class MenuProcessingRequest(BaseModel):
    """Request model for menu processing endpoint (Requirements 8.1, 8.2, 2.2)"""
    target_language: SupportedLanguage
    source_language: Optional[SupportedLanguage] = None
    include_images: bool = Field(default=True)
    max_images_per_item: int = Field(default=3, ge=1, le=10, description="Maximum number of images per menu item")
    
    @validator('target_language')
    def validate_target_language(cls, v):
        """Validate target language is supported (Requirement 8.1)"""
        if v not in SupportedLanguage:
            raise ValueError(f"Unsupported target language: {v}")
        return v
    
    @validator('max_images_per_item')
    def validate_image_limits(cls, v):
        """Validate image limits (Requirement 2.2)"""
        if v < 1 or v > 10:
            raise ValueError("max_images_per_item must be between 1 and 10")
        return v


class ExtractedMenuItem(BaseModel):
    """Model for individual extracted and translated menu items (Requirements 8.1, 8.2)"""
    original_text: str
    translated_text: str
    confidence_score: float = Field(ge=0.0, le=1.0, description="OCR confidence score")
    bounding_box: List[int] = Field(description="Bounding box coordinates [x1, y1, x2, y2]")
    food_images: List[str] = Field(default_factory=list, description="URLs of food images")
    processing_status: str = Field(default="success", description="Processing status: success, partial, failed")
    error_details: Optional[str] = None
    
    @validator('bounding_box')
    def validate_bounding_box(cls, v):
        """Validate bounding box format (Requirement 8.1)"""
        if len(v) != 4:
            raise ValueError("bounding_box must contain exactly 4 coordinates [x1, y1, x2, y2]")
        if not all(isinstance(coord, int) and coord >= 0 for coord in v):
            raise ValueError("bounding_box coordinates must be non-negative integers")
        return v
    
    @validator('processing_status')
    def validate_processing_status(cls, v):
        """Validate processing status values (Requirement 8.1)"""
        valid_statuses = {"success", "partial", "failed"}
        if v not in valid_statuses:
            raise ValueError(f"processing_status must be one of: {valid_statuses}")
        return v


class MenuProcessingResponse(BaseModel):
    """Response model for menu processing endpoint (Requirements 8.1, 8.2)"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    processing_time_ms: int = Field(ge=0, description="Processing time in milliseconds")
    source_language_detected: str
    menu_items: List[ExtractedMenuItem]
    total_items_found: int = Field(ge=0, description="Total number of menu items found")
    success: bool
    partial_results: bool = Field(default=False, description="Indicates if some items failed processing")
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('total_items_found')
    def validate_total_items(cls, v, values):
        """Validate total items count matches menu items (Requirement 8.1)"""
        menu_items = values.get('menu_items', [])
        if v != len(menu_items):
            raise ValueError("total_items_found must match the number of menu_items")
        return v


class BatchProcessingRequest(BaseModel):
    """Request model for batch processing endpoint (Requirements 8.1, 2.2)"""
    requests: List[MenuProcessingRequest] = Field(max_items=10, description="Maximum 10 requests per batch")
    
    @validator('requests')
    def validate_batch_size(cls, v):
        """Validate batch size limits (Requirement 2.2)"""
        if len(v) == 0:
            raise ValueError("Batch must contain at least one request")
        if len(v) > 10:
            raise ValueError("Batch cannot contain more than 10 requests")
        return v


class BatchProcessingResponse(BaseModel):
    """Response model for batch processing endpoint (Requirements 8.1, 8.2)"""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    results: List[MenuProcessingResponse]
    total_processed: int = Field(ge=0, description="Total number of requests processed")
    total_successful: int = Field(ge=0, description="Number of successful requests")
    processing_time_ms: int = Field(ge=0, description="Total batch processing time")
    
    @validator('total_processed')
    def validate_total_processed(cls, v, values):
        """Validate total processed matches results (Requirement 8.1)"""
        results = values.get('results', [])
        if v != len(results):
            raise ValueError("total_processed must match the number of results")
        return v
    
    @validator('total_successful')
    def validate_total_successful(cls, v, values):
        """Validate successful count (Requirement 8.1)"""
        results = values.get('results', [])
        actual_successful = sum(1 for result in results if result.success)
        if v != actual_successful:
            raise ValueError("total_successful must match the number of successful results")
        return v


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint (Requirements 8.1, 8.2)"""
    status: str = Field(description="Overall system status: healthy, degraded, unhealthy")
    models_status: Dict[str, bool] = Field(description="Status of individual AI models")
    uptime_seconds: int = Field(ge=0, description="System uptime in seconds")
    version: str = Field(description="API version")
    concurrent_requests: int = Field(ge=0, description="Number of concurrent requests being processed")
    memory_usage_mb: float = Field(ge=0.0, description="Current memory usage in MB")
    
    @validator('status')
    def validate_status(cls, v):
        """Validate health status values (Requirement 8.1)"""
        valid_statuses = {"healthy", "degraded", "unhealthy"}
        if v not in valid_statuses:
            raise ValueError(f"status must be one of: {valid_statuses}")
        return v


class StandardErrorResponse(BaseModel):
    """Standardized error response format (Requirements 8.1, 8.2)"""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('error_code')
    def validate_error_code(cls, v):
        """Validate error code format (Requirement 8.1)"""
        if not v or not v.isupper():
            raise ValueError("error_code must be a non-empty uppercase string")
        return v
