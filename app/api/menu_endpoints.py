"""
Menu processing API endpoints.

Implements the main menu processing functionality including:
- POST /process-menu: Single menu image processing
- File upload handling with validation
- Integration with OCR, translation, and food image services
- Request validation and response formatting

Requirements: 2.1, 2.2, 3.1, 4.1, 5.1
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import time
import uuid
from datetime import datetime

from app.models import (
    MenuProcessingRequest,
    MenuProcessingResponse,
    ExtractedMenuItem,
    SupportedLanguage,
    StandardErrorResponse,
    ErrorCode
)
from app.core.dependencies import (
    get_processing_pipeline,
    get_concurrency_manager,
    get_request_id
)
from app.services import (
    ImageValidationError,
    OCRProcessingError,
    TranslationFailureError
)
from app.core.processing_pipeline import ProcessingPipeline
from app.core.concurrency_manager import ConcurrencyManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["menu-processing"])


@router.post(
    "/process-menu",
    response_model=MenuProcessingResponse,
    responses={
        400: {"model": StandardErrorResponse, "description": "Invalid request or image format"},
        413: {"model": StandardErrorResponse, "description": "Image too large"},
        422: {"model": StandardErrorResponse, "description": "Validation error"},
        500: {"model": StandardErrorResponse, "description": "Internal server error"},
        503: {"model": StandardErrorResponse, "description": "Service unavailable"}
    }
)
async def process_menu(
    request: Request,
    image: UploadFile = File(..., description="Menu image file (JPEG, PNG, WebP)"),
    target_language: SupportedLanguage = SupportedLanguage.ENGLISH,
    source_language: Optional[SupportedLanguage] = None,
    include_images: bool = True,
    max_images_per_item: int = 3,
    processing_pipeline: ProcessingPipeline = Depends(get_processing_pipeline),
    concurrency_manager: ConcurrencyManager = Depends(get_concurrency_manager),
    request_id: str = Depends(get_request_id)
) -> MenuProcessingResponse:
    """
    Process a menu image to extract and translate text with optional food images.
    
    This endpoint handles the complete menu processing pipeline with concurrency management:
    1. Validates uploaded image format and size (Requirements 2.1, 2.2)
    2. Extracts text using OCR with confidence scores (Requirement 3.1)
    3. Translates extracted text to target language (Requirement 4.1)
    4. Retrieves corresponding food images (Requirement 5.1)
    5. Implements error handling and monitoring (Requirements 6.1, 6.2, 6.3, 6.4)
    
    Args:
        image: Uploaded menu image file
        target_language: Target language for translation
        source_language: Source language (auto-detected if not provided)
        include_images: Whether to include food images in response
        max_images_per_item: Maximum number of images per menu item (1-10)
        processing_pipeline: Injected processing pipeline service
        concurrency_manager: Injected concurrency manager service
        request_id: Injected request ID
        
    Returns:
        MenuProcessingResponse with extracted and translated menu items
        
    Raises:
        HTTPException: For various error conditions with appropriate status codes
    """
    start_time = time.time()
    
    logger.info(
        f"Processing menu image request {request_id}",
        extra={
            'request_id': request_id,
            'target_language': target_language.value,
            'source_language': source_language.value if source_language else 'auto-detect',
            'include_images': include_images,
            'max_images_per_item': max_images_per_item,
            'image_filename': image.filename,
            'image_content_type': image.content_type
        }
    )
    
    try:
        # Create processing request model
        processing_request = MenuProcessingRequest(
            target_language=target_language,
            source_language=source_language,
            include_images=include_images,
            max_images_per_item=max_images_per_item
        )
        
        # Validate image file
        if not image.filename:
            raise HTTPException(
                status_code=400,
                detail=StandardErrorResponse(
                    error_code=ErrorCode.INVALID_IMAGE_FORMAT.value,
                    message="No image file provided",
                    request_id=request_id
                ).dict()
            )
        
        # Read image data
        try:
            image_data = await image.read()
        except Exception as e:
            logger.error(f"Failed to read image data: {str(e)}", extra={'request_id': request_id})
            raise HTTPException(
                status_code=400,
                detail=StandardErrorResponse(
                    error_code=ErrorCode.IMAGE_CORRUPTED.value,
                    message="Failed to read image data",
                    request_id=request_id
                ).dict()
            )
        
        # Check image size (Requirement 2.2)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(image_data) > max_size:
            raise HTTPException(
                status_code=413,
                detail=StandardErrorResponse(
                    error_code=ErrorCode.IMAGE_TOO_LARGE.value,
                    message=f"Image size ({len(image_data)} bytes) exceeds maximum allowed size ({max_size} bytes)",
                    request_id=request_id
                ).dict()
            )
        
        # Process the menu image through the pipeline with concurrency management
        # Implements Requirements 6.1, 6.2, 6.3, 6.4 for error handling and monitoring
        async def process_with_pipeline():
            return await processing_pipeline.process_menu_image(
                image_data=image_data,
                request=processing_request,
                request_id=request_id
            )
        
        # Use concurrency manager to handle the request with proper resource management
        response = await concurrency_manager.process_request(
            request_id=request_id,
            processing_func=process_with_pipeline
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        response.processing_time_ms = processing_time
        
        logger.info(
            f"Menu processing completed for request {request_id}",
            extra={
                'request_id': request_id,
                'processing_time_ms': processing_time,
                'items_found': response.total_items_found,
                'success': response.success,
                'partial_results': response.partial_results
            }
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except ImageValidationError as e:
        logger.warning(f"Image validation failed: {str(e)}", extra={'request_id': request_id})
        raise HTTPException(
            status_code=400,
            detail=StandardErrorResponse(
                error_code=ErrorCode.INVALID_IMAGE_FORMAT.value,
                message=str(e),
                request_id=request_id
            ).dict()
        )
        
    except OCRProcessingError as e:
        logger.error(f"OCR processing failed: {str(e)}", extra={'request_id': request_id})
        raise HTTPException(
            status_code=500,
            detail=StandardErrorResponse(
                error_code=ErrorCode.OCR_PROCESSING_FAILED.value,
                message="Failed to extract text from image",
                request_id=request_id
            ).dict()
        )
        
    except TranslationFailureError as e:
        logger.error(f"Translation failed: {str(e)}", extra={'request_id': request_id})
        raise HTTPException(
            status_code=500,
            detail=StandardErrorResponse(
                error_code=ErrorCode.TRANSLATION_FAILED.value,
                message="Failed to translate extracted text",
                request_id=request_id
            ).dict()
        )
        
    except Exception as e:
        logger.error(
            f"Unexpected error processing menu: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=StandardErrorResponse(
                error_code=ErrorCode.INTERNAL_SERVER_ERROR.value,
                message="Internal server error occurred during processing",
                request_id=request_id
            ).dict()
        )