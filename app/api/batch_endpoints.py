"""
Batch processing API endpoints.

Implements batch menu processing functionality including:
- POST /process-menu-batch: Multiple menu image processing
- Concurrent processing with resource management
- Proper error handling and partial results
- Request queuing and timeout management

Requirements: 2.3, 7.1, 7.3
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
import time
import uuid
import asyncio
from datetime import datetime

from app.models import (
    BatchProcessingRequest,
    BatchProcessingResponse,
    MenuProcessingRequest,
    MenuProcessingResponse,
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

router = APIRouter(prefix="/api/v1", tags=["batch-processing"])


@router.post(
    "/process-menu-batch",
    response_model=BatchProcessingResponse,
    responses={
        400: {"model": StandardErrorResponse, "description": "Invalid request or image format"},
        413: {"model": StandardErrorResponse, "description": "Too many images or images too large"},
        422: {"model": StandardErrorResponse, "description": "Validation error"},
        429: {"model": StandardErrorResponse, "description": "Too many concurrent requests"},
        500: {"model": StandardErrorResponse, "description": "Internal server error"},
        503: {"model": StandardErrorResponse, "description": "Service unavailable"}
    }
)
async def process_menu_batch(
    request: Request,
    images: List[UploadFile] = File(..., description="Menu image files (JPEG, PNG, WebP)"),
    target_language: SupportedLanguage = Form(SupportedLanguage.ENGLISH),
    source_language: Optional[SupportedLanguage] = Form(None),
    include_images: bool = Form(True),
    max_images_per_item: int = Form(3),
    processing_pipeline: ProcessingPipeline = Depends(get_processing_pipeline),
    concurrency_manager: ConcurrencyManager = Depends(get_concurrency_manager),
    request_id: str = Depends(get_request_id)
) -> BatchProcessingResponse:
    """
    Process multiple menu images in batch with concurrent processing.
    
    This endpoint handles batch processing of menu images with:
    1. Concurrent processing of multiple images (Requirement 7.1)
    2. Resource management and queuing (Requirement 7.1)
    3. Timeout handling for long-running operations (Requirement 7.3)
    4. Proper error handling with partial results (Requirement 2.3)
    5. Integrated error handling and monitoring (Requirements 6.1, 6.2, 6.3, 6.4)
    
    Args:
        images: List of uploaded menu image files (max 10)
        target_language: Target language for translation
        source_language: Source language (auto-detected if not provided)
        include_images: Whether to include food images in response
        max_images_per_item: Maximum number of images per menu item (1-10)
        processing_pipeline: Injected processing pipeline service
        concurrency_manager: Injected concurrency manager service
        request_id: Injected request ID
        
    Returns:
        BatchProcessingResponse with results for all processed images
        
    Raises:
        HTTPException: For various error conditions with appropriate status codes
    """
    batch_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        f"Processing batch request {batch_id} with {len(images)} images",
        extra={
            'request_id': request_id,
            'batch_id': batch_id,
            'image_count': len(images),
            'target_language': target_language.value,
            'source_language': source_language.value if source_language else 'auto-detect',
            'include_images': include_images,
            'max_images_per_item': max_images_per_item
        }
    )
    
    # Validate batch size (Requirement 2.3)
    max_batch_size = 10
    if len(images) > max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=StandardErrorResponse(
                error_code=ErrorCode.IMAGE_TOO_LARGE.value,
                message=f"Batch size ({len(images)}) exceeds maximum allowed ({max_batch_size})",
                request_id=request_id
            ).dict()
        )
    
    if len(images) == 0:
        raise HTTPException(
            status_code=400,
            detail=StandardErrorResponse(
                error_code=ErrorCode.INVALID_IMAGE_FORMAT.value,
                message="No images provided for batch processing",
                request_id=request_id
            ).dict()
        )
    
    try:
        # Create processing request template
        processing_request_template = MenuProcessingRequest(
            target_language=target_language,
            source_language=source_language,
            include_images=include_images,
            max_images_per_item=max_images_per_item
        )
        
        # Use injected services for batch processing
        # The processing pipeline and concurrency manager are already configured
        # with proper error handling and monitoring (Requirements 6.1, 6.2, 6.3, 6.4)
        
        # Prepare batch processing tasks
        processing_tasks = []
        image_data_list = []
        
        # Read and validate all images first
        for i, image in enumerate(images):
            try:
                if not image.filename:
                    raise ImageValidationError(f"Image {i+1}: No filename provided")
                
                # Read image data
                image_data = await image.read()
                
                # Check individual image size
                max_size = 10 * 1024 * 1024  # 10MB per image
                if len(image_data) > max_size:
                    raise ImageValidationError(
                        f"Image {i+1} ({image.filename}): Size ({len(image_data)} bytes) exceeds maximum ({max_size} bytes)"
                    )
                
                image_data_list.append((image_data, f"{batch_id}_{i+1}", image.filename))
                
            except Exception as e:
                logger.warning(
                    f"Failed to read image {i+1} ({image.filename}): {str(e)}",
                    extra={'request_id': request_id, 'batch_id': batch_id, 'image_index': i}
                )
                # Add placeholder for failed image
                image_data_list.append((None, f"{batch_id}_{i+1}", image.filename))
        
        # Process images concurrently with resource management
        async def process_single_image(image_data_tuple):
            """Process a single image with error handling."""
            image_data, image_request_id, filename = image_data_tuple
            
            if image_data is None:
                # Return error response for failed image read
                return MenuProcessingResponse(
                    request_id=image_request_id,
                    processing_time_ms=0,
                    source_language_detected=SupportedLanguage.ENGLISH.value,
                    menu_items=[],
                    total_items_found=0,
                    success=False,
                    partial_results=False,
                    error_message=f"Failed to read image data for {filename}",
                    timestamp=datetime.utcnow()
                )
            
            try:
                # Process through concurrency manager with timeout
                async def process_with_pipeline():
                    return await processing_pipeline.process_menu_image(
                        image_data=image_data,
                        request=processing_request_template,
                        request_id=image_request_id
                    )
                
                result = await concurrency_manager.process_request(
                    request_id=image_request_id,
                    processing_func=process_with_pipeline
                )
                
                return result
                
            except asyncio.TimeoutError:
                logger.warning(
                    f"Image processing timed out for {filename}",
                    extra={'request_id': request_id, 'image_request_id': image_request_id}
                )
                
                return MenuProcessingResponse(
                    request_id=image_request_id,
                    processing_time_ms=0,
                    source_language_detected=SupportedLanguage.ENGLISH.value,
                    menu_items=[],
                    total_items_found=0,
                    success=False,
                    partial_results=False,
                    error_message=f"Processing timeout for {filename}",
                    timestamp=datetime.utcnow()
                )
                
            except Exception as e:
                logger.error(
                    f"Failed to process image {filename}: {str(e)}",
                    extra={'request_id': request_id, 'image_request_id': image_request_id},
                    exc_info=True
                )
                
                return MenuProcessingResponse(
                    request_id=image_request_id,
                    processing_time_ms=0,
                    source_language_detected=SupportedLanguage.ENGLISH.value,
                    menu_items=[],
                    total_items_found=0,
                    success=False,
                    partial_results=False,
                    error_message=f"Processing failed for {filename}: {str(e)}",
                    timestamp=datetime.utcnow()
                )
        
        # Execute batch processing with concurrency control
        try:
            # Use asyncio.gather with timeout for the entire batch
            batch_timeout = 300  # 5 minutes for entire batch
            
            results = await asyncio.wait_for(
                asyncio.gather(
                    *[process_single_image(image_data_tuple) for image_data_tuple in image_data_list],
                    return_exceptions=True
                ),
                timeout=batch_timeout
            )
            
        except asyncio.TimeoutError:
            logger.error(
                f"Batch processing timed out after {batch_timeout} seconds",
                extra={'request_id': request_id, 'batch_id': batch_id}
            )
            
            raise HTTPException(
                status_code=408,
                detail=StandardErrorResponse(
                    error_code=ErrorCode.PROCESSING_TIMEOUT.value,
                    message=f"Batch processing timed out after {batch_timeout} seconds",
                    request_id=request_id
                ).dict()
            )
        
        # Process results and handle exceptions
        processed_results = []
        successful_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exception during processing
                logger.error(
                    f"Exception during batch processing for image {i+1}: {str(result)}",
                    extra={'request_id': request_id, 'batch_id': batch_id, 'image_index': i}
                )
                
                error_response = MenuProcessingResponse(
                    request_id=f"{batch_id}_{i+1}",
                    processing_time_ms=0,
                    source_language_detected=SupportedLanguage.ENGLISH.value,
                    menu_items=[],
                    total_items_found=0,
                    success=False,
                    partial_results=False,
                    error_message=f"Processing exception: {str(result)}",
                    timestamp=datetime.utcnow()
                )
                processed_results.append(error_response)
                
            else:
                # Valid result
                processed_results.append(result)
                if result.success:
                    successful_count += 1
        
        # Calculate total processing time
        total_processing_time = int((time.time() - start_time) * 1000)
        
        # Create batch response
        batch_response = BatchProcessingResponse(
            batch_id=batch_id,
            results=processed_results,
            total_processed=len(processed_results),
            total_successful=successful_count,
            processing_time_ms=total_processing_time
        )
        
        logger.info(
            f"Batch processing completed for {batch_id}",
            extra={
                'request_id': request_id,
                'batch_id': batch_id,
                'total_processed': len(processed_results),
                'total_successful': successful_count,
                'processing_time_ms': total_processing_time,
                'success_rate': (successful_count / len(processed_results)) * 100 if processed_results else 0
            }
        )
        
        return batch_response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        logger.error(
            f"Unexpected error during batch processing: {str(e)}",
            extra={'request_id': request_id, 'batch_id': batch_id},
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail=StandardErrorResponse(
                error_code=ErrorCode.INTERNAL_SERVER_ERROR.value,
                message="Internal server error occurred during batch processing",
                request_id=request_id
            ).dict()
        )