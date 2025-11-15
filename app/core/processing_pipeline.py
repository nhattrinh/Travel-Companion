"""
Processing pipeline for menu image processing.

Orchestrates the complete workflow from image to translated results:
1. Image validation and preprocessing
2. OCR text extraction
3. Language detection and translation
4. Food image retrieval
5. Response formatting with error handling

Implements graceful degradation and partial result handling.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

from app.models import (
    MenuProcessingRequest,
    MenuProcessingResponse,
    ExtractedMenuItem,
    ProcessingContext,
    OCRResult,
    TranslationResult,
    FoodImage,
    SupportedLanguage,
    ErrorCode
)
from app.services import (
    OCRService,
    TranslationService,
    FoodImageService,
    ImageValidationError,
    OCRProcessingError,
    TranslationFailureError,
    FoodImageServiceError
)
from app.core.models import ModelManager

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """
    Orchestrates the complete menu processing workflow.
    
    Handles the integration of OCR, translation, and food image services
    with comprehensive error handling and graceful degradation.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        ocr_service: OCRService,
        translation_service: TranslationService,
        food_image_service: FoodImageService
    ):
        self.model_manager = model_manager
        self.ocr_service = ocr_service
        self.translation_service = translation_service
        self.food_image_service = food_image_service
        self.logger = logging.getLogger(__name__)
    
    async def process_menu_image(
        self,
        image_data: bytes,
        request: MenuProcessingRequest,
        request_id: str
    ) -> MenuProcessingResponse:
        """
        Process a single menu image through the complete pipeline.
        
        Args:
            image_data: Raw image bytes
            request: Processing request parameters
            request_id: Unique request identifier
            
        Returns:
            MenuProcessingResponse with processed results
        """
        context = ProcessingContext(
            request_id=request_id,
            start_time=datetime.utcnow().timestamp()
        )
        
        try:
            # Step 1: Validate and preprocess image
            context.processing_stage = "image_validation"
            validated_image = await self._validate_and_preprocess_image(image_data, context)
            
            # Step 2: Extract text using OCR
            context.processing_stage = "ocr_extraction"
            ocr_results = await self._extract_text_with_ocr(validated_image, context)
            
            if not ocr_results:
                # No text found, return empty response
                return self._create_empty_response(request_id, context)
            
            # Step 3: Detect source language if not provided
            context.processing_stage = "language_detection"
            detected_language = await self._detect_source_language(
                ocr_results, request.source_language, context
            )
            
            # Step 4: Translate extracted text
            context.processing_stage = "translation"
            translation_results = await self._translate_text(
                ocr_results, detected_language, request.target_language, context
            )
            
            # Step 5: Retrieve food images if requested
            context.processing_stage = "image_retrieval"
            food_images = {}
            if request.include_images:
                food_images = await self._retrieve_food_images(
                    translation_results, request.max_images_per_item, context
                )
            
            # Step 6: Format response
            context.processing_stage = "response_formatting"
            response = await self._format_response(
                ocr_results,
                translation_results,
                food_images,
                detected_language,
                request_id,
                context
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                f"Pipeline processing failed at stage {context.processing_stage}: {str(e)}",
                extra={'request_id': request_id, 'stage': context.processing_stage},
                exc_info=True
            )
            
            # Return partial results if available
            return await self._handle_pipeline_failure(context, request_id, e)
    
    async def _validate_and_preprocess_image(
        self, 
        image_data: bytes, 
        context: ProcessingContext
    ) -> Any:
        """Validate image format and preprocess for OCR."""
        try:
            # Use OCR service's image validation
            validated_image = await self.ocr_service.validate_image(image_data)
            
            # Preprocess for better OCR results
            preprocessed_image = await self.ocr_service.preprocess_image(validated_image)
            
            return preprocessed_image
            
        except ImageValidationError as e:
            context.add_error("image_validation", str(e))
            raise
        except Exception as e:
            context.add_error("image_validation", f"Unexpected validation error: {str(e)}")
            raise ImageValidationError(f"Image validation failed: {str(e)}")
    
    async def _extract_text_with_ocr(
        self, 
        image: Any, 
        context: ProcessingContext
    ) -> List[OCRResult]:
        """Extract text from image using OCR service."""
        try:
            ocr_results = await self.ocr_service.extract_text(image)
            
            # Filter low confidence results
            filtered_results = self.ocr_service.filter_low_confidence_results(ocr_results)
            
            # Group menu items by spatial proximity
            grouped_results = self.ocr_service.group_menu_items(filtered_results)
            
            # Flatten grouped results back to list
            final_results = []
            for group_id, group_results in grouped_results.items():
                for result in group_results:
                    result.group_id = group_id
                    final_results.append(result)
            
            context.ocr_results = final_results
            
            self.logger.info(
                f"OCR extracted {len(final_results)} text items",
                extra={'request_id': context.request_id, 'items_count': len(final_results)}
            )
            
            return final_results
            
        except OCRProcessingError as e:
            context.add_error("ocr_extraction", str(e))
            raise
        except Exception as e:
            context.add_error("ocr_extraction", f"Unexpected OCR error: {str(e)}")
            raise OCRProcessingError(f"OCR processing failed: {str(e)}")
    
    async def _detect_source_language(
        self,
        ocr_results: List[OCRResult],
        provided_language: Optional[SupportedLanguage],
        context: ProcessingContext
    ) -> SupportedLanguage:
        """Detect source language from OCR results."""
        if provided_language:
            return provided_language
        
        try:
            # Combine text from all OCR results for language detection
            combined_text = " ".join([result.text for result in ocr_results[:5]])  # Use first 5 items
            
            if not combined_text.strip():
                # Default to English if no text available
                return SupportedLanguage.ENGLISH
            
            detected_language = await self.translation_service.detect_language(combined_text)
            
            self.logger.info(
                f"Detected source language: {detected_language.value}",
                extra={'request_id': context.request_id, 'detected_language': detected_language.value}
            )
            
            return detected_language
            
        except Exception as e:
            context.add_error("language_detection", f"Language detection failed: {str(e)}")
            # Default to English on detection failure
            return SupportedLanguage.ENGLISH
    
    async def _translate_text(
        self,
        ocr_results: List[OCRResult],
        source_language: SupportedLanguage,
        target_language: SupportedLanguage,
        context: ProcessingContext
    ) -> List[TranslationResult]:
        """Translate extracted text to target language."""
        translation_results = []
        
        # Skip translation if source and target are the same
        if source_language == target_language:
            for ocr_result in ocr_results:
                translation_results.append(
                    TranslationResult(
                        translated_text=ocr_result.text,
                        source_language=source_language.value,
                        confidence=1.0
                    )
                )
            context.translation_results = translation_results
            return translation_results
        
        # Batch translate for efficiency
        texts_to_translate = [result.text for result in ocr_results]
        
        try:
            batch_results = await self.translation_service.batch_translate(
                texts_to_translate,
                target_language,
                source_language
            )
            
            translation_results = batch_results
            context.translation_results = translation_results
            
            self.logger.info(
                f"Translated {len(translation_results)} text items",
                extra={'request_id': context.request_id, 'items_count': len(translation_results)}
            )
            
        except TranslationFailureError as e:
            context.add_error("translation", str(e))
            
            # Graceful degradation: return original text with error indicators
            for ocr_result in ocr_results:
                translation_results.append(
                    TranslationResult(
                        translated_text=ocr_result.text,
                        source_language=source_language.value,
                        confidence=0.0,
                        error_indicator="Translation service unavailable"
                    )
                )
            
            context.translation_results = translation_results
            
        except Exception as e:
            context.add_error("translation", f"Unexpected translation error: {str(e)}")
            
            # Graceful degradation: return original text
            for ocr_result in ocr_results:
                translation_results.append(
                    TranslationResult(
                        translated_text=ocr_result.text,
                        source_language=source_language.value,
                        confidence=0.0,
                        error_indicator="Translation failed"
                    )
                )
            
            context.translation_results = translation_results
        
        return translation_results
    
    async def _retrieve_food_images(
        self,
        translation_results: List[TranslationResult],
        max_images_per_item: int,
        context: ProcessingContext
    ) -> Dict[str, List[FoodImage]]:
        """Retrieve food images for translated menu items."""
        food_images = {}
        
        # Process each translated item
        for i, translation_result in enumerate(translation_results):
            food_name = translation_result.translated_text.strip()
            
            if not food_name:
                continue
            
            try:
                images = await self.food_image_service.get_most_relevant_images(
                    food_name, 
                    limit=max_images_per_item
                )
                
                if images:
                    food_images[food_name] = images
                else:
                    # Handle no images found
                    placeholder_images = await self.food_image_service.handle_no_images_found(food_name)
                    food_images[food_name] = placeholder_images
                
            except FoodImageServiceError as e:
                context.add_partial_failure(food_name, f"Image retrieval failed: {str(e)}")
                
                # Graceful degradation: use placeholder
                try:
                    placeholder_images = await self.food_image_service.handle_image_service_failure(food_name)
                    food_images[food_name] = placeholder_images
                except Exception:
                    # Complete failure, skip images for this item
                    food_images[food_name] = []
                
            except Exception as e:
                context.add_partial_failure(food_name, f"Unexpected image error: {str(e)}")
                food_images[food_name] = []
        
        context.food_images = food_images
        
        self.logger.info(
            f"Retrieved images for {len(food_images)} menu items",
            extra={'request_id': context.request_id, 'items_with_images': len(food_images)}
        )
        
        return food_images
    
    async def _format_response(
        self,
        ocr_results: List[OCRResult],
        translation_results: List[TranslationResult],
        food_images: Dict[str, List[FoodImage]],
        detected_language: SupportedLanguage,
        request_id: str,
        context: ProcessingContext
    ) -> MenuProcessingResponse:
        """Format the final response with all processed data."""
        menu_items = []
        
        # Combine OCR and translation results
        for i, (ocr_result, translation_result) in enumerate(zip(ocr_results, translation_results)):
            # Get food images for this item
            item_images = food_images.get(translation_result.translated_text.strip(), [])
            image_urls = [img.url for img in item_images if not img.is_placeholder]
            
            # Determine processing status
            processing_status = "success"
            error_details = None
            
            if translation_result.error_indicator:
                processing_status = "partial"
                error_details = translation_result.error_indicator
            elif ocr_result.confidence < 0.5:
                processing_status = "partial"
                error_details = "Low OCR confidence"
            
            menu_item = ExtractedMenuItem(
                original_text=ocr_result.text,
                translated_text=translation_result.translated_text,
                confidence_score=min(ocr_result.confidence, translation_result.confidence),
                bounding_box=[ocr_result.bbox[0], ocr_result.bbox[1], ocr_result.bbox[2], ocr_result.bbox[3]],
                food_images=image_urls,
                processing_status=processing_status,
                error_details=error_details
            )
            
            menu_items.append(menu_item)
        
        # Determine overall success status
        success = len(context.errors) == 0
        partial_results = len(context.partial_failures) > 0 or any(
            item.processing_status != "success" for item in menu_items
        )
        
        # Create response
        response = MenuProcessingResponse(
            request_id=request_id,
            processing_time_ms=0,  # Will be set by endpoint
            source_language_detected=detected_language.value,
            menu_items=menu_items,
            total_items_found=len(menu_items),
            success=success,
            partial_results=partial_results,
            error_message="; ".join(context.errors) if context.errors else None,
            timestamp=datetime.utcnow()
        )
        
        return response
    
    def _create_empty_response(self, request_id: str, context: ProcessingContext) -> MenuProcessingResponse:
        """Create response when no text is detected."""
        return MenuProcessingResponse(
            request_id=request_id,
            processing_time_ms=0,
            source_language_detected=SupportedLanguage.ENGLISH.value,
            menu_items=[],
            total_items_found=0,
            success=True,
            partial_results=False,
            error_message="No text detected in image",
            timestamp=datetime.utcnow()
        )
    
    async def _handle_pipeline_failure(
        self, 
        context: ProcessingContext, 
        request_id: str, 
        error: Exception
    ) -> MenuProcessingResponse:
        """Handle complete pipeline failure with partial results if available."""
        # Try to return partial results if any stage completed successfully
        menu_items = []
        
        if context.ocr_results and context.translation_results:
            # Both OCR and translation completed, format what we have
            for ocr_result, translation_result in zip(context.ocr_results, context.translation_results):
                menu_item = ExtractedMenuItem(
                    original_text=ocr_result.text,
                    translated_text=translation_result.translated_text,
                    confidence_score=min(ocr_result.confidence, translation_result.confidence),
                    bounding_box=[ocr_result.bbox[0], ocr_result.bbox[1], ocr_result.bbox[2], ocr_result.bbox[3]],
                    food_images=[],
                    processing_status="partial",
                    error_details="Processing incomplete due to pipeline failure"
                )
                menu_items.append(menu_item)
        
        elif context.ocr_results:
            # Only OCR completed, return original text
            for ocr_result in context.ocr_results:
                menu_item = ExtractedMenuItem(
                    original_text=ocr_result.text,
                    translated_text=ocr_result.text,
                    confidence_score=ocr_result.confidence,
                    bounding_box=[ocr_result.bbox[0], ocr_result.bbox[1], ocr_result.bbox[2], ocr_result.bbox[3]],
                    food_images=[],
                    processing_status="failed",
                    error_details="Translation failed"
                )
                menu_items.append(menu_item)
        
        return MenuProcessingResponse(
            request_id=request_id,
            processing_time_ms=0,
            source_language_detected=SupportedLanguage.ENGLISH.value,
            menu_items=menu_items,
            total_items_found=len(menu_items),
            success=False,
            partial_results=len(menu_items) > 0,
            error_message=f"Pipeline failed at {context.processing_stage}: {str(error)}",
            timestamp=datetime.utcnow()
        )