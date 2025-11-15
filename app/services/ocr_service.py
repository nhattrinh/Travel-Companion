"""
OCR Service for menu text extraction and image processing.

This module provides OCR functionality for extracting text from menu images
with confidence scores, spatial layout preservation, and image preprocessing.
"""

import asyncio
import logging
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import time
from abc import ABC, abstractmethod

from ..models.internal_models import OCRResult, ProcessingStage, ErrorCode
from .image_processor import ImageProcessor, ImageValidationError


class OCRProcessingError(Exception):
    """Raised when OCR processing fails (Requirement 3.4)"""
    pass


class BaseOCRModel(ABC):
    """Abstract base class for OCR models (Requirement 3.1)"""
    
    @abstractmethod
    async def extract_text_from_image(self, image: Image.Image) -> List[OCRResult]:
        """Extract text from image with confidence scores and bounding boxes"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if OCR model is healthy and ready"""
        pass


class MockOCRModel(BaseOCRModel):
    """Mock OCR model for testing and development (Requirement 3.1)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_healthy = True
    
    async def extract_text_from_image(self, image: Image.Image) -> List[OCRResult]:
        """Mock text extraction with simulated results (Requirement 3.1)"""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Mock OCR results with varying confidence scores
        mock_results = [
            OCRResult(
                text="Grilled Salmon",
                confidence=0.95,
                bbox=(100, 50, 300, 80),
                group_id="menu_item_1"
            ),
            OCRResult(
                text="Fresh Atlantic salmon with herbs",
                confidence=0.88,
                bbox=(100, 85, 400, 110),
                group_id="menu_item_1"
            ),
            OCRResult(
                text="$24.99",
                confidence=0.92,
                bbox=(450, 50, 520, 80),
                group_id="menu_item_1"
            ),
            OCRResult(
                text="Pasta Carbonara",
                confidence=0.91,
                bbox=(100, 150, 280, 180),
                group_id="menu_item_2"
            ),
            OCRResult(
                text="Traditional Italian pasta",
                confidence=0.85,
                bbox=(100, 185, 350, 210),
                group_id="menu_item_2"
            ),
            OCRResult(
                text="$18.50",
                confidence=0.89,
                bbox=(450, 150, 520, 180),
                group_id="menu_item_2"
            )
        ]
        
        return mock_results
    
    async def health_check(self) -> bool:
        """Mock health check (Requirement 3.1)"""
        return self.is_healthy


class OCRService:
    """
    OCR service for extracting text from menu images.
    
    Provides text extraction with confidence scores, spatial layout preservation,
    image validation, and preprocessing capabilities.
    """
    
    # OCR processing parameters (Requirement 3.3)
    MIN_CONFIDENCE_THRESHOLD = 0.3
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    
    def __init__(self, ocr_model: Optional[BaseOCRModel] = None):
        """Initialize OCR service with optional model"""
        self.ocr_model = ocr_model or MockOCRModel()
        self.image_processor = ImageProcessor()
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = self.DEFAULT_CONFIDENCE_THRESHOLD
    
    async def extract_text(self, image: Image.Image) -> List[OCRResult]:
        """
        Extract text from image with confidence scores and spatial layout.
        
        Args:
            image: PIL Image object to process
            
        Returns:
            List of OCRResult objects with text, confidence, and bounding boxes
            
        Raises:
            OCRProcessingError: If OCR processing fails
            
        Requirements: 3.1, 3.2, 3.3
        """
        try:
            start_time = time.time()
            
            # Preprocess image for better OCR results
            preprocessed_image = await self.image_processor.preprocess_for_ocr(image)
            
            # Extract text using OCR model
            ocr_results = await self.ocr_model.extract_text_from_image(preprocessed_image)
            
            # Filter results based on confidence threshold
            filtered_results = self.filter_low_confidence_results(ocr_results)
            
            # Group menu items by spatial proximity
            grouped_results = self.group_menu_items(filtered_results)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Add processing time to results
            for result in filtered_results:
                result.processing_time_ms = processing_time
            
            self.logger.info(
                f"OCR extraction completed: {len(filtered_results)} items found "
                f"in {processing_time}ms"
            )
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {str(e)}")
            raise OCRProcessingError(f"Failed to extract text from image: {str(e)}")
    

    
    async def validate_image(self, image_data: bytes) -> Image.Image:
        """
        Validate image format, size, and integrity.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Validated PIL Image object
            
        Raises:
            ImageValidationError: If validation fails
            
        Requirements: 2.1, 2.2, 2.4
        """
        return await self.image_processor.validate_image(image_data)
    
    async def process_batch_images(self, images_data: List[bytes]) -> List[List[OCRResult]]:
        """
        Process multiple images in batch.
        
        Args:
            images_data: List of raw image bytes
            
        Returns:
            List of OCR results for each image
            
        Requirements: 2.3
        """
        results = []
        
        # Use image processor for batch validation and preprocessing
        try:
            validated_images = await self.image_processor.process_batch_images(
                images_data, preprocess=True
            )
            
            # Process each validated image with OCR
            for i, image in enumerate(validated_images):
                try:
                    ocr_results = await self.extract_text(image)
                    results.append(ocr_results)
                    
                    self.logger.info(f"Batch OCR processing: image {i+1}/{len(validated_images)} completed")
                    
                except OCRProcessingError as e:
                    self.logger.error(f"Batch OCR processing: image {i+1} failed: {str(e)}")
                    # Add empty results for failed OCR processing
                    results.append([])
                    
        except ImageValidationError as e:
            self.logger.error(f"Batch image validation failed: {str(e)}")
            raise
        
        return results
    
    def filter_low_confidence_results(self, results: List[OCRResult]) -> List[OCRResult]:
        """
        Filter OCR results based on confidence threshold.
        
        Args:
            results: List of OCRResult objects
            
        Returns:
            Filtered list of OCRResult objects
            
        Requirements: 3.3
        """
        filtered_results = [
            result for result in results 
            if result.confidence >= self.confidence_threshold
        ]
        
        if len(filtered_results) < len(results):
            filtered_count = len(results) - len(filtered_results)
            self.logger.info(
                f"Filtered {filtered_count} low-confidence results "
                f"(threshold: {self.confidence_threshold})"
            )
        
        return filtered_results
    
    def group_menu_items(self, results: List[OCRResult]) -> Dict[str, List[OCRResult]]:
        """
        Group OCR results by spatial proximity for menu item organization.
        
        Args:
            results: List of OCRResult objects
            
        Returns:
            Dictionary mapping group IDs to lists of OCRResult objects
            
        Requirements: 3.2
        """
        grouped_results = {}
        
        for result in results:
            group_id = result.group_id or "ungrouped"
            
            if group_id not in grouped_results:
                grouped_results[group_id] = []
            
            grouped_results[group_id].append(result)
        
        # Sort groups by vertical position (top to bottom)
        for group_id, group_results in grouped_results.items():
            group_results.sort(key=lambda r: r.bbox[1])  # Sort by y1 coordinate
        
        self.logger.info(f"Grouped {len(results)} results into {len(grouped_results)} groups")
        
        return grouped_results
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Set the confidence threshold for filtering results.
        
        Args:
            threshold: Confidence threshold (0.0 to 1.0)
            
        Requirements: 3.3
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        self.confidence_threshold = threshold
        self.logger.info(f"Confidence threshold set to {threshold}")
    
    async def health_check(self) -> bool:
        """
        Check if OCR service is healthy and ready.
        
        Returns:
            True if service is healthy, False otherwise
            
        Requirements: 3.1
        """
        try:
            return await self.ocr_model.health_check()
        except Exception as e:
            self.logger.error(f"OCR service health check failed: {str(e)}")
            return False
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats"""
        return self.image_processor.get_supported_formats()
    
    def get_size_limits(self) -> Dict[str, int]:
        """Get image size limits"""
        return self.image_processor.get_size_limits()