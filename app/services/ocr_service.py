"""
OCR Service for menu text extraction and image processing.

This module provides OCR functionality for extracting text from menu images
with confidence scores, spatial layout preservation, and image preprocessing.
"""

import asyncio
import logging
import os
import base64
import json
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import io
import time
from abc import ABC, abstractmethod

from google import genai
from google.genai import types

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


class GeminiVisionOCRModel(BaseOCRModel):
    """Google Gemini Vision-based OCR model for menu text extraction"""
    
    def __init__(self, target_language: str = "en"):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.target_language = target_language
        
    async def extract_text_from_image(self, image: Image.Image) -> List[OCRResult]:
        """Extract text from image using Google Gemini Vision API"""
        if not self.client:
            self.logger.warning("No GOOGLE_API_KEY, falling back to mock")
            return await MockOCRModel().extract_text_from_image(image)
        
        try:
            # Convert PIL Image to bytes
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            image_bytes = buffered.getvalue()
            
            lang_name = {
                "en": "English", "vi": "Vietnamese", "ko": "Korean"
            }.get(self.target_language, "English")
            
            prompt = f"""You are a menu OCR and translation assistant.
Analyze this menu image and extract ALL food/drink items.

Return a JSON array where each item has:
- text: original text exactly as shown on menu
- translated: {lang_name} translation of the food name
- item_type: "food" for food/drink, "price" for standalone prices
- confidence: 0.0-1.0 based on text clarity
- y_position: vertical position (0=top, 100=bottom)
- price: the price if visible near this item (e.g., "₩6,000")
- image_url: a real, working image URL of this food dish from the web (use Wikipedia Commons, Korean food sites, or well-known food image sources)

IMPORTANT for image_url:
- Provide actual working URLs to images of the specific dish
- For Korean dishes, use URLs from Korean food sites or Wikipedia
- Example: "https://upload.wikimedia.org/wikipedia/commons/thumb/..." 
- If you cannot find a specific image URL, use null

Rules:
- Translate food names accurately to {lang_name}
- Keep prices in original format (₩, $, etc.)
- Skip standalone price-only items (item_type="price")
- Return ONLY valid JSON array, no markdown or explanation."""

            # Call Gemini Vision API (run sync in thread pool to avoid SSL issues)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model='gemini-2.0-flash',
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/jpeg'
                    )
                ]
            )
            
            # Parse response
            content = response.text
            self.logger.info(f"Gemini response: {content[:500]}...")
            
            # Clean up response - remove markdown code blocks if present
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                # Remove first and last lines (```json and ```)
                content = "\n".join(lines[1:-1])
            content = content.strip()
            
            # Parse JSON
            items = json.loads(content)
            
            # Convert to OCRResult objects
            results = []
            img_height = image.height
            img_width = image.width
            
            for i, item in enumerate(items):
                if isinstance(item, dict) and "text" in item:
                    text = item.get("text", "")
                    translated = item.get("translated", text)
                    item_type = item.get("item_type", "food")
                    price = item.get("price")
                    image_url = item.get("image_url")
                    confidence = float(item.get("confidence", 0.8))
                    y_pos = float(item.get("y_position", i * 10))
                    
                    # Skip standalone price items
                    if item_type == "price":
                        continue
                    
                    # Calculate bounding box based on y_position
                    y1 = int((y_pos / 100) * img_height)
                    y2 = min(y1 + 30, img_height)
                    x1 = 50
                    x2 = img_width - 50
                    
                    results.append(OCRResult(
                        text=text,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        group_id=f"menu_item_{i}",
                        translated_text=translated,
                        item_type=item_type,
                        price=price,
                        image_url=image_url
                    ))
            
            self.logger.info(f"Extracted {len(results)} food items with Gemini")
            return results
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Gemini response as JSON: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Gemini Vision OCR failed: {e}")
            raise OCRProcessingError(f"Vision OCR failed: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check if Gemini API is accessible"""
        return self.client is not None


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
        # Use OpenAI Vision model by default if API key is available
        if ocr_model:
            self.ocr_model = ocr_model
        elif os.getenv("OPENAI_API_KEY"):
            self.ocr_model = OpenAIVisionOCRModel()
        else:
            self.ocr_model = MockOCRModel()
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