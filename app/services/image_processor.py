"""
Image processing and validation utilities for menu images.

This module provides comprehensive image validation, preprocessing,
and optimization functions for OCR processing.
"""

import asyncio
import logging
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import hashlib
from enum import Enum

from ..models.internal_models import ErrorCode


class ImageFormat(str, Enum):
    """Supported image formats (Requirement 2.1)"""
    JPEG = "JPEG"
    JPG = "JPEG"  # Alias for JPEG
    PNG = "PNG"
    WEBP = "WEBP"


class ImageValidationError(Exception):
    """Raised when image validation fails (Requirements 2.1, 2.2, 2.4)"""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.IMAGE_CORRUPTED):
        super().__init__(message)
        self.error_code = error_code


class ImageProcessor:
    """
    Comprehensive image processing and validation service.
    
    Handles image format validation, size checking, preprocessing for OCR,
    and error handling for corrupted images.
    """
    
    # Image format validation (Requirement 2.1)
    SUPPORTED_FORMATS = {
        ImageFormat.JPEG.value,
        ImageFormat.PNG.value, 
        ImageFormat.WEBP.value
    }
    
    # Size limits (Requirement 2.2)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MIN_FILE_SIZE = 1024  # 1KB
    MAX_DIMENSION = 4096  # pixels
    MIN_DIMENSION = 32  # pixels
    
    # Processing parameters
    DEFAULT_QUALITY = 85
    CONTRAST_ENHANCEMENT = 1.2
    SHARPNESS_ENHANCEMENT = 1.1
    NOISE_REDUCTION_SIZE = 3
    
    def __init__(self):
        """Initialize image processor"""
        self.logger = logging.getLogger(__name__)
    
    async def validate_image_format(self, image_data: bytes) -> str:
        """
        Validate image format against supported formats.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Validated image format string
            
        Raises:
            ImageValidationError: If format is not supported
            
        Requirements: 2.1
        """
        try:
            # Try to determine format from image data
            image = Image.open(io.BytesIO(image_data))
            format_name = image.format
            
            if format_name not in self.SUPPORTED_FORMATS:
                raise ImageValidationError(
                    f"Unsupported image format: {format_name}. "
                    f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}",
                    ErrorCode.INVALID_IMAGE_FORMAT
                )
            
            self.logger.debug(f"Image format validation passed: {format_name}")
            return format_name
            
        except IOError as e:
            raise ImageValidationError(
                f"Cannot determine image format: {str(e)}",
                ErrorCode.INVALID_IMAGE_FORMAT
            )
    
    async def validate_image_size(self, image_data: bytes) -> Tuple[int, int]:
        """
        Validate image file size and dimensions.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Tuple of (width, height) in pixels
            
        Raises:
            ImageValidationError: If size validation fails
            
        Requirements: 2.2
        """
        # Check file size
        file_size = len(image_data)
        
        if file_size > self.MAX_FILE_SIZE:
            raise ImageValidationError(
                f"Image file size {file_size} bytes exceeds maximum "
                f"allowed size of {self.MAX_FILE_SIZE} bytes",
                ErrorCode.IMAGE_TOO_LARGE
            )
        
        if file_size < self.MIN_FILE_SIZE:
            raise ImageValidationError(
                f"Image file size {file_size} bytes is below minimum "
                f"required size of {self.MIN_FILE_SIZE} bytes",
                ErrorCode.INVALID_IMAGE_FORMAT
            )
        
        try:
            # Check image dimensions
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            
            if width > self.MAX_DIMENSION or height > self.MAX_DIMENSION:
                raise ImageValidationError(
                    f"Image dimensions {width}x{height} exceed maximum "
                    f"allowed dimensions of {self.MAX_DIMENSION}x{self.MAX_DIMENSION}",
                    ErrorCode.IMAGE_TOO_LARGE
                )
            
            if width < self.MIN_DIMENSION or height < self.MIN_DIMENSION:
                raise ImageValidationError(
                    f"Image dimensions {width}x{height} are below minimum "
                    f"required dimensions of {self.MIN_DIMENSION}x{self.MIN_DIMENSION}",
                    ErrorCode.INVALID_IMAGE_FORMAT
                )
            
            self.logger.debug(f"Image size validation passed: {width}x{height}, {file_size} bytes")
            return width, height
            
        except IOError as e:
            raise ImageValidationError(
                f"Cannot read image dimensions: {str(e)}",
                ErrorCode.IMAGE_CORRUPTED
            )
    
    async def validate_image_integrity(self, image_data: bytes) -> bool:
        """
        Validate image integrity and detect corruption.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            True if image is valid and not corrupted
            
        Raises:
            ImageValidationError: If image is corrupted
            
        Requirements: 2.4
        """
        try:
            # Try to open and verify the image
            image = Image.open(io.BytesIO(image_data))
            
            # Verify image integrity
            image.verify()
            
            # Try to load image data to ensure it's readable
            image = Image.open(io.BytesIO(image_data))
            image.load()
            
            self.logger.debug("Image integrity validation passed")
            return True
            
        except (IOError, OSError, ValueError) as e:
            raise ImageValidationError(
                f"Image is corrupted or invalid: {str(e)}",
                ErrorCode.IMAGE_CORRUPTED
            )
    
    async def validate_image(self, image_data: bytes) -> Image.Image:
        """
        Comprehensive image validation including format, size, and integrity.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Validated PIL Image object
            
        Raises:
            ImageValidationError: If any validation fails
            
        Requirements: 2.1, 2.2, 2.4
        """
        # Validate format
        image_format = await self.validate_image_format(image_data)
        
        # Validate size
        width, height = await self.validate_image_size(image_data)
        
        # Validate integrity
        await self.validate_image_integrity(image_data)
        
        # Return validated image
        image = Image.open(io.BytesIO(image_data))
        
        self.logger.info(
            f"Image validation successful: {image_format} "
            f"{width}x{height}, {len(image_data)} bytes"
        )
        
        return image
    
    async def preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for optimal OCR performance.
        
        Args:
            image: PIL Image object to preprocess
            
        Returns:
            Preprocessed PIL Image object optimized for OCR
            
        Requirements: 3.4
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                self.logger.debug(f"Converted image from {image.mode} to RGB")
            
            # Resize if image is too large (maintain aspect ratio)
            width, height = image.size
            if width > self.MAX_DIMENSION or height > self.MAX_DIMENSION:
                ratio = min(self.MAX_DIMENSION / width, self.MAX_DIMENSION / height)
                new_size = (int(width * ratio), int(height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                self.logger.debug(f"Resized image from {width}x{height} to {new_size}")
            
            # Enhance contrast for better text recognition
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(self.CONTRAST_ENHANCEMENT)
            
            # Enhance sharpness for clearer text
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(self.SHARPNESS_ENHANCEMENT)
            
            # Apply noise reduction
            image = image.filter(ImageFilter.MedianFilter(size=self.NOISE_REDUCTION_SIZE))
            
            # Auto-orient image based on EXIF data
            image = ImageOps.exif_transpose(image)
            
            self.logger.debug("Image preprocessing for OCR completed")
            return image
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {str(e)}")
            raise ImageValidationError(
                f"Failed to preprocess image for OCR: {str(e)}",
                ErrorCode.OCR_PROCESSING_FAILED
            )
    
    async def optimize_image_quality(self, image: Image.Image) -> Image.Image:
        """
        Optimize image quality for better processing performance.
        
        Args:
            image: PIL Image object to optimize
            
        Returns:
            Quality-optimized PIL Image object
        """
        try:
            # Adjust brightness if image is too dark or bright
            enhancer = ImageEnhance.Brightness(image)
            
            # Calculate average brightness
            grayscale = image.convert('L')
            histogram = grayscale.histogram()
            pixels = sum(histogram)
            brightness = sum(i * histogram[i] for i in range(256)) / pixels
            
            # Adjust brightness if needed
            if brightness < 100:  # Too dark
                image = enhancer.enhance(1.2)
                self.logger.debug("Enhanced brightness for dark image")
            elif brightness > 200:  # Too bright
                image = enhancer.enhance(0.8)
                self.logger.debug("Reduced brightness for bright image")
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Image quality optimization failed: {str(e)}")
            return image  # Return original image if optimization fails
    
    async def process_batch_images(
        self, 
        images_data: List[bytes], 
        preprocess: bool = True
    ) -> List[Image.Image]:
        """
        Process multiple images in batch with validation and preprocessing.
        
        Args:
            images_data: List of raw image bytes
            preprocess: Whether to apply OCR preprocessing
            
        Returns:
            List of processed PIL Image objects
            
        Raises:
            ImageValidationError: If any image validation fails
            
        Requirements: 2.3
        """
        processed_images = []
        
        for i, image_data in enumerate(images_data):
            try:
                # Validate image
                image = await self.validate_image(image_data)
                
                # Apply preprocessing if requested
                if preprocess:
                    image = await self.preprocess_for_ocr(image)
                    image = await self.optimize_image_quality(image)
                
                processed_images.append(image)
                
                self.logger.debug(f"Batch processing: image {i+1}/{len(images_data)} completed")
                
            except ImageValidationError as e:
                self.logger.error(f"Batch processing: image {i+1} failed: {str(e)}")
                raise  # Re-raise validation errors
        
        self.logger.info(f"Batch processing completed: {len(processed_images)} images processed")
        return processed_images
    
    def calculate_image_hash(self, image_data: bytes) -> str:
        """
        Calculate hash for image deduplication and caching.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            SHA-256 hash of the image data
        """
        return hashlib.sha256(image_data).hexdigest()
    
    def get_image_info(self, image: Image.Image) -> Dict[str, Union[str, int, float]]:
        """
        Get comprehensive information about an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with image information
        """
        return {
            'format': image.format,
            'mode': image.mode,
            'width': image.size[0],
            'height': image.size[1],
            'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
            'dpi': image.info.get('dpi', (72, 72)),
            'exif_data': bool(image.getexif()) if hasattr(image, 'getexif') else False
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats"""
        return list(self.SUPPORTED_FORMATS)
    
    def get_size_limits(self) -> Dict[str, int]:
        """Get image size limits and constraints"""
        return {
            'max_file_size_bytes': self.MAX_FILE_SIZE,
            'min_file_size_bytes': self.MIN_FILE_SIZE,
            'max_dimension_pixels': self.MAX_DIMENSION,
            'min_dimension_pixels': self.MIN_DIMENSION
        }
    
    def get_processing_parameters(self) -> Dict[str, Union[int, float]]:
        """Get current image processing parameters"""
        return {
            'contrast_enhancement': self.CONTRAST_ENHANCEMENT,
            'sharpness_enhancement': self.SHARPNESS_ENHANCEMENT,
            'noise_reduction_size': self.NOISE_REDUCTION_SIZE,
            'default_quality': self.DEFAULT_QUALITY
        }