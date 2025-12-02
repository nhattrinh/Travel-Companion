# Business logic services

from .ocr_service import OCRService, BaseOCRModel, MockOCRModel, OCRProcessingError
from .image_processor import ImageProcessor, ImageValidationError, ImageFormat
from .translation_service import (
    TranslationService, 
    BaseTranslationModel, 
    MockTranslationModel,
    UnsupportedLanguageError,
    TranslationFailureError
)
from .food_image_service import (
    FoodImageService,
    FoodImageServiceError,
    ImageSearchError,
    CacheError
)

# Enhanced ML-powered services with modern PyTorch optimizations
from .enhanced_ocr_service import EnhancedOCRService, EnhancedOCRConfig
from .enhanced_translation_service import (
    EnhancedTranslationService,
    EnhancedTranslationConfig
)
from .enhanced_navigation_service import (
    EnhancedNavigationService,
    NavigationServiceConfig,
    TravelContext,
    POIResult,
    create_enhanced_navigation_service
)

# ML Models package
from .ml_models import (
    BaseMLModel,
    ModelConfig,
    DeviceType,
    PrecisionMode,
    TrOCRModel,
    EasyOCRModel,
    NLLBTranslationModel,
    MBARTTranslationModel,
    GPUImagePreprocessor,
)


# Factory functions for dependency injection
async def create_food_image_service(
    cache_client=None,
    external_api_client=None
) -> FoodImageService:
    """
    Factory function to create FoodImageService with proper dependencies.
    
    Args:
        cache_client: Optional cache client (will use default if None)
        external_api_client: Optional external API client
        
    Returns:
        Configured FoodImageService instance
    """
    from app.core.cache_client import get_cache_client
    
    if cache_client is None:
        cache_client = await get_cache_client()
    
    return FoodImageService(
        cache_client=cache_client,
        external_api_client=external_api_client
    )


__all__ = [
    # Legacy/Mock services
    'OCRService',
    'BaseOCRModel',
    'MockOCRModel',
    'OCRProcessingError',
    'ImageProcessor',
    'ImageValidationError',
    'ImageFormat',
    'TranslationService',
    'BaseTranslationModel',
    'MockTranslationModel',
    'UnsupportedLanguageError',
    'TranslationFailureError',
    'FoodImageService',
    'FoodImageServiceError',
    'ImageSearchError',
    'CacheError',
    'create_food_image_service',
    # Enhanced ML-powered services
    'EnhancedOCRService',
    'EnhancedOCRConfig',
    'EnhancedTranslationService',
    'EnhancedTranslationConfig',
    'EnhancedNavigationService',
    'NavigationServiceConfig',
    'TravelContext',
    'POIResult',
    'create_enhanced_navigation_service',
    # ML Models
    'BaseMLModel',
    'ModelConfig',
    'DeviceType',
    'PrecisionMode',
    'TrOCRModel',
    'EasyOCRModel',
    'NLLBTranslationModel',
    'MBARTTranslationModel',
    'GPUImagePreprocessor',
    'AdaptiveImageAnalyzer',
    'OCRPreprocessor',
]
