"""
Unit tests for enhanced ML models and services.

Tests the modern PyTorch-based implementations for:
- Base ML model configuration
- OCR services (TrOCR, EasyOCR)
- Translation services (NLLB, mBART)
- Navigation services
- Image preprocessing
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock


class TestDeviceTypeEnum:
    """Tests for DeviceType enum."""

    def test_device_type_values(self):
        """Test DeviceType enum has correct values."""
        from app.services.ml_models.base import DeviceType
        
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.MPS.value == "mps"

    def test_device_type_is_string_enum(self):
        """Test DeviceType is a string enum."""
        from app.services.ml_models.base import DeviceType
        
        assert isinstance(DeviceType.CPU, str)
        assert DeviceType.CPU.value == "cpu"


class TestPrecisionModeEnum:
    """Tests for PrecisionMode enum."""

    def test_precision_mode_values(self):
        """Test PrecisionMode enum has correct values."""
        from app.services.ml_models.base import PrecisionMode
        
        assert PrecisionMode.FP32.value == "fp32"
        assert PrecisionMode.FP16.value == "fp16"
        assert PrecisionMode.BF16.value == "bf16"
        assert PrecisionMode.INT8.value == "int8"


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_defaults(self):
        """Test ModelConfig has sensible defaults."""
        from app.services.ml_models.base import ModelConfig, DeviceType, PrecisionMode
        
        config = ModelConfig(model_name="test-model")
        
        assert config.model_name == "test-model"
        assert config.device_type == DeviceType.CPU
        assert config.precision_mode == PrecisionMode.FP16
        assert config.max_batch_size == 8
        assert config.cache_enabled is True

    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        from app.services.ml_models.base import ModelConfig, DeviceType, PrecisionMode
        
        config = ModelConfig(
            model_name="custom-model",
            device_type=DeviceType.CUDA,
            precision_mode=PrecisionMode.BF16,
            max_batch_size=16,
            cache_enabled=False,
            warmup_iterations=5
        )
        
        assert config.model_name == "custom-model"
        assert config.precision_mode == PrecisionMode.BF16
        assert config.max_batch_size == 16
        assert config.cache_enabled is False
        assert config.warmup_iterations == 5

    def test_model_config_compile_settings(self):
        """Test ModelConfig compile settings."""
        from app.services.ml_models.base import ModelConfig
        
        config = ModelConfig(
            model_name="test-model",
            use_compile=True,
            compile_mode="max-autotune"
        )
        
        # Note: use_compile may be False if torch is not available
        assert config.compile_mode == "max-autotune"


class TestEnhancedOCRConfig:
    """Tests for EnhancedOCRConfig."""

    def test_ocr_config_defaults(self):
        """Test EnhancedOCRConfig default values."""
        from app.services.enhanced_ocr_service import EnhancedOCRConfig
        from app.services.ml_models.base import DeviceType, PrecisionMode
        
        config = EnhancedOCRConfig()
        
        assert config.mode == "hybrid"
        assert config.device == DeviceType.CUDA
        assert config.precision == PrecisionMode.FP16
        assert config.min_confidence == 0.5
        assert config.use_compile is True
        assert config.adaptive_preprocessing is True
        # Default languages should include travel companion languages
        assert "en" in config.languages
        assert "ja" in config.languages

    def test_ocr_config_custom_mode(self):
        """Test EnhancedOCRConfig with custom mode."""
        from app.services.enhanced_ocr_service import EnhancedOCRConfig
        
        config = EnhancedOCRConfig(mode="fast", min_confidence=0.8)
        
        assert config.mode == "fast"
        assert config.min_confidence == 0.8


class TestEnhancedTranslationConfig:
    """Tests for EnhancedTranslationConfig."""

    def test_translation_config_defaults(self):
        """Test EnhancedTranslationConfig default values."""
        from app.services.enhanced_translation_service import EnhancedTranslationConfig
        from app.services.ml_models.base import DeviceType, PrecisionMode
        
        config = EnhancedTranslationConfig()
        
        assert config.model_type == "nllb"
        assert config.model_size == "600M"
        assert config.device == DeviceType.CUDA
        assert config.precision == PrecisionMode.FP16
        assert config.num_beams == 4
        assert config.max_length == 256

    def test_translation_config_mbart(self):
        """Test EnhancedTranslationConfig with mBART."""
        from app.services.enhanced_translation_service import EnhancedTranslationConfig
        
        config = EnhancedTranslationConfig(model_type="mbart")
        
        assert config.model_type == "mbart"


class TestNavigationServiceConfig:
    """Tests for NavigationServiceConfig."""

    def test_navigation_config_defaults(self):
        """Test NavigationServiceConfig default values."""
        from app.services.enhanced_navigation_service import NavigationServiceConfig
        
        config = NavigationServiceConfig()
        
        assert config.cache_ttl_seconds == 300
        assert config.cache_prefix == "nav"
        assert config.enable_semantic_search is True
        assert config.enable_personalization is True
        assert config.default_radius_m == 1000
        assert config.max_results == 50


class TestTravelContextEnum:
    """Tests for TravelContext enum."""

    def test_travel_context_values(self):
        """Test TravelContext enum has expected contexts."""
        from app.services.enhanced_navigation_service import TravelContext
        
        assert TravelContext.BUSINESS.value == "business"
        assert TravelContext.TOURISM.value == "tourism"
        assert TravelContext.DINING.value == "dining"
        assert TravelContext.SHOPPING.value == "shopping"
        assert TravelContext.CULTURAL.value == "cultural"
        assert TravelContext.EMERGENCY.value == "emergency"


class TestPOIResult:
    """Tests for POIResult dataclass."""

    def test_poi_result_creation(self):
        """Test POIResult can be created."""
        from app.services.enhanced_navigation_service import POIResult
        
        poi = POIResult(
            name="Test Restaurant",
            category="restaurant",
            latitude=35.6762,
            longitude=139.6503,
            distance_m=150.5,
            etiquette_notes=["Don't tip"]
        )
        
        assert poi.name == "Test Restaurant"
        assert poi.category == "restaurant"
        assert poi.latitude == 35.6762
        assert poi.distance_m == 150.5
        assert len(poi.etiquette_notes) == 1

    def test_poi_result_defaults(self):
        """Test POIResult default field values."""
        from app.services.enhanced_navigation_service import POIResult
        
        poi = POIResult(
            name="Test",
            category="test",
            latitude=0.0,
            longitude=0.0,
            distance_m=0.0,
            etiquette_notes=[]
        )
        
        assert poi.cultural_tips == []
        assert poi.relevance_score == 1.0
        assert poi.accessibility_info is None
        assert poi.language_support == []

    def test_poi_result_to_dict(self):
        """Test POIResult.to_dict() method."""
        from app.services.enhanced_navigation_service import POIResult
        
        poi = POIResult(
            name="Tokyo Station",
            category="transit",
            latitude=35.6812,
            longitude=139.7671,
            distance_m=500.0,
            etiquette_notes=["Queue orderly"]
        )
        
        result = poi.to_dict()
        
        assert isinstance(result, dict)
        assert result["name"] == "Tokyo Station"
        assert result["category"] == "transit"
        assert result["etiquette_notes"] == ["Queue orderly"]


class TestTranslationLanguage:
    """Tests for TranslationLanguage enum."""

    def test_translation_language_values(self):
        """Test TranslationLanguage enum has expected language codes."""
        from app.services.ml_models.translation_models import TranslationLanguage
        
        assert TranslationLanguage.ENGLISH.value == "eng_Latn"
        assert TranslationLanguage.JAPANESE.value == "jpn_Jpan"
        assert TranslationLanguage.SPANISH.value == "spa_Latn"
        assert TranslationLanguage.VIETNAMESE.value == "vie_Latn"

    def test_translation_language_from_iso(self):
        """Test TranslationLanguage.from_iso() method."""
        from app.services.ml_models.translation_models import TranslationLanguage
        
        assert TranslationLanguage.from_iso("en") == TranslationLanguage.ENGLISH
        assert TranslationLanguage.from_iso("ja") == TranslationLanguage.JAPANESE
        assert TranslationLanguage.from_iso("es") == TranslationLanguage.SPANISH
        assert TranslationLanguage.from_iso("vi") == TranslationLanguage.VIETNAMESE
        assert TranslationLanguage.from_iso("unknown") is None


class TestPreprocessConfig:
    """Tests for PreprocessConfig dataclass."""

    def test_preprocess_config_defaults(self):
        """Test PreprocessConfig default values."""
        from app.services.ml_models.image_preprocessing import PreprocessConfig
        from app.services.ml_models.base import DeviceType
        
        config = PreprocessConfig()
        
        assert config.device == DeviceType.CPU
        assert config.target_size is None
        assert config.normalize is True
        assert config.enhance_contrast == 1.2
        assert config.adaptive_preprocessing is True


class TestOCRModelConfig:
    """Tests for OCRModelConfig dataclass."""

    def test_ocr_model_config_defaults(self):
        """Test OCRModelConfig default values."""
        from app.services.ml_models.ocr_models import OCRModelConfig
        
        config = OCRModelConfig(model_name="test")
        
        assert config.min_confidence == 0.5
        assert config.max_new_tokens == 128
        assert config.num_beams == 4
        assert config.languages == ["en"]
        assert config.detect_only is False

    def test_ocr_model_config_custom_languages(self):
        """Test OCRModelConfig with custom languages."""
        from app.services.ml_models.ocr_models import OCRModelConfig
        
        config = OCRModelConfig(
            model_name="test",
            languages=["ja", "en", "zh"]
        )
        
        assert config.languages == ["ja", "en", "zh"]


class TestEnhancedOCRService:
    """Tests for EnhancedOCRService class."""

    def test_service_instantiation(self):
        """Test EnhancedOCRService can be instantiated."""
        from app.services.enhanced_ocr_service import (
            EnhancedOCRService,
            EnhancedOCRConfig
        )
        
        config = EnhancedOCRConfig()
        service = EnhancedOCRService(config)
        
        assert service is not None
        assert service.config == config

    def test_service_default_config(self):
        """Test EnhancedOCRService with default config."""
        from app.services.enhanced_ocr_service import EnhancedOCRService
        
        service = EnhancedOCRService()
        
        assert service.config is not None
        assert service.config.mode == "hybrid"


class TestEnhancedTranslationService:
    """Tests for EnhancedTranslationService class."""

    def test_service_instantiation(self):
        """Test EnhancedTranslationService can be instantiated."""
        from app.services.enhanced_translation_service import (
            EnhancedTranslationService,
            EnhancedTranslationConfig
        )
        
        config = EnhancedTranslationConfig()
        service = EnhancedTranslationService(config)
        
        assert service is not None
        assert service.config == config

    def test_service_default_config(self):
        """Test EnhancedTranslationService with default config."""
        from app.services.enhanced_translation_service import EnhancedTranslationService
        
        service = EnhancedTranslationService()
        
        assert service.config is not None
        assert service.config.model_type == "nllb"


class TestEnhancedNavigationService:
    """Tests for EnhancedNavigationService class."""

    def test_service_instantiation(self):
        """Test EnhancedNavigationService can be instantiated."""
        from app.services.enhanced_navigation_service import (
            EnhancedNavigationService,
            NavigationServiceConfig
        )
        
        config = NavigationServiceConfig()
        # Mock dependencies
        maps_client = Mock()
        cache_client = Mock()
        
        service = EnhancedNavigationService(
            config=config,
            maps_client=maps_client,
            cache_client=cache_client
        )
        
        assert service is not None
        assert service.config == config


class TestEtiquetteDatabase:
    """Tests for etiquette database content."""

    def test_japan_restaurant_etiquette(self):
        """Test Japanese restaurant etiquette is available."""
        from app.services.enhanced_navigation_service import ETIQUETTE_DATABASE
        
        assert "JP" in ETIQUETTE_DATABASE
        assert "restaurant" in ETIQUETTE_DATABASE["JP"]
        
        etiquette = ETIQUETTE_DATABASE["JP"]["restaurant"]
        assert len(etiquette) > 0
        # Check for specific etiquette items
        etiquette_text = " ".join(etiquette).lower()
        assert "itadakimasu" in etiquette_text or "tip" in etiquette_text

    def test_spain_etiquette(self):
        """Test Spanish etiquette is available."""
        from app.services.enhanced_navigation_service import ETIQUETTE_DATABASE
        
        assert "ES" in ETIQUETTE_DATABASE

    def test_vietnam_etiquette(self):
        """Test Vietnamese etiquette is available."""
        from app.services.enhanced_navigation_service import ETIQUETTE_DATABASE
        
        assert "VN" in ETIQUETTE_DATABASE


class TestCategoryKeywords:
    """Tests for category keyword matching."""

    def test_category_keywords_exist(self):
        """Test category keywords are defined."""
        from app.services.enhanced_navigation_service import CATEGORY_KEYWORDS
        
        assert "restaurant" in CATEGORY_KEYWORDS
        assert "transit" in CATEGORY_KEYWORDS
        assert "cultural" in CATEGORY_KEYWORDS
        assert "shopping" in CATEGORY_KEYWORDS
        assert "emergency" in CATEGORY_KEYWORDS

    def test_restaurant_keywords(self):
        """Test restaurant category has expected keywords."""
        from app.services.enhanced_navigation_service import CATEGORY_KEYWORDS
        
        keywords = CATEGORY_KEYWORDS["restaurant"]
        assert "food" in keywords
        assert "dining" in keywords
        assert "cafe" in keywords


class TestServicesExports:
    """Tests for services package exports."""

    def test_ml_models_exports(self):
        """Test ml_models package exports expected classes."""
        from app.services.ml_models import (
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
        
        assert BaseMLModel is not None
        assert ModelConfig is not None
        assert DeviceType is not None
        assert PrecisionMode is not None
        assert TrOCRModel is not None
        assert EasyOCRModel is not None
        assert NLLBTranslationModel is not None
        assert MBARTTranslationModel is not None
        assert GPUImagePreprocessor is not None

    def test_services_exports(self):
        """Test services package exports enhanced services."""
        from app.services import (
            EnhancedOCRService,
            EnhancedOCRConfig,
            EnhancedTranslationService,
            EnhancedTranslationConfig,
            EnhancedNavigationService,
            NavigationServiceConfig,
            TravelContext,
            POIResult,
        )
        
        assert EnhancedOCRService is not None
        assert EnhancedOCRConfig is not None
        assert EnhancedTranslationService is not None
        assert EnhancedTranslationConfig is not None
        assert EnhancedNavigationService is not None
        assert NavigationServiceConfig is not None
        assert TravelContext is not None
        assert POIResult is not None
