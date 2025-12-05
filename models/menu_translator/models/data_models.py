"""
Data models for Menu Translator
"""

from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass


class SupportedLanguage(str, Enum):
    """Supported languages for OCR and translation."""
    EN = "en"
    KO = "ko"
    VI = "vi"
    OTHER = "other"


@dataclass
class TextBox:
    """OCR detected text box."""
    bbox: list[float]  # [x_min, y_min, x_max, y_max]
    text: str
    confidence: float
    language: SupportedLanguage = SupportedLanguage.OTHER
    block_id: int = 0
    line_id: int = 0


@dataclass
class OCRResult:
    """Complete OCR result for an image."""
    text_boxes: list[TextBox]
    processing_time_ms: float
    image_width: int
    image_height: int
    avg_confidence: float = 0.0


@dataclass
class LocalizedText:
    """Text with translations in all supported languages."""
    en: Optional[str] = None
    ko: Optional[str] = None
    vi: Optional[str] = None


@dataclass
class TranslationResult:
    """Translation result for a text."""
    source_text: str
    source_language: SupportedLanguage
    translated_text: LocalizedText
    explanation: LocalizedText
    confidence: float = 0.0
    processing_time_ms: float = 0.0


@dataclass
class DishPrediction:
    """EfficientNet dish classification result."""
    dish_class_id: str
    dish_class_name: str
    confidence: float
    category: Optional[str] = None
    rank: int = 0


@dataclass
class DishClassificationResult:
    """Complete dish classification result."""
    predictions: list[DishPrediction]
    processing_time_ms: float
    model_name: str = "efficientnet_b4"


@dataclass
class MenuItem:
    """Parsed menu item."""
    id: str
    raw_name: str
    raw_description: Optional[str] = None
    raw_price: Optional[str] = None
    section: Optional[str] = None
    source_language: SupportedLanguage = SupportedLanguage.OTHER
    normalized_name: Optional[str] = None
    normalized_tags: list[str] = None
    canonical_dish_id: Optional[str] = None
    ocr_confidence: float = 0.0
    bbox: list[float] = None

    def __post_init__(self):
        if self.normalized_tags is None:
            self.normalized_tags = []
        if self.bbox is None:
            self.bbox = []
