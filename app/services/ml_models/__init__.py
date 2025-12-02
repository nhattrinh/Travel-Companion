"""
ML Models package for Travel Companion.

This package contains modern PyTorch-based implementations for:
- OCR (Optical Character Recognition) using TrOCR and EasyOCR
- Translation using NLLB-200 and mBART models
- Image preprocessing with GPU acceleration

All models utilize latest PyTorch optimizations:
- torch.compile for JIT compilation
- torch.autocast for mixed precision
- Efficient batching and caching strategies
"""

from .base import BaseMLModel, ModelConfig, DeviceType, PrecisionMode
from .ocr_models import TrOCRModel, EasyOCRModel, HybridOCRModel
from .translation_models import NLLBTranslationModel, MBARTTranslationModel
from .image_preprocessing import GPUImagePreprocessor

__all__ = [
    "BaseMLModel",
    "ModelConfig",
    "DeviceType",
    "PrecisionMode",
    "TrOCRModel",
    "EasyOCRModel",
    "HybridOCRModel",
    "NLLBTranslationModel",
    "MBARTTranslationModel",
    "GPUImagePreprocessor",
]
