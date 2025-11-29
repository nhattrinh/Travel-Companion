"""
Enhanced OCR Service with modern PyTorch models.

This service provides high-performance OCR capabilities by combining:
- EasyOCR for fast, production-ready text detection
- TrOCR for high-accuracy text recognition
- GPU acceleration with torch.compile
- Mixed precision inference (FP16/BF16)
- Adaptive preprocessing based on image quality

Usage:
    service = EnhancedOCRService()
    await service.initialize()
    results = await service.extract_text(image)
"""

import asyncio
import logging
from typing import List, Optional, Union
from PIL import Image
from dataclasses import dataclass

from ..models.internal_models import OCRResult
from .ml_models.base import DeviceType, PrecisionMode
from .ml_models.ocr_models import (
    OCRModelConfig,
    TrOCRModel,
    EasyOCRModel,
    HybridOCRModel,
)
from .ml_models.image_preprocessing import (
    GPUImagePreprocessor,
    PreprocessConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedOCRConfig:
    """
    Configuration for the enhanced OCR service.

    Attributes:
        mode: OCR mode - "fast" (EasyOCR), "accurate" (TrOCR),
              or "hybrid" (EasyOCR detection + TrOCR recognition)
        device: Compute device (cuda, cpu, mps)
        precision: Inference precision (fp16, bf16, fp32)
        languages: List of language codes for EasyOCR
        min_confidence: Minimum confidence threshold
        use_compile: Enable torch.compile optimization
        adaptive_preprocessing: Enable quality-based preprocessing
        trocr_model: TrOCR model name for accurate/hybrid modes
    """
    mode: str = "hybrid"
    device: DeviceType = DeviceType.CUDA
    precision: PrecisionMode = PrecisionMode.FP16
    languages: List[str] = None
    min_confidence: float = 0.5
    use_compile: bool = True
    adaptive_preprocessing: bool = True
    trocr_model: str = "microsoft/trocr-base-printed"

    def __post_init__(self):
        if self.languages is None:
            # Default to primary travel companion languages
            self.languages = ["en", "ja", "es", "vi"]


class EnhancedOCRService:
    """
    Enhanced OCR service with modern PyTorch optimizations.

    This service provides three operating modes:

    1. **Fast Mode** (EasyOCR only):
       - Best for: High throughput, good enough accuracy
       - Latency: ~200-400ms per image
       - Languages: 80+ supported

    2. **Accurate Mode** (TrOCR only):
       - Best for: Maximum accuracy on known text regions
       - Latency: ~300-600ms per image
       - Accuracy: State-of-the-art for printed text

    3. **Hybrid Mode** (EasyOCR + TrOCR):
       - Best for: Best of both worlds
       - Latency: ~400-800ms per image
       - Uses EasyOCR for detection, TrOCR for recognition

    All modes support:
    - GPU acceleration (CUDA, MPS)
    - torch.compile for JIT optimization
    - Mixed precision (FP16/BF16)
    - Adaptive image preprocessing
    - Batch processing
    """

    def __init__(self, config: Optional[EnhancedOCRConfig] = None):
        """
        Initialize the enhanced OCR service.

        Args:
            config: Service configuration (uses defaults if None)
        """
        self.config = config or EnhancedOCRConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        self._model: Optional[Union[EasyOCRModel, TrOCRModel, HybridOCRModel]] = None  # noqa: E501
        self._preprocessor: Optional[GPUImagePreprocessor] = None
        self._is_initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """
        Initialize the OCR service and load models.

        This method is idempotent and thread-safe.
        """
        async with self._init_lock:
            if self._is_initialized:
                return

            self.logger.info(
                f"Initializing Enhanced OCR Service in {self.config.mode} mode"
            )

            # Initialize preprocessor
            preprocess_config = PreprocessConfig(
                device=self.config.device,
                adaptive_preprocessing=self.config.adaptive_preprocessing,
            )
            self._preprocessor = GPUImagePreprocessor(preprocess_config)

            # Initialize OCR model based on mode
            if self.config.mode == "fast":
                await self._init_fast_mode()
            elif self.config.mode == "accurate":
                await self._init_accurate_mode()
            elif self.config.mode == "hybrid":
                await self._init_hybrid_mode()
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")

            self._is_initialized = True
            self.logger.info("Enhanced OCR Service initialized successfully")

    async def _init_fast_mode(self) -> None:
        """Initialize fast mode with EasyOCR."""
        config = OCRModelConfig(
            model_name="easyocr",
            device=self.config.device,
            precision=self.config.precision,
            use_compile=False,  # EasyOCR doesn't support torch.compile
            languages=self.config.languages,
            min_confidence=self.config.min_confidence,
        )
        self._model = EasyOCRModel(config)
        await self._model.load()

    async def _init_accurate_mode(self) -> None:
        """Initialize accurate mode with TrOCR."""
        config = OCRModelConfig(
            model_name=self.config.trocr_model,
            device=self.config.device,
            precision=self.config.precision,
            use_compile=self.config.use_compile,
            min_confidence=self.config.min_confidence,
        )
        self._model = TrOCRModel(config)
        await self._model.load()

    async def _init_hybrid_mode(self) -> None:
        """Initialize hybrid mode with EasyOCR + TrOCR."""
        # Detection config (EasyOCR)
        detection_config = OCRModelConfig(
            model_name="easyocr",
            device=self.config.device,
            precision=self.config.precision,
            use_compile=False,
            languages=self.config.languages,
            min_confidence=self.config.min_confidence,
        )

        # Recognition config (TrOCR)
        recognition_config = OCRModelConfig(
            model_name=self.config.trocr_model,
            device=self.config.device,
            precision=self.config.precision,
            use_compile=self.config.use_compile,
            min_confidence=self.config.min_confidence,
        )

        self._model = HybridOCRModel(detection_config, recognition_config)
        await self._model.load()

    async def extract_text(
        self,
        image: Union[Image.Image, bytes, str],
        preprocess: bool = True,
    ) -> List[OCRResult]:
        """
        Extract text from an image.

        Args:
            image: PIL Image, bytes, or file path
            preprocess: Whether to apply preprocessing

        Returns:
            List of OCRResult with detected text, confidence, and bboxes
        """
        if not self._is_initialized:
            await self.initialize()

        # Load image if needed
        if isinstance(image, bytes):
            image = Image.open(__import__("io").BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)

        # Preprocess
        if preprocess and self._preprocessor:
            image = await self._preprocessor.preprocess(image)

        # Run inference
        results = await self._model.inference(image)

        # Filter by confidence
        results = [
            r for r in results
            if r.confidence >= self.config.min_confidence
        ]

        self.logger.info(f"Extracted {len(results)} text regions")
        return results

    async def extract_text_batch(
        self,
        images: List[Union[Image.Image, bytes, str]],
        preprocess: bool = True,
    ) -> List[List[OCRResult]]:
        """
        Extract text from multiple images.

        Args:
            images: List of images
            preprocess: Whether to apply preprocessing

        Returns:
            List of results for each image
        """
        if not self._is_initialized:
            await self.initialize()

        # Process images
        processed = []
        for img in images:
            if isinstance(img, bytes):
                img = Image.open(__import__("io").BytesIO(img))
            elif isinstance(img, str):
                img = Image.open(img)

            if preprocess and self._preprocessor:
                img = await self._preprocessor.preprocess(img)

            processed.append(img)

        # Run batch inference
        all_results = []
        for img in processed:
            results = await self._model.inference(img)
            filtered = [
                r for r in results
                if r.confidence >= self.config.min_confidence
            ]
            all_results.append(filtered)

        return all_results

    async def validate_image(
        self,
        image_data: bytes,
    ) -> tuple[bool, str, Optional[Image.Image]]:
        """
        Validate image data for OCR.

        Args:
            image_data: Raw image bytes

        Returns:
            Tuple of (is_valid, error_message, loaded_image)
        """
        if self._preprocessor:
            return await self._preprocessor.validate_image(image_data)

        # Fallback validation
        try:
            import io
            image = Image.open(io.BytesIO(image_data))
            image.verify()
            image = Image.open(io.BytesIO(image_data))
            return True, "", image
        except Exception as e:
            return False, str(e), None

    async def health_check(self) -> bool:
        """Check if the OCR service is healthy."""
        if not self._is_initialized:
            return False

        if self._model:
            return await self._model.health_check()

        return False

    def get_stats(self) -> dict:
        """Get service statistics."""
        stats = {
            "mode": self.config.mode,
            "device": self.config.device.value,
            "precision": self.config.precision.value,
            "languages": self.config.languages,
            "is_initialized": self._is_initialized,
        }

        if self._model:
            stats["model"] = self._model.get_stats()

        return stats

    async def shutdown(self) -> None:
        """Shutdown the service and release resources."""
        if self._model:
            await self._model.unload()
            self._model = None

        self._is_initialized = False
        self.logger.info("Enhanced OCR Service shutdown complete")


# Factory function for easy instantiation
def create_ocr_service(
    mode: str = "hybrid",
    device: str = "auto",
    languages: Optional[List[str]] = None,
) -> EnhancedOCRService:
    """
    Create an enhanced OCR service with sensible defaults.

    Args:
        mode: "fast", "accurate", or "hybrid"
        device: "auto", "cuda", "cpu", or "mps"
        languages: List of language codes

    Returns:
        Configured EnhancedOCRService instance
    """
    import torch

    # Auto-detect device
    if device == "auto":
        if torch.cuda.is_available():
            device_type = DeviceType.CUDA
        elif torch.backends.mps.is_available():
            device_type = DeviceType.MPS
        else:
            device_type = DeviceType.CPU
    else:
        device_type = DeviceType(device)

    # Adjust precision for device
    if device_type == DeviceType.CUDA:
        if torch.cuda.is_bf16_supported():
            precision = PrecisionMode.BF16
        else:
            precision = PrecisionMode.FP16
    else:
        precision = PrecisionMode.FP32

    config = EnhancedOCRConfig(
        mode=mode,
        device=device_type,
        precision=precision,
        languages=languages,
    )

    return EnhancedOCRService(config)
