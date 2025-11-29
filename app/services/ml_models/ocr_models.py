"""
OCR Models with modern PyTorch optimizations.

Provides state-of-the-art OCR capabilities:
- TrOCR: Transformer-based OCR for printed and handwritten text
- EasyOCR: Production-ready multilingual OCR
- HybridOCR: Combines detection and recognition for best results

All models support:
- torch.compile for JIT optimization
- Mixed precision inference (FP16/BF16)
- Batch processing with dynamic batching
- GPU acceleration with automatic fallback
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from PIL import Image
import io

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        AutoProcessor,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

from .base import BaseMLModel, ModelConfig, DeviceType, PrecisionMode
from ...models.internal_models import OCRResult

logger = logging.getLogger(__name__)


@dataclass
class OCRModelConfig(ModelConfig):
    """
    Configuration specific to OCR models.

    Additional attributes:
        min_confidence: Minimum confidence threshold for results
        max_new_tokens: Maximum tokens to generate per text region
        num_beams: Number of beams for beam search decoding
        languages: List of language codes for EasyOCR
        detect_only: If True, only run text detection
        paragraph_mode: Group text into paragraphs
    """
    min_confidence: float = 0.5
    max_new_tokens: int = 128
    num_beams: int = 4
    languages: List[str] = None
    detect_only: bool = False
    paragraph_mode: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.languages is None:
            self.languages = ["en"]


class TrOCRModel(BaseMLModel[Image.Image]):
    """
    TrOCR model for high-accuracy text recognition.

    Uses Microsoft's TrOCR (Transformer-based OCR) which combines
    a Vision Transformer encoder with a text Transformer decoder.

    Optimizations:
    - torch.compile for inference acceleration
    - Mixed precision with autocast
    - Static cache for decoder generation
    - Batched inference support

    Example:
        config = OCRModelConfig(
            model_name="microsoft/trocr-base-printed",
            device=DeviceType.CUDA,
            precision=PrecisionMode.FP16,
        )
        model = TrOCRModel(config)
        await model.load()
        results = await model.inference(image)
    """

    def __init__(self, config: OCRModelConfig):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers required for TrOCR. "
                "Install with: pip install transformers"
            )
        if not HAS_TORCH:
            raise ImportError(
                "torch required for TrOCR. "
                "Install with: pip install torch"
            )

        super().__init__(config)
        self.ocr_config = config
        self._generation_config = None

    async def _load_model(self) -> None:
        """Load TrOCR model and processor."""
        loop = asyncio.get_event_loop()

        # Load processor
        self._processor = await loop.run_in_executor(
            None,
            lambda: TrOCRProcessor.from_pretrained(self.config.model_name)
        )

        # Load model with optimizations
        self._model = await loop.run_in_executor(
            None,
            lambda: VisionEncoderDecoderModel.from_pretrained(
                self.config.model_name,
                torch_dtype=self.dtype,
                device_map=self.device if self.config.device == DeviceType.CUDA else None,  # noqa: E501
            )
        )

        # Move to device if not using device_map
        if self.config.device != DeviceType.CUDA:
            self._model = self._model.to(self.device)

        # Set to evaluation mode
        self._model.eval()

        # Configure generation settings for speed
        self._generation_config = {
            "max_new_tokens": self.ocr_config.max_new_tokens,
            "num_beams": self.ocr_config.num_beams,
            "use_cache": True,
            "cache_implementation": "static",  # PyTorch 2.x static cache
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        self.logger.info(
            f"TrOCR loaded: {self.config.model_name} "
            f"on {self.device} with {self.config.precision.value}"
        )

    async def _run_warmup_iteration(self) -> None:
        """Run warmup with dummy image."""
        # Create a small dummy image
        dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        await self._run_inference(dummy_image)

    async def _run_inference(
        self,
        inputs: Union[Image.Image, List[Image.Image]]
    ) -> List[OCRResult]:
        """
        Run TrOCR inference on image(s).

        Args:
            inputs: Single image or list of images

        Returns:
            List of OCRResult with recognized text
        """
        if isinstance(inputs, Image.Image):
            inputs = [inputs]

        loop = asyncio.get_event_loop()

        # Process images
        pixel_values = await loop.run_in_executor(
            None,
            lambda: self._processor(
                images=inputs,
                return_tensors="pt"
            ).pixel_values.to(self.device, dtype=self.dtype)
        )

        # Generate text
        outputs = await loop.run_in_executor(
            None,
            lambda: self._model.generate(
                pixel_values,
                **self._generation_config
            )
        )

        # Decode outputs
        generated_ids = outputs.sequences
        scores = outputs.sequences_scores if hasattr(outputs, 'sequences_scores') else None  # noqa: E501

        texts = await loop.run_in_executor(
            None,
            lambda: self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
        )

        # Build results
        results = []
        for i, text in enumerate(texts):
            confidence = float(torch.sigmoid(scores[i]).item()) if scores is not None else 0.9  # noqa: E501
            results.append(OCRResult(
                text=text.strip(),
                confidence=confidence,
                bbox=(0, 0, inputs[i].width, inputs[i].height),
                group_id=f"trocr_{i}"
            ))

        return results

    async def recognize_regions(
        self,
        image: Image.Image,
        regions: List[Tuple[int, int, int, int]]
    ) -> List[OCRResult]:
        """
        Recognize text in specific image regions.

        Args:
            image: Full image
            regions: List of (x1, y1, x2, y2) bounding boxes

        Returns:
            List of OCRResult for each region
        """
        if not regions:
            return []

        # Crop regions
        cropped = []
        for x1, y1, x2, y2 in regions:
            crop = image.crop((x1, y1, x2, y2))
            cropped.append(crop)

        # Batch inference
        results = await self.inference(cropped)

        # Update bboxes to original coordinates
        for i, (x1, y1, x2, y2) in enumerate(regions):
            if i < len(results):
                results[i] = OCRResult(
                    text=results[i].text,
                    confidence=results[i].confidence,
                    bbox=(x1, y1, x2, y2),
                    group_id=f"trocr_region_{i}"
                )

        return results


class EasyOCRModel(BaseMLModel[Image.Image]):
    """
    EasyOCR wrapper with optimized settings.

    EasyOCR provides:
    - 80+ language support
    - Built-in text detection (CRAFT) + recognition
    - Good balance of speed and accuracy
    - Lower memory requirements than transformer models

    Optimizations:
    - GPU acceleration when available
    - Batch processing for multiple images
    - Configurable detection parameters
    """

    def __init__(self, config: OCRModelConfig):
        if not HAS_EASYOCR:
            raise ImportError(
                "easyocr required. Install with: pip install easyocr"
            )

        super().__init__(config)
        self.ocr_config = config
        self._reader = None

    async def _load_model(self) -> None:
        """Initialize EasyOCR reader."""
        loop = asyncio.get_event_loop()

        gpu = self.config.device == DeviceType.CUDA

        self._reader = await loop.run_in_executor(
            None,
            lambda: easyocr.Reader(
                self.ocr_config.languages,
                gpu=gpu,
                model_storage_directory=None,  # Use default cache
                download_enabled=True,
            )
        )

        self.logger.info(
            f"EasyOCR loaded for languages: {self.ocr_config.languages}"
        )

    async def _run_warmup_iteration(self) -> None:
        """Run warmup with dummy image."""
        import numpy as np
        dummy = np.ones((100, 100, 3), dtype=np.uint8) * 255
        await self._run_inference(Image.fromarray(dummy))

    async def _run_inference(
        self,
        inputs: Union[Image.Image, List[Image.Image]]
    ) -> List[OCRResult]:
        """
        Run EasyOCR inference.

        Args:
            inputs: Single image or list of images

        Returns:
            List of OCRResult with detected text
        """
        import numpy as np

        if isinstance(inputs, Image.Image):
            inputs = [inputs]

        loop = asyncio.get_event_loop()
        all_results = []

        for img in inputs:
            # Convert PIL to numpy
            img_array = np.array(img)

            # Run OCR
            raw_results = await loop.run_in_executor(
                None,
                lambda arr=img_array: self._reader.readtext(
                    arr,
                    detail=1,
                    paragraph=self.ocr_config.paragraph_mode,
                    decoder='beamsearch',
                    beamWidth=5,
                    batch_size=self.config.max_batch_size,
                )
            )

            # Convert to OCRResult
            for i, (bbox, text, conf) in enumerate(raw_results):
                if conf >= self.ocr_config.min_confidence:
                    # EasyOCR returns [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)

                    all_results.append(OCRResult(
                        text=text,
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        group_id=f"easyocr_{i}"
                    ))

        return all_results

    async def detect_text_regions(
        self,
        image: Image.Image
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions without recognition.

        Args:
            image: Input image

        Returns:
            List of (x1, y1, x2, y2) bounding boxes
        """
        import numpy as np

        loop = asyncio.get_event_loop()
        img_array = np.array(image)

        horizontal_list, free_list = await loop.run_in_executor(
            None,
            lambda: self._reader.detect(img_array)
        )

        regions = []

        # Process horizontal boxes [x_min, x_max, y_min, y_max]
        for h in horizontal_list[0] if horizontal_list else []:
            x1, x2, y1, y2 = h
            regions.append((int(x1), int(y1), int(x2), int(y2)))

        # Process free-form boxes
        for f in free_list[0] if free_list else []:
            x_coords = [p[0] for p in f]
            y_coords = [p[1] for p in f]
            regions.append((
                int(min(x_coords)),
                int(min(y_coords)),
                int(max(x_coords)),
                int(max(y_coords))
            ))

        return regions


class HybridOCRModel(BaseMLModel[Image.Image]):
    """
    Hybrid OCR combining EasyOCR detection with TrOCR recognition.

    This approach provides:
    - Fast and accurate text detection from EasyOCR
    - High-quality text recognition from TrOCR
    - Best of both worlds for menu/sign reading

    The pipeline:
    1. Detect text regions using EasyOCR's CRAFT detector
    2. Crop and recognize each region with TrOCR
    3. Return combined results with precise bounding boxes
    """

    def __init__(
        self,
        detection_config: OCRModelConfig,
        recognition_config: OCRModelConfig
    ):
        # Use detection config as base
        super().__init__(detection_config)

        self.detection_config = detection_config
        self.recognition_config = recognition_config

        self._detector: Optional[EasyOCRModel] = None
        self._recognizer: Optional[TrOCRModel] = None

    async def _load_model(self) -> None:
        """Load both detection and recognition models."""
        # Initialize detector (EasyOCR)
        self._detector = EasyOCRModel(self.detection_config)
        await self._detector.load()

        # Initialize recognizer (TrOCR)
        self._recognizer = TrOCRModel(self.recognition_config)
        await self._recognizer.load()

        self.logger.info("Hybrid OCR loaded (EasyOCR + TrOCR)")

    async def _run_warmup_iteration(self) -> None:
        """Run warmup for both models."""
        dummy = Image.new("RGB", (224, 224), color=(255, 255, 255))
        await self._run_inference(dummy)

    async def _run_inference(
        self,
        inputs: Union[Image.Image, List[Image.Image]]
    ) -> List[OCRResult]:
        """
        Run hybrid OCR pipeline.

        Args:
            inputs: Image or list of images

        Returns:
            List of OCRResult with high-quality recognition
        """
        if isinstance(inputs, Image.Image):
            inputs = [inputs]

        all_results = []

        for image in inputs:
            # Step 1: Detect text regions
            regions = await self._detector.detect_text_regions(image)

            if not regions:
                continue

            # Step 2: Recognize each region with TrOCR
            recognition_results = await self._recognizer.recognize_regions(
                image, regions
            )

            all_results.extend(recognition_results)

        return all_results

    async def health_check(self) -> bool:
        """Check both models are healthy."""
        if self._detector is None or self._recognizer is None:
            return False

        detector_ok = await self._detector.health_check()
        recognizer_ok = await self._recognizer.health_check()

        return detector_ok and recognizer_ok

    async def unload(self) -> None:
        """Unload both models."""
        if self._detector:
            await self._detector.unload()
        if self._recognizer:
            await self._recognizer.unload()

        await super().unload()

    def get_stats(self) -> dict:
        """Get combined stats from both models."""
        base_stats = super().get_stats()

        if self._detector:
            base_stats["detector"] = self._detector.get_stats()
        if self._recognizer:
            base_stats["recognizer"] = self._recognizer.get_stats()

        return base_stats
