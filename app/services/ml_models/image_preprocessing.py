"""
GPU-Accelerated Image Preprocessing for OCR and ML inference.

Provides high-performance image preprocessing using:
- torchvision transforms with GPU support
- Adaptive preprocessing based on image quality
- Batch processing optimization
- Memory-efficient streaming for large images
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from enum import Enum
from PIL import Image
import io

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import torchvision.transforms.v2 as T
    import torchvision.transforms.v2.functional as TF
    HAS_TORCHVISION = True
except ImportError:
    try:
        import torchvision.transforms as T
        import torchvision.transforms.functional as TF
        HAS_TORCHVISION = True
    except ImportError:
        HAS_TORCHVISION = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .base import DeviceType, ModelConfig

logger = logging.getLogger(__name__)


class ImageQuality(str, Enum):
    """Detected image quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class PreprocessConfig:
    """
    Configuration for image preprocessing.

    Attributes:
        device: Target device for preprocessing
        target_size: Target image size (width, height) or None for no resize
        normalize: Whether to normalize pixel values
        enhance_contrast: Contrast enhancement factor (1.0 = no change)
        enhance_sharpness: Sharpness enhancement factor
        denoise_strength: Denoising strength (0.0 = disabled)
        auto_orient: Automatically correct image orientation
        adaptive_preprocessing: Enable quality-based adaptive preprocessing
        jpeg_quality: JPEG quality for compression (0-100)
    """
    device: DeviceType = DeviceType.CPU
    target_size: Optional[Tuple[int, int]] = None
    normalize: bool = True
    enhance_contrast: float = 1.2
    enhance_sharpness: float = 1.1
    denoise_strength: float = 0.0
    auto_orient: bool = True
    adaptive_preprocessing: bool = True
    jpeg_quality: int = 85


class GPUImagePreprocessor:
    """
    High-performance image preprocessor with GPU acceleration.

    Features:
    - Automatic device selection (GPU/CPU)
    - Quality-adaptive preprocessing pipeline
    - Batch processing support
    - Memory-efficient operations

    The preprocessing pipeline includes:
    1. Image validation and decoding
    2. Orientation correction (EXIF)
    3. Quality assessment
    4. Adaptive enhancement based on quality
    5. Normalization for ML inference

    Example:
        config = PreprocessConfig(
            device=DeviceType.CUDA,
            target_size=(640, 480),
            adaptive_preprocessing=True,
        )
        preprocessor = GPUImagePreprocessor(config)
        processed = await preprocessor.preprocess(image)
    """

    # Image quality thresholds
    LOW_BRIGHTNESS_THRESHOLD = 50
    HIGH_BRIGHTNESS_THRESHOLD = 200
    LOW_CONTRAST_THRESHOLD = 30
    BLUR_THRESHOLD = 100

    def __init__(self, config: PreprocessConfig):
        if not HAS_TORCH:
            logger.warning(
                "torch not available, using CPU-only preprocessing"
            )
        if not HAS_TORCHVISION:
            logger.warning(
                "torchvision not available, limited preprocessing"
            )

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Determine actual device
        if config.device == DeviceType.CUDA and HAS_TORCH:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.logger.warning("CUDA not available, using CPU")
                self.device = torch.device("cpu")
        elif config.device == DeviceType.MPS and HAS_TORCH:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu") if HAS_TORCH else None

        # Build transform pipeline
        self._build_transforms()

    def _build_transforms(self) -> None:
        """Build the torchvision transform pipeline."""
        if not HAS_TORCHVISION or not HAS_TORCH:
            self._transforms = None
            return

        transforms = []

        # Convert to tensor
        transforms.append(T.ToImage())
        transforms.append(T.ToDtype(torch.float32, scale=True))

        # Resize if target size specified
        if self.config.target_size:
            w, h = self.config.target_size
            transforms.append(
                T.Resize(
                    (h, w),
                    interpolation=T.InterpolationMode.BILINEAR,
                    antialias=True
                )
            )

        # Normalize for ML models
        if self.config.normalize:
            transforms.append(
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

        self._transforms = T.Compose(transforms)

    async def preprocess(
        self,
        image: Union[Image.Image, bytes, str],
    ) -> Image.Image:
        """
        Preprocess a single image for OCR/ML inference.

        Args:
            image: PIL Image, bytes, or file path

        Returns:
            Preprocessed PIL Image
        """
        # Load image if needed
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Auto-orient based on EXIF
        if self.config.auto_orient:
            image = await self._auto_orient(image)

        # Assess quality if adaptive preprocessing enabled
        if self.config.adaptive_preprocessing:
            quality = await self._assess_quality(image)
            image = await self._apply_adaptive_enhancement(image, quality)
        else:
            image = await self._apply_basic_enhancement(image)

        return image

    async def preprocess_for_tensor(
        self,
        image: Union[Image.Image, bytes, str],
    ) -> "torch.Tensor":
        """
        Preprocess image and convert to tensor for ML inference.

        Args:
            image: Input image

        Returns:
            Preprocessed tensor on configured device
        """
        # First do PIL preprocessing
        processed = await self.preprocess(image)

        if not HAS_TORCH or self._transforms is None:
            raise RuntimeError("torch/torchvision required for tensor output")

        loop = asyncio.get_event_loop()

        # Apply torch transforms
        tensor = await loop.run_in_executor(
            None,
            lambda: self._transforms(processed)
        )

        # Move to device
        tensor = tensor.to(self.device)

        # Add batch dimension
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        return tensor

    async def preprocess_batch(
        self,
        images: List[Union[Image.Image, bytes, str]],
    ) -> List[Image.Image]:
        """
        Preprocess multiple images.

        Args:
            images: List of images

        Returns:
            List of preprocessed PIL Images
        """
        tasks = [self.preprocess(img) for img in images]
        return await asyncio.gather(*tasks)

    async def preprocess_batch_tensor(
        self,
        images: List[Union[Image.Image, bytes, str]],
    ) -> "torch.Tensor":
        """
        Preprocess multiple images and stack as tensor batch.

        Args:
            images: List of images

        Returns:
            Batched tensor of shape (N, C, H, W)
        """
        tensors = []
        for img in images:
            t = await self.preprocess_for_tensor(img)
            tensors.append(t)

        return torch.cat(tensors, dim=0)

    async def _auto_orient(self, image: Image.Image) -> Image.Image:
        """Apply EXIF orientation correction."""
        from PIL import ImageOps

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: ImageOps.exif_transpose(image)
        )

    async def _assess_quality(self, image: Image.Image) -> ImageQuality:
        """
        Assess image quality for adaptive preprocessing.

        Analyzes:
        - Brightness level
        - Contrast level
        - Blur detection (Laplacian variance)

        Returns:
            ImageQuality enum value
        """
        if not HAS_NUMPY:
            return ImageQuality.MEDIUM

        loop = asyncio.get_event_loop()

        def analyze():
            img_array = np.array(image.convert("L"))

            # Brightness (mean pixel value)
            brightness = np.mean(img_array)

            # Contrast (standard deviation)
            contrast = np.std(img_array)

            # Blur detection using Laplacian variance
            try:
                import cv2
                laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
            except ImportError:
                # Fallback: simple gradient-based blur detection
                gx = np.gradient(img_array.astype(float), axis=1)
                gy = np.gradient(img_array.astype(float), axis=0)
                laplacian_var = np.mean(gx**2 + gy**2)

            return brightness, contrast, laplacian_var

        brightness, contrast, blur_score = await loop.run_in_executor(
            None, analyze
        )

        # Determine quality based on metrics
        issues = 0

        if brightness < self.LOW_BRIGHTNESS_THRESHOLD:
            issues += 1
            self.logger.debug(f"Low brightness detected: {brightness:.1f}")
        elif brightness > self.HIGH_BRIGHTNESS_THRESHOLD:
            issues += 1
            self.logger.debug(f"High brightness detected: {brightness:.1f}")

        if contrast < self.LOW_CONTRAST_THRESHOLD:
            issues += 1
            self.logger.debug(f"Low contrast detected: {contrast:.1f}")

        if blur_score < self.BLUR_THRESHOLD:
            issues += 1
            self.logger.debug(f"Blur detected: {blur_score:.1f}")

        if issues >= 3:
            return ImageQuality.VERY_LOW
        elif issues == 2:
            return ImageQuality.LOW
        elif issues == 1:
            return ImageQuality.MEDIUM
        else:
            return ImageQuality.HIGH

    async def _apply_adaptive_enhancement(
        self,
        image: Image.Image,
        quality: ImageQuality
    ) -> Image.Image:
        """
        Apply quality-adaptive enhancement.

        Adjusts enhancement parameters based on detected quality:
        - VERY_LOW: Aggressive enhancement
        - LOW: Moderate enhancement
        - MEDIUM: Slight enhancement
        - HIGH: Minimal changes
        """
        from PIL import ImageEnhance, ImageFilter

        loop = asyncio.get_event_loop()

        # Define enhancement factors based on quality
        enhancement_map = {
            ImageQuality.VERY_LOW: {
                "contrast": 1.5,
                "sharpness": 1.4,
                "brightness": 1.2,
                "denoise": True,
            },
            ImageQuality.LOW: {
                "contrast": 1.3,
                "sharpness": 1.2,
                "brightness": 1.1,
                "denoise": True,
            },
            ImageQuality.MEDIUM: {
                "contrast": 1.15,
                "sharpness": 1.1,
                "brightness": 1.0,
                "denoise": False,
            },
            ImageQuality.HIGH: {
                "contrast": 1.05,
                "sharpness": 1.0,
                "brightness": 1.0,
                "denoise": False,
            },
        }

        params = enhancement_map[quality]

        def enhance():
            img = image

            # Denoise if needed
            if params["denoise"]:
                img = img.filter(ImageFilter.MedianFilter(size=3))

            # Brightness
            if params["brightness"] != 1.0:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(params["brightness"])

            # Contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(params["contrast"])

            # Sharpness
            if params["sharpness"] != 1.0:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(params["sharpness"])

            return img

        enhanced = await loop.run_in_executor(None, enhance)
        self.logger.debug(f"Applied {quality.value} quality enhancement")

        return enhanced

    async def _apply_basic_enhancement(
        self,
        image: Image.Image
    ) -> Image.Image:
        """Apply basic enhancement without quality assessment."""
        from PIL import ImageEnhance

        loop = asyncio.get_event_loop()

        def enhance():
            img = image

            # Contrast
            if self.config.enhance_contrast != 1.0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(self.config.enhance_contrast)

            # Sharpness
            if self.config.enhance_sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(self.config.enhance_sharpness)

            return img

        return await loop.run_in_executor(None, enhance)

    async def validate_image(
        self,
        image_data: bytes,
        max_size_mb: float = 10.0,
        min_dimension: int = 32,
        max_dimension: int = 4096,
    ) -> Tuple[bool, str, Optional[Image.Image]]:
        """
        Validate image data.

        Args:
            image_data: Raw image bytes
            max_size_mb: Maximum file size in MB
            min_dimension: Minimum width/height
            max_dimension: Maximum width/height

        Returns:
            Tuple of (is_valid, error_message, loaded_image)
        """
        # Check file size
        size_mb = len(image_data) / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"Image too large: {size_mb:.1f}MB > {max_size_mb}MB", None  # noqa: E501

        if len(image_data) < 1024:
            return False, "Image too small: less than 1KB", None

        # Try to open image
        try:
            image = Image.open(io.BytesIO(image_data))
            image.verify()
            # Re-open after verify
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return False, f"Invalid or corrupted image: {e}", None

        # Check format
        if image.format not in ["JPEG", "PNG", "WEBP", "GIF"]:
            return False, f"Unsupported format: {image.format}", None

        # Check dimensions
        w, h = image.size
        if w < min_dimension or h < min_dimension:
            return False, f"Image too small: {w}x{h}", None
        if w > max_dimension or h > max_dimension:
            return False, f"Image too large: {w}x{h}", None

        return True, "", image

    def get_optimal_batch_size(self, image_size: Tuple[int, int]) -> int:
        """
        Calculate optimal batch size based on image size and available memory.

        Args:
            image_size: (width, height) of images

        Returns:
            Recommended batch size
        """
        if not HAS_TORCH or self.device.type != "cuda":
            return 4  # Conservative default for CPU

        try:
            # Get available GPU memory
            free_memory = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            available = (free_memory - allocated) * 0.8  # Use 80%

            # Estimate memory per image (rough heuristic)
            w, h = image_size
            bytes_per_image = w * h * 3 * 4 * 4  # RGB, float32, ~4x overhead

            batch_size = max(1, int(available / bytes_per_image))
            return min(batch_size, 32)  # Cap at 32

        except Exception:
            return 8  # Safe default
