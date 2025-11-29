"""
Base classes and configuration for ML models.

Provides abstract base classes and shared configuration for all ML models
in the Travel Companion application, following latest PyTorch best practices.
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Dict, List, TypeVar, Generic

# Lazy import for torch to allow module loading without torch installed
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """Supported compute devices."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


class PrecisionMode(str, Enum):
    """Precision modes for inference."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


@dataclass
class ModelConfig:
    """
    Configuration for ML models.

    Attributes:
        model_name: HuggingFace model identifier or local path
        device: Target device (cpu, cuda, mps)
        precision: Inference precision mode
        use_compile: Whether to use torch.compile for optimization
        compile_mode: torch.compile mode (reduce-overhead, max-autotune)
        max_batch_size: Maximum batch size for inference
        cache_enabled: Whether to enable model output caching
        cache_ttl_seconds: Cache TTL in seconds
        warmup_iterations: Number of warmup runs before serving
    """
    model_name: str
    device_type: DeviceType = DeviceType.CPU
    precision_mode: PrecisionMode = PrecisionMode.FP16
    use_compile: bool = True
    compile_mode: str = "reduce-overhead"
    max_batch_size: int = 8
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    warmup_iterations: int = 3

    def __post_init__(self):
        """Validate and adjust configuration based on device capabilities."""
        if not TORCH_AVAILABLE:
            self.device_type = DeviceType.CPU
            self.use_compile = False
            return

        # Auto-detect best device on first access
        self._validate_device()
        self._validate_precision()

    def _validate_device(self) -> None:
        """Validate and adjust device based on availability."""
        if not TORCH_AVAILABLE:
            return

        if self.device_type == DeviceType.CUDA:
            if not torch.cuda.is_available():
                logger.warning(
                    "CUDA requested but not available, falling back to CPU"
                )
                self.device_type = DeviceType.CPU

        if self.device_type == DeviceType.MPS:
            if not (
                hasattr(torch.backends, 'mps') and
                torch.backends.mps.is_available()
            ):
                logger.warning(
                    "MPS requested but not available, falling back to CPU"
                )
                self.device_type = DeviceType.CPU

    def _validate_precision(self) -> None:
        """Validate and adjust precision based on device."""
        if not TORCH_AVAILABLE:
            return

        if self.device_type == DeviceType.CPU:
            if self.precision_mode == PrecisionMode.FP16:
                logger.info("FP16 not optimal on CPU, using BF16")
                self.precision_mode = PrecisionMode.BF16

        if self.precision_mode == PrecisionMode.BF16:
            if self.device_type == DeviceType.CUDA:
                if not torch.cuda.is_bf16_supported():
                    logger.warning("BF16 not supported, using FP16")
                    self.precision_mode = PrecisionMode.FP16

    @property
    def torch_device(self) -> Any:
        """Get torch device object."""
        if not TORCH_AVAILABLE:
            return None
        return torch.device(self.device_type.value)

    @property
    def torch_dtype(self) -> Any:
        """Get torch dtype for the configured precision."""
        if not TORCH_AVAILABLE:
            return None
        dtype_map = {
            PrecisionMode.FP32: torch.float32,
            PrecisionMode.FP16: torch.float16,
            PrecisionMode.BF16: torch.bfloat16,
            PrecisionMode.INT8: torch.int8,
        }
        return dtype_map.get(self.precision_mode, torch.float32)


T = TypeVar('T')


class BaseMLModel(ABC, Generic[T]):
    """
    Abstract base class for all ML models.

    Provides common functionality:
    - Device management
    - torch.compile optimization
    - Mixed precision inference
    - Health checks
    - Model warmup
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the base ML model.

        Args:
            config: Model configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model: Any = None
        self._processor: Any = None
        self._is_compiled = False
        self._is_loaded = False
        self._initialized = False
        self._load_lock = asyncio.Lock()

    @property
    def device(self) -> Any:
        """Get the compute device."""
        return self.config.torch_device

    @property
    def dtype(self) -> Any:
        """Get the torch dtype."""
        return self.config.torch_dtype

    @abstractmethod
    async def _load_model(self) -> None:
        """
        Load the model and processor.

        Must be implemented by subclasses to load their specific model.
        """
        pass

    @abstractmethod
    async def _run_inference(self, inputs: T) -> Any:
        """
        Run model inference.

        Args:
            inputs: Model inputs (type varies by model)

        Returns:
            Model outputs
        """
        pass

    async def load(self) -> None:
        """
        Load and optimize the model.

        Thread-safe model loading with warmup.
        """
        async with self._load_lock:
            if self._is_loaded:
                return

            self.logger.info(f"Loading model: {self.config.model_name}")

            # Load the model
            await self._load_model()

            # Apply torch.compile if enabled and supported
            if self.config.use_compile and self._model is not None:
                await self._compile_model()

            # Run warmup iterations
            await self._warmup()

            self._is_loaded = True
            self.logger.info(f"Model loaded successfully on {self.config.device.value}")

    async def _compile_model(self) -> None:
        """
        Compile the model using torch.compile.

        Uses the configured compile mode for optimization.
        """
        if self._is_compiled:
            return

        try:
            self.logger.info(f"Compiling model with mode: {self.config.compile_mode}")

            # Run compilation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: torch.compile(
                    self._model,
                    mode=self.config.compile_mode,
                    fullgraph=False,  # Allow graph breaks for flexibility
                )
            )

            self._is_compiled = True
            self.logger.info("Model compiled successfully")

        except Exception as e:
            self.logger.warning(f"torch.compile failed: {e}, using eager mode")
            self._is_compiled = False

    async def _warmup(self) -> None:
        """
        Run warmup iterations to optimize JIT compilation.

        Executes dummy inputs to trigger compilation and caching.
        """
        if self.config.warmup_iterations <= 0:
            return

        self.logger.info(f"Running {self.config.warmup_iterations} warmup iterations")

        for i in range(self.config.warmup_iterations):
            try:
                await self._run_warmup_iteration()
            except Exception as e:
                self.logger.warning(f"Warmup iteration {i} failed: {e}")

    async def _run_warmup_iteration(self) -> None:
        """
        Run a single warmup iteration.

        Override in subclasses for model-specific warmup.
        """
        pass

    async def inference(self, inputs: T) -> Any:
        """
        Run inference with automatic mixed precision.

        Args:
            inputs: Model inputs

        Returns:
            Model outputs
        """
        if not self._is_loaded:
            await self.load()

        # Run with autocast for mixed precision
        if self.config.device == DeviceType.CUDA:
            with torch.autocast(
                device_type="cuda",
                dtype=self.dtype,
                enabled=self.config.precision != PrecisionMode.FP32
            ):
                with torch.inference_mode():
                    return await self._run_inference(inputs)
        else:
            # CPU or MPS path
            device_type = "cpu" if self.config.device == DeviceType.CPU else "mps"
            with torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16 if self.config.precision == PrecisionMode.BF16 else torch.float32,
                enabled=self.config.precision in (PrecisionMode.BF16,)
            ):
                with torch.inference_mode():
                    return await self._run_inference(inputs)

    async def health_check(self) -> bool:
        """
        Check if the model is healthy and ready for inference.

        Returns:
            True if model is ready, False otherwise
        """
        try:
            if not self._is_loaded:
                return False

            # Run a quick inference to verify model is working
            await self._run_warmup_iteration()
            return True

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get model statistics and configuration.

        Returns:
            Dictionary with model stats
        """
        stats = {
            "model_name": self.config.model_name,
            "device": self.config.device.value,
            "precision": self.config.precision.value,
            "is_loaded": self._is_loaded,
            "is_compiled": self._is_compiled,
            "compile_mode": self.config.compile_mode if self.config.use_compile else None,
        }

        if torch.cuda.is_available() and self.config.device == DeviceType.CUDA:
            stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024

        return stats

    async def unload(self) -> None:
        """
        Unload the model and free resources.
        """
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._is_loaded = False
        self._is_compiled = False

        # Clear CUDA cache if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Model unloaded")
