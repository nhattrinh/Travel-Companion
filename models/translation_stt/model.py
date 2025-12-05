"""
Whisper Speech-to-Text ML Model.

Provides speech recognition and translation using OpenAI's Whisper model
for the Travel Companion application.

Features:
- Multiple model sizes (tiny, base, small, medium, large, turbo)
- Multilingual transcription (99+ languages)
- Speech translation to English
- Language detection
- Word-level timestamps
- GPU acceleration (CUDA, MPS)
"""

import asyncio
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .metrics import (
    record_transcription_latency,
    record_translation_latency,
    record_language_detection,
    record_detected_language,
    record_model_load_time,
    record_transcription_error,
    record_translation_error,
)

# Lazy import for whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    whisper = None  # type: ignore
    WHISPER_AVAILABLE = False

# Lazy import for torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class WhisperModelSize(str, Enum):
    """Available Whisper model sizes."""
    TINY = "tiny"
    TINY_EN = "tiny.en"
    BASE = "base"
    BASE_EN = "base.en"
    SMALL = "small"
    SMALL_EN = "small.en"
    MEDIUM = "medium"
    MEDIUM_EN = "medium.en"
    LARGE = "large"
    TURBO = "turbo"


class WhisperSTTModel:
    """
    Whisper Speech-to-Text ML Model.

    Wraps OpenAI's Whisper for transcription and translation
    with async support and metrics collection.

    Example:
        model = WhisperSTTModel(model_size="turbo", device="cuda")
        await model.load()

        # Transcribe audio
        result = await model.transcribe("audio.mp3")
        print(result["text"])

        # Translate to English
        result = await model.translate("japanese.wav", language="ja")
        print(result["text"])

        # Detect language
        lang, probs = await model.detect_language("speech.mp3")
    """

    def __init__(
        self,
        model_size: str = "turbo",
        device: Optional[str] = None,
        download_root: Optional[str] = None,
        in_memory: bool = False,
    ):
        """
        Initialize the Whisper STT model.

        Args:
            model_size: Model size (tiny, base, small, medium, large, turbo)
            device: Device to use (cuda, cpu, mps). Auto-detected if None.
            download_root: Custom directory for model downloads
            in_memory: Load model in memory (useful for containers)
        """
        self.model_size = model_size
        self.download_root = download_root
        self.in_memory = in_memory

        # Auto-detect device
        if device is None:
            self.device = self._detect_device()
        else:
            self.device = device

        self._model: Any = None
        self._is_loaded = False
        self._load_lock = asyncio.Lock()

        self.logger = logging.getLogger(self.__class__.__name__)

    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        if not TORCH_AVAILABLE:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        if (
            hasattr(torch.backends, 'mps') and
            torch.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @property
    def is_multilingual(self) -> bool:
        """Check if model supports multiple languages."""
        if self._model is not None:
            return self._model.is_multilingual
        return not self.model_size.endswith('.en')

    @property
    def num_parameters(self) -> Optional[int]:
        """Get total number of model parameters."""
        if self._model is None:
            return None
        return sum(np.prod(p.shape) for p in self._model.parameters())

    @property
    def dims(self) -> Optional[Any]:
        """Get model dimensions."""
        if self._model is None:
            return None
        return self._model.dims

    async def load(self) -> None:
        """
        Load the Whisper model.

        Thread-safe loading with timing metrics.
        """
        async with self._load_lock:
            if self._is_loaded:
                return

            if not WHISPER_AVAILABLE:
                raise RuntimeError(
                    "Whisper is not installed. "
                    "Install with: pip install openai-whisper"
                )

            self.logger.info(
                f"Loading Whisper {self.model_size} on {self.device}"
            )

            start_time = time.perf_counter()

            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                self._load_model_sync
            )

            load_time_ms = (time.perf_counter() - start_time) * 1000.0
            record_model_load_time(load_time_ms)

            self._is_loaded = True

            # Log model info
            if self.is_multilingual:
                lang_type = 'multilingual'
            else:
                lang_type = 'English-only'

            self.logger.info(
                f"Loaded in {load_time_ms:.0f}ms - "
                f"{lang_type}, {self.num_parameters:,} params"
            )

    def _load_model_sync(self) -> Any:
        """Synchronous model loading."""
        kwargs: Dict[str, Any] = {
            "device": self.device,
            "in_memory": self.in_memory,
        }

        if self.download_root:
            kwargs["download_root"] = self.download_root

        return whisper.load_model(self.model_size, **kwargs)

    async def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        temperature: float = 0.0,
        beam_size: int = 5,
        best_of: int = 5,
        fp16: bool = True,
        word_timestamps: bool = False,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        no_speech_threshold: float = 0.6,
        compression_ratio_threshold: float = 2.4,
        logprob_threshold: float = -1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio file path or numpy array (16kHz mono)
            language: Source language code (auto-detect if None)
            task: "transcribe" or "translate" (to English)
            temperature: Sampling temperature (0 for greedy)
            beam_size: Beam search width
            best_of: Number of candidates for best-of-n
            fp16: Use FP16 inference (GPU only)
            word_timestamps: Include word-level timestamps
            condition_on_previous_text: Use previous text as context
            initial_prompt: Optional prompt for context
            no_speech_threshold: Threshold for no-speech detection
            compression_ratio_threshold: Max compression ratio
            logprob_threshold: Min log probability threshold

        Returns:
            Dict with keys:
            - text: Full transcription
            - segments: List of segments with timestamps
            - language: Detected language code
        """
        if not self._is_loaded:
            await self.load()

        # Get audio duration for metrics
        audio_duration = await self._get_audio_duration(audio)

        # Disable fp16 on CPU
        if self.device == "cpu":
            fp16 = False

        options = {
            "task": task,
            "temperature": temperature,
            "beam_size": beam_size,
            "best_of": best_of,
            "fp16": fp16,
            "word_timestamps": word_timestamps,
            "condition_on_previous_text": condition_on_previous_text,
            "no_speech_threshold": no_speech_threshold,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            **kwargs,
        }

        if language:
            options["language"] = language
        if initial_prompt:
            options["initial_prompt"] = initial_prompt

        try:
            with record_transcription_latency(audio_duration):
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._model.transcribe(
                        str(audio) if isinstance(audio, Path) else audio,
                        **options
                    )
                )
        except Exception as e:
            record_transcription_error()
            self.logger.error(f"Transcription failed: {e}")
            raise

        # Record detected language
        detected_lang = result.get("language", "unknown")
        record_detected_language(detected_lang)

        return result

    async def translate(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Translate audio to English text.

        Note: Turbo model doesn't support translation well.
        Use medium or large for best results.

        Args:
            audio: Audio file path or numpy array
            language: Source language (auto-detect if None)
            **kwargs: Additional options for transcribe()

        Returns:
            Dict with English translation
        """
        if not self._is_loaded:
            await self.load()

        if self.model_size == "turbo":
            self.logger.warning(
                "Turbo model has limited translation support. "
                "Consider using medium or large."
            )

        # Get audio duration for metrics
        audio_duration = await self._get_audio_duration(audio)

        options = {
            "task": "translate",
            **kwargs,
        }
        if language:
            options["language"] = language

        try:
            with record_translation_latency(audio_duration):
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._model.transcribe(
                        str(audio) if isinstance(audio, Path) else audio,
                        **options
                    )
                )
        except Exception as e:
            record_translation_error()
            self.logger.error(f"Translation failed: {e}")
            raise

        detected_lang = result.get("language", "unknown")
        record_detected_language(detected_lang)

        return result

    async def detect_language(
        self,
        audio: Union[str, Path, np.ndarray],
    ) -> tuple[str, Dict[str, float]]:
        """
        Detect the language of audio.

        Args:
            audio: Audio file path or numpy array

        Returns:
            Tuple of (language_code, probability_dict)
        """
        if not self._is_loaded:
            await self.load()

        with record_language_detection():
            loop = asyncio.get_event_loop()

            # Load and preprocess audio
            if isinstance(audio, (str, Path)):
                audio_data = await loop.run_in_executor(
                    None,
                    lambda: whisper.load_audio(str(audio))
                )
            else:
                audio_data = audio

            audio_data = whisper.pad_or_trim(audio_data)

            # Generate mel spectrogram
            mel = whisper.log_mel_spectrogram(
                audio_data,
                n_mels=self._model.dims.n_mels
            ).to(self._model.device)

            # Detect language
            _, probs = await loop.run_in_executor(
                None,
                lambda: self._model.detect_language(mel)
            )

        detected = max(probs, key=probs.get)
        record_detected_language(detected)

        return detected, dict(probs)

    async def decode(
        self,
        mel: Any,
        language: Optional[str] = None,
        task: str = "transcribe",
        without_timestamps: bool = True,
    ) -> Any:
        """
        Low-level decoding from mel spectrogram.

        Args:
            mel: Log-mel spectrogram tensor
            language: Language code
            task: "transcribe" or "translate"
            without_timestamps: Disable timestamp prediction

        Returns:
            DecodingResult object
        """
        if not self._is_loaded:
            await self.load()

        options = whisper.DecodingOptions(
            language=language,
            task=task,
            without_timestamps=without_timestamps,
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: whisper.decode(self._model, mel, options)
        )

        return result

    async def _get_audio_duration(
        self,
        audio: Union[str, Path, np.ndarray],
    ) -> Optional[float]:
        """Get audio duration in seconds."""
        try:
            if isinstance(audio, (str, Path)):
                loop = asyncio.get_event_loop()
                audio_data = await loop.run_in_executor(
                    None,
                    lambda: whisper.load_audio(str(audio))
                )
                # Whisper loads at 16kHz
                return len(audio_data) / 16000.0
            elif isinstance(audio, np.ndarray):
                return len(audio) / 16000.0
        except Exception:
            pass
        return None

    async def unload(self) -> None:
        """Unload model to free memory."""
        async with self._load_lock:
            if self._model is not None:
                del self._model
                self._model = None
                self._is_loaded = False

                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.logger.info("Model unloaded")

    @staticmethod
    def available_models() -> List[str]:
        """Get list of available model names."""
        if WHISPER_AVAILABLE:
            return whisper.available_models()
        return [size.value for size in WhisperModelSize]

    @staticmethod
    def load_audio(path: Union[str, Path]) -> np.ndarray:
        """
        Load audio file as numpy array.

        Args:
            path: Path to audio file

        Returns:
            Audio as float32 numpy array at 16kHz
        """
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper not installed")
        return whisper.load_audio(str(path))

    @staticmethod
    def pad_or_trim(audio: np.ndarray, length: int = 480000) -> np.ndarray:
        """
        Pad or trim audio to specified length.

        Args:
            audio: Audio array
            length: Target length (default 30s at 16kHz)

        Returns:
            Padded/trimmed audio
        """
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper not installed")
        return whisper.pad_or_trim(audio, length)
