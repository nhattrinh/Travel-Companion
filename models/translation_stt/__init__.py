"""
Speech-to-Text (STT) Translation Module.

This module provides speech recognition and translation capabilities
using OpenAI's Whisper model for the Travel Companion application.

Components:
    - WhisperSTTModel: Core Whisper ML model for transcription/translation
    - Metrics: Latency and accuracy metrics collection
"""

from .model import (
    WhisperSTTModel,
    WhisperModelSize,
)
from .metrics import (
    record_transcription_latency,
    record_translation_latency,
    record_language_detection,
    snapshot_stt_metrics,
    reset_stt_metrics,
)

__all__ = [
    # Model
    "WhisperSTTModel",
    "WhisperModelSize",
    # Metrics
    "record_transcription_latency",
    "record_translation_latency",
    "record_language_detection",
    "snapshot_stt_metrics",
    "reset_stt_metrics",
]
