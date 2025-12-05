"""
Speech-to-Text metrics instrumentation.

Provides latency tracking and statistics for Whisper-based
transcription and translation operations.
"""

import time
from contextlib import contextmanager
from typing import Optional
from collections import defaultdict

# Timing storage
_transcription_timings_ms: list[float] = []
_translation_timings_ms: list[float] = []
_language_detection_timings_ms: list[float] = []

# Language detection stats
_detected_languages: dict[str, int] = defaultdict(int)

# Model loading stats
_model_load_time_ms: Optional[float] = None

# Error tracking
_transcription_errors: int = 0
_translation_errors: int = 0

# Audio duration tracking for RTF calculation
_total_audio_duration_seconds: float = 0.0
_total_processing_time_seconds: float = 0.0


@contextmanager
def record_transcription_latency(
    audio_duration_seconds: Optional[float] = None
):
    """
    Record transcription latency.
    
    Args:
        audio_duration_seconds: Duration of the audio being transcribed,
                                used for RTF (Real-Time Factor) calculation.
    
    Yields:
        None
    
    Example:
        with record_transcription_latency(audio_duration_seconds=5.0):
            result = model.transcribe(audio)
    """
    global _total_audio_duration_seconds, _total_processing_time_seconds
    
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_seconds = time.perf_counter() - start
        elapsed_ms = elapsed_seconds * 1000.0
        _transcription_timings_ms.append(elapsed_ms)
        
        if audio_duration_seconds is not None:
            _total_audio_duration_seconds += audio_duration_seconds
            _total_processing_time_seconds += elapsed_seconds


@contextmanager
def record_translation_latency(audio_duration_seconds: Optional[float] = None):
    """
    Record translation latency (speech to English translation).
    
    Args:
        audio_duration_seconds: Duration of the audio being translated.
    
    Yields:
        None
    """
    global _total_audio_duration_seconds, _total_processing_time_seconds
    
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_seconds = time.perf_counter() - start
        elapsed_ms = elapsed_seconds * 1000.0
        _translation_timings_ms.append(elapsed_ms)
        
        if audio_duration_seconds is not None:
            _total_audio_duration_seconds += audio_duration_seconds
            _total_processing_time_seconds += elapsed_seconds


@contextmanager
def record_language_detection():
    """
    Record language detection latency.
    
    Yields:
        None
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _language_detection_timings_ms.append(elapsed_ms)


def record_detected_language(language_code: str) -> None:
    """
    Record a detected language.
    
    Args:
        language_code: ISO 639-1 language code (e.g., 'en', 'ja', 'vi')
    """
    _detected_languages[language_code] += 1


def record_model_load_time(load_time_ms: float) -> None:
    """
    Record model loading time.
    
    Args:
        load_time_ms: Time taken to load the model in milliseconds
    """
    global _model_load_time_ms
    _model_load_time_ms = load_time_ms


def record_transcription_error() -> None:
    """Increment transcription error counter."""
    global _transcription_errors
    _transcription_errors += 1


def record_translation_error() -> None:
    """Increment translation error counter."""
    global _translation_errors
    _translation_errors += 1


def _calculate_percentiles(values: list[float]) -> dict:
    """
    Calculate percentile statistics for a list of values.
    
    Args:
        values: List of timing values in milliseconds
    
    Returns:
        Dictionary with count, avg, p50, p95, p99, min, max
    """
    if not values:
        return {
            "count": 0,
            "avg_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    
    sorted_vals = sorted(values)
    count = len(sorted_vals)
    
    def _percentile(p: float) -> float:
        idx = int(round(p * (count - 1)))
        return sorted_vals[idx]
    
    return {
        "count": count,
        "avg_ms": sum(sorted_vals) / count,
        "p50_ms": _percentile(0.50),
        "p95_ms": _percentile(0.95),
        "p99_ms": _percentile(0.99),
        "min_ms": sorted_vals[0],
        "max_ms": sorted_vals[-1],
    }


def _calculate_rtf() -> Optional[float]:
    """
    Calculate Real-Time Factor (RTF).
    
    RTF < 1.0 means faster than real-time processing.
    
    Returns:
        RTF value or None if no audio has been processed
    """
    if _total_audio_duration_seconds == 0:
        return None
    return _total_processing_time_seconds / _total_audio_duration_seconds


def snapshot_stt_metrics() -> dict:
    """
    Get a snapshot of all STT metrics.
    
    Returns:
        Dictionary containing:
        - transcription: Transcription latency percentiles
        - translation: Translation latency percentiles
        - language_detection: Language detection latency percentiles
        - detected_languages: Count of detected languages
        - rtf: Real-Time Factor (processing speed)
        - model_load_time_ms: Model loading time
        - errors: Error counts
    """
    lang_detect_stats = _calculate_percentiles(_language_detection_timings_ms)
    return {
        "transcription": _calculate_percentiles(_transcription_timings_ms),
        "translation": _calculate_percentiles(_translation_timings_ms),
        "language_detection": lang_detect_stats,
        "detected_languages": dict(_detected_languages),
        "rtf": _calculate_rtf(),
        "model_load_time_ms": _model_load_time_ms,
        "errors": {
            "transcription": _transcription_errors,
            "translation": _translation_errors,
        },
        "total_audio_processed_seconds": _total_audio_duration_seconds,
    }


def reset_stt_metrics() -> None:
    """Reset all STT metrics to initial state."""
    global _transcription_timings_ms, _translation_timings_ms
    global _language_detection_timings_ms, _detected_languages
    global _model_load_time_ms, _transcription_errors, _translation_errors
    global _total_audio_duration_seconds, _total_processing_time_seconds
    
    _transcription_timings_ms = []
    _translation_timings_ms = []
    _language_detection_timings_ms = []
    _detected_languages = defaultdict(int)
    _model_load_time_ms = None
    _transcription_errors = 0
    _translation_errors = 0
    _total_audio_duration_seconds = 0.0
    _total_processing_time_seconds = 0.0
