"""
Metrics collection for Menu Translator models.

Provides Prometheus-compatible metrics for:
- OCR performance (CER, WER, detection accuracy, latency)
- Translation quality (BLEU, confidence, latency)
- Classification accuracy (Top-1, Top-5, F1, latency)
- System metrics (request counts, errors, throughput)
"""

import time
import logging
from typing import Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from functools import wraps

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        REGISTRY,
        generate_latest,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


# Histogram buckets for latency measurements
LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
    0.75, 1.0, 2.5, 5.0, 7.5, 10.0
)

# Buckets for confidence scores
CONFIDENCE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


@dataclass
class MetricValue:
    """Simple metric value holder for non-Prometheus mode."""
    count: int = 0
    total: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    last_val: float = 0.0

    def observe(self, value: float):
        self.count += 1
        self.total += value
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        self.last_val = value

    def inc(self, amount: float = 1.0):
        self.count += int(amount)
        self.total += amount
        self.last_val = self.total

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0


class MetricsCollector:
    """
    Centralized metrics collector for all models.
    
    Supports both Prometheus metrics (when available) and
    simple in-memory metrics for testing/development.
    
    Categories:
    - OCR Metrics: Request counts, latency, CER, WER, confidence
    - Translation Metrics: Request counts, latency, BLEU, pairs
    - Classifier Metrics: Request counts, latency, accuracy, F1
    - Parser Metrics: Items extracted, sections, prices
    - System Metrics: Errors, throughput, model status
    """

    def __init__(self, use_prometheus: bool = True):
        """
        Initialize metrics collector.
        
        Args:
            use_prometheus: Whether to use Prometheus metrics
        """
        self.use_prometheus = use_prometheus and PROMETHEUS_AVAILABLE
        self._simple_metrics: dict[str, MetricValue] = defaultdict(MetricValue)

        if self.use_prometheus:
            self._init_prometheus_metrics()
        else:
            logger.info("Using simple in-memory metrics")

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # ===== OCR Metrics =====
        self.ocr_requests = Counter(
            'menu_ocr_requests_total',
            'Total OCR requests'
        )
        self.ocr_processing_time = Histogram(
            'menu_ocr_processing_seconds',
            'OCR processing time in seconds',
            buckets=LATENCY_BUCKETS
        )
        self.ocr_text_boxes = Histogram(
            'menu_ocr_text_boxes_detected',
            'Number of text boxes detected per image',
            buckets=[1, 5, 10, 20, 50, 100, 200]
        )
        self.ocr_confidence = Histogram(
            'menu_ocr_confidence_score',
            'OCR confidence scores',
            buckets=CONFIDENCE_BUCKETS
        )
        self.ocr_language = Counter(
            'menu_ocr_language_detected_total',
            'Language detection counts',
            ['language']
        )
        self.ocr_errors = Counter(
            'menu_ocr_errors_total',
            'OCR error counts',
            ['error_type']
        )
        self.ocr_cer = Histogram(
            'menu_ocr_character_error_rate',
            'Character Error Rate distribution',
            buckets=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
        )
        self.ocr_wer = Histogram(
            'menu_ocr_word_error_rate',
            'Word Error Rate distribution',
            buckets=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
        )

        # ===== Translation Metrics =====
        self.translation_requests = Counter(
            'menu_translation_requests_total',
            'Total translation requests'
        )
        self.translation_time = Histogram(
            'menu_translation_processing_seconds',
            'Translation processing time',
            buckets=LATENCY_BUCKETS
        )
        self.translation_pairs = Counter(
            'menu_translation_language_pairs_total',
            'Translation requests by language pair',
            ['source', 'target']
        )
        self.translation_confidence = Histogram(
            'menu_translation_confidence',
            'Translation confidence scores',
            buckets=CONFIDENCE_BUCKETS
        )
        self.translation_glossary_hits = Counter(
            'menu_translation_glossary_hits_total',
            'Glossary term matches'
        )
        self.translation_errors = Counter(
            'menu_translation_errors_total',
            'Translation errors',
            ['error_type']
        )
        self.translation_bleu = Histogram(
            'menu_translation_bleu_score',
            'BLEU score distribution',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        # ===== Classifier Metrics =====
        self.classifier_requests = Counter(
            'menu_classifier_requests_total',
            'Total classification requests'
        )
        self.classifier_time = Histogram(
            'menu_classifier_processing_seconds',
            'Classification processing time',
            buckets=LATENCY_BUCKETS
        )
        self.classifier_top1_conf = Histogram(
            'menu_classifier_top1_confidence',
            'Top-1 prediction confidence',
            buckets=CONFIDENCE_BUCKETS
        )
        self.classifier_predictions = Counter(
            'menu_classifier_predictions_total',
            'Prediction class counts',
            ['class_name']
        )
        self.classifier_errors = Counter(
            'menu_classifier_errors_total',
            'Classification errors',
            ['error_type']
        )
        self.classifier_model_loaded = Gauge(
            'menu_classifier_model_loaded',
            'Whether model is loaded (1=yes, 0=no)'
        )
        self.classifier_top1_acc = Gauge(
            'menu_classifier_top1_accuracy',
            'Top-1 accuracy on evaluation set'
        )
        self.classifier_top5_acc = Gauge(
            'menu_classifier_top5_accuracy',
            'Top-5 accuracy on evaluation set'
        )
        self.classifier_f1 = Gauge(
            'menu_classifier_f1_score',
            'Macro F1 score on evaluation set'
        )

        # ===== Parser Metrics =====
        self.parser_items = Histogram(
            'menu_parser_items_extracted',
            'Number of menu items extracted',
            buckets=[1, 5, 10, 20, 50, 100]
        )
        self.parser_sections = Counter(
            'menu_parser_sections_detected_total',
            'Number of sections detected'
        )
        self.parser_prices = Counter(
            'menu_parser_prices_detected_total',
            'Number of prices detected'
        )
        self.parser_time = Histogram(
            'menu_parser_processing_seconds',
            'Parser processing time',
            buckets=LATENCY_BUCKETS
        )
        self.parser_grouping_acc = Gauge(
            'menu_parser_grouping_accuracy',
            'Menu item grouping accuracy'
        )

        # ===== System Metrics =====
        self.total_requests = Counter(
            'menu_translator_requests_total',
            'Total API requests',
            ['endpoint']
        )
        self.request_latency = Histogram(
            'menu_translator_request_seconds',
            'End-to-end request latency',
            ['endpoint'],
            buckets=LATENCY_BUCKETS
        )

    # ===== OCR Metric Methods =====

    def increment_ocr_requests(self):
        """Increment OCR request counter."""
        if self.use_prometheus:
            self.ocr_requests.inc()
        else:
            self._simple_metrics['ocr_requests'].inc()

    def record_ocr_processing_time(self, seconds: float):
        """Record OCR processing time."""
        if self.use_prometheus:
            self.ocr_processing_time.observe(seconds)
        else:
            self._simple_metrics['ocr_processing_time'].observe(seconds)

    def record_text_boxes_detected(self, count: int):
        """Record number of text boxes detected."""
        if self.use_prometheus:
            self.ocr_text_boxes.observe(count)
        else:
            self._simple_metrics['ocr_text_boxes'].observe(count)

    def record_ocr_confidence(self, confidence: float):
        """Record OCR confidence score."""
        if self.use_prometheus:
            self.ocr_confidence.observe(confidence)
        else:
            self._simple_metrics['ocr_confidence'].observe(confidence)

    def increment_language_detected(self, language: str):
        """Increment language detection counter."""
        if self.use_prometheus:
            self.ocr_language.labels(language=language).inc()
        else:
            self._simple_metrics[f'ocr_language_{language}'].inc()

    def record_ocr_error(self, error_type: str):
        """Record OCR error."""
        if self.use_prometheus:
            self.ocr_errors.labels(error_type=error_type).inc()
        else:
            self._simple_metrics[f'ocr_error_{error_type}'].inc()

    def record_cer(self, cer: float):
        """Record Character Error Rate."""
        if self.use_prometheus:
            self.ocr_cer.observe(cer)
        else:
            self._simple_metrics['ocr_cer'].observe(cer)

    def record_wer(self, wer: float):
        """Record Word Error Rate."""
        if self.use_prometheus:
            self.ocr_wer.observe(wer)
        else:
            self._simple_metrics['ocr_wer'].observe(wer)

    # ===== Translation Metric Methods =====

    def increment_translation_requests(self):
        """Increment translation request counter."""
        if self.use_prometheus:
            self.translation_requests.inc()
        else:
            self._simple_metrics['translation_requests'].inc()

    def record_translation_time(self, seconds: float):
        """Record translation processing time."""
        if self.use_prometheus:
            self.translation_time.observe(seconds)
        else:
            self._simple_metrics['translation_time'].observe(seconds)

    def record_translation_pair(self, source: str, target: str):
        """Record translation language pair."""
        if self.use_prometheus:
            self.translation_pairs.labels(source=source, target=target).inc()
        else:
            self._simple_metrics[f'translation_pair_{source}_{target}'].inc()

    def record_translation_confidence(self, confidence: float):
        """Record translation confidence."""
        if self.use_prometheus:
            self.translation_confidence.observe(confidence)
        else:
            self._simple_metrics['translation_confidence'].observe(confidence)

    def increment_glossary_hits(self):
        """Increment glossary hit counter."""
        if self.use_prometheus:
            self.translation_glossary_hits.inc()
        else:
            self._simple_metrics['translation_glossary_hits'].inc()

    def record_translation_error(self, error_type: str):
        """Record translation error."""
        if self.use_prometheus:
            self.translation_errors.labels(error_type=error_type).inc()
        else:
            self._simple_metrics[f'translation_error_{error_type}'].inc()

    def record_bleu_score(self, bleu: float):
        """Record BLEU score."""
        if self.use_prometheus:
            self.translation_bleu.observe(bleu)
        else:
            self._simple_metrics['translation_bleu'].observe(bleu)

    # ===== Classifier Metric Methods =====

    def increment_classifier_requests(self):
        """Increment classifier request counter."""
        if self.use_prometheus:
            self.classifier_requests.inc()
        else:
            self._simple_metrics['classifier_requests'].inc()

    def record_classifier_time(self, seconds: float):
        """Record classification processing time."""
        if self.use_prometheus:
            self.classifier_time.observe(seconds)
        else:
            self._simple_metrics['classifier_time'].observe(seconds)

    def record_top1_confidence(self, confidence: float):
        """Record top-1 prediction confidence."""
        if self.use_prometheus:
            self.classifier_top1_conf.observe(confidence)
        else:
            self._simple_metrics['classifier_top1_conf'].observe(confidence)

    def record_prediction_class(self, class_name: str):
        """Record prediction class."""
        if self.use_prometheus:
            self.classifier_predictions.labels(class_name=class_name).inc()
        else:
            self._simple_metrics[f'classifier_pred_{class_name}'].inc()

    def record_classifier_error(self, error_type: str):
        """Record classifier error."""
        if self.use_prometheus:
            self.classifier_errors.labels(error_type=error_type).inc()
        else:
            self._simple_metrics[f'classifier_error_{error_type}'].inc()

    def set_model_loaded(self, loaded: bool):
        """Set model loaded status."""
        if self.use_prometheus:
            self.classifier_model_loaded.set(1 if loaded else 0)
        else:
            self._simple_metrics['classifier_model_loaded'].last_val = (
                1 if loaded else 0
            )

    def record_top1_accuracy(self, accuracy: float):
        """Record top-1 accuracy."""
        if self.use_prometheus:
            self.classifier_top1_acc.set(accuracy)
        else:
            self._simple_metrics['classifier_top1_acc'].last_val = accuracy

    def record_top5_accuracy(self, accuracy: float):
        """Record top-5 accuracy."""
        if self.use_prometheus:
            self.classifier_top5_acc.set(accuracy)
        else:
            self._simple_metrics['classifier_top5_acc'].last_val = accuracy

    def record_f1_score(self, f1: float):
        """Record F1 score."""
        if self.use_prometheus:
            self.classifier_f1.set(f1)
        else:
            self._simple_metrics['classifier_f1'].last_val = f1

    # ===== Parser Metric Methods =====

    def record_items_extracted(self, count: int):
        """Record number of menu items extracted."""
        if self.use_prometheus:
            self.parser_items.observe(count)
        else:
            self._simple_metrics['parser_items'].observe(count)

    def increment_sections_detected(self):
        """Increment sections detected counter."""
        if self.use_prometheus:
            self.parser_sections.inc()
        else:
            self._simple_metrics['parser_sections'].inc()

    def increment_prices_detected(self):
        """Increment prices detected counter."""
        if self.use_prometheus:
            self.parser_prices.inc()
        else:
            self._simple_metrics['parser_prices'].inc()

    def record_parser_time(self, seconds: float):
        """Record parser processing time."""
        if self.use_prometheus:
            self.parser_time.observe(seconds)
        else:
            self._simple_metrics['parser_time'].observe(seconds)

    def record_grouping_accuracy(self, accuracy: float):
        """Record menu item grouping accuracy."""
        if self.use_prometheus:
            self.parser_grouping_acc.set(accuracy)
        else:
            self._simple_metrics['parser_grouping_acc'].last_val = accuracy

    # ===== System Metric Methods =====

    def record_request(self, endpoint: str, latency: float):
        """Record API request with latency."""
        if self.use_prometheus:
            self.total_requests.labels(endpoint=endpoint).inc()
            self.request_latency.labels(endpoint=endpoint).observe(latency)
        else:
            self._simple_metrics[f'request_{endpoint}'].inc()
            self._simple_metrics[f'latency_{endpoint}'].observe(latency)

    # ===== Utility Methods =====

    def get_metrics(self) -> dict:
        """
        Get all metrics as dictionary.
        
        Useful for debugging and non-Prometheus environments.
        """
        result = {}
        for name, metric in self._simple_metrics.items():
            result[name] = {
                'count': metric.count,
                'total': metric.total,
                'avg': metric.avg,
                'min': metric.min_val if metric.min_val != float('inf') else 0,
                'max': metric.max_val if metric.max_val != float('-inf') else 0,
                'last': metric.last_val,
            }
        return result

    def export_prometheus(self) -> bytes:
        """Export metrics in Prometheus format."""
        if self.use_prometheus:
            return generate_latest(REGISTRY)
        else:
            # Return simple text format
            lines = []
            for name, values in self.get_metrics().items():
                lines.append(f"# {name}")
                lines.append(f"{name}_count {values['count']}")
                lines.append(f"{name}_total {values['total']}")
                lines.append(f"{name}_avg {values['avg']}")
            return "\n".join(lines).encode()

    def reset(self):
        """Reset all simple metrics (for testing)."""
        self._simple_metrics.clear()


def timed(metric_name: str):
    """
    Decorator to time function execution.
    
    Usage:
        @timed('my_function')
        def my_function():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                metrics._simple_metrics[metric_name].observe(elapsed)
        return wrapper
    return decorator


# Global metrics instance
metrics = MetricsCollector(use_prometheus=PROMETHEUS_AVAILABLE)
