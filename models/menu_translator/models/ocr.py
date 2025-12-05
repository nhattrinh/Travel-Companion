"""
OCR Model using PaddleOCR for multilingual menu text recognition.

Supports English (en), Korean (ko), and Vietnamese (vi).
"""

import time
import logging
from typing import Optional
from dataclasses import dataclass, field

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

import numpy as np
from PIL import Image

from .data_models import TextBox, OCRResult, SupportedLanguage
from .metrics import metrics

logger = logging.getLogger(__name__)


# Language detection patterns for quick identification
KOREAN_CHARS = set("가나다라마바사아자차카타파하")
VIETNAMESE_DIACRITICS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệ"
                            "ìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụ"
                            "ưừứửữựỳýỷỹỵđ")


@dataclass
class OCRConfig:
    """Configuration for OCR model."""
    use_gpu: bool = False
    use_angle_cls: bool = True
    lang: str = "en"  # Primary language hint
    det_model_dir: Optional[str] = None
    rec_model_dir: Optional[str] = None
    show_log: bool = False
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False


class OCRModel:
    """
    Multilingual OCR model for menu text extraction.
    
    Uses PaddleOCR for text detection and recognition with
    language detection for en/ko/vi.
    
    Metrics tracked:
    - ocr_requests_total: Total OCR requests
    - ocr_processing_seconds: Processing time histogram
    - ocr_text_boxes_detected: Number of text boxes per image
    - ocr_confidence_score: Average confidence scores
    - ocr_language_detected: Language detection counts
    - ocr_errors_total: Error counts by type
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        """Initialize OCR model with configuration."""
        self.config = config or OCRConfig()
        self._ocr_engines: dict[str, PaddleOCR] = {}
        self._initialized = False

        if not PADDLEOCR_AVAILABLE:
            logger.warning(
                "PaddleOCR not available. Install with: pip install paddleocr"
            )

    def _get_engine(self, lang: str = "en") -> Optional[PaddleOCR]:
        """Get or create OCR engine for specified language."""
        if not PADDLEOCR_AVAILABLE:
            return None

        if lang not in self._ocr_engines:
            try:
                # Map our language codes to PaddleOCR language codes
                paddle_lang_map = {
                    "en": "en",
                    "ko": "korean",
                    "vi": "vi",
                }
                paddle_lang = paddle_lang_map.get(lang, "en")

                self._ocr_engines[lang] = PaddleOCR(
                    use_angle_cls=self.config.use_angle_cls,
                    lang=paddle_lang,
                    use_gpu=self.config.use_gpu,
                    show_log=self.config.show_log,
                    use_doc_orientation_classify=(
                        self.config.use_doc_orientation_classify
                    ),
                    use_doc_unwarping=self.config.use_doc_unwarping,
                    use_textline_orientation=self.config.use_textline_orientation,
                )
                logger.info(f"Initialized OCR engine for language: {lang}")
            except Exception as e:
                logger.error(f"Failed to initialize OCR engine for {lang}: {e}")
                metrics.record_ocr_error("initialization_error")
                return None

        return self._ocr_engines[lang]

    def detect_language(self, text: str) -> SupportedLanguage:
        """
        Detect language of text, limited to en/ko/vi.
        
        Uses character-based heuristics first, then langdetect as fallback.
        """
        if not text or len(text.strip()) == 0:
            return SupportedLanguage.OTHER

        text_lower = text.lower()

        # Check for Korean characters
        if any(char in KOREAN_CHARS for char in text):
            return SupportedLanguage.KO

        # Check for Korean Unicode range (Hangul syllables)
        if any('\uAC00' <= char <= '\uD7A3' for char in text):
            return SupportedLanguage.KO

        # Check for Vietnamese diacritics
        if any(char in VIETNAMESE_DIACRITICS for char in text_lower):
            return SupportedLanguage.VI

        # Use langdetect for further detection
        if LANGDETECT_AVAILABLE:
            try:
                detected = detect(text)
                if detected == "ko":
                    return SupportedLanguage.KO
                elif detected == "vi":
                    return SupportedLanguage.VI
                elif detected == "en":
                    return SupportedLanguage.EN
            except Exception:
                pass

        # Default to English for ASCII text
        if text.isascii():
            return SupportedLanguage.EN

        return SupportedLanguage.OTHER

    def run_ocr(
        self,
        image: np.ndarray | Image.Image | bytes,
        hint_language: str = "en"
    ) -> OCRResult:
        """
        Run OCR on an image.
        
        Args:
            image: Input image as numpy array, PIL Image, or bytes
            hint_language: Language hint for OCR engine selection
            
        Returns:
            OCRResult with detected text boxes and metadata
        """
        start_time = time.time()
        metrics.increment_ocr_requests()

        # Convert image to numpy array if needed
        if isinstance(image, bytes):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            image_array = np.array(image)
            img_width, img_height = image.size
        else:
            image_array = image
            img_height, img_width = image_array.shape[:2]

        text_boxes: list[TextBox] = []

        engine = self._get_engine(hint_language)
        if engine is None:
            # Return empty result if no engine available
            processing_time = (time.time() - start_time) * 1000
            metrics.record_ocr_processing_time(processing_time / 1000)
            return OCRResult(
                text_boxes=[],
                processing_time_ms=processing_time,
                image_width=img_width,
                image_height=img_height,
                avg_confidence=0.0
            )

        try:
            # Run OCR
            result = engine.ocr(image_array, cls=self.config.use_angle_cls)

            if result and result[0]:
                for idx, line in enumerate(result[0]):
                    if len(line) >= 2:
                        bbox_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        text_info = line[1]  # (text, confidence)

                        # Convert polygon to bounding box
                        x_coords = [p[0] for p in bbox_points]
                        y_coords = [p[1] for p in bbox_points]
                        bbox = [
                            min(x_coords),
                            min(y_coords),
                            max(x_coords),
                            max(y_coords)
                        ]

                        text = text_info[0]
                        confidence = float(text_info[1])

                        # Detect language for this text
                        language = self.detect_language(text)
                        metrics.increment_language_detected(language.value)

                        text_boxes.append(TextBox(
                            bbox=bbox,
                            text=text,
                            confidence=confidence,
                            language=language,
                            block_id=0,
                            line_id=idx
                        ))

        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            metrics.record_ocr_error("processing_error")

        processing_time = (time.time() - start_time) * 1000

        # Calculate average confidence
        avg_confidence = 0.0
        if text_boxes:
            avg_confidence = sum(tb.confidence for tb in text_boxes) / len(text_boxes)

        # Record metrics
        metrics.record_ocr_processing_time(processing_time / 1000)
        metrics.record_text_boxes_detected(len(text_boxes))
        metrics.record_ocr_confidence(avg_confidence)

        return OCRResult(
            text_boxes=text_boxes,
            processing_time_ms=processing_time,
            image_width=img_width,
            image_height=img_height,
            avg_confidence=avg_confidence
        )

    def calculate_cer(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate Character Error Rate (CER).
        
        CER = (substitutions + insertions + deletions) / len(ground_truth)
        Uses Levenshtein distance.
        """
        if not ground_truth:
            return 0.0 if not predicted else 1.0

        # Simple Levenshtein distance
        m, n = len(predicted), len(ground_truth)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if predicted[i-1] == ground_truth[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # deletion
                        dp[i][j-1],      # insertion
                        dp[i-1][j-1]     # substitution
                    )

        cer = dp[m][n] / n
        metrics.record_cer(cer)
        return cer

    def calculate_wer(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate Word Error Rate (WER).
        
        WER = (substitutions + insertions + deletions) / num_words(ground_truth)
        """
        pred_words = predicted.split()
        gt_words = ground_truth.split()

        if not gt_words:
            return 0.0 if not pred_words else 1.0

        m, n = len(pred_words), len(gt_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_words[i-1] == gt_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],
                        dp[i][j-1],
                        dp[i-1][j-1]
                    )

        wer = dp[m][n] / n
        metrics.record_wer(wer)
        return wer
