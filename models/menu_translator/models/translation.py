"""
Translation Model for menu text translation.

Supports translation between English (en), Korean (ko), and Vietnamese (vi).
"""

import time
import logging
from typing import Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .data_models import (
    LocalizedText,
    TranslationResult,
    SupportedLanguage
)
from .metrics import metrics

logger = logging.getLogger(__name__)


# Food-specific glossary for better translations
FOOD_GLOSSARY = {
    # Korean dishes
    "ko": {
        "비빔밥": ("bibimbap", "mixed rice with vegetables and meat"),
        "불고기": ("bulgogi", "marinated grilled beef"),
        "김치": ("kimchi", "fermented vegetables"),
        "떡볶이": ("tteokbokki", "spicy rice cakes"),
        "삼겹살": ("samgyeopsal", "grilled pork belly"),
        "냉면": ("naengmyeon", "cold noodles"),
        "갈비": ("galbi", "grilled short ribs"),
        "김밥": ("gimbap", "seaweed rice roll"),
        "순두부": ("sundubu", "soft tofu"),
        "찌개": ("jjigae", "stew"),
        "국": ("guk", "soup"),
        "밥": ("bap", "rice"),
        "면": ("myeon", "noodles"),
    },
    # Vietnamese dishes
    "vi": {
        "phở": ("pho", "Vietnamese noodle soup"),
        "bánh mì": ("banh mi", "Vietnamese sandwich"),
        "bún": ("bun", "rice vermicelli"),
        "gỏi cuốn": ("goi cuon", "fresh spring rolls"),
        "chả giò": ("cha gio", "fried spring rolls"),
        "cơm": ("com", "rice"),
        "bún chả": ("bun cha", "grilled pork with noodles"),
        "bánh xèo": ("banh xeo", "Vietnamese crepe"),
        "cà phê": ("ca phe", "coffee"),
        "nước mắm": ("nuoc mam", "fish sauce"),
    }
}


class TranslationBackend(ABC):
    """Abstract base class for translation backends."""

    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> tuple[str, float]:
        """
        Translate text from source to target language.
        
        Returns:
            Tuple of (translated_text, confidence_score)
        """
        pass


class MockTranslationBackend(TranslationBackend):
    """
    Mock translation backend for testing.
    
    Returns the original text with a language prefix.
    """

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> tuple[str, float]:
        if source_lang == target_lang:
            return text, 1.0

        # Check glossary first
        if source_lang in FOOD_GLOSSARY:
            glossary = FOOD_GLOSSARY[source_lang]
            for term, (transliteration, description) in glossary.items():
                if term in text:
                    if target_lang == "en":
                        return f"{transliteration} ({description})", 0.9

        # Mock translation
        return f"[{target_lang}] {text}", 0.5


class HuggingFaceTranslationBackend(TranslationBackend):
    """
    HuggingFace NLLB (No Language Left Behind) translation backend.
    
    Supports 200+ languages including Korean, Vietnamese, and English.
    Uses facebook/nllb-200-distilled-600M by default (smaller, faster).
    For better quality, use facebook/nllb-200-3.3B.
    """

    # NLLB language codes
    LANG_CODES = {
        "en": "eng_Latn",
        "ko": "kor_Hang",
        "vi": "vie_Latn",
    }

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "auto",
        use_fp16: bool = True
    ):
        """
        Initialize HuggingFace translation backend.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('auto', 'cpu', 'cuda')
            use_fp16: Use half precision for faster inference
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self._model = None
        self._tokenizer = None
        self._initialized = False

    def _load_model(self):
        """Lazy load the translation model."""
        if self._initialized:
            return

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch

            logger.info(f"Loading translation model: {self.model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            dtype = torch.float16 if self.use_fp16 else torch.float32
            if self.device == "auto":
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map="auto"
                )
            else:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype
                )
                if self.device != "cpu":
                    self._model = self._model.to(self.device)

            self._model.eval()
            self._initialized = True
            logger.info("Translation model loaded successfully")

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Install: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> tuple[str, float]:
        """
        Translate text using NLLB model.
        
        Returns:
            Tuple of (translated_text, confidence_score)
        """
        if source_lang == target_lang:
            return text, 1.0

        # Get NLLB language codes
        src_code = self.LANG_CODES.get(source_lang)
        tgt_code = self.LANG_CODES.get(target_lang)

        if not src_code or not tgt_code:
            logger.warning(f"Unsupported language pair: {source_lang}->{target_lang}")
            return text, 0.0

        self._load_model()

        try:
            import torch

            # Set source language for tokenizer
            self._tokenizer.src_lang = src_code

            # Tokenize input
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # Move to model device
            if hasattr(self._model, 'device'):
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            # Generate translation
            with torch.no_grad():
                generated_tokens = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt_code),
                    max_length=512,
                    num_beams=5,
                    early_stopping=True
                )

            # Decode output
            translated = self._tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )[0]

            # Estimate confidence (based on length ratio heuristic)
            len_ratio = len(translated) / max(len(text), 1)
            confidence = min(1.0, max(0.5, 1.0 - abs(1.0 - len_ratio) * 0.5))

            return translated, confidence

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text, 0.0


class MarianMTBackend(TranslationBackend):
    """
    MarianMT translation backend using Helsinki-NLP models.
    
    Uses separate models for each language pair.
    Faster than NLLB but requires downloading multiple models.
    """

    # Model mappings for each language pair
    MODEL_MAPPINGS = {
        ("en", "ko"): "Helsinki-NLP/opus-mt-en-ko",
        ("ko", "en"): "Helsinki-NLP/opus-mt-ko-en",
        ("en", "vi"): "Helsinki-NLP/opus-mt-en-vi",
        ("vi", "en"): "Helsinki-NLP/opus-mt-vi-en",
        # For ko-vi, we'll use English as pivot
    }

    def __init__(self, device: str = "cpu", use_fp16: bool = False):
        """
        Initialize MarianMT backend.
        
        Args:
            device: Device to run on
            use_fp16: Use half precision
        """
        self.device = device
        self.use_fp16 = use_fp16
        self._models = {}
        self._tokenizers = {}

    def _get_model(self, source_lang: str, target_lang: str):
        """Get or load model for language pair."""
        pair = (source_lang, target_lang)

        if pair not in self._models:
            model_name = self.MODEL_MAPPINGS.get(pair)
            if not model_name:
                return None, None

            try:
                from transformers import MarianMTModel, MarianTokenizer
                import torch

                logger.info(f"Loading MarianMT model: {model_name}")

                tokenizer = MarianTokenizer.from_pretrained(model_name)
                dtype = torch.float16 if self.use_fp16 else torch.float32
                model = MarianMTModel.from_pretrained(
                    model_name,
                    torch_dtype=dtype
                )

                if self.device != "cpu":
                    model = model.to(self.device)

                model.eval()
                self._models[pair] = model
                self._tokenizers[pair] = tokenizer

            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                return None, None

        return self._models.get(pair), self._tokenizers.get(pair)

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> tuple[str, float]:
        """Translate using MarianMT models."""
        if source_lang == target_lang:
            return text, 1.0

        # Handle ko-vi by pivoting through English
        if (source_lang, target_lang) in [("ko", "vi"), ("vi", "ko")]:
            # First translate to English
            en_text, conf1 = self.translate(text, source_lang, "en")
            # Then translate to target
            result, conf2 = self.translate(en_text, "en", target_lang)
            return result, conf1 * conf2

        model, tokenizer = self._get_model(source_lang, target_lang)
        if model is None:
            logger.warning(f"No model for {source_lang}->{target_lang}")
            return text, 0.0

        try:
            import torch

            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=512)

            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated, 0.85

        except Exception as e:
            logger.error(f"MarianMT translation error: {e}")
            return text, 0.0


class TranslationModel:
    """
    Translation model for menu items.
    
    Supports many-to-many translation between en, ko, vi.
    Uses a pluggable backend (LLM, MT API, or mock).
    
    Metrics tracked:
    - translation_requests_total: Total translation requests
    - translation_processing_seconds: Processing time histogram
    - translation_language_pairs: Counts by source-target pair
    - translation_confidence: Confidence score distribution
    - translation_glossary_hits: Glossary term matches
    - translation_errors_total: Error counts
    """

    SUPPORTED_PAIRS = [
        ("en", "ko"), ("ko", "en"),
        ("en", "vi"), ("vi", "en"),
        ("ko", "vi"), ("vi", "ko"),
    ]

    def __init__(
        self,
        backend: Optional[TranslationBackend] = None,
        use_glossary: bool = True
    ):
        """
        Initialize translation model.
        
        Args:
            backend: Translation backend (defaults to mock)
            use_glossary: Whether to use food glossary
        """
        self.backend = backend or MockTranslationBackend()
        self.use_glossary = use_glossary
        self._glossary = FOOD_GLOSSARY

    def _apply_glossary(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Optional[tuple[str, str]]:
        """
        Check if text matches a glossary term.
        
        Returns:
            Tuple of (translated_term, explanation) or None
        """
        if not self.use_glossary:
            return None

        if source_lang not in self._glossary:
            return None

        glossary = self._glossary[source_lang]
        text_lower = text.lower().strip()

        for term, (transliteration, description) in glossary.items():
            if term.lower() == text_lower or term in text:
                metrics.increment_glossary_hits()
                if target_lang == "en":
                    return transliteration, description
                else:
                    # For non-English targets, use transliteration
                    return transliteration, description

        return None

    def translate(
        self,
        text: str,
        source_lang: SupportedLanguage,
        target_langs: Optional[list[SupportedLanguage]] = None
    ) -> TranslationResult:
        """
        Translate text to multiple target languages.
        
        Args:
            text: Source text to translate
            source_lang: Source language
            target_langs: Target languages (defaults to all)
            
        Returns:
            TranslationResult with translations and explanations
        """
        start_time = time.time()
        metrics.increment_translation_requests()

        if target_langs is None:
            target_langs = [
                SupportedLanguage.EN,
                SupportedLanguage.KO,
                SupportedLanguage.VI
            ]

        translations = {"en": None, "ko": None, "vi": None}
        explanations = {"en": None, "ko": None, "vi": None}
        total_confidence = 0.0
        num_translations = 0

        src = source_lang.value

        for target in target_langs:
            tgt = target.value

            # Skip same language
            if src == tgt:
                translations[tgt] = text
                explanations[tgt] = ""
                continue

            # Check if pair is supported
            if (src, tgt) not in self.SUPPORTED_PAIRS:
                continue

            try:
                # Try glossary first
                glossary_result = self._apply_glossary(text, src, tgt)
                if glossary_result:
                    trans_text, explanation = glossary_result
                    translations[tgt] = trans_text
                    explanations[tgt] = explanation
                    total_confidence += 0.95
                    num_translations += 1
                else:
                    # Use backend
                    trans_text, confidence = self.backend.translate(
                        text, src, tgt
                    )
                    translations[tgt] = trans_text
                    explanations[tgt] = ""
                    total_confidence += confidence
                    num_translations += 1

                metrics.record_translation_pair(src, tgt)

            except Exception as e:
                logger.error(f"Translation error {src}->{tgt}: {e}")
                metrics.record_translation_error("backend_error")

        processing_time = (time.time() - start_time) * 1000
        avg_confidence = (
            total_confidence / num_translations if num_translations > 0 else 0.0
        )

        # Record metrics
        metrics.record_translation_time(processing_time / 1000)
        metrics.record_translation_confidence(avg_confidence)

        return TranslationResult(
            source_text=text,
            source_language=source_lang,
            translated_text=LocalizedText(
                en=translations["en"],
                ko=translations["ko"],
                vi=translations["vi"]
            ),
            explanation=LocalizedText(
                en=explanations["en"],
                ko=explanations["ko"],
                vi=explanations["vi"]
            ),
            confidence=avg_confidence,
            processing_time_ms=processing_time
        )

    def calculate_bleu(
        self,
        prediction: str,
        reference: str,
        max_n: int = 4
    ) -> float:
        """
        Calculate BLEU score for translation evaluation.
        
        Simplified implementation of corpus BLEU.
        """
        from collections import Counter
        import math

        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        # Calculate n-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            pred_ngrams = Counter(
                tuple(pred_tokens[i:i+n])
                for i in range(len(pred_tokens) - n + 1)
            )
            ref_ngrams = Counter(
                tuple(ref_tokens[i:i+n])
                for i in range(len(ref_tokens) - n + 1)
            )

            matches = sum(
                min(count, ref_ngrams[ngram])
                for ngram, count in pred_ngrams.items()
            )
            total = sum(pred_ngrams.values())

            if total == 0:
                precisions.append(0.0)
            else:
                precisions.append(matches / total)

        # Geometric mean of precisions
        if 0 in precisions:
            bleu = 0.0
        else:
            log_precisions = [math.log(p) for p in precisions]
            bleu = math.exp(sum(log_precisions) / len(log_precisions))

        # Brevity penalty
        if len(pred_tokens) < len(ref_tokens):
            bp = math.exp(1 - len(ref_tokens) / len(pred_tokens))
            bleu *= bp

        metrics.record_bleu_score(bleu)
        return bleu
