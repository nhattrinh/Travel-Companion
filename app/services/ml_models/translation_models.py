"""
Translation Models with modern PyTorch optimizations.

Provides state-of-the-art neural machine translation:
- NLLB-200: Meta's No Language Left Behind for 200+ languages
- mBART: Facebook's multilingual BART for translation

All models support:
- torch.compile for JIT optimization
- Mixed precision inference (FP16/BF16)
- Static caching for faster generation
- Batch translation with dynamic batching
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union
from enum import Enum

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        NllbTokenizer,
        pipeline,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from .base import BaseMLModel, ModelConfig, DeviceType, PrecisionMode

logger = logging.getLogger(__name__)


class TranslationLanguage(str, Enum):
    """
    Supported language codes for translation.

    Uses NLLB language codes (Flores-200) for consistency.
    """
    # Primary supported languages
    ENGLISH = "eng_Latn"
    JAPANESE = "jpn_Jpan"
    SPANISH = "spa_Latn"
    VIETNAMESE = "vie_Latn"

    # Additional languages
    FRENCH = "fra_Latn"
    GERMAN = "deu_Latn"
    ITALIAN = "ita_Latn"
    PORTUGUESE = "por_Latn"
    CHINESE_SIMPLIFIED = "zho_Hans"
    CHINESE_TRADITIONAL = "zho_Hant"
    KOREAN = "kor_Hang"
    THAI = "tha_Thai"
    ARABIC = "arb_Arab"
    HINDI = "hin_Deva"
    RUSSIAN = "rus_Cyrl"

    @classmethod
    def from_iso(cls, iso_code: str) -> Optional["TranslationLanguage"]:
        """
        Convert ISO 639-1 code to TranslationLanguage.

        Args:
            iso_code: Two-letter ISO language code

        Returns:
            TranslationLanguage or None if not found
        """
        iso_map = {
            "en": cls.ENGLISH,
            "ja": cls.JAPANESE,
            "es": cls.SPANISH,
            "vi": cls.VIETNAMESE,
            "fr": cls.FRENCH,
            "de": cls.GERMAN,
            "it": cls.ITALIAN,
            "pt": cls.PORTUGUESE,
            "zh": cls.CHINESE_SIMPLIFIED,
            "ko": cls.KOREAN,
            "th": cls.THAI,
            "ar": cls.ARABIC,
            "hi": cls.HINDI,
            "ru": cls.RUSSIAN,
        }
        return iso_map.get(iso_code.lower())


@dataclass
class TranslationResult:
    """
    Result of a translation operation.

    Attributes:
        translated_text: The translated text
        source_language: Detected or specified source language
        target_language: Target language
        confidence: Translation confidence score
        alternatives: Alternative translations if available
    """
    translated_text: str
    source_language: str
    target_language: str
    confidence: float = 0.0
    alternatives: List[str] = field(default_factory=list)


@dataclass
class TranslationModelConfig(ModelConfig):
    """
    Configuration for translation models.

    Additional attributes:
        default_source_lang: Default source language for translation
        default_target_lang: Default target language for translation
        max_length: Maximum token length for input/output
        num_beams: Number of beams for beam search
        num_return_sequences: Number of alternative translations
        early_stopping: Whether to use early stopping in generation
        use_static_cache: Use PyTorch 2.x static cache for generation
    """
    default_source_lang: TranslationLanguage = TranslationLanguage.ENGLISH
    default_target_lang: TranslationLanguage = TranslationLanguage.JAPANESE
    max_length: int = 256
    num_beams: int = 4
    num_return_sequences: int = 1
    early_stopping: bool = True
    use_static_cache: bool = True


class NLLBTranslationModel(BaseMLModel[str]):
    """
    NLLB-200 translation model for 200+ languages.

    Meta's No Language Left Behind model provides:
    - Support for 200+ languages including low-resource ones
    - High-quality translations for the Travel Companion use cases
    - Distilled versions for faster inference

    Recommended models:
    - "facebook/nllb-200-distilled-600M": Fast, good quality
    - "facebook/nllb-200-distilled-1.3B": Better quality, slower
    - "facebook/nllb-200-3.3B": Best quality, requires more GPU

    Optimizations:
    - torch.compile with reduce-overhead mode
    - Mixed precision with FP16/BF16
    - Static cache for decoder efficiency
    - Batched translation support
    """

    def __init__(self, config: TranslationModelConfig):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers required for NLLB. "
                "Install with: pip install transformers"
            )
        if not HAS_TORCH:
            raise ImportError(
                "torch required for NLLB. "
                "Install with: pip install torch"
            )

        super().__init__(config)
        self.translation_config = config
        self._tokenizer = None
        self._generation_config = None

    async def _load_model(self) -> None:
        """Load NLLB model and tokenizer."""
        loop = asyncio.get_event_loop()

        # Load tokenizer
        self._tokenizer = await loop.run_in_executor(
            None,
            lambda: AutoTokenizer.from_pretrained(
                self.config.model_name,
                src_lang=self.translation_config.default_source_lang.value,
            )
        )

        # Load model with optimizations
        self._model = await loop.run_in_executor(
            None,
            lambda: AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                torch_dtype=self.dtype,
                device_map="auto" if self.config.device == DeviceType.CUDA else None,  # noqa: E501
            )
        )

        # Move to device if not using device_map
        if self.config.device != DeviceType.CUDA:
            self._model = self._model.to(self.device)

        self._model.eval()

        # Set up generation config
        self._generation_config = {
            "max_length": self.translation_config.max_length,
            "num_beams": self.translation_config.num_beams,
            "num_return_sequences": self.translation_config.num_return_sequences,  # noqa: E501
            "early_stopping": self.translation_config.early_stopping,
            "use_cache": True,
        }

        # Use static cache for PyTorch 2.x optimization
        if self.translation_config.use_static_cache:
            self._generation_config["cache_implementation"] = "static"

        self.logger.info(
            f"NLLB loaded: {self.config.model_name} "
            f"on {self.device} with {self.config.precision.value}"
        )

    async def _run_warmup_iteration(self) -> None:
        """Run warmup with simple translation."""
        await self._run_inference("Hello world")

    async def _run_inference(
        self,
        inputs: Union[str, List[str]],
        source_lang: Optional[TranslationLanguage] = None,
        target_lang: Optional[TranslationLanguage] = None,
    ) -> List[TranslationResult]:
        """
        Translate text(s) from source to target language.

        Args:
            inputs: Text or list of texts to translate
            source_lang: Source language (uses config default if None)
            target_lang: Target language (uses config default if None)

        Returns:
            List of TranslationResult objects
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        source = source_lang or self.translation_config.default_source_lang
        target = target_lang or self.translation_config.default_target_lang

        loop = asyncio.get_event_loop()

        # Set source language for tokenizer
        self._tokenizer.src_lang = source.value

        # Tokenize inputs
        encoded = await loop.run_in_executor(
            None,
            lambda: self._tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.translation_config.max_length,
            )
        )

        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Get target language token ID
        forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(
            target.value
        )

        # Generate translations
        generated = await loop.run_in_executor(
            None,
            lambda: self._model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                **self._generation_config
            )
        )

        # Decode outputs
        translations = await loop.run_in_executor(
            None,
            lambda: self._tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )
        )

        # Build results
        results = []
        num_sequences = self.translation_config.num_return_sequences

        for i, text in enumerate(inputs):
            idx_start = i * num_sequences
            idx_end = idx_start + num_sequences

            main_translation = translations[idx_start]
            alternatives = translations[idx_start + 1:idx_end] if num_sequences > 1 else []  # noqa: E501

            results.append(TranslationResult(
                translated_text=main_translation,
                source_language=source.value,
                target_language=target.value,
                confidence=0.95,  # NLLB generally high confidence
                alternatives=alternatives,
            ))

        return results

    async def translate(
        self,
        text: str,
        source_lang: Optional[TranslationLanguage] = None,
        target_lang: Optional[TranslationLanguage] = None,
    ) -> TranslationResult:
        """
        Translate a single text string.

        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language

        Returns:
            TranslationResult
        """
        results = await self.inference(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        return results[0]

    async def translate_batch(
        self,
        texts: List[str],
        source_lang: Optional[TranslationLanguage] = None,
        target_lang: Optional[TranslationLanguage] = None,
    ) -> List[TranslationResult]:
        """
        Translate multiple texts in batch.

        Args:
            texts: List of texts to translate
            source_lang: Source language
            target_lang: Target language

        Returns:
            List of TranslationResult
        """
        # Split into batches based on config
        batch_size = self.config.max_batch_size
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results = await self.inference(
                batch,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            all_results.extend(results)

        return all_results


class MBARTTranslationModel(BaseMLModel[str]):
    """
    mBART-50 translation model for 50+ languages.

    Facebook's multilingual BART provides:
    - Many-to-many translation for 50+ languages
    - Good balance of quality and speed
    - Strong performance on common language pairs

    Recommended models:
    - "facebook/mbart-large-50-many-to-many-mmt": Full model
    - "facebook/mbart-large-50-one-to-many-mmt": English to many

    Optimizations:
    - torch.compile with inductor backend
    - SDPA (Scaled Dot Product Attention) when available
    - Mixed precision inference
    """

    # mBART language code mapping
    LANG_CODE_MAP = {
        TranslationLanguage.ENGLISH: "en_XX",
        TranslationLanguage.JAPANESE: "ja_XX",
        TranslationLanguage.SPANISH: "es_XX",
        TranslationLanguage.VIETNAMESE: "vi_VN",
        TranslationLanguage.FRENCH: "fr_XX",
        TranslationLanguage.GERMAN: "de_DE",
        TranslationLanguage.ITALIAN: "it_IT",
        TranslationLanguage.PORTUGUESE: "pt_XX",
        TranslationLanguage.CHINESE_SIMPLIFIED: "zh_CN",
        TranslationLanguage.KOREAN: "ko_KR",
        TranslationLanguage.THAI: "th_TH",
        TranslationLanguage.ARABIC: "ar_AR",
        TranslationLanguage.HINDI: "hi_IN",
        TranslationLanguage.RUSSIAN: "ru_RU",
    }

    def __init__(self, config: TranslationModelConfig):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers required for mBART. "
                "Install with: pip install transformers"
            )
        if not HAS_TORCH:
            raise ImportError(
                "torch required for mBART. "
                "Install with: pip install torch"
            )

        super().__init__(config)
        self.translation_config = config
        self._tokenizer = None
        self._generation_config = None

    def _get_mbart_code(self, lang: TranslationLanguage) -> str:
        """Convert TranslationLanguage to mBART language code."""
        return self.LANG_CODE_MAP.get(lang, "en_XX")

    async def _load_model(self) -> None:
        """Load mBART model and tokenizer."""
        loop = asyncio.get_event_loop()

        # Load tokenizer
        self._tokenizer = await loop.run_in_executor(
            None,
            lambda: AutoTokenizer.from_pretrained(self.config.model_name)
        )

        # Load model with SDPA attention (PyTorch 2.x optimization)
        self._model = await loop.run_in_executor(
            None,
            lambda: AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                torch_dtype=self.dtype,
                attn_implementation="sdpa",  # Scaled dot product attention
                device_map="auto" if self.config.device == DeviceType.CUDA else None,  # noqa: E501
            )
        )

        if self.config.device != DeviceType.CUDA:
            self._model = self._model.to(self.device)

        self._model.eval()

        # Generation config
        self._generation_config = {
            "max_length": self.translation_config.max_length,
            "num_beams": self.translation_config.num_beams,
            "num_return_sequences": self.translation_config.num_return_sequences,  # noqa: E501
            "early_stopping": self.translation_config.early_stopping,
            "use_cache": True,
        }

        if self.translation_config.use_static_cache:
            self._generation_config["cache_implementation"] = "static"

        self.logger.info(
            f"mBART loaded: {self.config.model_name} "
            f"on {self.device} with SDPA attention"
        )

    async def _run_warmup_iteration(self) -> None:
        """Run warmup translation."""
        await self._run_inference("Hello world")

    async def _run_inference(
        self,
        inputs: Union[str, List[str]],
        source_lang: Optional[TranslationLanguage] = None,
        target_lang: Optional[TranslationLanguage] = None,
    ) -> List[TranslationResult]:
        """
        Translate text(s) using mBART.

        Args:
            inputs: Text or list of texts
            source_lang: Source language
            target_lang: Target language

        Returns:
            List of TranslationResult
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        source = source_lang or self.translation_config.default_source_lang
        target = target_lang or self.translation_config.default_target_lang

        src_code = self._get_mbart_code(source)
        tgt_code = self._get_mbart_code(target)

        loop = asyncio.get_event_loop()

        # Set source language
        self._tokenizer.src_lang = src_code

        # Tokenize
        encoded = await loop.run_in_executor(
            None,
            lambda: self._tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.translation_config.max_length,
            )
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Get target language token ID
        forced_bos_token_id = self._tokenizer.lang_code_to_id.get(
            tgt_code,
            self._tokenizer.convert_tokens_to_ids(tgt_code)
        )

        # Generate
        generated = await loop.run_in_executor(
            None,
            lambda: self._model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                **self._generation_config
            )
        )

        # Decode
        translations = await loop.run_in_executor(
            None,
            lambda: self._tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )
        )

        # Build results
        results = []
        num_seq = self.translation_config.num_return_sequences

        for i, text in enumerate(inputs):
            idx_start = i * num_seq
            idx_end = idx_start + num_seq

            main = translations[idx_start]
            alts = translations[idx_start + 1:idx_end] if num_seq > 1 else []

            results.append(TranslationResult(
                translated_text=main,
                source_language=source.value,
                target_language=target.value,
                confidence=0.93,
                alternatives=alts,
            ))

        return results

    async def translate(
        self,
        text: str,
        source_lang: Optional[TranslationLanguage] = None,
        target_lang: Optional[TranslationLanguage] = None,
    ) -> TranslationResult:
        """Translate single text."""
        results = await self.inference(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        return results[0]

    async def translate_batch(
        self,
        texts: List[str],
        source_lang: Optional[TranslationLanguage] = None,
        target_lang: Optional[TranslationLanguage] = None,
    ) -> List[TranslationResult]:
        """Translate multiple texts in batches."""
        batch_size = self.config.max_batch_size
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results = await self.inference(
                batch,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            all_results.extend(results)

        return all_results


def create_translation_model(
    model_type: str = "nllb",
    model_size: str = "600M",
    device: DeviceType = None,
    precision: PrecisionMode = PrecisionMode.FP16,
) -> BaseMLModel:
    """
    Factory function to create translation models.

    Args:
        model_type: "nllb" or "mbart"
        model_size: Model size variant (600M, 1.3B, 3.3B for NLLB)
        device: Target device
        precision: Inference precision

    Returns:
        Configured translation model instance
    """
    if device is None:
        device = DeviceType.CUDA if torch.cuda.is_available() else DeviceType.CPU  # noqa: E501

    if model_type == "nllb":
        model_name = f"facebook/nllb-200-distilled-{model_size}"
        config = TranslationModelConfig(
            model_name=model_name,
            device=device,
            precision=precision,
        )
        return NLLBTranslationModel(config)

    elif model_type == "mbart":
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        config = TranslationModelConfig(
            model_name=model_name,
            device=device,
            precision=precision,
        )
        return MBARTTranslationModel(config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
