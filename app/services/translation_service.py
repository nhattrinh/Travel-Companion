"""
Translation service for the menu translation backend.

This module provides translation functionality including text translation,
language detection, and batch processing capabilities. It implements the
TranslationService interface as specified in the design document.

Requirements addressed:
- 4.1: Translate text to specified target language
- 4.2: Auto-detect source language when not specified
- 4.3: Handle translation failures gracefully
- 4.4: Validate language support
"""

import logging
import asyncio
from typing import List, Optional
from abc import ABC, abstractmethod

from app.models.api_models import SupportedLanguage
from app.models.internal_models import TranslationResult, ErrorCode


class UnsupportedLanguageError(Exception):
    """Raised when target language is not supported (Requirement 4.4)"""
    pass


class TranslationFailureError(Exception):
    """Raised when translation fails (Requirement 4.3)"""
    pass


class BaseTranslationModel(ABC):
    """Abstract base class for translation models"""
    
    @abstractmethod
    async def translate(
        self, 
        text: str, 
        target_language: str, 
        source_language: Optional[str] = None
    ) -> dict:
        """Translate text using the underlying model"""
        pass
    
    @abstractmethod
    async def detect_language(self, text: str) -> str:
        """Detect the language of input text"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the translation model is healthy"""
        pass


class MockTranslationModel(BaseTranslationModel):
    """Mock translation model for development and testing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._supported_languages = {lang.value for lang in SupportedLanguage}
        self._translation_count = 0
        self._detection_count = 0
    
    async def translate(
        self, 
        text: str, 
        target_language: str, 
        source_language: Optional[str] = None
    ) -> dict:
        """Mock translation implementation"""
        self._translation_count += 1
        
        # Simulate variable processing time based on text length
        processing_time = min(0.1, len(text) / 1000)
        await asyncio.sleep(processing_time)
        
        # Simulate occasional failures for testing error handling
        if self._translation_count % 50 == 0:  # Fail every 50th translation
            raise TranslationFailureError("Mock translation service temporarily unavailable")
        
        # Mock translation logic - in real implementation, this would call actual translation API
        if target_language == source_language:
            translated_text = text  # No translation needed
            confidence = 1.0
        else:
            translated_text = f"[{target_language.upper()}] {text}"
            confidence = 0.95
        
        detected_source = source_language or await self.detect_language(text)
        
        return {
            "translated_text": translated_text,
            "source_language": detected_source,
            "confidence": confidence
        }
    
    async def detect_language(self, text: str) -> str:
        """Mock language detection"""
        self._detection_count += 1
        
        # Simulate processing time
        await asyncio.sleep(0.02)
        
        if not text or not text.strip():
            return "en"
        
        # Simple mock detection based on common words and patterns
        text_lower = text.lower()
        
        # English indicators
        if any(word in text_lower for word in ["the", "and", "is", "are", "this", "that", "with", "have"]):
            return "en"
        
        # Spanish indicators
        elif any(word in text_lower for word in ["el", "la", "es", "son", "este", "con", "de", "en"]):
            return "es"
        
        # French indicators
        elif any(word in text_lower for word in ["le", "la", "est", "sont", "ce", "avec", "de", "dans"]):
            return "fr"
        
        # German indicators
        elif any(word in text_lower for word in ["der", "die", "das", "ist", "sind", "mit", "von", "in"]):
            return "de"
        
        # Italian indicators
        elif any(word in text_lower for word in ["il", "la", "è", "sono", "questo", "con", "di", "in"]):
            return "it"
        
        # Portuguese indicators
        elif any(word in text_lower for word in ["o", "a", "é", "são", "este", "com", "de", "em"]):
            return "pt"
        
        # Chinese indicators (simplified check for Chinese characters)
        elif any('\u4e00' <= char <= '\u9fff' for char in text):
            return "zh"
        
        # Japanese indicators (hiragana, katakana, kanji)
        elif any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' or '\u4e00' <= char <= '\u9fff' for char in text):
            return "ja"
        
        # Korean indicators (Hangul)
        elif any('\uac00' <= char <= '\ud7af' for char in text):
            return "ko"
        
        else:
            return "en"  # Default to English
    
    async def health_check(self) -> bool:
        """Mock health check"""
        # Simulate occasional health check failures
        import random
        if random.random() < 0.05:  # 5% chance of failure
            return False
        return True
    
    def get_stats(self) -> dict:
        """Get mock model statistics"""
        return {
            "translations_performed": self._translation_count,
            "language_detections_performed": self._detection_count,
            "supported_languages": list(self._supported_languages)
        }


class TranslationService:
    """
    Translation service implementing core translation functionality.
    
    This service provides text translation, language detection, and batch processing
    capabilities as specified in Requirements 4.1, 4.2, 4.3, and 4.4.
    """
    
    def __init__(self, translation_model: Optional[BaseTranslationModel] = None):
        """
        Initialize the translation service.
        
        Args:
            translation_model: The underlying translation model to use.
                             If None, uses MockTranslationModel for development.
        """
        self.logger = logging.getLogger(__name__)
        self.translation_model = translation_model or MockTranslationModel()
        self._supported_languages = {lang.value for lang in SupportedLanguage}
        
        self.logger.info("TranslationService initialized")
    
    async def translate_text(
        self, 
        text: str, 
        target_language: SupportedLanguage, 
        source_language: Optional[SupportedLanguage] = None
    ) -> TranslationResult:
        """
        Translate text to target language (Requirement 4.1).
        
        Args:
            text: The text to translate
            target_language: The target language for translation
            source_language: The source language (optional, will auto-detect if None)
        
        Returns:
            TranslationResult containing translated text and metadata
        
        Raises:
            UnsupportedLanguageError: If target language is not supported
            TranslationFailureError: If translation fails
        """
        try:
            # Validate target language support (Requirement 4.4)
            if not await self.validate_language_support(target_language.value):
                raise UnsupportedLanguageError(f"Target language '{target_language.value}' is not supported")
            
            # Auto-detect source language if not provided (Requirement 4.2)
            detected_source = None
            if source_language is None:
                detected_source = await self.detect_language(text)
                self.logger.debug(f"Auto-detected source language: {detected_source}")
            else:
                # Validate source language support
                if not await self.validate_language_support(source_language.value):
                    self.logger.warning(f"Source language '{source_language.value}' not supported, auto-detecting")
                    detected_source = await self.detect_language(text)
                else:
                    detected_source = source_language.value
            
            # Perform translation
            self.logger.debug(f"Translating text from '{detected_source}' to '{target_language.value}'")
            
            translation_result = await self.translation_model.translate(
                text=text,
                target_language=target_language.value,
                source_language=detected_source
            )
            
            # Create structured result
            result = TranslationResult(
                translated_text=translation_result["translated_text"],
                source_language=translation_result["source_language"],
                confidence=translation_result["confidence"]
            )
            
            self.logger.debug(f"Translation completed successfully with confidence {result.confidence}")
            return result
            
        except UnsupportedLanguageError:
            # Re-raise language errors as-is
            raise
        except Exception as e:
            # Handle translation failures gracefully (Requirement 4.3)
            self.logger.error(f"Translation failed for text '{text[:50]}...': {str(e)}")
            return await self.handle_translation_failure(text, e)
    
    async def detect_language(self, text: str) -> SupportedLanguage:
        """
        Detect the language of input text (Requirement 4.2).
        
        Args:
            text: The text to analyze for language detection
        
        Returns:
            SupportedLanguage enum value for the detected language
        
        Raises:
            TranslationFailureError: If language detection fails
        """
        try:
            if not text or not text.strip():
                self.logger.warning("Empty text provided for language detection, defaulting to English")
                return SupportedLanguage.ENGLISH
            
            detected_lang = await self.translation_model.detect_language(text)
            
            # Validate that detected language is supported
            if detected_lang not in self._supported_languages:
                self.logger.warning(f"Detected language '{detected_lang}' not supported, defaulting to English")
                return SupportedLanguage.ENGLISH
            
            # Convert to SupportedLanguage enum
            for lang in SupportedLanguage:
                if lang.value == detected_lang:
                    return lang
            
            # Fallback to English if conversion fails
            return SupportedLanguage.ENGLISH
            
        except Exception as e:
            self.logger.error(f"Language detection failed for text '{text[:50]}...': {str(e)}")
            # Default to English on detection failure
            return SupportedLanguage.ENGLISH
    
    async def validate_language_support(self, language: str) -> bool:
        """
        Check if language is supported (Requirement 4.4).
        
        Args:
            language: Language code to validate
        
        Returns:
            True if language is supported, False otherwise
        """
        return language in self._supported_languages
    
    async def handle_translation_failure(self, text: str, error: Exception) -> TranslationResult:
        """
        Handle translation failures gracefully (Requirement 4.3).
        
        Args:
            text: The original text that failed to translate
            error: The exception that caused the failure
        
        Returns:
            TranslationResult with original text and error indicator
        """
        error_message = f"Translation failed: {str(error)}"
        self.logger.error(error_message)
        
        # Return original text with error indicator
        return TranslationResult(
            translated_text=text,  # Return original text
            source_language="unknown",
            confidence=0.0,
            error_indicator=error_message
        )
    
    async def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List of supported language codes
        """
        return list(self._supported_languages)
    
    async def batch_translate(
        self, 
        texts: List[str], 
        target_language: SupportedLanguage, 
        source_language: Optional[SupportedLanguage] = None
    ) -> List[TranslationResult]:
        """
        Translate multiple texts in batch (Requirements 4.1, 4.2, 4.3).
        
        This method processes multiple texts efficiently, handling failures gracefully
        and providing individual results for each text.
        
        Args:
            texts: List of texts to translate
            target_language: The target language for all translations
            source_language: The source language (optional, will auto-detect if None)
        
        Returns:
            List of TranslationResult objects, one for each input text
        """
        if not texts:
            self.logger.warning("Empty text list provided for batch translation")
            return []
        
        self.logger.info(f"Starting batch translation of {len(texts)} texts to {target_language.value}")
        
        # Validate target language once for the entire batch (Requirement 4.4)
        if not await self.validate_language_support(target_language.value):
            error_msg = f"Target language '{target_language.value}' is not supported"
            self.logger.error(error_msg)
            # Return error results for all texts
            return [
                TranslationResult(
                    translated_text=text,
                    source_language="unknown",
                    confidence=0.0,
                    error_indicator=error_msg
                )
                for text in texts
            ]
        
        # Process texts concurrently for better performance
        translation_tasks = []
        for i, text in enumerate(texts):
            task = self._translate_single_text_with_error_handling(
                text, target_language, source_language, i
            )
            translation_tasks.append(task)
        
        # Execute all translations concurrently
        results = await asyncio.gather(*translation_tasks, return_exceptions=True)
        
        # Process results and handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exceptions that weren't caught in individual tasks
                self.logger.error(f"Batch translation task {i} failed with exception: {str(result)}")
                error_result = await self.handle_translation_failure(texts[i], result)
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        successful_count = sum(1 for result in final_results if result.error_indicator is None)
        self.logger.info(f"Batch translation completed: {successful_count}/{len(texts)} successful")
        
        return final_results
    
    async def _translate_single_text_with_error_handling(
        self,
        text: str,
        target_language: SupportedLanguage,
        source_language: Optional[SupportedLanguage],
        index: int
    ) -> TranslationResult:
        """
        Helper method to translate a single text with comprehensive error handling.
        
        Args:
            text: Text to translate
            target_language: Target language
            source_language: Source language (optional)
            index: Index of the text in the batch (for logging)
        
        Returns:
            TranslationResult for the text
        """
        try:
            return await self.translate_text(text, target_language, source_language)
        except Exception as e:
            self.logger.error(f"Translation failed for batch item {index}: {str(e)}")
            return await self.handle_translation_failure(text, e)
    
    async def batch_detect_language(self, texts: List[str]) -> List[SupportedLanguage]:
        """
        Detect languages for multiple texts in batch (Requirement 4.2).
        
        Args:
            texts: List of texts to analyze for language detection
        
        Returns:
            List of SupportedLanguage enum values for detected languages
        """
        if not texts:
            self.logger.warning("Empty text list provided for batch language detection")
            return []
        
        self.logger.info(f"Starting batch language detection for {len(texts)} texts")
        
        # Process language detection concurrently
        detection_tasks = [
            self._detect_language_with_error_handling(text, i) 
            for i, text in enumerate(texts)
        ]
        
        results = await asyncio.gather(*detection_tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Language detection failed for batch item {i}: {str(result)}")
                final_results.append(SupportedLanguage.ENGLISH)  # Default fallback
            else:
                final_results.append(result)
        
        successful_count = sum(1 for result in final_results if result != SupportedLanguage.ENGLISH or 
                             any(word in texts[i].lower() for word in ["the", "and", "is"]))
        self.logger.info(f"Batch language detection completed: {successful_count}/{len(texts)} detected")
        
        return final_results
    
    async def _detect_language_with_error_handling(
        self, 
        text: str, 
        index: int
    ) -> SupportedLanguage:
        """
        Helper method to detect language for a single text with error handling.
        
        Args:
            text: Text to analyze
            index: Index of the text in the batch (for logging)
        
        Returns:
            SupportedLanguage enum value
        """
        try:
            return await self.detect_language(text)
        except Exception as e:
            self.logger.error(f"Language detection failed for batch item {index}: {str(e)}")
            return SupportedLanguage.ENGLISH  # Default fallback
    
    async def translate_with_auto_detection(
        self,
        text: str,
        target_language: SupportedLanguage
    ) -> TranslationResult:
        """
        Convenience method that always auto-detects source language (Requirement 4.2).
        
        Args:
            text: Text to translate
            target_language: Target language for translation
        
        Returns:
            TranslationResult with auto-detected source language
        """
        return await self.translate_text(text, target_language, source_language=None)
    
    async def get_translation_statistics(self) -> dict:
        """
        Get statistics about the translation service performance.
        
        Returns:
            Dictionary containing service statistics
        """
        try:
            is_healthy = await self.health_check()
            supported_langs = await self.get_supported_languages()
            
            return {
                "service_healthy": is_healthy,
                "supported_languages_count": len(supported_langs),
                "supported_languages": supported_langs,
                "model_type": type(self.translation_model).__name__
            }
        except Exception as e:
            self.logger.error(f"Failed to get translation statistics: {str(e)}")
            return {
                "service_healthy": False,
                "error": str(e)
            }
    
    async def health_check(self) -> bool:
        """
        Check if the translation service is healthy and ready.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            return await self.translation_model.health_check()
        except Exception as e:
            self.logger.error(f"Translation service health check failed: {str(e)}")
            return False