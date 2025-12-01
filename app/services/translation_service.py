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
    """
    Improved translation model for English, Korean, and Vietnamese.
    Uses dictionary-based translations for common phrases with fallback.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._supported_languages = {"en", "ko", "vi"}
        self._translation_count = 0
        self._detection_count = 0
        
        # Common phrase translations (English as base)
        self._translations = {
            # Greetings
            "hello": {"ko": "안녕하세요", "vi": "Xin chào", "en": "Hello"},
            "hi": {"ko": "안녕", "vi": "Chào", "en": "Hi"},
            "goodbye": {"ko": "안녕히 가세요", "vi": "Tạm biệt", "en": "Goodbye"},
            "bye": {"ko": "안녕", "vi": "Tạm biệt", "en": "Bye"},
            "good morning": {"ko": "좋은 아침이에요", "vi": "Chào buổi sáng", "en": "Good morning"},
            "good evening": {"ko": "좋은 저녁이에요", "vi": "Chào buổi tối", "en": "Good evening"},
            "good night": {"ko": "안녕히 주무세요", "vi": "Chúc ngủ ngon", "en": "Good night"},
            
            # Politeness
            "thank you": {"ko": "감사합니다", "vi": "Cảm ơn", "en": "Thank you"},
            "thanks": {"ko": "고마워요", "vi": "Cảm ơn", "en": "Thanks"},
            "please": {"ko": "제발", "vi": "Làm ơn", "en": "Please"},
            "sorry": {"ko": "죄송합니다", "vi": "Xin lỗi", "en": "Sorry"},
            "excuse me": {"ko": "실례합니다", "vi": "Xin lỗi", "en": "Excuse me"},
            "you're welcome": {"ko": "천만에요", "vi": "Không có gì", "en": "You're welcome"},
            
            # Questions
            "how are you": {"ko": "어떻게 지내세요?", "vi": "Bạn khỏe không?", "en": "How are you?"},
            "what is your name": {"ko": "이름이 뭐예요?", "vi": "Bạn tên gì?", "en": "What is your name?"},
            "where is": {"ko": "어디에 있어요?", "vi": "Ở đâu?", "en": "Where is?"},
            "how much": {"ko": "얼마예요?", "vi": "Bao nhiêu tiền?", "en": "How much?"},
            "what time": {"ko": "몇 시예요?", "vi": "Mấy giờ?", "en": "What time?"},
            
            # Restaurant
            "menu": {"ko": "메뉴", "vi": "Thực đơn", "en": "Menu"},
            "water": {"ko": "물", "vi": "Nước", "en": "Water"},
            "food": {"ko": "음식", "vi": "Thức ăn", "en": "Food"},
            "delicious": {"ko": "맛있어요", "vi": "Ngon", "en": "Delicious"},
            "bill please": {"ko": "계산서 주세요", "vi": "Tính tiền", "en": "Bill please"},
            "check please": {"ko": "계산해 주세요", "vi": "Tính tiền đi", "en": "Check please"},
            "i want to order": {"ko": "주문하고 싶어요", "vi": "Tôi muốn gọi món", "en": "I want to order"},
            "this one please": {"ko": "이거 주세요", "vi": "Cho tôi cái này", "en": "This one please"},
            "is this spicy": {"ko": "이거 매워요?", "vi": "Cái này cay không?", "en": "Is this spicy?"},
            "not spicy please": {"ko": "안 맵게 해주세요", "vi": "Không cay nhé", "en": "Not spicy please"},
            
            # Directions
            "left": {"ko": "왼쪽", "vi": "Bên trái", "en": "Left"},
            "right": {"ko": "오른쪽", "vi": "Bên phải", "en": "Right"},
            "straight": {"ko": "직진", "vi": "Thẳng", "en": "Straight"},
            "turn left": {"ko": "왼쪽으로 돌아가세요", "vi": "Rẽ trái", "en": "Turn left"},
            "turn right": {"ko": "오른쪽으로 돌아가세요", "vi": "Rẽ phải", "en": "Turn right"},
            "go straight": {"ko": "직진하세요", "vi": "Đi thẳng", "en": "Go straight"},
            
            # Transportation
            "taxi": {"ko": "택시", "vi": "Taxi", "en": "Taxi"},
            "bus": {"ko": "버스", "vi": "Xe buýt", "en": "Bus"},
            "train": {"ko": "기차", "vi": "Tàu hỏa", "en": "Train"},
            "subway": {"ko": "지하철", "vi": "Tàu điện ngầm", "en": "Subway"},
            "airport": {"ko": "공항", "vi": "Sân bay", "en": "Airport"},
            "station": {"ko": "역", "vi": "Ga", "en": "Station"},
            
            # Hotel
            "hotel": {"ko": "호텔", "vi": "Khách sạn", "en": "Hotel"},
            "room": {"ko": "방", "vi": "Phòng", "en": "Room"},
            "reservation": {"ko": "예약", "vi": "Đặt phòng", "en": "Reservation"},
            "check in": {"ko": "체크인", "vi": "Nhận phòng", "en": "Check in"},
            "check out": {"ko": "체크아웃", "vi": "Trả phòng", "en": "Check out"},
            
            # Shopping
            "how much is this": {"ko": "이거 얼마예요?", "vi": "Cái này bao nhiêu?", "en": "How much is this?"},
            "too expensive": {"ko": "너무 비싸요", "vi": "Đắt quá", "en": "Too expensive"},
            "discount": {"ko": "할인", "vi": "Giảm giá", "en": "Discount"},
            "i'll take it": {"ko": "이거 살게요", "vi": "Tôi mua cái này", "en": "I'll take it"},
            
            # Emergency
            "help": {"ko": "도와주세요", "vi": "Cứu tôi", "en": "Help"},
            "emergency": {"ko": "응급", "vi": "Khẩn cấp", "en": "Emergency"},
            "hospital": {"ko": "병원", "vi": "Bệnh viện", "en": "Hospital"},
            "police": {"ko": "경찰", "vi": "Cảnh sát", "en": "Police"},
            "i need help": {"ko": "도움이 필요해요", "vi": "Tôi cần giúp đỡ", "en": "I need help"},
            
            # Common phrases
            "yes": {"ko": "네", "vi": "Vâng", "en": "Yes"},
            "no": {"ko": "아니요", "vi": "Không", "en": "No"},
            "i don't understand": {"ko": "이해가 안 돼요", "vi": "Tôi không hiểu", "en": "I don't understand"},
            "i don't speak korean": {"ko": "한국어를 못해요", "vi": "Tôi không nói được tiếng Hàn", "en": "I don't speak Korean"},
            "i don't speak vietnamese": {"ko": "베트남어를 못해요", "vi": "Tôi không nói được tiếng Việt", "en": "I don't speak Vietnamese"},
            "do you speak english": {"ko": "영어 하세요?", "vi": "Bạn nói tiếng Anh không?", "en": "Do you speak English?"},
            "i am lost": {"ko": "길을 잃었어요", "vi": "Tôi bị lạc", "en": "I am lost"},
            "where is the bathroom": {"ko": "화장실이 어디예요?", "vi": "Nhà vệ sinh ở đâu?", "en": "Where is the bathroom?"},
            "where is the restroom": {"ko": "화장실이 어디예요?", "vi": "Nhà vệ sinh ở đâu?", "en": "Where is the restroom?"},
            "where is the toilet": {"ko": "화장실이 어디예요?", "vi": "Nhà vệ sinh ở đâu?", "en": "Where is the toilet?"},
            "bathroom": {"ko": "화장실", "vi": "Nhà vệ sinh", "en": "Bathroom"},
            "toilet": {"ko": "화장실", "vi": "Nhà vệ sinh", "en": "Toilet"},
        }
        
        # Additional Korean phrase mappings (variations)
        self._korean_phrases = {
            "화장실 어디예요": "where is the bathroom",
            "화장실이 어디예요": "where is the bathroom",
            "이거 뭐예요": "what is this",
            "뭐예요": "what is this",
            "감사합니다": "thank you",
            "고맙습니다": "thank you",
            "고마워요": "thanks",
            "안녕하세요": "hello",
            "안녕": "hi",
            "네": "yes",
            "아니요": "no",
            "아니": "no",
            "죄송합니다": "sorry",
            "미안해요": "sorry",
            "도와주세요": "help",
            "얼마예요": "how much",
            "이거 얼마예요": "how much is this",
            "물 주세요": "water please",
            "계산해 주세요": "check please",
            "맛있어요": "delicious",
            "주문할게요": "i want to order",
            "메뉴 주세요": "menu please",
        }
        
        # Additional Vietnamese phrase mappings (variations)
        self._vietnamese_phrases = {
            "xin chào": "hello",
            "chào": "hi",
            "cảm ơn": "thank you",
            "cám ơn": "thank you",
            "xin lỗi": "sorry",
            "vâng": "yes",
            "không": "no",
            "bao nhiêu tiền": "how much",
            "cái này bao nhiêu": "how much is this",
            "giúp tôi": "help",
            "tôi cần giúp đỡ": "i need help",
            "nhà vệ sinh ở đâu": "where is the bathroom",
            "ngon": "delicious",
            "ngon quá": "delicious",
            "tính tiền": "check please",
            "cho tôi": "give me",
            "nước": "water",
        }
        
        # Reverse mappings for Korean and Vietnamese to English
        self._ko_to_en = {}
        self._vi_to_en = {}
        for eng_phrase, translations in self._translations.items():
            if "ko" in translations:
                self._ko_to_en[translations["ko"].lower()] = eng_phrase
            if "vi" in translations:
                self._vi_to_en[translations["vi"].lower()] = eng_phrase
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        return text.lower().strip().rstrip('?!.,')
    
    def _find_translation(
        self, text: str, source_lang: str, target_lang: str
    ) -> Optional[str]:
        """Find translation in dictionary."""
        normalized = self._normalize_text(text)
        
        if source_lang == "en":
            # English to Korean/Vietnamese
            if normalized in self._translations:
                return self._translations[normalized].get(target_lang)
        elif source_lang == "ko":
            # First check additional Korean phrases
            if normalized in self._korean_phrases:
                eng_phrase = self._korean_phrases[normalized]
                if target_lang == "en":
                    return self._translations.get(eng_phrase, {}).get("en", eng_phrase.title())
                else:
                    return self._translations.get(eng_phrase, {}).get(target_lang)
            # Then check reverse mapping
            if normalized in self._ko_to_en:
                eng_phrase = self._ko_to_en[normalized]
                if target_lang == "en":
                    return self._translations[eng_phrase]["en"]
                else:
                    return self._translations[eng_phrase].get(target_lang)
        elif source_lang == "vi":
            # First check additional Vietnamese phrases
            if normalized in self._vietnamese_phrases:
                eng_phrase = self._vietnamese_phrases[normalized]
                if target_lang == "en":
                    return self._translations.get(eng_phrase, {}).get("en", eng_phrase.title())
                else:
                    return self._translations.get(eng_phrase, {}).get(target_lang)
            # Then check reverse mapping
            if normalized in self._vi_to_en:
                eng_phrase = self._vi_to_en[normalized]
                if target_lang == "en":
                    return self._translations[eng_phrase]["en"]
                else:
                    return self._translations[eng_phrase].get(target_lang)
        
        return None
    
    async def translate(
        self, 
        text: str, 
        target_language: str, 
        source_language: Optional[str] = None
    ) -> dict:
        """Translate text with improved accuracy for EN/KO/VI."""
        self._translation_count += 1
        
        # Simulate small processing time
        await asyncio.sleep(0.01)
        
        # Detect source language if not provided
        detected_source = source_language or await self.detect_language(text)
        
        # Same language - no translation needed
        if target_language == detected_source:
            return {
                "translated_text": text,
                "source_language": detected_source,
                "confidence": 1.0
            }
        
        # Try dictionary lookup first
        translation = self._find_translation(text, detected_source, target_language)
        
        if translation:
            return {
                "translated_text": translation,
                "source_language": detected_source,
                "confidence": 0.98
            }
        
        # Fallback: Try partial matching for longer phrases
        normalized = self._normalize_text(text)
        words = normalized.split()
        
        # Try to translate word by word for simple phrases
        if len(words) <= 5:
            translated_words = []
            found_any = False
            for word in words:
                word_trans = self._find_translation(word, detected_source, target_language)
                if word_trans:
                    translated_words.append(word_trans)
                    found_any = True
                else:
                    translated_words.append(word)
            
            if found_any:
                return {
                    "translated_text": " ".join(translated_words),
                    "source_language": detected_source,
                    "confidence": 0.75
                }
        
        # Final fallback - indicate translation not available
        lang_names = {"en": "English", "ko": "Korean", "vi": "Vietnamese"}
        return {
            "translated_text": f"[{lang_names.get(target_language, target_language)}] {text}",
            "source_language": detected_source,
            "confidence": 0.5
        }
    
    async def detect_language(self, text: str) -> str:
        """Detect language with improved accuracy for EN/KO/VI."""
        self._detection_count += 1
        
        if not text or not text.strip():
            return "en"
        
        # Check for Korean (Hangul) - highest priority for Korean characters
        korean_count = sum(1 for char in text if '\uac00' <= char <= '\ud7af' or '\u1100' <= char <= '\u11ff')
        if korean_count > 0:
            return "ko"
        
        # Check for Vietnamese (special diacritics)
        vietnamese_chars = set('ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵĂÂĐÊÔƠƯÀẢÃÁẠẰẲẴẮẶẦẨẪẤẬÈẺẼÉẸỀỂỄẾỆÌỈĨÍỊÒỎÕÓỌỒỔỖỐỘỜỞỠỚỢÙỦŨÚỤỪỬỮỨỰỲỶỸÝỴ')
        if any(char in vietnamese_chars for char in text):
            return "vi"
        
        # Default to English for Latin alphabet
        return "en"
    
    async def health_check(self) -> bool:
        """Health check always returns True for this implementation."""
        return True
    
    def get_stats(self) -> dict:
        """Get translation statistics."""
        return {
            "translations_performed": self._translation_count,
            "language_detections_performed": self._detection_count,
            "supported_languages": list(self._supported_languages),
            "dictionary_size": len(self._translations)
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