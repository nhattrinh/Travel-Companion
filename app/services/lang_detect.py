"""Very lightweight heuristic language detection helper."""

def detect_language(text: str) -> str:
    """
    Detect language using simple heuristics.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Language code (en, es, fr, ja, ko, vi, zh) - defaults to 'en'
    """
    if not text.strip():
        return "en"
    
    # Check for Asian scripts first (character-based detection)
    for char in text:
        # Korean (Hangul)
        if '\uac00' <= char <= '\ud7af' or '\u1100' <= char <= '\u11ff':
            return "ko"
        # Japanese (Hiragana, Katakana)
        if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff':
            return "ja"
        # Chinese (CJK Unified Ideographs) - check after Japanese
        if '\u4e00' <= char <= '\u9fff':
            # Could be Chinese or Japanese Kanji - check for hiragana/katakana
            if any('\u3040' <= c <= '\u30ff' for c in text):
                return "ja"
            return "zh"
    
    # Vietnamese detection (special diacritics)
    vietnamese_chars = 'ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ'
    if any(char in vietnamese_chars for char in text.lower()):
        return "vi"
    
    # Word-based detection for European languages
    lower = text.lower()
    
    # Spanish
    for word in ["el", "la", "es", "con", "que", "de", "los", "las"]:
        if f" {word} " in f" {lower} ":
            return "es"
    
    # French
    for word in ["le", "avec", "est", "les", "des", "une", "que", "dans"]:
        if f" {word} " in f" {lower} ":
            return "fr"
    
    return "en"
