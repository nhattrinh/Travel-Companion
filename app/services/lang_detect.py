"""Very lightweight heuristic language detection helper."""

def detect_language(text: str) -> str:
    """
    Detect language using simple heuristics.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Language code (en, ko, vi) - defaults to 'en'
    """
    if not text.strip():
        return "en"
    
    # Check for Asian scripts first (character-based detection)
    for char in text:
        # Korean (Hangul)
        if '\uac00' <= char <= '\ud7af' or '\u1100' <= char <= '\u11ff':
            return "ko"
    
    # Vietnamese detection (special diacritics)
    vietnamese_chars = 'ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ'
    if any(char in vietnamese_chars for char in text.lower()):
        return "vi"
    
    return "en"
