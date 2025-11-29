"""Very lightweight heuristic language detection helper."""

def detect_language(text: str) -> str:
    """
    Detect language using simple heuristics.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Language code (en, es, fr, ja) - defaults to 'en'
    """
    if not text.strip():
        return "en"
    lower = text.lower()
    for word in ["el", "la", "es", "con"]:
        if f" {word} " in lower:
            return "es"
    for word in ["le", "avec", "est"]:
        if f" {word} " in lower:
            return "fr"
    if any('\u3040' <= c <= '\u30ff' for c in text):
        return "ja"
    return "en"
