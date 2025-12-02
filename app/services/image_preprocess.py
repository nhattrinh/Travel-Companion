"""Simple image preprocessing utilities wrapper for User Story 1."""
from PIL import Image, ImageEnhance
import io

def enhance_contrast(image_bytes: bytes) -> Image.Image:
    """
    Enhance image contrast for better OCR accuracy.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        PIL Image with enhanced contrast (1.5x)
    """
    img = Image.open(io.BytesIO(image_bytes))
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(1.5)
