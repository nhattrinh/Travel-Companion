from app.services.image_preprocess import enhance_contrast
from PIL import Image
import io

def test_enhance_contrast_basic():
    # create simple image
    img = Image.new('RGB', (10,10), color='grey')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    out = enhance_contrast(buf.getvalue())
    assert out.size == (10,10)
