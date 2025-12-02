import time
from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io

client = TestClient(app)

def test_translation_latency_smoke():
    img = Image.new('RGB', (80,80), color='white')
    buf = io.BytesIO(); img.save(buf, format='PNG')
    files = {'image': ('frame.png', buf.getvalue(), 'image/png')}
    start = time.time()
    r = client.post('/translation/live-frame?target_language=ja', files=files)
    assert r.status_code == 200
    elapsed_ms = (time.time() - start) * 1000
    # Loose budget for mock
    assert elapsed_ms < 1500
