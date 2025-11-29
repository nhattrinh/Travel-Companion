from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_live_frame_invalid_image_returns_envelope_error():
    # send non-image bytes to trigger preprocess/ocr failure
    files = {'image': ('bad.bin', b'not-an-image', 'application/octet-stream')}
    r = client.post('/translation/live-frame?target_language=ja', files=files)
    assert r.status_code == 400
    body = r.json()
    assert body.get('status') == 'error'
    assert body.get('data') is None
    assert isinstance(body.get('error'), str) and 'OCR_PROCESSING_FAILED' in body.get('error')
