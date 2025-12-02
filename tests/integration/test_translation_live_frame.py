from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io

client = TestClient(app)

def create_image_bytes():
    img = Image.new('RGB', (40,40), color='white')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

def test_live_frame_endpoint():
    img_bytes = create_image_bytes()
    files = {'image': ('frame.png', img_bytes, 'image/png')}
    r = client.post('/translation/live-frame?target_language=ja', files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['status'] == 'ok'
    data = body['data']
    assert 'segments' in data
    assert data['target_language'] == 'ja'
