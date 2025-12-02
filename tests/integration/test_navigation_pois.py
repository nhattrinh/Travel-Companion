from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_pois_endpoint():
    r = client.get('/navigation/pois?lat=35.68&lon=139.76&radius=1000')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'ok'
    assert 'pois' in body['data']
    assert len(body['data']['pois']) > 0
