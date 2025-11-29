from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_invalid_auth_login_returns_error():
    r = client.post('/auth/login', json={'email':'missing@example.com','password':'wrongpass'})
    assert r.status_code == 401
    body = r.json()
    assert body['error'] is not None or body.get('detail')
