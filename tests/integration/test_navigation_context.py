from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_pois_include_etiquette_notes():
    r = client.get('/navigation/pois?lat=35.68&lon=139.76')
    assert r.status_code == 200
    pois = r.json()['data']['pois']
    # check that at least one POI has notes
    assert any(p.get('etiquette_notes') for p in pois)
