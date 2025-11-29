import time
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_poi_endpoint_latency():
    start = time.time()
    r = client.get('/navigation/pois?lat=35.68&lon=139.76')
    assert r.status_code == 200
    elapsed_ms = (time.time() - start) * 1000
    # Phase 4 budget: p95 ≤800ms
    assert elapsed_ms < 800
