from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_register_login_refresh_flow():
    # register
    r = client.post("/auth/register", json={"email": "user@example.com", "password": "Passw0rd!", "preferences": {"lang_pairs": ["en-ja"]}})
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert data["user"]["email"] == "user@example.com"
    access = data["token"]["access_token"]
    refresh = data["token"]["refresh_token"]

    # login
    r2 = client.post("/auth/login", json={"email": "user@example.com", "password": "Passw0rd!"})
    assert r2.status_code == 200
    data2 = r2.json()["data"]
    assert data2["user"]["email"] == "user@example.com"

    # refresh
    r3 = client.post("/auth/refresh", json={"refresh_token": refresh})
    assert r3.status_code == 200
    data3 = r3.json()["data"]
    assert data3["token"]["access_token"] != access
