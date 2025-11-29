import pytest
from app.core.security import hash_password, verify_password, issue_token, verify_token


def test_password_hash_and_verify():
    raw = "SuperSecurePass123!"
    hashed = hash_password(raw)
    assert hashed != raw
    assert verify_password(raw, hashed)
    assert not verify_password("wrong", hashed)


def test_issue_and_verify_token():
    token = issue_token("user-1", expires_minutes=5)
    payload = verify_token(token)
    assert payload.get("sub") == "user-1"
    assert "exp" in payload
