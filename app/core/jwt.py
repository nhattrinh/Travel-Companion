"""JWT issue / verify utilities (access & refresh tokens)"""
from datetime import datetime, timedelta
from typing import Dict, Any
import os
import jwt

JWT_SECRET = os.getenv("JWT_SECRET", "please-change-me")
JWT_REFRESH_SECRET = os.getenv("JWT_REFRESH_SECRET", JWT_SECRET)
ALGORITHM = "HS256"

def _build_payload(subject: str, expires_minutes: int, scopes: list[str] | None = None) -> Dict[str, Any]:
    return {
        "sub": subject,
        "scopes": scopes or [],
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=expires_minutes),
    }

def create_access_token(subject: str, scopes: list[str] | None = None, expires_minutes: int = 60) -> str:
    return jwt.encode(_build_payload(subject, expires_minutes, scopes), JWT_SECRET, algorithm=ALGORITHM)

def create_refresh_token(subject: str, expires_minutes: int = 60 * 24 * 7) -> str:
    return jwt.encode(_build_payload(subject, expires_minutes), JWT_REFRESH_SECRET, algorithm=ALGORITHM)

def decode_token(token: str, refresh: bool = False) -> Dict[str, Any] | None:
    secret = JWT_REFRESH_SECRET if refresh else JWT_SECRET
    try:
        return jwt.decode(token, secret, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
