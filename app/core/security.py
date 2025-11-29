from datetime import datetime, timedelta
from typing import Any, Dict
import os
import jwt
import hashlib
import hmac

try:
    import bcrypt  # type: ignore
    _BCRYPT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BCRYPT_AVAILABLE = False

JWT_SECRET = os.getenv("JWT_SECRET", "please-change-me")
JWT_ALG = "HS256"

def issue_token(subject: str, expires_minutes: int = 60) -> str:
    payload: Dict[str, Any] = {
        "sub": subject,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def verify_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.PyJWTError:
        return {}


def hash_password(password: str) -> str:
    """Hash a password using bcrypt if available; fallback to PBKDF2."""
    if _BCRYPT_AVAILABLE:
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")
    # Fallback PBKDF2-HMAC-SHA256
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"pbkdf2_sha256$100000${salt.hex()}${dk.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against stored hash supporting bcrypt or PBKDF2 fallback."""
    if hashed.startswith("$2b$") or hashed.startswith("$2a$"):
        if not _BCRYPT_AVAILABLE:
            return False
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    if hashed.startswith("pbkdf2_sha256$"):
        try:
            _algo, iterations, salt_hex, dk_hex = hashed.split("$")[0:4]
            iterations_int = int(iterations)
            salt = bytes.fromhex(salt_hex)
            dk_check = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations_int)
            return hmac.compare_digest(dk_check.hex(), dk_hex)
        except Exception:
            return False
    return False
