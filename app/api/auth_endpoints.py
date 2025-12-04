"""Authentication endpoints for Travel Companion feature (Phase 2)."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import db_session
from app.models.user import User
from app.core.security import hash_password, verify_password
from app.core.jwt import create_access_token, create_refresh_token, decode_token
from app.schemas.user import UserCreate, UserRead, LoginRequest, Token, TokenRefresh
from app.schemas.base import Envelope

router = APIRouter(prefix="/auth", tags=["auth"])

def _envelope(data=None, error: str | None = None, status: str = "ok"):
    return {"status": status, "data": data, "error": error}

@router.post("/register", response_model=Envelope)
async def register_user(payload: UserCreate):
    with db_session() as session:  # type: Session
        existing = session.execute(select(User).where(User.email == payload.email)).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        user = User(email=payload.email, hashed_password=hash_password(payload.password), preferences=payload.preferences)
        session.add(user)
        session.flush()  # assign id
        access = create_access_token(str(user.id))
        refresh = create_refresh_token(str(user.id))
        return _envelope(data={"user": UserRead(id=user.id, email=user.email, preferences=user.preferences), "token": Token(access_token=access, refresh_token=refresh)})

@router.post("/login", response_model=Envelope)
async def login_user(payload: LoginRequest):
    with db_session() as session:
        user = session.execute(select(User).where(User.email == payload.email)).scalar_one_or_none()
        if not user or not verify_password(payload.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        access = create_access_token(str(user.id))
        refresh = create_refresh_token(str(user.id))
        return _envelope(data={"user": UserRead(id=user.id, email=user.email, preferences=user.preferences), "token": Token(access_token=access, refresh_token=refresh)})

@router.post("/refresh", response_model=Envelope)
async def refresh_token(payload: TokenRefresh):
    decoded = decode_token(payload.refresh_token, refresh=True)
    if not decoded:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    subject = decoded.get("sub")
    access = create_access_token(subject)
    new_refresh = create_refresh_token(subject)
    return _envelope(data={"token": Token(access_token=access, refresh_token=new_refresh)})


@router.get("/me", response_model=Envelope)
async def get_current_user(authorization: str | None = None):
    """Get current user from JWT token in Authorization header."""
    if not authorization:
        raise HTTPException(
            status_code=401, detail="Missing authorization header"
        )

    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401, detail="Invalid authorization header format"
        )

    token = parts[1]
    decoded = decode_token(token, refresh=False)
    if not decoded:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = decoded.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    with db_session() as session:
        user = session.execute(
            select(User).where(User.id == int(user_id))
        ).scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return _envelope(data={
            "user": UserRead(
                id=user.id,
                email=user.email,
                preferences=user.preferences
            )
        })
