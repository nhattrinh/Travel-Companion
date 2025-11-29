"""User profile endpoints: fetch & update preferences.

Envelope format: {"status": str, "data": object|None, "error": str|None}
Task: T142
"""
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse  # noqa: F401
from sqlalchemy.orm import Session

from app.core.db import db_session
from app.models.user import User

router = APIRouter(prefix="/user", tags=["user"])


def _get_user(session: Session, user_id: int) -> User:
    user = session.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="USER_NOT_FOUND")
    return user


@router.get("/profile")
def get_profile(user_id: int):
    """Return basic user profile & preferences.

    Authentication integration pending; user_id passed explicitly.
    """
    session: Session = db_session()
    try:
        user = _get_user(session, user_id)
        data = {
            "id": user.id,
            "email": user.email,
            "preferences": user.preferences or {},
            "created_at": (
                user.created_at.isoformat() if user.created_at else None
            ),
            "updated_at": (
                user.updated_at.isoformat() if user.updated_at else None
            ),
        }
        return {"status": "ok", "data": data, "error": None}
    finally:
        session.close()


@router.patch("/profile/preferences")
def update_preferences(
    user_id: int,
    preferences: dict = Body(
        ...,
        description="JSON preferences, e.g. { 'language_pairs': ['en-ja'] }",
    )
):
    """Update user preferences JSONB column.

    Performs simple merge: existing keys overwritten by provided keys.
    """
    session: Session = db_session()
    try:
        user = _get_user(session, user_id)
        current = user.preferences or {}
        current.update(preferences)
        user.preferences = current
        session.add(user)
        session.commit()
        return {
            "status": "ok",
            "data": {"preferences": user.preferences},
            "error": None,
        }
    finally:
        session.close()


@router.delete("/profile/preferences")
def clear_preferences(user_id: int):
    """Clear user preferences (sets to empty object)."""
    session: Session = db_session()
    try:
        user = _get_user(session, user_id)
        user.preferences = {}
        session.add(user)
        session.commit()
        return {"status": "ok", "data": {"preferences": {}}, "error": None}
    finally:
        session.close()
