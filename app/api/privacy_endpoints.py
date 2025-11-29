"""Privacy purge endpoint.

Task: T144
Removes user-associated transient data: translations, favorites, trips.
Cascade will migrate to dedicated service (T145) later.
Envelope format maintained.
"""
from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import delete  # noqa: F401

from app.core.db import db_session
from app.models.user import User
from app.services.privacy_purge_service import PrivacyPurgeService
from app.core.db import db_session as _session_factory

router = APIRouter(prefix="/privacy", tags=["privacy"])


def _get_user(session: Session, user_id: int) -> User:
    user = session.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="USER_NOT_FOUND")
    return user


@router.post("/purge")
def purge_user_data(user_id: int):
    """Delete user-associated data (retention reset)."""
    session: Session = db_session()
    try:
        _get_user(session, user_id)
        service = PrivacyPurgeService(_session_factory)
        summary = service.purge(user_id)
        return {
            "status": "ok",
            "data": {"purged": True, "summary": summary},
            "error": None,
        }
    finally:
        session.close()
