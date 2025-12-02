"""Privacy purge service.

Task: T145
Cascade deletion of user transient data. Returns summary counts.
"""
from sqlalchemy.orm import Session
from sqlalchemy import delete, select, func

from app.models.translation import Translation  # type: ignore
from app.models.favorite import Favorite  # type: ignore
from app.models.trip import Trip  # type: ignore


class PrivacyPurgeService:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def purge(self, user_id: int) -> dict:
        """Delete translations, favorites, trips for a user.

        Returns counts of deleted rows.
        """
        session: Session = self._session_factory()
        try:
            counts = {}

            def _count(model):
                return session.execute(
                    select(func.count()).where(model.user_id == user_id)
                ).scalar_one()

            counts["translations_before"] = _count(Translation)
            counts["favorites_before"] = _count(Favorite)
            counts["trips_before"] = _count(Trip)

            session.execute(
                delete(Translation).where(Translation.user_id == user_id)
            )
            session.execute(
                delete(Favorite).where(Favorite.user_id == user_id)
            )
            session.execute(
                delete(Trip).where(Trip.user_id == user_id)
            )
            session.commit()

            counts["translations_deleted"] = counts["translations_before"]
            counts["favorites_deleted"] = counts["favorites_before"]
            counts["trips_deleted"] = counts["trips_before"]
            return counts
        finally:
            session.close()
