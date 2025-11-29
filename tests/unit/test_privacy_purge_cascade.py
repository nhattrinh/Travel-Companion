"""Unit test for privacy purge cascade.

Task: T146
Ensures service deletes user data and returns counts.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.services.privacy_purge_service import PrivacyPurgeService
from app.models.translation import Translation
from app.models.favorite import Favorite
from app.models.trip import Trip
from app.models.user import User, Base as UserBase
from app.models.translation import Base as TranslationBase  # type: ignore
from app.models.favorite import Base as FavoriteBase  # type: ignore
from app.models.trip import Trip as TripModel  # noqa: F401


@pytest.fixture(scope="function")
def session_factory():
    # In-memory SQLite for fast unit testing
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Create tables from all Bases declaring own metadata
    UserBase.metadata.create_all(engine)
    TranslationBase.metadata.create_all(engine)
    FavoriteBase.metadata.create_all(engine)
    # Trip model uses app.core.db.Base so ensure creation through its metadata
    from app.core.db import Base as CoreBase
    CoreBase.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _seed(session, user_id: int):
    session.add(
        Translation(
            user_id=user_id,
            source_text="hola",
            target_text="hello",
            source_language="es",
            target_language="en",
        )
    )
    session.add(Favorite(user_id=user_id, target_type="phrase", target_id=1))
    session.add(
        Trip(
            user_id=user_id,
            destination="Tokyo",
            start_date="2025-11-01",
        )
    )
    session.commit()


def test_privacy_purge_cascade(session_factory):
    session = session_factory()
    try:
        # create user
        user = User(email="test@example.com", hashed_password="x")
        session.add(user)
        session.commit()

        _seed(session, user.id)

        svc = PrivacyPurgeService(session_factory)
        summary = svc.purge(user.id)

        assert summary["translations_before"] == 1
        assert summary["favorites_before"] == 1
        assert summary["trips_before"] == 1
        # After purge counts reflect deletions
        assert summary["translations_deleted"] == 1
        assert summary["favorites_deleted"] == 1
        assert summary["trips_deleted"] == 1

        # Verify DB empty for user associations
        assert (
            session.query(Translation).filter_by(user_id=user.id).count() == 0
        )
        assert session.query(Favorite).filter_by(user_id=user.id).count() == 0
        assert session.query(Trip).filter_by(user_id=user.id).count() == 0
    finally:
        session.close()
