from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from contextlib import contextmanager
from typing import Generator
import os

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://user:password@localhost:5432/travel_companion"
)

engine = create_engine(POSTGRES_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
)

# SQLAlchemy declarative base for models
Base = declarative_base()


@contextmanager
def db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
