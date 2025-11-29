"""
Trip model for persistent trip memory
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum as SQLEnum, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum

from app.core.db import Base


class TripStatus(str, enum.Enum):
    """Trip lifecycle status"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class Trip(Base):
    """
    Trip represents a user's travel session
    Tracks destination, dates, and aggregates translations/favorites
    """
    __tablename__ = "trips"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    destination = Column(String(255), nullable=False)
    start_date = Column(DateTime(timezone=True), nullable=False, index=True)
    end_date = Column(DateTime(timezone=True), nullable=True)
    status = Column(SQLEnum(TripStatus), default=TripStatus.ACTIVE, nullable=False, index=True)
    trip_metadata = Column(JSONB, nullable=True)  # Flexible storage for notes, preferences, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="trips")
    translations = relationship("Translation", back_populates="trip", cascade="all, delete-orphan")
