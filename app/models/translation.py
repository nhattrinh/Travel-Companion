from sqlalchemy.orm import relationship
from app.core.db import Base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, func


class Translation(Base):
    __tablename__ = "translations"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    trip_id = Column(Integer, ForeignKey("trips.id"), nullable=True, index=True)
    source_text = Column(Text, nullable=False)
    target_text = Column(Text, nullable=False)
    source_language = Column(String(8), nullable=False)
    target_language = Column(String(8), nullable=False)
    confidence = Column(Integer, nullable=True)  # store *100 for simplicity
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    trip = relationship("Trip", back_populates="translations")
