"""PhraseSuggestion model.

Task: T148
Links suggested phrases with relevance and optional user/context.
"""
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, func

Base = declarative_base()


class PhraseSuggestion(Base):
    __tablename__ = "phrase_suggestions"

    id = Column(Integer, primary_key=True)
    phrase_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    context = Column(String(64), nullable=True)
    relevance = Column(Float, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PhraseSuggestion id={self.id} phrase_id={self.phrase_id}>"
