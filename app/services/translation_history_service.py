"""Translation history persistence service."""
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.translation import Translation

class TranslationHistoryService:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def save(self, user_id: int | None, source_text: str, target_text: str, source_lang: str, target_lang: str, confidence: int | None = None) -> int:
        with self._session_factory() as session:  # type: Session
            t = Translation(user_id=user_id, source_text=source_text, target_text=target_text, source_language=source_lang, target_language=target_lang, confidence=confidence)
            session.add(t)
            session.flush()
            return t.id

    def list_recent(self, user_id: int | None, limit: int = 20):
        with self._session_factory() as session:
            stmt = select(Translation).where(Translation.user_id == user_id).order_by(Translation.id.desc()).limit(limit)
            return [r for r in session.execute(stmt).scalars()]
