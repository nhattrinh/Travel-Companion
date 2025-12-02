"""
Phrase Suggestion Service - Context-aware phrase recommendations
"""
import json
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.phrase import Phrase
from app.core.cache_client import CacheClient


class PhraseSuggestionService:
    """Provides context-aware phrase suggestions with caching"""

    def __init__(self, db: Session, cache_client: Optional[CacheClient] = None):
        self.db = db
        self.cache = cache_client

    async def get_suggestions(
        self,
        context: str,
        target_language: str,
        limit: int = 20
    ) -> list[dict]:
        """
        Get phrase suggestions for a context
        
        Args:
            context: Context category (restaurant, transit, lodging, general)
            target_language: Target language code (ja, es, etc.)
            limit: Maximum number of suggestions
            
        Returns:
            List of phrase dicts with canonical_text, translation, phonetic
        """
        from app.api.metrics_endpoints import metrics_collector
        
        cache_key = f"phrases:{context}:{target_language}:{limit}"
        
        # Try cache first
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                metrics_collector.record_cache_hit("phrase_suggestions")
                # Cache returns JSON string, parse it back to list
                return json.loads(cached) if isinstance(cached, str) else cached
            metrics_collector.record_cache_miss("phrase_suggestions")

        # Query phrases by context (sync execution)
        stmt = (
            select(Phrase)
            .where(Phrase.context_category == context)
            .limit(limit)
        )
        result = self.db.execute(stmt)
        phrases = result.scalars().all()

        # Format response with target language translation
        suggestions = []
        for phrase in phrases:
            translation = phrase.translations.get(target_language, "")
            if translation:
                suggestions.append({
                    "id": phrase.id,
                    "canonical_text": phrase.canonical_text,
                    "translation": translation,
                    "phonetic": phrase.phonetic,
                    "context_category": phrase.context_category,
                })

        # Cache for 10 minutes (phrases don't change frequently)
        if self.cache:
            # Serialize to JSON string for Redis
            await self.cache.set(
                cache_key, json.dumps(suggestions), ttl_seconds=600
            )

        return suggestions
