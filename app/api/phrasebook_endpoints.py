"""
Phrasebook API endpoints - Context-aware phrase suggestions and favorites
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.db import get_db
from app.core.cache_client import CacheClient, get_cache_client
from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.phrase import Phrase
from app.models.favorite import Favorite
from app.schemas.phrase import PhraseSuggestionResponse, PhraseCreate, PhraseRead
from app.schemas.favorite import FavoriteCreate, FavoriteRead
from app.schemas.api_models import Envelope
from app.services.phrase_suggestion_service import PhraseSuggestionService
from app.core.metrics_navigation_phrase import record_phrase_latency

router = APIRouter(prefix="/phrases", tags=["phrasebook"])


@router.get("", response_model=Envelope[PhraseSuggestionResponse])
async def get_phrase_suggestions(
    context: str,
    target_language: str = "ja",
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    cache_client: Optional[CacheClient] = Depends(get_cache_client),
    current_user: User = Depends(get_current_user),
):
    """
    Get context-aware phrase suggestions
    
    - **context**: Context category (restaurant, transit, lodging, general)
    - **target_language**: Target language code (default: ja)
    - **limit**: Max number of suggestions (default: 20)
    """
    service = PhraseSuggestionService(db, cache_client)
    with record_phrase_latency():
        suggestions = await service.get_suggestions(
            context, target_language, limit
        )
    
    return Envelope(
        status="ok",
        data=PhraseSuggestionResponse(
            context=context,
            target_language=target_language,
            suggestions=suggestions,
        )
    )


@router.post(
    "",
    response_model=Envelope[PhraseRead],
    status_code=status.HTTP_201_CREATED,
)
async def create_phrase(
    phrase: PhraseCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new phrase (admin/curator only - future: add authorization)
    
    - **canonical_text**: Source language phrase
    - **translations**: Dict of language_code -> translation
    - **phonetic**: Optional phonetic representation
    - **context_category**: Context (restaurant, transit, lodging, general)
    """
    new_phrase = Phrase(
        canonical_text=phrase.canonical_text,
        translations=phrase.translations,
        phonetic=phrase.phonetic,
        context_category=phrase.context_category,
    )
    db.add(new_phrase)
    await db.commit()
    await db.refresh(new_phrase)
    
    return Envelope(
        status="ok",
        data=PhraseRead(
            id=new_phrase.id,
            canonical_text=new_phrase.canonical_text,
            translations=new_phrase.translations,
            phonetic=new_phrase.phonetic,
            context_category=new_phrase.context_category,
            created_at=new_phrase.created_at,
        )
    )


@router.post("/{phrase_id}/favorite", response_model=Envelope[FavoriteRead])
async def toggle_favorite_phrase(
    phrase_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Toggle favorite status for a phrase (creates or removes favorite)
    
    - **phrase_id**: ID of phrase to favorite/unfavorite
    """
    # Check if phrase exists
    stmt = select(Phrase).where(Phrase.id == phrase_id)
    result = await db.execute(stmt)
    phrase = result.scalar_one_or_none()
    
    if not phrase:
        raise HTTPException(status_code=404, detail="Phrase not found")
    
    # Check if already favorited
    stmt = select(Favorite).where(
        Favorite.user_id == current_user.id,
        Favorite.target_type == "phrase",
        Favorite.target_id == phrase_id,
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()
    
    if existing:
        # Remove favorite
        await db.delete(existing)
        await db.commit()
        return Envelope(
            status="ok",
            data={"favorited": False, "phrase_id": phrase_id}
        )
    else:
        # Add favorite
        favorite = Favorite(
            user_id=current_user.id,
            target_type="phrase",
            target_id=phrase_id,
        )
        db.add(favorite)
        await db.commit()
        await db.refresh(favorite)
        
        return Envelope(
            status="ok",
            data=FavoriteRead(
                id=favorite.id,
                user_id=favorite.user_id,
                target_type=favorite.target_type,
                target_id=favorite.target_id,
                created_at=favorite.created_at,
            )
        )


@router.get("/favorites", response_model=Envelope[list[PhraseRead]])
async def get_favorite_phrases(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get user's favorite phrases
    """
    # Get all phrase favorites for current user
    stmt = (
        select(Favorite)
        .where(
            Favorite.user_id == current_user.id,
            Favorite.target_type == "phrase",
        )
    )
    result = await db.execute(stmt)
    favorites = result.scalars().all()
    
    # Fetch actual phrase objects
    phrase_ids = [f.target_id for f in favorites]
    if not phrase_ids:
        return Envelope(status="ok", data=[])
    
    stmt = select(Phrase).where(Phrase.id.in_(phrase_ids))
    result = await db.execute(stmt)
    phrases = result.scalars().all()
    
    phrase_list = [
        PhraseRead(
            id=p.id,
            canonical_text=p.canonical_text,
            translations=p.translations,
            phonetic=p.phonetic,
            context_category=p.context_category,
            created_at=p.created_at,
        )
        for p in phrases
    ]
    
    return Envelope(status="ok", data=phrase_list)
