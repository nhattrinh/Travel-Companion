"""
Unit tests for favorites toggle functionality
"""
import pytest
from app.models.favorite import Favorite
from app.models.phrase import Phrase
from app.models.user import User


@pytest.mark.asyncio
async def test_create_favorite(db_session):
    """Test creating a new favorite"""
    user = User(email="test@example.com", hashed_password="hash123")
    phrase = Phrase(
        canonical_text="Hello",
        translations={"ja": "こんにちは"},
        context_category="general"
    )
    db_session.add_all([user, phrase])
    await db_session.commit()
    
    favorite = Favorite(
        user_id=user.id,
        target_type="phrase",
        target_id=phrase.id
    )
    db_session.add(favorite)
    await db_session.commit()
    await db_session.refresh(favorite)
    
    assert favorite.id is not None
    assert favorite.user_id == user.id
    assert favorite.target_type == "phrase"
    assert favorite.target_id == phrase.id
    assert favorite.created_at is not None


@pytest.mark.asyncio
async def test_remove_favorite(db_session):
    """Test removing a favorite"""
    user = User(email="test@example.com", hashed_password="hash123")
    phrase = Phrase(
        canonical_text="Hello",
        translations={"ja": "こんにちは"},
        context_category="general"
    )
    favorite = Favorite(
        user_id=1,
        target_type="phrase",
        target_id=1
    )
    db_session.add_all([user, phrase, favorite])
    await db_session.commit()
    
    # Remove favorite
    await db_session.delete(favorite)
    await db_session.commit()
    
    # Verify removed
    from sqlalchemy import select
    stmt = select(Favorite).where(Favorite.id == favorite.id)
    result = await db_session.execute(stmt)
    assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_list_user_favorites(db_session):
    """Test listing all favorites for a user"""
    user1 = User(email="user1@example.com", hashed_password="hash1")
    user2 = User(email="user2@example.com", hashed_password="hash2")
    phrase1 = Phrase(canonical_text="P1", translations={"ja": "P1"}, context_category="general")
    phrase2 = Phrase(canonical_text="P2", translations={"ja": "P2"}, context_category="general")
    phrase3 = Phrase(canonical_text="P3", translations={"ja": "P3"}, context_category="general")
    
    db_session.add_all([user1, user2, phrase1, phrase2, phrase3])
    await db_session.commit()
    
    # User1 favorites 2 phrases
    fav1 = Favorite(user_id=user1.id, target_type="phrase", target_id=phrase1.id)
    fav2 = Favorite(user_id=user1.id, target_type="phrase", target_id=phrase2.id)
    # User2 favorites 1 phrase
    fav3 = Favorite(user_id=user2.id, target_type="phrase", target_id=phrase3.id)
    
    db_session.add_all([fav1, fav2, fav3])
    await db_session.commit()
    
    # List user1 favorites
    from sqlalchemy import select
    stmt = select(Favorite).where(
        Favorite.user_id == user1.id,
        Favorite.target_type == "phrase"
    )
    result = await db_session.execute(stmt)
    user1_favorites = result.scalars().all()
    
    assert len(user1_favorites) == 2
    assert {f.target_id for f in user1_favorites} == {phrase1.id, phrase2.id}


@pytest.mark.asyncio
async def test_favorite_different_target_types(db_session):
    """Test that favorites can reference different target types"""
    user = User(email="test@example.com", hashed_password="hash123")
    db_session.add(user)
    await db_session.commit()
    
    fav_phrase = Favorite(user_id=user.id, target_type="phrase", target_id=1)
    fav_poi = Favorite(user_id=user.id, target_type="poi", target_id=2)
    fav_translation = Favorite(user_id=user.id, target_type="translation", target_id=3)
    
    db_session.add_all([fav_phrase, fav_poi, fav_translation])
    await db_session.commit()
    
    # Verify all created
    from sqlalchemy import select
    stmt = select(Favorite).where(Favorite.user_id == user.id)
    result = await db_session.execute(stmt)
    favorites = result.scalars().all()
    
    assert len(favorites) == 3
    assert {f.target_type for f in favorites} == {"phrase", "poi", "translation"}
