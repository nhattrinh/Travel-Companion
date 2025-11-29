"""
Integration tests for phrase suggestions endpoint
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_phrase_suggestions_success(async_client: AsyncClient, authenticated_headers, db_session):
    """Test successful phrase suggestions retrieval"""
    from app.models.phrase import Phrase
    
    # Create test phrases
    phrases = [
        Phrase(
            canonical_text="Where is the restroom?",
            translations={"ja": "トイレはどこですか？", "es": "¿Dónde está el baño?"},
            phonetic="toire wa doko desu ka?",
            context_category="restaurant"
        ),
        Phrase(
            canonical_text="The check, please",
            translations={"ja": "お会計お願いします", "es": "La cuenta, por favor"},
            phonetic="okaikei onegai shimasu",
            context_category="restaurant"
        ),
    ]
    db_session.add_all(phrases)
    await db_session.commit()
    
    response = await async_client.get(
        "/phrases?context=restaurant&target_language=ja&limit=10",
        headers=authenticated_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "data" in data
    assert data["data"]["context"] == "restaurant"
    assert data["data"]["target_language"] == "ja"
    assert len(data["data"]["suggestions"]) == 2
    
    # Verify suggestion structure
    suggestion = data["data"]["suggestions"][0]
    assert "id" in suggestion
    assert "canonical_text" in suggestion
    assert "translation" in suggestion
    assert "phonetic" in suggestion
    assert "context_category" in suggestion


@pytest.mark.asyncio
async def test_phrase_suggestions_empty_result(async_client: AsyncClient, authenticated_headers):
    """Test phrase suggestions with no matching phrases"""
    response = await async_client.get(
        "/phrases?context=nonexistent&target_language=ja",
        headers=authenticated_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert len(data["data"]["suggestions"]) == 0


@pytest.mark.asyncio
async def test_phrase_suggestions_unauthorized(async_client: AsyncClient):
    """Test phrase suggestions without authentication"""
    response = await async_client.get("/phrases?context=general&target_language=ja")
    
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_create_phrase_success(async_client: AsyncClient, authenticated_headers):
    """Test creating a new phrase"""
    phrase_data = {
        "canonical_text": "Good morning",
        "translations": {"ja": "おはよう", "es": "Buenos días"},
        "phonetic": "ohayou",
        "context_category": "general"
    }
    
    response = await async_client.post(
        "/phrases",
        json=phrase_data,
        headers=authenticated_headers
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "ok"
    assert data["data"]["canonical_text"] == "Good morning"
    assert data["data"]["translations"]["ja"] == "おはよう"
    assert data["data"]["phonetic"] == "ohayou"
    assert data["data"]["context_category"] == "general"


@pytest.mark.asyncio
async def test_toggle_favorite_add(async_client: AsyncClient, authenticated_headers, db_session):
    """Test adding a phrase to favorites"""
    from app.models.phrase import Phrase
    
    phrase = Phrase(
        canonical_text="Test phrase",
        translations={"ja": "テスト"},
        context_category="general"
    )
    db_session.add(phrase)
    await db_session.commit()
    
    response = await async_client.post(
        f"/phrases/{phrase.id}/favorite",
        headers=authenticated_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    # Response could be FavoriteRead or dict with favorited status
    assert "data" in data


@pytest.mark.asyncio
async def test_toggle_favorite_remove(async_client: AsyncClient, authenticated_headers, db_session, test_user):
    """Test removing a phrase from favorites"""
    from app.models.phrase import Phrase
    from app.models.favorite import Favorite
    
    phrase = Phrase(
        canonical_text="Test phrase",
        translations={"ja": "テスト"},
        context_category="general"
    )
    db_session.add(phrase)
    await db_session.commit()
    
    # Add favorite first
    favorite = Favorite(
        user_id=test_user.id,
        target_type="phrase",
        target_id=phrase.id
    )
    db_session.add(favorite)
    await db_session.commit()
    
    # Remove favorite
    response = await async_client.post(
        f"/phrases/{phrase.id}/favorite",
        headers=authenticated_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_toggle_favorite_nonexistent_phrase(async_client: AsyncClient, authenticated_headers):
    """Test toggling favorite for non-existent phrase"""
    response = await async_client.post(
        "/phrases/99999/favorite",
        headers=authenticated_headers
    )
    
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_favorite_phrases(async_client: AsyncClient, authenticated_headers, db_session, test_user):
    """Test retrieving user's favorite phrases"""
    from app.models.phrase import Phrase
    from app.models.favorite import Favorite
    
    # Create test phrases
    phrase1 = Phrase(canonical_text="P1", translations={"ja": "P1"}, context_category="general")
    phrase2 = Phrase(canonical_text="P2", translations={"ja": "P2"}, context_category="general")
    db_session.add_all([phrase1, phrase2])
    await db_session.commit()
    
    # Add favorites
    fav1 = Favorite(user_id=test_user.id, target_type="phrase", target_id=phrase1.id)
    fav2 = Favorite(user_id=test_user.id, target_type="phrase", target_id=phrase2.id)
    db_session.add_all([fav1, fav2])
    await db_session.commit()
    
    response = await async_client.get(
        "/phrases/favorites",
        headers=authenticated_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert len(data["data"]) == 2
    assert {p["canonical_text"] for p in data["data"]} == {"P1", "P2"}
