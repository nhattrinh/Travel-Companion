"""
Unit tests for phrase suggestion scoring logic
"""
import pytest
from app.services.phrase_suggestion_service import PhraseSuggestionService
from app.models.phrase import Phrase


@pytest.mark.asyncio
async def test_context_filtering(db_session):
    """Test that phrases are filtered by context correctly"""
    # Create test phrases
    phrase1 = Phrase(
        canonical_text="Where is the bathroom?",
        translations={"ja": "トイレはどこですか？"},
        context_category="restaurant"
    )
    phrase2 = Phrase(
        canonical_text="How much is the ticket?",
        translations={"ja": "チケットはいくらですか？"},
        context_category="transit"
    )
    db_session.add_all([phrase1, phrase2])
    await db_session.commit()
    
    # Test filtering
    service = PhraseSuggestionService(db_session)
    suggestions = await service.get_suggestions("restaurant", "ja", limit=10)
    
    assert len(suggestions) == 1
    assert suggestions[0]["canonical_text"] == "Where is the bathroom?"
    assert suggestions[0]["translation"] == "トイレはどこですか？"


@pytest.mark.asyncio
async def test_target_language_filtering(db_session):
    """Test that only phrases with target language translation are returned"""
    phrase = Phrase(
        canonical_text="Thank you",
        translations={"ja": "ありがとう", "es": "Gracias"},
        context_category="general"
    )
    db_session.add(phrase)
    await db_session.commit()
    
    service = PhraseSuggestionService(db_session)
    
    # Request Japanese
    suggestions_ja = await service.get_suggestions("general", "ja", limit=10)
    assert len(suggestions_ja) == 1
    assert suggestions_ja[0]["translation"] == "ありがとう"
    
    # Request Spanish
    suggestions_es = await service.get_suggestions("general", "es", limit=10)
    assert len(suggestions_es) == 1
    assert suggestions_es[0]["translation"] == "Gracias"
    
    # Request unsupported language
    suggestions_fr = await service.get_suggestions("general", "fr", limit=10)
    assert len(suggestions_fr) == 0


@pytest.mark.asyncio
async def test_limit_respected(db_session):
    """Test that limit parameter is respected"""
    # Create 30 test phrases
    phrases = [
        Phrase(
            canonical_text=f"Phrase {i}",
            translations={"ja": f"フレーズ {i}"},
            context_category="general"
        )
        for i in range(30)
    ]
    db_session.add_all(phrases)
    await db_session.commit()
    
    service = PhraseSuggestionService(db_session)
    suggestions = await service.get_suggestions("general", "ja", limit=10)
    
    assert len(suggestions) == 10


@pytest.mark.asyncio
async def test_cache_hit(db_session, mock_cache_client):
    """Test that cache is used when available"""
    cached_data = [
        {
            "id": 999,
            "canonical_text": "Cached phrase",
            "translation": "キャッシュされたフレーズ",
            "phonetic": "kyasshu sareta furezu",
            "context_category": "general"
        }
    ]
    mock_cache_client.get.return_value = cached_data
    
    service = PhraseSuggestionService(db_session, mock_cache_client)
    suggestions = await service.get_suggestions("general", "ja", limit=10)
    
    # Should return cached data
    assert suggestions == cached_data
    mock_cache_client.get.assert_called_once_with("phrases:general:ja:10")


@pytest.mark.asyncio
async def test_cache_set_on_miss(db_session, mock_cache_client):
    """Test that results are cached after database query"""
    mock_cache_client.get.return_value = None
    
    phrase = Phrase(
        canonical_text="Hello",
        translations={"ja": "こんにちは"},
        context_category="general"
    )
    db_session.add(phrase)
    await db_session.commit()
    
    service = PhraseSuggestionService(db_session, mock_cache_client)
    suggestions = await service.get_suggestions("general", "ja", limit=10)
    
    # Should have set cache
    mock_cache_client.set.assert_called_once()
    args = mock_cache_client.set.call_args
    assert args[0][0] == "phrases:general:ja:10"
    assert len(args[0][1]) == 1
    assert args[1]["ttl"] == 600
