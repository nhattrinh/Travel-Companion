"""
Performance tests for phrase suggestion latency
"""
import pytest
import time
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_phrase_suggestions_latency_p95(async_client: AsyncClient, authenticated_headers, db_session):
    """Test that phrase suggestions meet p95 <= 300ms requirement"""
    from app.models.phrase import Phrase
    
    # Create realistic dataset (100 phrases)
    phrases = [
        Phrase(
            canonical_text=f"Common phrase {i}",
            translations={
                "ja": f"よくあるフレーズ {i}",
                "es": f"Frase común {i}",
                "fr": f"Phrase courante {i}"
            },
            phonetic=f"yoku aru furezu {i}",
            context_category=["restaurant", "transit", "lodging", "general"][i % 4]
        )
        for i in range(100)
    ]
    db_session.add_all(phrases)
    await db_session.commit()
    
    # Measure latency over 20 requests
    latencies = []
    for _ in range(20):
        start = time.perf_counter()
        response = await async_client.get(
            "/phrases?context=restaurant&target_language=ja&limit=20",
            headers=authenticated_headers
        )
        end = time.perf_counter()
        
        assert response.status_code == 200
        latencies.append((end - start) * 1000)  # Convert to ms
    
    # Calculate p95
    latencies.sort()
    p95_index = int(len(latencies) * 0.95)
    p95_latency = latencies[p95_index]
    
    print(f"\nPhrase suggestion latencies (ms): {latencies}")
    print(f"p95 latency: {p95_latency:.2f}ms")
    
    # Assert p95 <= 300ms
    assert p95_latency <= 300.0, f"p95 latency {p95_latency:.2f}ms exceeds 300ms threshold"


@pytest.mark.asyncio
async def test_phrase_suggestions_with_cache_performance(
    async_client: AsyncClient,
    authenticated_headers,
    db_session,
    mock_cache_client
):
    """Test that caching significantly improves latency"""
    from app.models.phrase import Phrase
    
    # Create phrases
    phrases = [
        Phrase(
            canonical_text=f"Phrase {i}",
            translations={"ja": f"フレーズ {i}"},
            context_category="general"
        )
        for i in range(50)
    ]
    db_session.add_all(phrases)
    await db_session.commit()
    
    # First request (cache miss)
    mock_cache_client.get.return_value = None
    start1 = time.perf_counter()
    response1 = await async_client.get(
        "/phrases?context=general&target_language=ja&limit=20",
        headers=authenticated_headers
    )
    latency1 = (time.perf_counter() - start1) * 1000
    
    assert response1.status_code == 200
    
    # Second request (cache hit)
    cached_data = response1.json()["data"]
    mock_cache_client.get.return_value = cached_data["suggestions"]
    
    start2 = time.perf_counter()
    response2 = await async_client.get(
        "/phrases?context=general&target_language=ja&limit=20",
        headers=authenticated_headers
    )
    latency2 = (time.perf_counter() - start2) * 1000
    
    assert response2.status_code == 200
    
    print(f"\nUncached latency: {latency1:.2f}ms")
    print(f"Cached latency: {latency2:.2f}ms")
    print(f"Speedup: {latency1/latency2:.2f}x")
    
    # Cache should be significantly faster (at least 2x)
    assert latency2 < latency1 / 2


@pytest.mark.asyncio
async def test_toggle_favorite_latency(async_client: AsyncClient, authenticated_headers, db_session):
    """Test that favoriting is fast (<100ms p95)"""
    from app.models.phrase import Phrase
    
    # Create test phrase
    phrase = Phrase(
        canonical_text="Test",
        translations={"ja": "テスト"},
        context_category="general"
    )
    db_session.add(phrase)
    await db_session.commit()
    
    # Measure toggle latency
    latencies = []
    for i in range(10):
        start = time.perf_counter()
        response = await async_client.post(
            f"/phrases/{phrase.id}/favorite",
            headers=authenticated_headers
        )
        latency = (time.perf_counter() - start) * 1000
        
        assert response.status_code == 200
        latencies.append(latency)
    
    latencies.sort()
    p95_index = int(len(latencies) * 0.95)
    p95_latency = latencies[p95_index]
    
    print(f"\nToggle favorite latencies (ms): {latencies}")
    print(f"p95 latency: {p95_latency:.2f}ms")
    
    # Assert p95 < 100ms (very fast operation)
    assert p95_latency < 100.0


@pytest.mark.asyncio
async def test_get_favorites_latency(async_client: AsyncClient, authenticated_headers, db_session, test_user):
    """Test that retrieving favorites is fast"""
    from app.models.phrase import Phrase
    from app.models.favorite import Favorite
    
    # Create 20 favorite phrases
    phrases = [
        Phrase(
            canonical_text=f"Fav {i}",
            translations={"ja": f"お気に入り {i}"},
            context_category="general"
        )
        for i in range(20)
    ]
    db_session.add_all(phrases)
    await db_session.commit()
    
    favorites = [
        Favorite(user_id=test_user.id, target_type="phrase", target_id=p.id)
        for p in phrases
    ]
    db_session.add_all(favorites)
    await db_session.commit()
    
    # Measure latency
    latencies = []
    for _ in range(10):
        start = time.perf_counter()
        response = await async_client.get(
            "/phrases/favorites",
            headers=authenticated_headers
        )
        latency = (time.perf_counter() - start) * 1000
        
        assert response.status_code == 200
        assert len(response.json()["data"]) == 20
        latencies.append(latency)
    
    latencies.sort()
    p95_index = int(len(latencies) * 0.95)
    p95_latency = latencies[p95_index]
    
    print(f"\nGet favorites latencies (ms): {latencies}")
    print(f"p95 latency: {p95_latency:.2f}ms")
    
    # Should be fast (<200ms p95)
    assert p95_latency < 200.0
