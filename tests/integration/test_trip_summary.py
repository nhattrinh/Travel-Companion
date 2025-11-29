"""
Integration tests for trip summary endpoint
"""
import pytest
from datetime import datetime, timedelta
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_trip_summary_success(async_client: AsyncClient, authenticated_headers, db_session, test_user):
    """Test successful trip summary retrieval with statistics"""
    from app.models.trip import Trip, TripStatus
    from app.models.translation import Translation
    from app.models.favorite import Favorite
    from app.models.phrase import Phrase
    
    # Create trip
    trip = Trip(
        user_id=test_user.id,
        destination="Tokyo, Japan",
        start_date=datetime.utcnow(),
        status=TripStatus.ACTIVE
    )
    db_session.add(trip)
    await db_session.commit()
    await db_session.refresh(trip)
    
    # Create translations for this trip
    trans1 = Translation(
        user_id=test_user.id,
        trip_id=trip.id,
        source_text="Hello",
        target_text="こんにちは",
        source_language="en",
        target_language="ja"
    )
    trans2 = Translation(
        user_id=test_user.id,
        trip_id=trip.id,
        source_text="Thank you",
        target_text="ありがとう",
        source_language="en",
        target_language="ja"
    )
    db_session.add_all([trans1, trans2])
    
    # Create favorites during trip period
    phrase = Phrase(
        canonical_text="Where is the bathroom?",
        translations={"ja": "トイレはどこですか？"},
        context_category="general"
    )
    db_session.add(phrase)
    await db_session.commit()
    
    favorite = Favorite(
        user_id=test_user.id,
        target_type="phrase",
        target_id=phrase.id
    )
    db_session.add(favorite)
    await db_session.commit()
    
    # Request trip summary
    response = await async_client.get(
        f"/trips/{trip.id}/summary",
        headers=authenticated_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "data" in data
    
    summary = data["data"]
    assert summary["trip"]["id"] == trip.id
    assert summary["trip"]["destination"] == "Tokyo, Japan"
    assert summary["translation_count"] == 2
    assert summary["favorite_count"] >= 1
    assert len(summary["recent_translations"]) == 2


@pytest.mark.asyncio
async def test_get_trip_summary_empty_trip(async_client: AsyncClient, authenticated_headers, db_session, test_user):
    """Test trip summary with no translations or favorites"""
    from app.models.trip import Trip, TripStatus
    
    trip = Trip(
        user_id=test_user.id,
        destination="Empty Trip",
        start_date=datetime.utcnow(),
        status=TripStatus.ACTIVE
    )
    db_session.add(trip)
    await db_session.commit()
    await db_session.refresh(trip)
    
    response = await async_client.get(
        f"/trips/{trip.id}/summary",
        headers=authenticated_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    summary = data["data"]
    
    assert summary["translation_count"] == 0
    assert summary["favorite_count"] == 0
    assert len(summary["recent_translations"]) == 0


@pytest.mark.asyncio
async def test_get_trip_summary_unauthorized(async_client: AsyncClient, db_session, test_user):
    """Test trip summary access without authentication"""
    from app.models.trip import Trip, TripStatus
    
    trip = Trip(
        user_id=test_user.id,
        destination="Test",
        start_date=datetime.utcnow(),
        status=TripStatus.ACTIVE
    )
    db_session.add(trip)
    await db_session.commit()
    
    response = await async_client.get(f"/trips/{trip.id}/summary")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_trip_summary_not_found(async_client: AsyncClient, authenticated_headers):
    """Test trip summary for non-existent trip"""
    response = await async_client.get(
        "/trips/99999/summary",
        headers=authenticated_headers
    )
    
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_trip_endpoint(async_client: AsyncClient, authenticated_headers):
    """Test creating trip via API"""
    trip_data = {
        "destination": "Paris, France",
        "start_date": datetime.utcnow().isoformat(),
        "end_date": (datetime.utcnow() + timedelta(days=5)).isoformat()
    }
    
    response = await async_client.post(
        "/trips",
        json=trip_data,
        headers=authenticated_headers
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "ok"
    assert data["data"]["destination"] == "Paris, France"
    assert data["data"]["status"] == "active"


@pytest.mark.asyncio
async def test_list_trips_endpoint(async_client: AsyncClient, authenticated_headers, db_session, test_user):
    """Test listing user trips"""
    from app.models.trip import Trip, TripStatus
    
    # Create multiple trips
    trip1 = Trip(
        user_id=test_user.id,
        destination="Trip 1",
        start_date=datetime.utcnow(),
        status=TripStatus.ACTIVE
    )
    trip2 = Trip(
        user_id=test_user.id,
        destination="Trip 2",
        start_date=datetime.utcnow(),
        status=TripStatus.COMPLETED
    )
    db_session.add_all([trip1, trip2])
    await db_session.commit()
    
    response = await async_client.get(
        "/trips",
        headers=authenticated_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 2


@pytest.mark.asyncio
async def test_complete_trip_endpoint(async_client: AsyncClient, authenticated_headers, db_session, test_user):
    """Test completing trip via API"""
    from app.models.trip import Trip, TripStatus
    
    trip = Trip(
        user_id=test_user.id,
        destination="To Complete",
        start_date=datetime.utcnow(),
        status=TripStatus.ACTIVE
    )
    db_session.add(trip)
    await db_session.commit()
    await db_session.refresh(trip)
    
    response = await async_client.post(
        f"/trips/{trip.id}/complete",
        headers=authenticated_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["status"] == "completed"
    assert data["data"]["end_date"] is not None
