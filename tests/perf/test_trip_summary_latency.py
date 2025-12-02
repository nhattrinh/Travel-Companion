"""
Performance tests for trip summary latency
"""
import pytest
import time
from datetime import datetime, timedelta
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_trip_summary_latency_p95(async_client: AsyncClient, authenticated_headers, db_session, test_user):
    """Test that trip summary meets p95 <= 500ms requirement"""
    from app.models.trip import Trip, TripStatus
    from app.models.translation import Translation
    
    # Create trip with realistic data (20 translations)
    trip = Trip(
        user_id=test_user.id,
        destination="Performance Test Trip",
        start_date=datetime.utcnow(),
        status=TripStatus.ACTIVE
    )
    db_session.add(trip)
    await db_session.commit()
    await db_session.refresh(trip)
    
    # Create 20 translations
    translations = [
        Translation(
            user_id=test_user.id,
            trip_id=trip.id,
            source_text=f"Source text {i}" * 10,  # ~150 chars
            target_text=f"Target text {i}" * 10,
            source_language="en",
            target_language="ja"
        )
        for i in range(20)
    ]
    db_session.add_all(translations)
    await db_session.commit()
    
    # Measure latency over 20 requests
    latencies = []
    for _ in range(20):
        start = time.perf_counter()
        response = await async_client.get(
            f"/trips/{trip.id}/summary",
            headers=authenticated_headers
        )
        end = time.perf_counter()
        
        assert response.status_code == 200
        latencies.append((end - start) * 1000)  # Convert to ms
    
    # Calculate p95
    latencies.sort()
    p95_index = int(len(latencies) * 0.95)
    p95_latency = latencies[p95_index]
    
    print(f"\nTrip summary latencies (ms): {latencies}")
    print(f"p95 latency: {p95_latency:.2f}ms")
    
    # Assert p95 <= 500ms
    assert p95_latency <= 500.0, f"p95 latency {p95_latency:.2f}ms exceeds 500ms threshold"


@pytest.mark.asyncio
async def test_list_trips_latency(async_client: AsyncClient, authenticated_headers, db_session, test_user):
    """Test list trips endpoint performance"""
    from app.models.trip import Trip, TripStatus
    
    # Create 10 trips
    trips = [
        Trip(
            user_id=test_user.id,
            destination=f"Trip {i}",
            start_date=datetime.utcnow() - timedelta(days=i*10),
            status=TripStatus.ACTIVE if i % 2 == 0 else TripStatus.COMPLETED
        )
        for i in range(10)
    ]
    db_session.add_all(trips)
    await db_session.commit()
    
    # Measure latency
    latencies = []
    for _ in range(10):
        start = time.perf_counter()
        response = await async_client.get(
            "/trips",
            headers=authenticated_headers
        )
        latency = (time.perf_counter() - start) * 1000
        
        assert response.status_code == 200
        latencies.append(latency)
    
    p95_index = int(len(latencies) * 0.95)
    p95_latency = latencies[p95_index] if latencies else 0
    
    print(f"\nList trips latencies (ms): {latencies}")
    print(f"p95 latency: {p95_latency:.2f}ms")
    
    # Should be fast (<200ms p95)
    assert p95_latency < 200.0


@pytest.mark.asyncio
async def test_create_trip_latency(async_client: AsyncClient, authenticated_headers):
    """Test trip creation performance"""
    trip_data = {
        "destination": "Perf Test",
        "start_date": datetime.utcnow().isoformat(),
        "end_date": (datetime.utcnow() + timedelta(days=7)).isoformat()
    }
    
    latencies = []
    for i in range(10):
        trip_data["destination"] = f"Perf Test {i}"
        
        start = time.perf_counter()
        response = await async_client.post(
            "/trips",
            json=trip_data,
            headers=authenticated_headers
        )
        latency = (time.perf_counter() - start) * 1000
        
        assert response.status_code == 201
        latencies.append(latency)
    
    p95_index = int(len(latencies) * 0.95)
    p95_latency = latencies[p95_index]
    
    print(f"\nCreate trip latencies (ms): {latencies}")
    print(f"p95 latency: {p95_latency:.2f}ms")
    
    # Should be fast (<300ms p95)
    assert p95_latency < 300.0


@pytest.mark.asyncio
async def test_complete_trip_latency(async_client: AsyncClient, authenticated_headers, db_session, test_user):
    """Test trip completion performance"""
    from app.models.trip import Trip, TripStatus
    
    # Create trips to complete
    trips = [
        Trip(
            user_id=test_user.id,
            destination=f"To Complete {i}",
            start_date=datetime.utcnow(),
            status=TripStatus.ACTIVE
        )
        for i in range(10)
    ]
    db_session.add_all(trips)
    await db_session.commit()
    
    latencies = []
    for trip in trips:
        start = time.perf_counter()
        response = await async_client.post(
            f"/trips/{trip.id}/complete",
            headers=authenticated_headers
        )
        latency = (time.perf_counter() - start) * 1000
        
        assert response.status_code == 200
        latencies.append(latency)
    
    p95_index = int(len(latencies) * 0.95)
    p95_latency = latencies[p95_index]
    
    print(f"\nComplete trip latencies (ms): {latencies}")
    print(f"p95 latency: {p95_latency:.2f}ms")
    
    # Should be fast (<200ms p95)
    assert p95_latency < 200.0
