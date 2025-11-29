"""
Unit tests for trip service lifecycle operations
"""
import pytest
from datetime import datetime, timedelta
from app.services.trip_service import TripService
from app.models.trip import Trip, TripStatus
from app.schemas.trip import TripCreate, TripUpdate


@pytest.mark.asyncio
async def test_create_trip(db_session, test_user):
    """Test creating a new trip"""
    service = TripService(db_session)
    
    trip_data = TripCreate(
        destination="Tokyo, Japan",
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow() + timedelta(days=7)
    )
    
    trip = await service.create_trip(test_user.id, trip_data)
    
    assert trip.id is not None
    assert trip.user_id == test_user.id
    assert trip.destination == "Tokyo, Japan"
    assert trip.status == TripStatus.ACTIVE


@pytest.mark.asyncio
async def test_get_active_trip(db_session, test_user):
    """Test retrieving active trip"""
    service = TripService(db_session)
    
    # Create active trip
    trip_data = TripCreate(
        destination="Paris, France",
        start_date=datetime.utcnow()
    )
    created_trip = await service.create_trip(test_user.id, trip_data)
    
    # Retrieve active trip
    active_trip = await service.get_active_trip(test_user.id)
    
    assert active_trip is not None
    assert active_trip.id == created_trip.id
    assert active_trip.status == TripStatus.ACTIVE


@pytest.mark.asyncio
async def test_complete_trip(db_session, test_user):
    """Test completing a trip"""
    service = TripService(db_session)
    
    # Create trip
    trip_data = TripCreate(
        destination="London, UK",
        start_date=datetime.utcnow()
    )
    trip = await service.create_trip(test_user.id, trip_data)
    
    # Complete trip
    completed = await service.complete_trip(trip.id, test_user.id)
    
    assert completed is not None
    assert completed.status == TripStatus.COMPLETED
    assert completed.end_date is not None


@pytest.mark.asyncio
async def test_update_trip(db_session, test_user):
    """Test updating trip details"""
    service = TripService(db_session)
    
    # Create trip
    trip_data = TripCreate(
        destination="Berlin, Germany",
        start_date=datetime.utcnow()
    )
    trip = await service.create_trip(test_user.id, trip_data)
    
    # Update destination
    update_data = TripUpdate(destination="Munich, Germany")
    updated = await service.update_trip(trip.id, test_user.id, update_data)
    
    assert updated is not None
    assert updated.destination == "Munich, Germany"


@pytest.mark.asyncio
async def test_list_user_trips(db_session, test_user):
    """Test listing user's trips with status filter"""
    service = TripService(db_session)
    
    # Create multiple trips
    trip1_data = TripCreate(destination="Trip 1", start_date=datetime.utcnow())
    trip2_data = TripCreate(destination="Trip 2", start_date=datetime.utcnow())
    
    trip1 = await service.create_trip(test_user.id, trip1_data)
    trip2 = await service.create_trip(test_user.id, trip2_data)
    
    # Complete one trip
    await service.complete_trip(trip1.id, test_user.id)
    
    # List all trips
    all_trips = await service.list_user_trips(test_user.id)
    assert len(all_trips) == 2
    
    # List only active trips
    active_trips = await service.list_user_trips(test_user.id, status=TripStatus.ACTIVE)
    assert len(active_trips) == 1
    assert active_trips[0].id == trip2.id
    
    # List only completed trips
    completed_trips = await service.list_user_trips(test_user.id, status=TripStatus.COMPLETED)
    assert len(completed_trips) == 1
    assert completed_trips[0].id == trip1.id


@pytest.mark.asyncio
async def test_get_trip_security_scoping(db_session, test_user):
    """Test that users can only access their own trips"""
    from app.models.user import User
    
    service = TripService(db_session)
    
    # Create another user
    other_user = User(email="other@example.com", hashed_password="hash456")
    db_session.add(other_user)
    await db_session.commit()
    
    # Create trip for test_user
    trip_data = TripCreate(destination="Private Trip", start_date=datetime.utcnow())
    trip = await service.create_trip(test_user.id, trip_data)
    
    # Try to access with wrong user_id
    accessed = await service.get_trip(trip.id, other_user.id)
    assert accessed is None


@pytest.mark.asyncio
async def test_trip_with_metadata(db_session, test_user):
    """Test trip creation with metadata"""
    service = TripService(db_session)
    
    trip_data = TripCreate(
        destination="Barcelona, Spain",
        start_date=datetime.utcnow(),
        metadata={"notes": "Food tour", "budget": "medium"}
    )
    
    trip = await service.create_trip(test_user.id, trip_data)
    
    assert trip.metadata is not None
    assert trip.metadata["notes"] == "Food tour"
    assert trip.metadata["budget"] == "medium"


@pytest.mark.asyncio
async def test_complete_trip_sets_end_date(db_session, test_user):
    """Test that completing trip sets end_date if not already set"""
    service = TripService(db_session)
    
    # Create trip without end_date
    trip_data = TripCreate(
        destination="Rome, Italy",
        start_date=datetime.utcnow()
    )
    trip = await service.create_trip(test_user.id, trip_data)
    assert trip.end_date is None
    
    # Complete trip
    completed = await service.complete_trip(trip.id, test_user.id)
    
    assert completed.end_date is not None
    assert completed.end_date >= trip.start_date
