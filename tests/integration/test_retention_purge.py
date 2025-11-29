"""
Integration tests for retention purge logic
"""
import pytest
from datetime import datetime, timedelta
from app.services.trip_service import TripService
from app.models.trip import Trip, TripStatus
from app.models.translation import Translation
from sqlalchemy import select


@pytest.mark.asyncio
async def test_purge_old_completed_trips(db_session, test_user):
    """Test that old completed trips are purged"""
    service = TripService(db_session)
    
    # Create old completed trip (40 days ago)
    old_trip = Trip(
        user_id=test_user.id,
        destination="Old Trip",
        start_date=datetime.utcnow() - timedelta(days=45),
        end_date=datetime.utcnow() - timedelta(days=40),
        status=TripStatus.COMPLETED,
        created_at=datetime.utcnow() - timedelta(days=45),
        updated_at=datetime.utcnow() - timedelta(days=40)
    )
    
    # Create recent completed trip (10 days ago)
    recent_trip = Trip(
        user_id=test_user.id,
        destination="Recent Trip",
        start_date=datetime.utcnow() - timedelta(days=15),
        end_date=datetime.utcnow() - timedelta(days=10),
        status=TripStatus.COMPLETED,
        created_at=datetime.utcnow() - timedelta(days=15),
        updated_at=datetime.utcnow() - timedelta(days=10)
    )
    
    # Create active trip
    active_trip = Trip(
        user_id=test_user.id,
        destination="Active Trip",
        start_date=datetime.utcnow(),
        status=TripStatus.ACTIVE
    )
    
    db_session.add_all([old_trip, recent_trip, active_trip])
    await db_session.commit()
    
    old_trip_id = old_trip.id
    recent_trip_id = recent_trip.id
    active_trip_id = active_trip.id
    
    # Run purge with 30-day retention
    deleted_count = await service.purge_old_completed_trips(retention_days=30)
    
    # Should delete only old completed trip
    assert deleted_count == 1
    
    # Verify old trip deleted
    stmt = select(Trip).where(Trip.id == old_trip_id)
    result = await db_session.execute(stmt)
    assert result.scalar_one_or_none() is None
    
    # Verify recent trip still exists
    stmt = select(Trip).where(Trip.id == recent_trip_id)
    result = await db_session.execute(stmt)
    assert result.scalar_one_or_none() is not None
    
    # Verify active trip still exists
    stmt = select(Trip).where(Trip.id == active_trip_id)
    result = await db_session.execute(stmt)
    assert result.scalar_one_or_none() is not None


@pytest.mark.asyncio
async def test_purge_cascades_translations(db_session, test_user):
    """Test that purging trips cascades to translations"""
    service = TripService(db_session)
    
    # Create old completed trip
    old_trip = Trip(
        user_id=test_user.id,
        destination="Old Trip",
        start_date=datetime.utcnow() - timedelta(days=45),
        end_date=datetime.utcnow() - timedelta(days=40),
        status=TripStatus.COMPLETED,
        created_at=datetime.utcnow() - timedelta(days=45),
        updated_at=datetime.utcnow() - timedelta(days=40)
    )
    db_session.add(old_trip)
    await db_session.commit()
    
    # Add translation to old trip
    trans = Translation(
        user_id=test_user.id,
        trip_id=old_trip.id,
        source_text="Old translation",
        target_text="古い翻訳",
        source_language="en",
        target_language="ja"
    )
    db_session.add(trans)
    await db_session.commit()
    trans_id = trans.id
    
    # Run purge
    deleted_count = await service.purge_old_completed_trips(retention_days=30)
    assert deleted_count == 1
    
    # Verify translation was cascaded
    stmt = select(Translation).where(Translation.id == trans_id)
    result = await db_session.execute(stmt)
    assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_purge_respects_retention_days(db_session, test_user):
    """Test that retention_days parameter is respected"""
    service = TripService(db_session)
    
    # Create completed trip 15 days old
    trip = Trip(
        user_id=test_user.id,
        destination="15 Day Old Trip",
        start_date=datetime.utcnow() - timedelta(days=20),
        end_date=datetime.utcnow() - timedelta(days=15),
        status=TripStatus.COMPLETED,
        created_at=datetime.utcnow() - timedelta(days=20),
        updated_at=datetime.utcnow() - timedelta(days=15)
    )
    db_session.add(trip)
    await db_session.commit()
    trip_id = trip.id
    
    # Purge with 30-day retention - should NOT delete
    deleted_count = await service.purge_old_completed_trips(retention_days=30)
    assert deleted_count == 0
    
    # Verify trip still exists
    stmt = select(Trip).where(Trip.id == trip_id)
    result = await db_session.execute(stmt)
    assert result.scalar_one_or_none() is not None
    
    # Purge with 10-day retention - should delete
    deleted_count = await service.purge_old_completed_trips(retention_days=10)
    assert deleted_count == 1
    
    # Verify trip deleted
    stmt = select(Trip).where(Trip.id == trip_id)
    result = await db_session.execute(stmt)
    assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_purge_ignores_active_and_archived(db_session, test_user):
    """Test that purge only affects completed trips"""
    service = TripService(db_session)
    
    # Create old active trip
    active_trip = Trip(
        user_id=test_user.id,
        destination="Old Active",
        start_date=datetime.utcnow() - timedelta(days=45),
        status=TripStatus.ACTIVE,
        created_at=datetime.utcnow() - timedelta(days=45),
        updated_at=datetime.utcnow() - timedelta(days=40)
    )
    
    # Create old archived trip
    archived_trip = Trip(
        user_id=test_user.id,
        destination="Old Archived",
        start_date=datetime.utcnow() - timedelta(days=45),
        status=TripStatus.ARCHIVED,
        created_at=datetime.utcnow() - timedelta(days=45),
        updated_at=datetime.utcnow() - timedelta(days=40)
    )
    
    db_session.add_all([active_trip, archived_trip])
    await db_session.commit()
    
    # Run purge
    deleted_count = await service.purge_old_completed_trips(retention_days=30)
    
    # Should not delete active or archived trips
    assert deleted_count == 0
    
    # Verify both still exist
    stmt = select(Trip).where(Trip.user_id == test_user.id)
    result = await db_session.execute(stmt)
    remaining = result.scalars().all()
    assert len(remaining) == 2
