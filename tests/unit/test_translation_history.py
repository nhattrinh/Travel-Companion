"""
Unit tests for translation history filtering by trip_id
"""
import pytest
from datetime import datetime
from app.models.translation import Translation
from app.models.trip import Trip, TripStatus
from sqlalchemy import select


@pytest.mark.asyncio
async def test_filter_translations_by_trip(db_session, test_user):
    """Test filtering translations by trip_id"""
    # Create two trips
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
        status=TripStatus.ACTIVE
    )
    db_session.add_all([trip1, trip2])
    await db_session.commit()
    
    # Create translations for trip1
    trans1 = Translation(
        user_id=test_user.id,
        trip_id=trip1.id,
        source_text="Hello from trip 1",
        target_text="안녕하세요",
        source_language="en",
        target_language="ko"
    )
    trans2 = Translation(
        user_id=test_user.id,
        trip_id=trip1.id,
        source_text="Goodbye from trip 1",
        target_text="안녕히 가세요",
        source_language="en",
        target_language="ko"
    )
    
    # Create translation for trip2
    trans3 = Translation(
        user_id=test_user.id,
        trip_id=trip2.id,
        source_text="Hello from trip 2",
        target_text="Xin chào",
        source_language="en",
        target_language="vi"
    )
    
    db_session.add_all([trans1, trans2, trans3])
    await db_session.commit()
    
    # Filter by trip1
    stmt = select(Translation).where(Translation.trip_id == trip1.id)
    result = await db_session.execute(stmt)
    trip1_translations = result.scalars().all()
    
    assert len(trip1_translations) == 2
    assert all(t.trip_id == trip1.id for t in trip1_translations)
    
    # Filter by trip2
    stmt = select(Translation).where(Translation.trip_id == trip2.id)
    result = await db_session.execute(stmt)
    trip2_translations = result.scalars().all()
    
    assert len(trip2_translations) == 1
    assert trip2_translations[0].trip_id == trip2.id


@pytest.mark.asyncio
async def test_translations_without_trip(db_session, test_user):
    """Test that translations can exist without trip_id"""
    trans = Translation(
        user_id=test_user.id,
        trip_id=None,
        source_text="No trip translation",
        target_text="여행 없음",
        source_language="en",
        target_language="ko"
    )
    db_session.add(trans)
    await db_session.commit()
    
    stmt = select(Translation).where(Translation.trip_id.is_(None))
    result = await db_session.execute(stmt)
    no_trip_translations = result.scalars().all()
    
    assert len(no_trip_translations) >= 1


@pytest.mark.asyncio
async def test_translation_cascade_delete_on_trip_deletion(db_session, test_user):
    """Test that translations are deleted when trip is deleted (cascade)"""
    trip = Trip(
        user_id=test_user.id,
        destination="Delete Me",
        start_date=datetime.utcnow(),
        status=TripStatus.ACTIVE
    )
    db_session.add(trip)
    await db_session.commit()
    trip_id = trip.id
    
    trans = Translation(
        user_id=test_user.id,
        trip_id=trip.id,
        source_text="Will be deleted",
        target_text="삭제될 것",
        source_language="en",
        target_language="ko"
    )
    db_session.add(trans)
    await db_session.commit()
    trans_id = trans.id
    
    # Delete trip
    await db_session.delete(trip)
    await db_session.commit()
    
    # Verify translation was cascaded
    stmt = select(Translation).where(Translation.id == trans_id)
    result = await db_session.execute(stmt)
    deleted_trans = result.scalar_one_or_none()
    
    assert deleted_trans is None
