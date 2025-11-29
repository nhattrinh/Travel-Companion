"""
Trip Service - Manages trip lifecycle and aggregations
"""
from typing import Optional, List
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.trip import Trip, TripStatus
from app.models.translation import Translation
from app.models.favorite import Favorite
from app.schemas.trip import TripCreate, TripUpdate, TripRead, TripSummaryResponse


class TripService:
    """Manages trip CRUD operations and summary aggregations"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_trip(self, user_id: int, trip_data: TripCreate) -> Trip:
        """
        Create a new trip for a user
        
        Args:
            user_id: User ID
            trip_data: Trip creation data
            
        Returns:
            Created trip
        """
        trip = Trip(
            user_id=user_id,
            destination=trip_data.destination,
            start_date=trip_data.start_date,
            end_date=trip_data.end_date,
            status=TripStatus.ACTIVE,
            metadata=trip_data.metadata or {},
        )
        self.db.add(trip)
        await self.db.commit()
        await self.db.refresh(trip)
        return trip

    async def get_trip(self, trip_id: int, user_id: int) -> Optional[Trip]:
        """
        Get a trip by ID (scoped to user)
        
        Args:
            trip_id: Trip ID
            user_id: User ID (for security scoping)
            
        Returns:
            Trip or None
        """
        stmt = select(Trip).where(
            Trip.id == trip_id,
            Trip.user_id == user_id
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_active_trip(self, user_id: int) -> Optional[Trip]:
        """
        Get user's active trip
        
        Args:
            user_id: User ID
            
        Returns:
            Active trip or None
        """
        stmt = select(Trip).where(
            Trip.user_id == user_id,
            Trip.status == TripStatus.ACTIVE
        ).order_by(Trip.start_date.desc())
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_user_trips(
        self,
        user_id: int,
        status: Optional[TripStatus] = None,
        limit: int = 20
    ) -> List[Trip]:
        """
        List user's trips with optional status filter
        
        Args:
            user_id: User ID
            status: Optional status filter
            limit: Max results
            
        Returns:
            List of trips
        """
        stmt = select(Trip).where(Trip.user_id == user_id)
        
        if status:
            stmt = stmt.where(Trip.status == status)
        
        stmt = stmt.order_by(Trip.start_date.desc()).limit(limit)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def update_trip(
        self,
        trip_id: int,
        user_id: int,
        trip_data: TripUpdate
    ) -> Optional[Trip]:
        """
        Update a trip
        
        Args:
            trip_id: Trip ID
            user_id: User ID (for security scoping)
            trip_data: Update data
            
        Returns:
            Updated trip or None
        """
        trip = await self.get_trip(trip_id, user_id)
        if not trip:
            return None
        
        update_fields = trip_data.model_dump(exclude_unset=True)
        for field, value in update_fields.items():
            setattr(trip, field, value)
        
        await self.db.commit()
        await self.db.refresh(trip)
        return trip

    async def complete_trip(self, trip_id: int, user_id: int) -> Optional[Trip]:
        """
        Mark a trip as completed
        
        Args:
            trip_id: Trip ID
            user_id: User ID
            
        Returns:
            Completed trip or None
        """
        trip = await self.get_trip(trip_id, user_id)
        if not trip:
            return None
        
        trip.status = TripStatus.COMPLETED
        trip.end_date = trip.end_date or datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(trip)
        return trip

    async def get_trip_summary(self, trip_id: int, user_id: int) -> Optional[TripSummaryResponse]:
        """
        Get trip with aggregated statistics
        
        Args:
            trip_id: Trip ID
            user_id: User ID
            
        Returns:
            Trip summary with counts and recent translations
        """
        trip = await self.get_trip(trip_id, user_id)
        if not trip:
            return None
        
        # Count translations
        stmt = select(func.count(Translation.id)).where(Translation.trip_id == trip_id)
        result = await self.db.execute(stmt)
        translation_count = result.scalar() or 0
        
        # Count favorites (during this trip)
        # Favorites don't have trip_id, so count based on created_at range
        stmt = select(func.count(Favorite.id)).where(
            Favorite.user_id == user_id,
            Favorite.created_at >= trip.start_date
        )
        if trip.end_date:
            stmt = stmt.where(Favorite.created_at <= trip.end_date)
        result = await self.db.execute(stmt)
        favorite_count = result.scalar() or 0
        
        # Get recent translations (last 5)
        stmt = (
            select(Translation)
            .where(Translation.trip_id == trip_id)
            .order_by(Translation.created_at.desc())
            .limit(5)
        )
        result = await self.db.execute(stmt)
        recent = result.scalars().all()
        
        recent_translations = [
            {
                "id": t.id,
                "source_text": t.source_text[:100],  # Truncate for preview
                "target_text": t.target_text[:100],
                "source_language": t.source_language,
                "target_language": t.target_language,
                "created_at": t.created_at.isoformat(),
            }
            for t in recent
        ]
        
        return TripSummaryResponse(
            trip=TripRead.model_validate(trip),
            translation_count=translation_count,
            favorite_count=favorite_count,
            recent_translations=recent_translations,
        )

    async def purge_old_completed_trips(self, retention_days: int = 30) -> int:
        """
        Delete completed trips older than retention period
        
        Args:
            retention_days: Days to retain completed trips
            
        Returns:
            Number of trips deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        stmt = select(Trip).where(
            Trip.status == TripStatus.COMPLETED,
            Trip.updated_at < cutoff_date
        )
        result = await self.db.execute(stmt)
        old_trips = result.scalars().all()
        
        count = len(old_trips)
        for trip in old_trips:
            await self.db.delete(trip)
        
        await self.db.commit()
        return count
