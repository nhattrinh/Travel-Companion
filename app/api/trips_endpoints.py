"""
Trip API endpoints - Trip lifecycle and summary retrieval
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.trip import TripStatus
from app.schemas.trip import TripCreate, TripRead, TripUpdate, TripSummaryResponse
from app.schemas.api_models import Envelope
from app.services.trip_service import TripService

router = APIRouter(prefix="/trips", tags=["trips"])


@router.post("", response_model=Envelope[TripRead], status_code=status.HTTP_201_CREATED)
async def create_trip(
    trip_data: TripCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new trip
    
    - **destination**: Trip destination (e.g., "Tokyo, Japan")
    - **start_date**: Trip start date/time
    - **end_date**: Optional trip end date/time
    - **metadata**: Optional flexible metadata (notes, preferences)
    """
    service = TripService(db)
    trip = await service.create_trip(current_user.id, trip_data)
    
    return Envelope(
        status="ok",
        data=TripRead.model_validate(trip)
    )


@router.get("", response_model=Envelope[list[TripRead]])
async def list_trips(
    status_filter: Optional[TripStatus] = Query(None, alias="status"),
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List user's trips with optional status filter
    
    - **status**: Optional filter (active, completed, archived)
    - **limit**: Max results (default 20, max 100)
    """
    service = TripService(db)
    trips = await service.list_user_trips(current_user.id, status_filter, limit)
    
    trip_list = [TripRead.model_validate(t) for t in trips]
    
    return Envelope(status="ok", data=trip_list)


@router.get("/active", response_model=Envelope[Optional[TripRead]])
async def get_active_trip(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get user's current active trip
    """
    service = TripService(db)
    trip = await service.get_active_trip(current_user.id)
    
    if not trip:
        return Envelope(status="ok", data=None)
    
    return Envelope(
        status="ok",
        data=TripRead.model_validate(trip)
    )


@router.get("/{trip_id}", response_model=Envelope[TripRead])
async def get_trip(
    trip_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get a specific trip by ID
    """
    service = TripService(db)
    trip = await service.get_trip(trip_id, current_user.id)
    
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    return Envelope(
        status="ok",
        data=TripRead.model_validate(trip)
    )


@router.get("/{trip_id}/summary", response_model=Envelope[TripSummaryResponse])
async def get_trip_summary(
    trip_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get trip summary with aggregated statistics
    
    Returns trip details plus:
    - Translation count
    - Favorite count (during trip period)
    - Recent translations (last 5)
    """
    service = TripService(db)
    summary = await service.get_trip_summary(trip_id, current_user.id)
    
    if not summary:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    return Envelope(status="ok", data=summary)


@router.put("/{trip_id}", response_model=Envelope[TripRead])
async def update_trip(
    trip_id: int,
    trip_data: TripUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update a trip
    
    All fields optional - only provided fields will be updated
    """
    service = TripService(db)
    trip = await service.update_trip(trip_id, current_user.id, trip_data)
    
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    return Envelope(
        status="ok",
        data=TripRead.model_validate(trip)
    )


@router.post("/{trip_id}/complete", response_model=Envelope[TripRead])
async def complete_trip(
    trip_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Mark a trip as completed
    
    Sets status to 'completed' and end_date to now if not already set
    """
    service = TripService(db)
    trip = await service.complete_trip(trip_id, current_user.id)
    
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    return Envelope(
        status="ok",
        data=TripRead.model_validate(trip)
    )
