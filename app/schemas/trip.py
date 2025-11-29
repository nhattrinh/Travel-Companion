"""
Trip schemas for API requests/responses
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from app.models.trip import TripStatus


class TripCreate(BaseModel):
    """Schema for creating a new trip"""
    destination: str = Field(..., min_length=1, max_length=255)
    start_date: datetime
    end_date: Optional[datetime] = None
    metadata: Optional[dict] = None


class TripUpdate(BaseModel):
    """Schema for updating a trip"""
    destination: Optional[str] = Field(None, min_length=1, max_length=255)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: Optional[TripStatus] = None
    metadata: Optional[dict] = None


class TripRead(BaseModel):
    """Schema for trip read response"""
    id: int
    user_id: int
    destination: str
    start_date: datetime
    end_date: Optional[datetime]
    status: TripStatus
    metadata: Optional[dict]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TripSummaryResponse(BaseModel):
    """Schema for trip summary with aggregated statistics"""
    trip: TripRead
    translation_count: int = 0
    favorite_count: int = 0
    recent_translations: list[dict] = []  # Last 5 translations

    class Config:
        from_attributes = True
