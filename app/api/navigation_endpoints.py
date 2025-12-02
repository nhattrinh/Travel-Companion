"""Navigation endpoints (Phase 4 US2)."""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from pydantic import BaseModel
from app.services.navigation_service import NavigationService
from app.services.walking_directions_service import (
    get_walking_directions_service,
    WalkingRoute
)
from app.services.unsplash_service import get_unsplash_service
from app.core.metrics_navigation_phrase import record_poi_latency

router = APIRouter(prefix="/navigation", tags=["navigation"])
nav_service = NavigationService()


# ============================================================================
# Request/Response Models
# ============================================================================

class DirectionsRequest(BaseModel):
    """Request model for walking directions"""
    start_location: str
    end_location: str


class DirectionsResponse(BaseModel):
    """Response model for walking directions"""
    status: str
    data: Optional[WalkingRoute] = None
    error: Optional[str] = None


class ImageSearchRequest(BaseModel):
    """Request model for image search"""
    query: str
    max_images: int = 3


class ImageSearchResponse(BaseModel):
    """Response model for image search"""
    status: str
    images: List[str] = []
    error: Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/images", response_model=ImageSearchResponse)
async def search_location_images(request: ImageSearchRequest):
    """
    Search for images of a location or landmark.
    
    Args:
        request: ImageSearchRequest with query and max_images
        
    Returns:
        ImageSearchResponse with list of image URLs
    """
    if not request.query:
        return ImageSearchResponse(
            status="error",
            images=[],
            error="Query is required"
        )
    
    try:
        service = get_unsplash_service()
        images = await service.get_location_images(
            request.query,
            request.max_images
        )
        return ImageSearchResponse(
            status="ok",
            images=images,
            error=None
        )
    except Exception as e:
        return ImageSearchResponse(
            status="error",
            images=[],
            error=str(e)
        )

@router.post("/directions", response_model=DirectionsResponse)
async def get_walking_directions(request: DirectionsRequest):
    """
    Get walking directions between two locations using AI.
    
    This endpoint uses OpenAI to generate natural language walking
    directions with waypoints and step-by-step instructions.
    
    Args:
        request: DirectionsRequest with start_location and end_location
        
    Returns:
        DirectionsResponse with WalkingRoute data or error
    """
    if not request.start_location or not request.end_location:
        return DirectionsResponse(
            status="error",
            data=None,
            error="Both start_location and end_location are required"
        )
    
    try:
        directions_service = get_walking_directions_service()
        route = await directions_service.get_walking_directions(
            start_location=request.start_location,
            end_location=request.end_location
        )
        return DirectionsResponse(
            status="ok",
            data=route,
            error=None
        )
    except ValueError as e:
        # API key not configured or invalid response
        return DirectionsResponse(
            status="error",
            data=None,
            error=str(e)
        )
    except Exception as e:
        # Unexpected error
        return DirectionsResponse(
            status="error",
            data=None,
            error=f"Failed to get directions: {str(e)}"
        )

 
@router.get("/pois")
async def get_pois(
    latitude: Optional[float] = Query(None),
    longitude: Optional[float] = Query(None),
    radius_m: int = Query(1000),
    # Legacy parameter names for backward compatibility
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    radius: Optional[int] = Query(None)
):
    # Support both new (latitude/longitude) and legacy (lat/lon) parameter names
    final_lat = latitude if latitude is not None else lat
    final_lon = longitude if longitude is not None else lon
    final_radius = radius_m if radius is None else radius
    
    if final_lat is None or final_lon is None:
        return {"status": "error", "data": None, "error": "latitude and longitude are required"}
    
    with record_poi_latency():
        pois = await nav_service.get_nearby_pois(final_lat, final_lon, final_radius)
    payload = {"pois": pois}
    return {"status": "ok", "data": payload, "error": None}
