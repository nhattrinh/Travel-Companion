"""Navigation endpoints (Phase 4 US2)."""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from pydantic import BaseModel
from app.services.navigation_service import NavigationService
from app.services.walking_directions_service import (
    get_walking_directions_service,
    WalkingRoute
)
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


# ============================================================================
# Endpoints
# ============================================================================

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
