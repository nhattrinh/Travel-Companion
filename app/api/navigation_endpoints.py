"""Navigation endpoints (Phase 4 US2)."""
from fastapi import APIRouter, Query
from typing import Optional
from app.services.navigation_service import NavigationService
from app.core.metrics_navigation_phrase import record_poi_latency

router = APIRouter(prefix="/navigation", tags=["navigation"])
nav_service = NavigationService()

 
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
