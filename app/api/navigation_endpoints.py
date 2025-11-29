"""Navigation endpoints (Phase 4 US2)."""
from fastapi import APIRouter
from app.services.navigation_service import NavigationService
from app.core.metrics_navigation_phrase import record_poi_latency
from app.schemas.poi import POIRead

router = APIRouter(prefix="/navigation", tags=["navigation"])
nav_service = NavigationService()

 
@router.get("/pois")
async def get_pois(lat: float, lon: float, radius: int = 1000):
    with record_poi_latency():
        pois = await nav_service.get_nearby_pois(lat, lon, radius)
    payload = {"pois": [POIRead(**p).dict() for p in pois]}
    return {"status": "ok", "data": payload, "error": None}
