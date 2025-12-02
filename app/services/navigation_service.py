"""Navigation service (Phase 4 US2)."""
import math
import json
from app.services.maps_client import MapsClient
from app.services.etiquette_data import get_context_notes
from app.core.cache_client import CacheClient

class NavigationService:
    def __init__(self, cache_client: CacheClient | None = None):
        self.maps = MapsClient()
        self.cache = cache_client or CacheClient()

    async def get_nearby_pois(self, user_lat: float, user_lon: float, radius_m: int = 1000):
        # Try cache first (geohash-based key for simplicity)
        from app.api.metrics_endpoints import metrics_collector
        
        cache_key = f"pois:{int(user_lat*100)}:{int(user_lon*100)}:{radius_m}"
        cached = await self.cache.get(cache_key)
        if cached:
            metrics_collector.record_cache_hit("navigation_pois")
            return json.loads(cached)
        
        metrics_collector.record_cache_miss("navigation_pois")
        
        raw = await self.maps.search_nearby_pois(user_lat, user_lon, radius_m)
        pois = []
        for p in raw:
            dist = self._haversine(user_lat, user_lon, p["lat"], p["lon"])
            notes = p.get("notes") or "; ".join(get_context_notes(p["category"]))
            pois.append({
                "name": p["name"], "category": p["category"],
                "latitude": p["lat"], "longitude": p["lon"],
                "etiquette_notes": notes, "distance_m": dist
            })
        result = sorted(pois, key=lambda x: x["distance_m"])
        
        # Cache for 5 minutes
        await self.cache.set(cache_key, json.dumps(result), ttl_seconds=300)
        return result

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi, dlam = math.radians(lat2-lat1), math.radians(lon2-lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
