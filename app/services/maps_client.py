"""Mock maps provider client (Phase 4 US2)."""

class MapsClient:
    async def search_nearby_pois(self, lat: float, lon: float, radius_m: int = 1000):
        # Mock POI data
        return [
            {"name": "Tokyo Station", "category": "transit", "lat": lat+0.001, "lon": lon+0.001, "notes": "Queue in orderly lines"},
            {"name": "Sushi Restaurant", "category": "restaurant", "lat": lat-0.001, "lon": lon, "notes": "Shoes off at entrance"},
        ]
