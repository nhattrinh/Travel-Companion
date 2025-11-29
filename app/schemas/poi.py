from pydantic import BaseModel

class POIRead(BaseModel):
    id: int
    name: str
    category: str
    latitude: float
    longitude: float
    etiquette_notes: str | None = None
    distance_m: float | None = None

class POIListResponse(BaseModel):
    pois: list[POIRead]
