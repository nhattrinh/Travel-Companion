from pydantic import BaseModel
from typing import Optional

class POIRead(BaseModel):
    id: Optional[int] = None
    name: str
    category: str
    latitude: float
    longitude: float
    etiquette_notes: Optional[str] = None
    distance_m: Optional[float] = None

class POIListResponse(BaseModel):
    pois: list[POIRead]
