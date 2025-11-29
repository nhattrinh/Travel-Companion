from pydantic import BaseModel, Field
from typing import Optional

class TranslationCreate(BaseModel):
    source_text: str
    target_language: str
    source_language: Optional[str] = None

class TranslationRead(BaseModel):
    id: int
    source_text: str
    target_text: str
    source_language: str
    target_language: str
    confidence: Optional[int] = Field(None, description="Confidence *100")

class LiveFrameSegment(BaseModel):
    text: str
    translated: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

class LiveFrameResponse(BaseModel):
    segments: list[LiveFrameSegment]
    source_language: str
    target_language: str
    latency_ms: float
