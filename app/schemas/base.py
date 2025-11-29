from pydantic import BaseModel
from typing import Any, Optional

class Envelope(BaseModel):
    status: str
    data: Any | None = None
    error: Optional[str] = None

class Message(BaseModel):
    message: str
