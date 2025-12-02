from pydantic import BaseModel
from typing import Any, Optional, Generic, TypeVar

T = TypeVar('T')


class Envelope(BaseModel, Generic[T]):
    status: str
    data: Optional[T] = None
    error: Optional[str] = None


class Message(BaseModel):
    message: str
