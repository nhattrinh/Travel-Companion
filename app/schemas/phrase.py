from pydantic import BaseModel
from typing import Any


class PhraseRead(BaseModel):
    id: int
    canonical_text: str
    translations: dict[str, str]
    phonetic: str | None = None
    context_category: str

class PhraseCreate(BaseModel):
    canonical_text: str
    translations: dict[str, str]
    phonetic: str | None = None
    context_category: str

class PhraseSuggestionResponse(BaseModel):
    context: str
    target_language: str
    suggestions: list[dict[str, Any]]
