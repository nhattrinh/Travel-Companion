from pydantic import BaseModel

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
    phrases: list[PhraseRead]
    context: str
