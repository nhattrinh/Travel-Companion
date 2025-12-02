from pydantic import BaseModel

class FavoriteCreate(BaseModel):
    target_type: str  # phrase, poi, translation
    target_id: int

class FavoriteRead(BaseModel):
    id: int
    user_id: int
    target_type: str
    target_id: int
