from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    preferences: Optional[Dict[str, Any]] = None

class UserRead(BaseModel):
    id: int
    email: EmailStr
    preferences: Optional[Dict[str, Any]] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenRefresh(BaseModel):
    refresh_token: str
