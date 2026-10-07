"""Request and response models for the auth API (see openapi/auth.yaml)."""

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class UserOut(BaseModel):
    id: str
    email: EmailStr
