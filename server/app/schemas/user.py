"""
Pydantic schemas for User-related API operations.

This module defines the data validation schemas for user registration,
authentication, and user information responses.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    """Base user schema with common fields."""
    email: EmailStr = Field(..., description="User's email address")
    full_name: Optional[str] = Field(None, description="User's full name")
    role: str = Field(default="user", description="User role (user/admin)")

class UserCreate(UserBase):
    """Schema for user registration."""
    password: str = Field(..., min_length=8, description="User password (minimum 8 characters)")

class UserUpdate(BaseModel):
    """Schema for user profile updates."""
    full_name: Optional[str] = Field(None, description="Updated full name")
    email: Optional[EmailStr] = Field(None, description="Updated email address")

class UserInDB(UserBase):
    """Schema representing user data stored in database."""
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class User(UserInDB):
    """Public user schema for API responses (excludes sensitive data)."""
    pass

class UserLogin(BaseModel):
    """Schema for user login requests."""
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")

class Token(BaseModel):
    """Schema for JWT token response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    user: User = Field(..., description="Authenticated user information")

class TokenData(BaseModel):
    """Schema for token payload data."""
    email: Optional[str] = None