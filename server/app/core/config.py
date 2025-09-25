"""
Configuration settings for the FastAPI application.

This module handles all configuration settings including database URLs,
JWT secrets, and other environment-specific configurations.
"""

import os
from pydantic import BaseModel
from typing import Optional

class Settings(BaseModel):
    """
    Application settings loaded from environment variables.
    Uses Pydantic for validation and type safety.
    """
    
    # Database configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://localhost/geomindflow")
    
    # JWT Configuration
    SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-here-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "GeoMindFlow API"
    
    # External API Keys (for future integrations)
    OPENWEATHER_API_KEY: Optional[str] = os.getenv("OPENWEATHER_API_KEY")
    TWILIO_ACCOUNT_SID: Optional[str] = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN: Optional[str] = os.getenv("TWILIO_AUTH_TOKEN")
    
    # Security settings
    CORS_ORIGINS: list = ["*"]  # In production, specify allowed origins

# Global settings instance
settings = Settings()