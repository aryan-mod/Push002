"""
Pydantic schemas for Site-related API operations.

This module defines data validation schemas for monitoring sites,
including geographical data validation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class SiteBase(BaseModel):
    """Base site schema with common fields."""
    name: str = Field(..., min_length=1, max_length=255, description="Site name")
    description: Optional[str] = Field(None, description="Site description")
    lat: float = Field(..., ge=-90.0, le=90.0, description="Site latitude")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Site longitude")
    radius: float = Field(default=1000.0, ge=0.0, description="Monitoring radius in meters")
    risk_type: Optional[str] = Field(None, description="Type of risk monitored")

class SiteCreate(SiteBase):
    """Schema for creating new monitoring sites."""
    pass

class SiteUpdate(BaseModel):
    """Schema for updating existing sites."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None)
    radius: Optional[float] = Field(None, ge=0.0)
    risk_type: Optional[str] = Field(None)
    is_active: Optional[bool] = Field(None, description="Site active status")

class SiteInDB(SiteBase):
    """Schema representing site data stored in database."""
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class Site(SiteInDB):
    """Complete site schema for API responses."""
    pass

class SiteWithStats(Site):
    """Site schema with additional statistics."""
    total_alerts: int = Field(default=0, description="Total alerts from this site")
    active_alerts: int = Field(default=0, description="Active alerts from this site")
    recent_predictions: int = Field(default=0, description="Recent predictions count")

class LocationSearch(BaseModel):
    """Schema for location-based site searches."""
    lat: float = Field(..., ge=-90.0, le=90.0, description="Search center latitude")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Search center longitude")
    radius: float = Field(default=50000.0, ge=0.0, description="Search radius in meters")
    limit: int = Field(default=50, ge=1, le=500, description="Maximum number of results")