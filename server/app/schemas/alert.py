"""
Pydantic schemas for Alert-related API operations.

This module defines data validation schemas for creating, updating,
and retrieving alert information.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

class AlertCategory(str, Enum):
    """Enumeration of alert risk categories."""
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL_RISK = "critical_risk"

class AlertSeverity(str, Enum):
    """Enumeration of alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    """Enumeration of alert status values."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"

class AlertBase(BaseModel):
    """Base alert schema with common fields."""
    probability: float = Field(..., ge=0.0, le=1.0, description="Risk probability (0.0 to 1.0)")
    category: AlertCategory = Field(..., description="Risk category classification")
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude coordinate")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude coordinate")
    weather: Optional[str] = Field(None, description="Weather condition")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    humidity: Optional[float] = Field(None, ge=0.0, le=100.0, description="Humidity percentage")
    wind_speed: Optional[float] = Field(None, ge=0.0, description="Wind speed in km/h")
    title: Optional[str] = Field(None, description="Alert title")
    message: Optional[str] = Field(None, description="Detailed alert message")
    severity: AlertSeverity = Field(default=AlertSeverity.MEDIUM, description="Alert severity level")

class AlertCreate(AlertBase):
    """Schema for creating new alerts."""
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Alert timestamp")
    site_id: Optional[int] = Field(None, description="Associated monitoring site ID")

class AlertUpdate(BaseModel):
    """Schema for updating existing alerts."""
    status: AlertStatus = Field(..., description="Updated alert status")
    message: Optional[str] = Field(None, description="Update message or notes")

class AlertInDB(AlertBase):
    """Schema representing alert data stored in database."""
    id: int
    timestamp: datetime
    status: AlertStatus
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    user_id: Optional[int] = None
    site_id: Optional[int] = None

    class Config:
        from_attributes = True

class Alert(AlertInDB):
    """Complete alert schema for API responses."""
    user: Optional[dict] = Field(None, description="User who created the alert")
    site: Optional[dict] = Field(None, description="Associated monitoring site")

class AlertStats(BaseModel):
    """Schema for alert statistics."""
    total_alerts: int = Field(..., description="Total number of alerts")
    active_alerts: int = Field(..., description="Number of active alerts")
    high_severity_alerts: int = Field(..., description="Number of high/critical severity alerts")
    category_breakdown: dict = Field(..., description="Breakdown of alerts by category")

class LocationAlerts(BaseModel):
    """Schema for location-based alert queries."""
    lat: float = Field(..., ge=-90.0, le=90.0, description="Center latitude")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Center longitude")
    radius: float = Field(default=10000.0, ge=0.0, description="Search radius in meters")
    limit: int = Field(default=50, ge=1, le=500, description="Maximum number of results")