"""
Pydantic schemas for ML Prediction-related API operations.

This module defines data validation schemas for ML prediction requests
and responses, including input validation and output formatting.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class RiskCategory(str, Enum):
    """Enumeration of risk prediction categories."""
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL_RISK = "critical_risk"

class PredictionInput(BaseModel):
    """Schema for ML prediction input data."""
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude coordinate")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude coordinate")
    
    # Environmental data
    rainfall: Optional[float] = Field(None, ge=0.0, description="Rainfall amount in mm")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    humidity: Optional[float] = Field(None, ge=0.0, le=100.0, description="Humidity percentage")
    wind_speed: Optional[float] = Field(None, ge=0.0, description="Wind speed in km/h")
    pressure: Optional[float] = Field(None, ge=0.0, description="Atmospheric pressure in hPa")
    
    # Geological data
    slope: Optional[float] = Field(None, ge=0.0, le=90.0, description="Slope angle in degrees")
    soil_moisture: Optional[float] = Field(None, ge=0.0, le=100.0, description="Soil moisture percentage")
    vibration: Optional[float] = Field(None, ge=0.0, description="Vibration sensor reading")
    
    # Optional metadata
    site_id: Optional[int] = Field(None, description="Associated site ID")
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Additional sensor data")

class PredictionResponse(BaseModel):
    """Schema for ML prediction response."""
    probability: float = Field(..., ge=0.0, le=1.0, description="Risk probability (0.0 to 1.0)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    category: RiskCategory = Field(..., description="Predicted risk category")
    
    # Input coordinates
    lat: float = Field(..., description="Input latitude")
    lon: float = Field(..., description="Input longitude")
    
    # Weather information (from external API)
    weather: Optional[str] = Field(None, description="Current weather condition")
    temperature: Optional[float] = Field(None, description="Current temperature in Celsius")
    
    # Prediction metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    model_version: Optional[str] = Field(None, description="Model version used")
    processing_time: Optional[float] = Field(None, description="Prediction processing time in seconds")
    
    # Risk factors breakdown
    risk_factors: Optional[Dict[str, float]] = Field(
        None, 
        description="Breakdown of contributing risk factors"
    )
    
    # Recommendations
    recommendations: Optional[List[str]] = Field(
        None,
        description="Risk mitigation recommendations"
    )

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests."""
    inputs: List[PredictionInput] = Field(..., min_length=1, max_length=100, description="List of prediction inputs")
    model_type: Optional[str] = Field(default="fusion", description="ML model type to use")

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction responses."""
    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    total_processed: int = Field(..., description="Total number of predictions processed")
    processing_time: float = Field(..., description="Total processing time in seconds")
    errors: Optional[List[str]] = Field(None, description="Any errors encountered during processing")

class ModelInfo(BaseModel):
    """Schema for ML model information."""
    id: int = Field(..., description="Model ID")
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type (cnn, lstm, fusion, etc.)")
    accuracy: Optional[float] = Field(None, description="Model accuracy score")
    is_active: bool = Field(..., description="Whether model is currently active")
    created_at: datetime = Field(..., description="Model creation timestamp")

class ModelPerformance(BaseModel):
    """Schema for model performance metrics."""
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Model accuracy")
    precision: float = Field(..., ge=0.0, le=1.0, description="Model precision")
    recall: float = Field(..., ge=0.0, le=1.0, description="Model recall")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    
class WeatherData(BaseModel):
    """Schema for weather information from external APIs."""
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Humidity percentage")
    pressure: float = Field(..., description="Atmospheric pressure in hPa")
    wind_speed: float = Field(..., ge=0.0, description="Wind speed in km/h")
    wind_direction: float = Field(..., ge=0.0, lt=360.0, description="Wind direction in degrees")
    visibility: float = Field(..., ge=0.0, description="Visibility in kilometers")
    condition: str = Field(..., description="Weather condition description")
    timestamp: datetime = Field(..., description="Weather data timestamp")