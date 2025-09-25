"""
ML prediction endpoints for risk assessment.

This module provides endpoints for making risk predictions using
machine learning models and retrieving prediction history.
"""

import time
import random
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.db.crud import create_prediction, get_recent_predictions, get_active_model
from app.db.models import User
from app.schemas.prediction import (
    PredictionInput, 
    PredictionResponse, 
    BatchPredictionRequest, 
    BatchPredictionResponse,
    WeatherData,
    RiskCategory
)
from app.core.websocket_manager import websocket_manager

router = APIRouter()

async def get_weather_data(lat: float, lon: float) -> WeatherData:
    """
    Fetch weather data from external API (mock implementation).
    
    In production, this would integrate with OpenWeatherMap or similar service.
    """
    # Mock weather data - replace with actual API call
    return WeatherData(
        temperature=round(20 + random.uniform(-10, 15), 1),
        humidity=round(40 + random.uniform(0, 40), 1),
        pressure=round(1000 + random.uniform(0, 50), 1),
        wind_speed=round(random.uniform(0, 25), 1),
        wind_direction=round(random.uniform(0, 360), 1),
        visibility=round(5 + random.uniform(0, 15), 1),
        condition=random.choice(["clear", "clouds", "rain", "snow"]),
        timestamp=time.time()
    )

async def run_ml_prediction(input_data: PredictionInput, weather: WeatherData) -> PredictionResponse:
    """
    Run ML model prediction (mock implementation).
    
    In production, this would load and run actual ML models.
    """
    start_time = time.time()
    
    # Mock prediction logic - replace with actual ML model
    base_risk = 0.1
    
    # Adjust risk based on weather
    if weather.condition == "rain":
        base_risk += 0.3
    elif weather.condition == "snow":
        base_risk += 0.2
    
    # Adjust based on input parameters
    if input_data.slope and input_data.slope > 30:
        base_risk += 0.2
    if input_data.rainfall and input_data.rainfall > 50:
        base_risk += 0.25
    if weather.wind_speed > 20:
        base_risk += 0.15
    
    # Add some randomness
    probability = min(base_risk + random.uniform(0, 0.3), 0.95)
    confidence = random.uniform(0.7, 0.95)
    
    # Determine category based on probability
    if probability < 0.3:
        category = RiskCategory.LOW_RISK
    elif probability < 0.6:
        category = RiskCategory.MEDIUM_RISK
    elif probability < 0.8:
        category = RiskCategory.HIGH_RISK
    else:
        category = RiskCategory.CRITICAL_RISK
    
    # Mock risk factors breakdown
    risk_factors = {
        "weather": min(0.4 if weather.condition in ["rain", "snow"] else 0.1, probability),
        "geological": min(0.3 if input_data.slope and input_data.slope > 30 else 0.1, probability),
        "environmental": random.uniform(0.05, 0.2),
        "historical": random.uniform(0.05, 0.15)
    }
    
    # Mock recommendations
    recommendations = []
    if probability > 0.7:
        recommendations.append("Avoid the area - high risk conditions detected")
        recommendations.append("Contact emergency services if in immediate danger")
    elif probability > 0.5:
        recommendations.append("Exercise caution and monitor conditions")
        recommendations.append("Consider alternative routes")
    else:
        recommendations.append("Normal precautions advised")
    
    processing_time = time.time() - start_time
    
    return PredictionResponse(
        probability=round(probability, 3),
        confidence=round(confidence, 3),
        category=category,
        lat=input_data.lat,
        lon=input_data.lon,
        weather=weather.condition,
        temperature=weather.temperature,
        processing_time=round(processing_time, 3),
        risk_factors=risk_factors,
        recommendations=recommendations
    )

@router.post("", response_model=PredictionResponse)
async def create_prediction_endpoint(
    input_data: PredictionInput,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate risk prediction for a specific location.
    
    This endpoint accepts geographical and environmental data and returns
    a risk assessment using ML models. Also fetches current weather data.
    
    - **lat**: Latitude coordinate (-90 to 90)
    - **lon**: Longitude coordinate (-180 to 180)
    - **rainfall**: Rainfall amount in mm (optional)
    - **temperature**: Temperature in Celsius (optional)
    - **slope**: Slope angle in degrees (optional)
    - **additional sensor data**: Optional environmental data
    """
    try:
        # Fetch current weather data
        weather = await get_weather_data(input_data.lat, input_data.lon)
        
        # Run ML prediction
        prediction = await run_ml_prediction(input_data, weather)
        
        # Save prediction to database
        await create_prediction(
            db=db,
            lat=input_data.lat,
            lon=input_data.lon,
            probability=prediction.probability,
            category=prediction.category.value,
            confidence=prediction.confidence,
            input_data=input_data.dict(),
            user_id=current_user.id,
            site_id=input_data.site_id
        )
        
        # Broadcast high-risk predictions via WebSocket
        if prediction.probability > 0.7:
            await websocket_manager.broadcast_prediction({
                "type": "high_risk_prediction",
                "prediction": prediction.dict(),
                "user_id": current_user.id
            })
        
        return prediction
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/batch", response_model=BatchPredictionResponse)
async def create_batch_predictions(
    batch_request: BatchPredictionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate multiple predictions in a single request.
    
    Useful for analyzing multiple locations or time series data.
    Maximum 100 predictions per batch request.
    """
    start_time = time.time()
    predictions = []
    errors = []
    
    for i, input_data in enumerate(batch_request.inputs):
        try:
            # Fetch weather data
            weather = await get_weather_data(input_data.lat, input_data.lon)
            
            # Run prediction
            prediction = await run_ml_prediction(input_data, weather)
            predictions.append(prediction)
            
            # Save to database
            await create_prediction(
                db=db,
                lat=input_data.lat,
                lon=input_data.lon,
                probability=prediction.probability,
                category=prediction.category.value,
                confidence=prediction.confidence,
                input_data=input_data.dict(),
                user_id=current_user.id,
                site_id=input_data.site_id
            )
            
        except Exception as e:
            errors.append(f"Prediction {i+1}: {str(e)}")
    
    processing_time = time.time() - start_time
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
        processing_time=round(processing_time, 3),
        errors=errors if errors else None
    )

@router.get("/history", response_model=List[dict])
async def get_prediction_history(
    limit: int = Query(default=50, ge=1, le=500, description="Number of predictions to retrieve"),
    site_id: Optional[int] = Query(default=None, description="Filter by site ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get prediction history for the current user.
    
    Returns recent predictions with optional filtering by site.
    """
    try:
        predictions = await get_recent_predictions(
            db=db,
            limit=limit,
            site_id=site_id
        )
        
        return [
            {
                "id": pred.id,
                "lat": pred.lat,
                "lon": pred.lon,
                "probability": pred.probability,
                "category": pred.category,
                "confidence": pred.confidence,
                "created_at": pred.created_at,
                "site_id": pred.site_id
            }
            for pred in predictions
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch prediction history: {str(e)}"
        )

@router.get("/weather")
async def get_location_weather(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude")
):
    """
    Get current weather data for a specific location.
    
    This endpoint fetches real-time weather information used in predictions.
    """
    try:
        weather = await get_weather_data(lat, lon)
        return weather
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch weather data: {str(e)}"
        )