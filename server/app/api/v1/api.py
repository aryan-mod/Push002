"""
Main API router that combines all v1 API endpoints.

This module serves as the main router for API version 1,
combining all endpoint routers into a single API router.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, predictions, alerts, sites, models

# Create main API router for version 1
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(predictions.router, prefix="/predict", tags=["predictions"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["alerts"])
api_router.include_router(sites.router, prefix="/sites", tags=["sites"])
api_router.include_router(models.router, prefix="/models", tags=["models"])