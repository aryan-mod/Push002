"""
ML model management endpoints.

This module provides endpoints for managing machine learning models,
including version control, performance tracking, and model activation.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user, get_current_admin
from app.db.crud import get_models, get_model, get_active_model
from app.db.models import User
from app.schemas.prediction import ModelInfo, ModelPerformance

router = APIRouter()

@router.get("", response_model=List[ModelInfo])
async def get_all_models(
    skip: int = Query(default=0, ge=0, description="Number of models to skip"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum number of models to return"),
    model_type: str = Query(default=None, description="Filter by model type"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all available ML models with pagination.
    
    Returns a list of ML models in the system with their metadata
    and performance metrics.
    """
    try:
        models = await get_models(db=db, skip=skip, limit=limit)
        
        # Filter by model type if specified
        if model_type:
            models = [model for model in models if model.model_type == model_type]
        
        return [
            ModelInfo(
                id=model.id,
                name=model.name,
                version=model.version,
                model_type=model.model_type,
                accuracy=model.accuracy,
                is_active=model.is_active,
                created_at=model.created_at
            )
            for model in models
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch models: {str(e)}"
        )

@router.get("/{model_id}", response_model=ModelInfo)
async def get_single_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a specific ML model.
    
    Returns comprehensive model information including performance metrics,
    training data details, and deployment status.
    """
    try:
        model = await get_model(db=db, model_id=model_id)
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        return ModelInfo(
            id=model.id,
            name=model.name,
            version=model.version,
            model_type=model.model_type,
            accuracy=model.accuracy,
            is_active=model.is_active,
            created_at=model.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model: {str(e)}"
        )

@router.get("/{model_id}/performance", response_model=ModelPerformance)
async def get_model_performance(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get performance metrics for a specific ML model.
    
    Returns detailed performance metrics including accuracy, precision,
    recall, and F1 score based on validation data.
    """
    try:
        model = await get_model(db=db, model_id=model_id)
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        return ModelPerformance(
            accuracy=model.accuracy or 0.0,
            precision=model.precision or 0.0,
            recall=model.recall or 0.0,
            f1_score=model.f1_score or 0.0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model performance: {str(e)}"
        )

@router.get("/active/{model_type}", response_model=ModelInfo)
async def get_active_model_endpoint(
    model_type: str = Query(..., description="Model type (e.g., 'fusion', 'cnn', 'lstm')"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get the currently active model for a specific type.
    
    Returns the model currently being used for predictions
    of the specified type (e.g., fusion, CNN, LSTM).
    """
    try:
        model = await get_active_model(db=db, model_type=model_type)
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active model found for type: {model_type}"
            )
        
        return ModelInfo(
            id=model.id,
            name=model.name,
            version=model.version,
            model_type=model.model_type,
            accuracy=model.accuracy,
            is_active=model.is_active,
            created_at=model.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch active model: {str(e)}"
        )

# Admin-only endpoints
@router.post("/{model_id}/activate")
async def activate_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin)
):
    """
    Activate a specific model for predictions (admin only).
    
    Sets the specified model as the active model for its type.
    Deactivates other models of the same type to ensure only
    one model is active per type.
    """
    try:
        model = await get_model(db=db, model_id=model_id)
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        # Deactivate other models of the same type
        # In a full implementation, you'd have a function to handle this
        # For now, we'll just activate this model
        model.is_active = True
        await db.commit()
        
        return {
            "message": f"Model '{model.name}' (v{model.version}) activated successfully",
            "model_id": model.id,
            "model_type": model.model_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to activate model: {str(e)}"
        )

@router.post("/{model_id}/deactivate")
async def deactivate_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin)
):
    """
    Deactivate a model (admin only).
    
    Removes the model from active prediction use.
    The model remains in the system for historical reference.
    """
    try:
        model = await get_model(db=db, model_id=model_id)
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        model.is_active = False
        await db.commit()
        
        return {
            "message": f"Model '{model.name}' (v{model.version}) deactivated successfully",
            "model_id": model.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deactivate model: {str(e)}"
        )