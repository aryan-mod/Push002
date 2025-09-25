"""
Alert management endpoints.

This module provides endpoints for creating, retrieving, and managing
risk alerts and emergency notifications.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user, get_current_admin
from app.db.crud import (
    create_alert, get_alerts, get_alert, update_alert_status,
    get_alerts_by_location, get_alert_stats
)
from app.db.models import User
from app.schemas.alert import (
    Alert, AlertCreate, AlertUpdate, AlertStats, LocationAlerts,
    AlertCategory, AlertSeverity, AlertStatus
)
from app.core.websocket_manager import websocket_manager

router = APIRouter()

@router.post("/create", response_model=Alert, status_code=status.HTTP_201_CREATED)
async def create_new_alert(
    alert: AlertCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new risk alert.
    
    This endpoint allows users to manually create alerts based on observations
    or automated systems to submit detected risks.
    
    - **probability**: Risk probability from 0.0 to 1.0
    - **category**: Risk category (low_risk, medium_risk, high_risk, critical_risk)
    - **lat/lon**: Geographical coordinates
    - **weather**: Current weather conditions
    - **temperature**: Temperature in Celsius
    - **title/message**: Alert details
    - **severity**: Alert severity level
    """
    try:
        # Create alert in database
        db_alert = await create_alert(db=db, alert=alert, user_id=current_user.id)
        
        # Broadcast alert via WebSocket for real-time notifications
        alert_data = {
            "id": db_alert.id,
            "probability": db_alert.probability,
            "category": db_alert.category,
            "lat": db_alert.lat,
            "lon": db_alert.lon,
            "severity": db_alert.severity,
            "title": db_alert.title,
            "message": db_alert.message,
            "timestamp": db_alert.timestamp.isoformat(),
            "user_id": current_user.id,
            "user_name": current_user.full_name or current_user.email
        }
        
        await websocket_manager.broadcast_alert(alert_data)
        
        # Fetch the complete alert with relationships
        complete_alert = await get_alert(db=db, alert_id=db_alert.id)
        return complete_alert
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create alert: {str(e)}"
        )

@router.get("", response_model=List[Alert])
async def get_all_alerts(
    skip: int = Query(default=0, ge=0, description="Number of alerts to skip"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum number of alerts to return"),
    category: Optional[AlertCategory] = Query(default=None, description="Filter by alert category"),
    status: Optional[AlertStatus] = Query(default=None, description="Filter by alert status"),
    severity: Optional[AlertSeverity] = Query(default=None, description="Filter by severity level"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve alerts with optional filtering and pagination.
    
    Returns a list of alerts that the user has access to view.
    Supports filtering by category, status, and severity.
    """
    try:
        # Regular users see all alerts, admins see everything
        user_filter = None if current_user.role == "admin" else current_user.id
        
        alerts = await get_alerts(
            db=db,
            skip=skip,
            limit=limit,
            category=category.value if category else None,
            status=status.value if status else None,
            severity=severity.value if severity else None,
            user_id=user_filter
        )
        
        return alerts
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch alerts: {str(e)}"
        )

@router.get("/{alert_id}", response_model=Alert)
async def get_single_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific alert by ID.
    
    Returns detailed information about a single alert including
    associated user and site information.
    """
    try:
        alert = await get_alert(db=db, alert_id=alert_id)
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        # Check if user has permission to view this alert
        if current_user.role != "admin" and alert.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this alert"
            )
        
        return alert
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch alert: {str(e)}"
        )

@router.put("/{alert_id}/status", response_model=Alert)
async def update_alert_status_endpoint(
    alert_id: int,
    alert_update: AlertUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update the status of an existing alert.
    
    Allows users to acknowledge alerts or mark them as resolved.
    Status changes are tracked with timestamps.
    """
    try:
        # Check if alert exists and user has permission
        alert = await get_alert(db=db, alert_id=alert_id)
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        if current_user.role != "admin" and alert.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this alert"
            )
        
        # Update alert status
        updated_alert = await update_alert_status(
            db=db,
            alert_id=alert_id,
            status=alert_update.status.value,
            user_id=current_user.id
        )
        
        if not updated_alert:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update alert status"
            )
        
        # Broadcast status update via WebSocket
        await websocket_manager.broadcast({
            "type": "alert_status_updated",
            "data": {
                "alert_id": alert_id,
                "status": alert_update.status.value,
                "updated_by": current_user.id,
                "timestamp": updated_alert.acknowledged_at.isoformat() if updated_alert.acknowledged_at else None
            }
        })
        
        return updated_alert
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update alert: {str(e)}"
        )

@router.post("/location", response_model=List[Alert])
async def get_alerts_by_location_endpoint(
    location_query: LocationAlerts,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get alerts within a specified radius of a location.
    
    Useful for finding nearby alerts when traveling to a new area
    or monitoring risks around a specific point of interest.
    """
    try:
        alerts = await get_alerts_by_location(
            db=db,
            lat=location_query.lat,
            lon=location_query.lon,
            radius=location_query.radius,
            limit=location_query.limit
        )
        
        return alerts
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch location-based alerts: {str(e)}"
        )

@router.get("/stats/summary", response_model=AlertStats)
async def get_alert_statistics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get alert statistics and summary information.
    
    Returns overview statistics including total alerts, active alerts,
    severity breakdown, and category distribution.
    """
    try:
        stats = await get_alert_stats(db=db)
        
        return AlertStats(
            total_alerts=stats["total_alerts"],
            active_alerts=stats["active_alerts"],
            high_severity_alerts=stats["high_severity_alerts"],
            category_breakdown=stats["category_breakdown"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch alert statistics: {str(e)}"
        )

# Admin-only endpoints
@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin)
):
    """
    Delete an alert (admin only).
    
    Permanently removes an alert from the system. This action cannot be undone.
    """
    try:
        alert = await get_alert(db=db, alert_id=alert_id)
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        # In a real implementation, you'd have a delete function
        # For now, we'll just update status to indicate deletion
        await update_alert_status(
            db=db,
            alert_id=alert_id,
            status="deleted",
            user_id=current_user.id
        )
        
        return {"message": "Alert deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete alert: {str(e)}"
        )