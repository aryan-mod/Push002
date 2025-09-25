"""
Monitoring site endpoints.

This module provides endpoints for managing monitoring sites,
including CRUD operations and geographical queries.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user, get_current_admin
from app.db.crud import (
    create_site, get_sites, get_site, get_sites_near_location
)
from app.db.models import User
from app.schemas.site import Site, SiteCreate, SiteUpdate, LocationSearch

router = APIRouter()

@router.post("", response_model=Site, status_code=status.HTTP_201_CREATED)
async def create_monitoring_site(
    site: SiteCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin)  # Only admins can create sites
):
    """
    Create a new monitoring site (admin only).
    
    Creates a new geographical monitoring site where sensors can be deployed
    and risk assessments are performed.
    
    - **name**: Site name/identifier
    - **description**: Detailed site description
    - **lat/lon**: Site coordinates
    - **radius**: Monitoring radius in meters
    - **risk_type**: Type of risk monitored (rockfall, landslide, etc.)
    """
    try:
        db_site = await create_site(db=db, site=site)
        return db_site
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create site: {str(e)}"
        )

@router.get("", response_model=List[Site])
async def get_all_sites(
    skip: int = Query(default=0, ge=0, description="Number of sites to skip"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum number of sites to return"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all monitoring sites with pagination.
    
    Returns a list of all active monitoring sites in the system.
    """
    try:
        sites = await get_sites(db=db, skip=skip, limit=limit)
        return sites
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch sites: {str(e)}"
        )

@router.get("/{site_id}", response_model=Site)
async def get_single_site(
    site_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific monitoring site by ID.
    
    Returns detailed information about a single monitoring site.
    """
    try:
        site = await get_site(db=db, site_id=site_id)
        
        if not site:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Site not found"
            )
        
        return site
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch site: {str(e)}"
        )

@router.post("/nearby", response_model=List[Site])
async def get_nearby_sites(
    location: LocationSearch,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Find monitoring sites near a specific location.
    
    Returns sites within the specified radius of the given coordinates.
    Useful for finding monitoring infrastructure in a target area.
    """
    try:
        sites = await get_sites_near_location(
            db=db,
            lat=location.lat,
            lon=location.lon,
            radius=location.radius
        )
        
        return sites[:location.limit]  # Apply limit
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch nearby sites: {str(e)}"
        )

@router.put("/{site_id}", response_model=Site)
async def update_site(
    site_id: int,
    site_update: SiteUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin)  # Only admins can update sites
):
    """
    Update an existing monitoring site (admin only).
    
    Updates site information such as description, monitoring radius,
    or operational status.
    """
    try:
        # Check if site exists
        existing_site = await get_site(db=db, site_id=site_id)
        
        if not existing_site:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Site not found"
            )
        
        # Update site fields
        update_data = site_update.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            setattr(existing_site, field, value)
        
        await db.commit()
        await db.refresh(existing_site)
        
        return existing_site
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update site: {str(e)}"
        )

@router.delete("/{site_id}")
async def delete_site(
    site_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin)  # Only admins can delete sites
):
    """
    Deactivate a monitoring site (admin only).
    
    Sets the site status to inactive rather than permanently deleting
    to preserve historical data and relationships.
    """
    try:
        site = await get_site(db=db, site_id=site_id)
        
        if not site:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Site not found"
            )
        
        # Deactivate site instead of deleting
        site.is_active = False
        await db.commit()
        
        return {"message": f"Site '{site.name}' deactivated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deactivate site: {str(e)}"
        )