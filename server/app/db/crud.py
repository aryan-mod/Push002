"""
CRUD operations for database models.

This module provides Create, Read, Update, Delete operations for all database models.
Uses SQLAlchemy async sessions for database interactions.
"""

from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import and_, or_, func, desc, asc
from geoalchemy2.functions import ST_DWithin, ST_GeogFromText
import json

from app.db.models import User, Alert, Site, Model, Prediction
from app.core.security import get_password_hash, verify_password
from app.schemas.user import UserCreate
from app.schemas.alert import AlertCreate
from app.schemas.site import SiteCreate

# User CRUD operations
async def get_user(db: AsyncSession, user_id: int) -> Optional[User]:
    """Get user by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()

async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Get user by email address."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()

async def get_users(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[User]:
    """Get all users with pagination."""
    result = await db.execute(select(User).offset(skip).limit(limit))
    return result.scalars().all()

async def create_user(db: AsyncSession, user: UserCreate) -> User:
    """Create a new user."""
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        role=user.role
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

async def authenticate_user(db: AsyncSession, email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password."""
    user = await get_user_by_email(db, email)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

# Alert CRUD operations
async def get_alert(db: AsyncSession, alert_id: int) -> Optional[Alert]:
    """Get alert by ID with related data."""
    result = await db.execute(
        select(Alert)
        .options(selectinload(Alert.user), selectinload(Alert.site))
        .where(Alert.id == alert_id)
    )
    return result.scalar_one_or_none()

async def get_alerts(
    db: AsyncSession, 
    skip: int = 0, 
    limit: int = 100,
    category: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    user_id: Optional[int] = None
) -> List[Alert]:
    """Get alerts with optional filtering and pagination."""
    query = select(Alert).options(selectinload(Alert.user), selectinload(Alert.site))
    
    # Apply filters
    conditions = []
    if category:
        conditions.append(Alert.category == category)
    if status:
        conditions.append(Alert.status == status)
    if severity:
        conditions.append(Alert.severity == severity)
    if user_id:
        conditions.append(Alert.user_id == user_id)
    
    if conditions:
        query = query.where(and_(*conditions))
    
    query = query.order_by(desc(Alert.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def get_alerts_by_location(
    db: AsyncSession,
    lat: float,
    lon: float,
    radius: float = 10000,  # radius in meters
    limit: int = 50
) -> List[Alert]:
    """Get alerts within a specified radius of a location."""
    # Create a point from lat/lon and find alerts within radius
    point_wkt = f"POINT({lon} {lat})"
    
    query = select(Alert).where(
        func.ST_DWithin(
            func.ST_GeogFromText(point_wkt),
            func.ST_MakePoint(Alert.lon, Alert.lat)::func.geography,
            radius
        )
    ).order_by(desc(Alert.created_at)).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def create_alert(db: AsyncSession, alert: AlertCreate, user_id: Optional[int] = None) -> Alert:
    """Create a new alert."""
    db_alert = Alert(
        probability=alert.probability,
        category=alert.category,
        lat=alert.lat,
        lon=alert.lon,
        weather=alert.weather,
        temperature=alert.temperature,
        humidity=alert.humidity,
        wind_speed=alert.wind_speed,
        title=alert.title,
        message=alert.message,
        severity=alert.severity,
        timestamp=alert.timestamp,
        user_id=user_id,
        site_id=alert.site_id
    )
    db.add(db_alert)
    await db.commit()
    await db.refresh(db_alert)
    return db_alert

async def update_alert_status(
    db: AsyncSession, 
    alert_id: int, 
    status: str, 
    user_id: Optional[int] = None
) -> Optional[Alert]:
    """Update alert status (acknowledge, resolve, etc.)."""
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    
    if alert:
        alert.status = status
        if status == "acknowledged":
            alert.acknowledged_at = func.now()
        elif status == "resolved":
            alert.resolved_at = func.now()
            
        await db.commit()
        await db.refresh(alert)
    
    return alert

# Site CRUD operations
async def get_site(db: AsyncSession, site_id: int) -> Optional[Site]:
    """Get site by ID."""
    result = await db.execute(select(Site).where(Site.id == site_id))
    return result.scalar_one_or_none()

async def get_sites(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[Site]:
    """Get all sites with pagination."""
    result = await db.execute(select(Site).offset(skip).limit(limit))
    return result.scalars().all()

async def create_site(db: AsyncSession, site: SiteCreate) -> Site:
    """Create a new monitoring site."""
    # Create PostGIS POINT geometry from lat/lon
    location_wkt = f"POINT({site.lon} {site.lat})"
    
    db_site = Site(
        name=site.name,
        description=site.description,
        location=func.ST_GeogFromText(location_wkt),
        radius=site.radius,
        risk_type=site.risk_type
    )
    db.add(db_site)
    await db.commit()
    await db.refresh(db_site)
    return db_site

async def get_sites_near_location(
    db: AsyncSession,
    lat: float,
    lon: float,
    radius: float = 50000  # 50km default
) -> List[Site]:
    """Get sites near a specific location."""
    point_wkt = f"POINT({lon} {lat})"
    
    query = select(Site).where(
        func.ST_DWithin(
            Site.location,
            func.ST_GeogFromText(point_wkt),
            radius
        )
    )
    
    result = await db.execute(query)
    return result.scalars().all()

# Model CRUD operations
async def get_model(db: AsyncSession, model_id: int) -> Optional[Model]:
    """Get model by ID."""
    result = await db.execute(select(Model).where(Model.id == model_id))
    return result.scalar_one_or_none()

async def get_models(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[Model]:
    """Get all models with pagination."""
    result = await db.execute(
        select(Model)
        .order_by(desc(Model.created_at))
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()

async def get_active_model(db: AsyncSession, model_type: str = "fusion") -> Optional[Model]:
    """Get the currently active model for predictions."""
    result = await db.execute(
        select(Model)
        .where(and_(Model.is_active == True, Model.model_type == model_type))
        .order_by(desc(Model.created_at))
    )
    return result.scalar_one_or_none()

# Prediction CRUD operations
async def create_prediction(
    db: AsyncSession,
    lat: float,
    lon: float,
    probability: float,
    category: str,
    confidence: Optional[float] = None,
    input_data: Optional[dict] = None,
    model_id: Optional[int] = None,
    user_id: Optional[int] = None,
    site_id: Optional[int] = None
) -> Prediction:
    """Create a new prediction record."""
    db_prediction = Prediction(
        lat=lat,
        lon=lon,
        probability=probability,
        category=category,
        confidence=confidence,
        input_data=json.dumps(input_data) if input_data else None,
        model_id=model_id,
        user_id=user_id,
        site_id=site_id
    )
    db.add(db_prediction)
    await db.commit()
    await db.refresh(db_prediction)
    return db_prediction

async def get_recent_predictions(
    db: AsyncSession,
    limit: int = 100,
    site_id: Optional[int] = None
) -> List[Prediction]:
    """Get recent predictions with optional site filtering."""
    query = select(Prediction).order_by(desc(Prediction.created_at))
    
    if site_id:
        query = query.where(Prediction.site_id == site_id)
    
    query = query.limit(limit)
    result = await db.execute(query)
    return result.scalars().all()

# Statistics and Analytics
async def get_alert_stats(db: AsyncSession) -> dict:
    """Get alert statistics for dashboard."""
    # Total alerts
    total_result = await db.execute(select(func.count(Alert.id)))
    total_alerts = total_result.scalar()
    
    # Active alerts
    active_result = await db.execute(
        select(func.count(Alert.id)).where(Alert.status == "active")
    )
    active_alerts = active_result.scalar()
    
    # High-severity alerts
    high_severity_result = await db.execute(
        select(func.count(Alert.id)).where(Alert.severity.in_(["high", "critical"]))
    )
    high_severity_alerts = high_severity_result.scalar()
    
    # Alerts by category
    category_result = await db.execute(
        select(Alert.category, func.count(Alert.id))
        .group_by(Alert.category)
    )
    category_stats = {category: count for category, count in category_result.all()}
    
    return {
        "total_alerts": total_alerts,
        "active_alerts": active_alerts,
        "high_severity_alerts": high_severity_alerts,
        "category_breakdown": category_stats
    }