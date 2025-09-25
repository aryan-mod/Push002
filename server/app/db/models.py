"""
SQLAlchemy database models for GeoMindFlow application.

This module defines all database tables and relationships using SQLAlchemy ORM.
Includes support for PostGIS geographical data types.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from geoalchemy2 import Geometry
import datetime

from app.core.database import Base

class User(Base):
    """
    User model for authentication and user management.
    
    Stores user credentials, profile information, and role-based permissions.
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user", nullable=False)  # 'user' or 'admin'
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship to alerts created by this user
    alerts = relationship("Alert", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"

class Site(Base):
    """
    Monitoring site model for geographical locations.
    
    Represents physical locations where monitoring equipment is deployed
    and risk assessments are performed.
    """
    __tablename__ = "sites"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    # PostGIS POINT geometry for precise location data
    location = Column(Geometry('POINT', srid=4326), nullable=False)
    radius = Column(Float, default=1000.0, nullable=False)  # Monitoring radius in meters
    risk_type = Column(String, nullable=True)  # Type of risk monitored (rockfall, landslide, etc.)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship to alerts from this site
    alerts = relationship("Alert", back_populates="site")
    
    def __repr__(self):
        return f"<Site(id={self.id}, name={self.name}, risk_type={self.risk_type})>"

class Alert(Base):
    """
    Alert model for storing risk alerts and predictions.
    
    Stores alert information including ML predictions, risk assessments,
    and geographical data for emergency response.
    """
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    probability = Column(Float, nullable=False)  # Risk probability (0.0 to 1.0)
    category = Column(String, nullable=False)    # Risk category (low, medium, high, critical)
    
    # Geographical coordinates
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    
    # Weather and environmental data
    weather = Column(String, nullable=True)      # Weather condition
    temperature = Column(Float, nullable=True)   # Temperature in Celsius
    humidity = Column(Float, nullable=True)      # Humidity percentage
    wind_speed = Column(Float, nullable=True)    # Wind speed in km/h
    
    # Alert details
    title = Column(String, nullable=True)
    message = Column(Text, nullable=True)
    severity = Column(String, default="medium")  # low, medium, high, critical
    status = Column(String, default="active")    # active, acknowledged, resolved
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.datetime.utcnow)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    
    # Foreign keys
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="alerts")
    site = relationship("Site", back_populates="alerts")
    
    def __repr__(self):
        return f"<Alert(id={self.id}, category={self.category}, probability={self.probability})>"

class Model(Base):
    """
    ML Model registry for version control and model management.
    
    Stores information about different versions of ML models used
    for risk prediction and analysis.
    """
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # 'cnn', 'lstm', 'fusion', etc.
    path = Column(String, nullable=False)        # File path or URL to model
    description = Column(Text, nullable=True)
    
    # Model performance metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Model status and metadata
    is_active = Column(Boolean, default=False)
    is_production = Column(Boolean, default=False)
    training_data_size = Column(Integer, nullable=True)
    features_count = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    trained_at = Column(DateTime(timezone=True), nullable=True)
    deployed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Model creator
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    def __repr__(self):
        return f"<Model(id={self.id}, name={self.name}, version={self.version}, type={self.model_type})>"

class Prediction(Base):
    """
    Prediction results model for storing ML model outputs.
    
    Stores individual prediction results from ML models including
    input data, output probabilities, and metadata.
    """
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Input data
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    input_data = Column(Text, nullable=True)  # JSON string of input features
    
    # Prediction results
    probability = Column(Float, nullable=False)  # Risk probability (0.0 to 1.0)
    confidence = Column(Float, nullable=True)    # Model confidence score
    category = Column(String, nullable=False)    # Predicted risk category
    
    # Model information
    model_id = Column(Integer, ForeignKey("models.id"), nullable=True)
    model_version = Column(String, nullable=True)
    
    # Associated site and user
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    prediction_time = Column(Float, nullable=True)  # Time taken for prediction in seconds
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, probability={self.probability}, category={self.category})>"

# Additional indexes for better query performance
from sqlalchemy import Index

# Geographic index for alerts
Index('idx_alerts_location', Alert.lat, Alert.lon)

# Time-based indexes for efficient querying
Index('idx_alerts_timestamp', Alert.timestamp)
Index('idx_alerts_created_at', Alert.created_at)
Index('idx_predictions_created_at', Prediction.created_at)

# Category and status indexes
Index('idx_alerts_category', Alert.category)
Index('idx_alerts_status', Alert.status)
Index('idx_alerts_severity', Alert.severity)