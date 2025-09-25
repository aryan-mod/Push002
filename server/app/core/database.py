"""
Database configuration and session management.

This module handles SQLAlchemy database setup, including PostgreSQL with PostGIS
extension for geographical data processing.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import text
import os

# Get database URL from environment with fallback for development
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Fallback for local development
    DATABASE_URL = "sqlite+aiosqlite:///./geomindflow.db"
    print("⚠️  Using SQLite fallback database for development")

# Convert postgresql:// to postgresql+asyncpg:// for async support
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Create async engine for database connections
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Set to False in production to reduce logging
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=300,  # Recycle connections every 5 minutes
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for all database models
Base = declarative_base()

async def get_db() -> AsyncSession:
    """
    Dependency function to get database session.
    
    This function creates a new database session for each request
    and ensures it's properly closed after use.
    
    Yields:
        AsyncSession: Database session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db():
    """
    Initialize database with required extensions and tables.
    
    This function creates the PostGIS extension if it doesn't exist
    and creates all database tables defined in the models.
    """
    async with engine.begin() as conn:
        # Enable PostGIS extension for geographical data
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
            print("✅ PostGIS extension enabled")
        except Exception as e:
            print(f"⚠️  PostGIS extension setup: {e}")
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        print("✅ Database tables created")

async def close_db():
    """
    Close database connections.
    Used during application shutdown.
    """
    await engine.dispose()
    print("🔐 Database connections closed")