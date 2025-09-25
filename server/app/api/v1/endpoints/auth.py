"""
Authentication endpoints for user registration and login.

This module provides JWT-based authentication endpoints including
user registration, login, and token validation.
"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.security import create_access_token, get_current_user, get_current_admin
from app.db.crud import get_user_by_email, create_user, authenticate_user
from app.schemas.user import User, UserCreate, UserLogin, Token

router = APIRouter()

@router.post("/signup", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user_signup(
    user: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new user account.
    
    - **email**: User's email address (must be unique)
    - **password**: User's password (minimum 8 characters)
    - **full_name**: User's full name (optional)
    - **role**: User role - defaults to 'user', only admins can create admin accounts
    """
    # Check if user already exists
    db_user = await get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Only allow admin role creation by existing admins
    if user.role == "admin":
        user.role = "user"  # Default to user role for signup
    
    # Create new user
    new_user = await create_user(db=db, user=user)
    return new_user

@router.post("/login", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate user and return JWT access token.
    
    - **username**: User's email address
    - **password**: User's password
    
    Returns JWT access token for subsequent API requests.
    """
    # Authenticate user
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

@router.post("/login-json", response_model=Token)
async def login_json(
    user_credentials: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """
    Alternative login endpoint that accepts JSON instead of form data.
    
    - **email**: User's email address
    - **password**: User's password
    """
    # Authenticate user
    user = await authenticate_user(db, user_credentials.email, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """
    Get current user information.
    
    Requires valid JWT token in Authorization header.
    """
    return current_user

@router.get("/verify-token")
async def verify_token(current_user: User = Depends(get_current_user)):
    """
    Verify if the provided JWT token is valid.
    
    Returns user information if token is valid, otherwise returns 401.
    """
    return {
        "valid": True,
        "user": current_user,
        "message": "Token is valid"
    }

# Admin-only endpoints
@router.post("/create-admin", response_model=User)
async def create_admin_user(
    user: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin)
):
    """
    Create a new admin user account.
    
    Only existing admin users can create new admin accounts.
    """
    # Check if user already exists
    db_user = await get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Set role to admin
    user.role = "admin"
    
    # Create new admin user
    new_user = await create_user(db=db, user=user)
    return new_user