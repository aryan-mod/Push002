"""
FastAPI Main Application for GeoMindFlow/Push002

This is the main entry point for the FastAPI backend server.
It configures the application, database, middleware, and routes.
"""

import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import json
from typing import List
import uvicorn

from app.core.config import settings
from app.core.database import engine, Base, init_db
from app.api.v1.api import api_router
from app.core.websocket_manager import WebSocketManager

# WebSocket manager instance for real-time connections
websocket_manager = WebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler to manage startup and shutdown events.
    Creates database tables on startup.
    """
    print("🚀 Starting GeoMindFlow FastAPI Backend...")
    
    # Initialize database and create tables
    print("📊 Initializing database...")
    await init_db()
    print("✅ Database initialized successfully")
    
    print("🌟 FastAPI backend is ready!")
    yield
    
    print("🔄 Shutting down FastAPI backend...")

# Initialize FastAPI app with lifespan events
app = FastAPI(
    title="GeoMindFlow API",
    description="Smart Tourist Safety System with ML-powered risk prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """
    Root endpoint that returns basic API information.
    """
    return {
        "message": "GeoMindFlow API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring service status.
    """
    return {
        "status": "healthy",
        "database": "connected"
    }

@app.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alert notifications.
    Clients can connect here to receive live updates about alerts and predictions.
    """
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Listen for client messages (optional - for client-to-server communication)
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                # Handle subscription requests
                await websocket.send_text(json.dumps({
                    "type": "subscribed",
                    "message": "Successfully subscribed to alerts"
                }))
            elif message.get("type") == "ping":
                # Handle ping requests for connection health
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": message.get("timestamp")
                }))
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
        print("📱 WebSocket client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        websocket_manager.disconnect(websocket)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc)
        }
    )

# Function to send test alerts via WebSocket
async def send_test_alert():
    """
    Utility function to send test alerts through WebSocket.
    This can be called from other parts of the application.
    """
    test_alert = {
        "type": "alert",
        "data": {
            "id": "test-alert-001",
            "probability": 0.85,
            "category": "high_risk",
            "lat": 28.7041,
            "lon": 77.1025,
            "weather": "rainy",
            "temperature": 25.5,
            "message": "High rockfall risk detected in the area",
            "timestamp": "2024-01-01T12:00:00Z"
        }
    }
    await websocket_manager.broadcast(test_alert)

if __name__ == "__main__":
    # Run the server directly with uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info"
    )