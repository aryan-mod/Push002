#!/usr/bin/env python3
"""
FastAPI Server Startup Script

This script starts the FastAPI backend server for GeoMindFlow/Push002.
It configures the server with proper settings for development and production.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

def main():
    """Main function to start the FastAPI server."""
    
    # Environment settings
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    environment = os.getenv("ENVIRONMENT", "development")
    
    # Development settings
    reload = environment == "development"
    log_level = "info" if environment == "production" else "debug"
    
    print(f"🚀 Starting GeoMindFlow FastAPI Backend")
    print(f"📍 Environment: {environment}")
    print(f"🌐 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Auto-reload: {reload}")
    print(f"📊 Docs available at: http://{host}:{port}/docs")
    print(f"🔧 ReDoc available at: http://{host}:{port}/redoc")
    print("")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True,
        app_dir=str(app_dir)
    )

if __name__ == "__main__":
    main()