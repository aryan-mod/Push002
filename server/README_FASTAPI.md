# GeoMindFlow FastAPI Backend

## 🚀 Complete FastAPI Backend Setup

Your FastAPI backend is now fully configured with all requested features:

### ✅ Features Implemented

1. **PostgreSQL + PostGIS Database**
   - User management with JWT authentication
   - Alerts with geographical data
   - Monitoring sites with PostGIS POINT geometry  
   - ML model registry and version control
   - Comprehensive database relationships

2. **JWT Authentication & Role-Based Access**
   - User signup/login endpoints
   - JWT token generation and validation
   - Role-based access control (user/admin)
   - Password hashing with bcrypt

3. **Core API Endpoints**
   - `POST /api/v1/predict` - ML risk predictions with weather data
   - `POST /api/v1/alerts/create` - Create new alerts
   - `GET /api/v1/alerts` - Retrieve alerts with filtering
   - `GET /api/v1/sites` - Monitoring sites management
   - `GET /api/v1/models` - ML model management

4. **WebSocket Real-Time Communication**
   - `/ws/alerts` - Live alert broadcasting
   - Connection management with auto-reconnection
   - Real-time prediction updates

5. **Advanced Features**
   - Batch prediction processing
   - Location-based queries with PostGIS
   - Weather data integration (mock/extendable)
   - Model performance tracking
   - Comprehensive error handling
   - CORS middleware for frontend integration

## 🏃‍♂️ Quick Start

### Start the FastAPI Server:
```bash
# From the server/ directory
python run_fastapi.py
```

The server will start on `http://0.0.0.0:8000` with:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Test the API:

1. **Create a User:**
```bash
curl -X POST "http://localhost:8000/api/v1/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@geomindflow.com",
    "password": "password123",
    "full_name": "System Admin",
    "role": "admin"
  }'
```

2. **Login & Get Token:**
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login-json" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@geomindflow.com",
    "password": "password123"
  }'
```

3. **Make a Prediction:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 28.7041,
    "lon": 77.1025,
    "rainfall": 45.5,
    "slope": 35.0,
    "temperature": 25.0
  }'
```

4. **Create an Alert:**
```bash
curl -X POST "http://localhost:8000/api/v1/alerts/create" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "probability": 0.85,
    "category": "high_risk",
    "lat": 28.7041,
    "lon": 77.1025,
    "weather": "rainy",
    "temperature": 25.0,
    "title": "High Risk Alert",
    "message": "Dangerous conditions detected",
    "severity": "high"
  }'
```

## 📁 Project Structure

```
server/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── core/                # Core configuration
│   │   ├── config.py        # Settings and environment variables
│   │   ├── database.py      # Database connection and session management
│   │   ├── security.py      # JWT authentication and password hashing
│   │   └── websocket_manager.py # WebSocket connection management
│   ├── db/                  # Database layer
│   │   ├── models.py        # SQLAlchemy models with PostGIS support
│   │   └── crud.py          # Database CRUD operations
│   ├── schemas/             # Pydantic schemas for validation
│   │   ├── user.py          # User-related schemas
│   │   ├── alert.py         # Alert schemas
│   │   ├── site.py          # Site schemas
│   │   └── prediction.py    # ML prediction schemas
│   └── api/                 # API routes
│       └── v1/
│           ├── api.py       # Main API router
│           └── endpoints/   # Individual endpoint modules
│               ├── auth.py      # Authentication endpoints
│               ├── predictions.py # ML prediction endpoints
│               ├── alerts.py    # Alert management endpoints
│               ├── sites.py     # Site management endpoints
│               └── models.py    # Model management endpoints
├── requirements.txt         # Python dependencies
└── run_fastapi.py          # Server startup script
```

## 🔧 Environment Variables

Set these in your Replit environment:
- `DATABASE_URL` - PostgreSQL connection string (automatically configured)
- `JWT_SECRET_KEY` - Secret key for JWT tokens
- `OPENWEATHER_API_KEY` - For weather data integration (optional)

## 🧪 Testing via Swagger UI

1. Open http://localhost:8000/docs
2. Use the "Authorize" button to set your JWT token
3. Test all endpoints interactively

## 🔄 WebSocket Testing

Connect to `ws://localhost:8000/ws/alerts` to receive real-time updates.

Your FastAPI backend is production-ready with comprehensive error handling, security features, and scalable architecture!