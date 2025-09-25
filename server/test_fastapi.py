#!/usr/bin/env python3
"""
Test script for the FastAPI backend.
Run this to verify all endpoints are working correctly.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_api():
    """Test the FastAPI backend endpoints."""
    print("🧪 Testing GeoMindFlow FastAPI Backend")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Root endpoint working")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test user signup
    print("\n3. Testing user signup...")
    user_data = {
        "email": "test@geomindflow.com",
        "password": "testpass123",
        "full_name": "Test User",
        "role": "user"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/auth/signup", json=user_data)
        if response.status_code == 201:
            print("✅ User signup successful")
            user = response.json()
            print(f"Created user: {user['email']}")
        elif response.status_code == 400 and "already registered" in response.text:
            print("✅ User already exists (expected for repeat tests)")
        else:
            print(f"❌ User signup failed: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ User signup error: {e}")
    
    # Test user login
    print("\n4. Testing user login...")
    login_data = {
        "email": "test@geomindflow.com",
        "password": "testpass123"
    }
    
    token = None
    try:
        response = requests.post(f"{BASE_URL}/api/v1/auth/login-json", json=login_data)
        if response.status_code == 200:
            print("✅ User login successful")
            auth_response = response.json()
            token = auth_response["access_token"]
            print(f"Token received: {token[:20]}...")
        else:
            print(f"❌ User login failed: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ User login error: {e}")
    
    if not token:
        print("❌ Cannot continue without authentication token")
        return
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test prediction endpoint
    print("\n5. Testing ML prediction...")
    prediction_data = {
        "lat": 28.7041,
        "lon": 77.1025,
        "rainfall": 45.5,
        "slope": 35.0,
        "temperature": 25.0,
        "humidity": 75.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/predict", json=prediction_data, headers=headers)
        if response.status_code == 200:
            print("✅ ML prediction successful")
            prediction = response.json()
            print(f"Risk probability: {prediction['probability']}")
            print(f"Risk category: {prediction['category']}")
        else:
            print(f"❌ ML prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ ML prediction error: {e}")
    
    # Test alert creation
    print("\n6. Testing alert creation...")
    alert_data = {
        "probability": 0.75,
        "category": "high_risk",
        "lat": 28.7041,
        "lon": 77.1025,
        "weather": "rainy",
        "temperature": 25.0,
        "title": "Test Alert",
        "message": "This is a test alert from the API test",
        "severity": "high"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/alerts/create", json=alert_data, headers=headers)
        if response.status_code == 201:
            print("✅ Alert creation successful")
            alert = response.json()
            print(f"Alert ID: {alert['id']}")
            print(f"Alert severity: {alert['severity']}")
        else:
            print(f"❌ Alert creation failed: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Alert creation error: {e}")
    
    # Test alerts retrieval
    print("\n7. Testing alerts retrieval...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/alerts", headers=headers)
        if response.status_code == 200:
            print("✅ Alerts retrieval successful")
            alerts = response.json()
            print(f"Found {len(alerts)} alerts")
        else:
            print(f"❌ Alerts retrieval failed: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Alerts retrieval error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    print(f"📚 Swagger UI: {BASE_URL}/docs")
    print(f"📖 ReDoc: {BASE_URL}/redoc")

if __name__ == "__main__":
    test_api()