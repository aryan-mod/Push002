# AI-Based Rockfall Prediction & Alert System

[![Build Status](https://github.com/your-repo/rockfall-prediction/workflows/CI/badge.svg)](https://github.com/your-repo/rockfall-prediction/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)

> **Smart India Hackathon 2025 - Problem Statement 25071**

A comprehensive, production-ready AI system for predicting rockfall events using multi-modal machine learning, real-time sensor monitoring, and advanced alert mechanisms.

## 🎯 Overview

This system combines cutting-edge AI/ML technologies with IoT sensor networks to provide early warning for rockfall events, potentially saving lives and infrastructure in mountainous and mining regions.

### Key Features

- **🧠 Multi-Modal AI Models**: CNN for drone imagery, LSTM for sensor time-series, fusion models
- **📊 Real-Time Monitoring**: Live sensor data processing with WebSocket updates
- **🚨 Advanced Alert System**: Multi-channel notifications (SMS, Email, WhatsApp, Push)
- **🗺️ Interactive Mapping**: Mapbox-powered risk visualization and site management
- **⚡ Edge Computing**: Raspberry Pi agents for local inference and offline operation
- **📈 Explainable AI**: SHAP and GradCAM for prediction interpretability
- **🔧 Production Ready**: Docker containers, monitoring, CI/CD, auto-scaling

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Edge Layer"
        Pi[Raspberry Pi Agents]
        Sensors[IoT Sensors]
        Drones[Drone Cameras]
    end
    
    subgraph "Communication"
        MQTT[MQTT Broker]
        API[REST API]
    end
    
    subgraph "Backend Services"
        FastAPI[FastAPI Server]
        Celery[Celery Workers]
        ML[ML Pipeline]
        DB[(PostgreSQL + PostGIS)]
        Redis[(Redis Cache)]
        MinIO[(Object Storage)]
    end
    
    subgraph "Frontend"
        Next[Next.js Dashboard]
        Maps[Mapbox GL JS]
        Charts[Real-time Charts]
    end
    
    subgraph "Monitoring"
        Prom[Prometheus]
        Graf[Grafana]
        Logs[Centralized Logging]
    end
    
    Pi --> MQTT
    Sensors --> Pi
    Drones --> Pi
    MQTT --> FastAPI
    FastAPI --> DB
    FastAPI --> Redis
    FastAPI --> MinIO
    FastAPI --> Celery
    Celery --> ML
    Next --> API
    FastAPI --> Prom
    Prom --> Graf
