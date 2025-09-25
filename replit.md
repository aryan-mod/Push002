# AI-Based Rockfall Prediction & Alert System

## Overview

This is a comprehensive AI-powered rockfall prediction and alert system designed for Smart India Hackathon 2025. The system combines cutting-edge machine learning with real-time IoT monitoring to provide early warning for rockfall events in mountainous and mining regions. It features multi-modal AI models (CNN for drone imagery, LSTM for sensor time series), real-time data processing, interactive mapping with risk visualization, and advanced multi-channel alert systems.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **React/TypeScript SPA**: Modern single-page application built with React 18+ and TypeScript
- **Component Library**: shadcn/ui components with Radix UI primitives and Tailwind CSS styling
- **State Management**: TanStack Query for server state management and caching
- **Real-time Updates**: WebSocket integration for live alerts and sensor data updates
- **Responsive Design**: Mobile-first approach with responsive layouts
- **Routing**: Wouter for lightweight client-side routing

### Backend Architecture
- **Node.js/Express Server**: TypeScript-based Express.js server with modern ES modules
- **Database Layer**: PostgreSQL with PostGIS for geospatial data and Drizzle ORM
- **Real-time Communication**: WebSocket server for live data streaming and alerts
- **API Design**: RESTful endpoints with structured error handling and validation
- **File Processing**: Multi-modal data ingestion for images, sensor data, and geospatial information

### Machine Learning Pipeline
- **Multi-Modal Fusion**: Combines CNN (drone imagery), LSTM (sensor time series), and tabular data
- **Model Architecture**: PyTorch-based models with ResNet50 backbone for image processing
- **Explainable AI**: SHAP and GradCAM integration for prediction interpretability
- **Edge Computing**: Raspberry Pi agents for local inference and offline operation
- **Model Management**: Version control and A/B testing capabilities

### Data Storage Strategy
- **PostgreSQL + PostGIS**: Primary database for structured data and geospatial features
- **Object Storage**: MinIO (S3-compatible) for images, DEM tiles, and large files
- **Time Series Data**: Optimized sensor reading storage with proper indexing
- **Caching Layer**: Redis for session management and real-time data caching

### Alert and Notification System
- **Multi-Channel Alerts**: SMS, email, WhatsApp, and push notifications
- **Severity-Based Routing**: Different notification channels based on risk levels
- **Real-time Broadcasting**: WebSocket-based alert distribution to connected clients
- **Alert Management**: Acknowledgment system and escalation procedures

## External Dependencies

### Third-Party Services
- **Neon Database**: Serverless PostgreSQL hosting with PostGIS extension
- **Twilio**: SMS and WhatsApp messaging services for critical alerts
- **SMTP Provider**: Email notification delivery
- **Mapbox GL**: Interactive mapping and geospatial visualization

### Development Tools
- **Vite**: Frontend build tool and development server
- **Drizzle Kit**: Database migrations and schema management
- **ESBuild**: Backend bundling for production deployment
- **TypeScript**: Type safety across the entire stack

### IoT and Communication
- **MQTT Protocol**: Device-to-cloud communication for sensor networks
- **WebSocket**: Real-time bidirectional communication
- **Raspberry Pi**: Edge computing devices for local data processing

### Machine Learning Stack
- **PyTorch**: Deep learning framework for model development
- **ONNX**: Model export format for cross-platform inference
- **OpenCV**: Computer vision and image processing
- **scikit-learn**: Traditional ML algorithms and model evaluation