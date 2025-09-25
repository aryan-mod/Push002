# GeoMindFlow - Smart Tourist Safety System

## Overview

GeoMindFlow is a comprehensive AI-powered tourist safety system that evolved from a rockfall prediction platform. The system combines cutting-edge machine learning with real-time location tracking to provide comprehensive safety monitoring for tourists in potentially hazardous environments. It features emergency alert systems, live tracking maps, location-based risk assessment, multi-channel notifications, and detailed safety reporting capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **React/TypeScript SPA**: Modern single-page application built with React 18+ and TypeScript
- **Component Library**: shadcn/ui components with Radix UI primitives and Tailwind CSS styling
- **State Management**: TanStack Query for server state management and caching
- **Real-time Updates**: WebSocket integration for live alerts and tourist location updates
- **Responsive Design**: Mobile-first approach with responsive layouts for mobile and desktop use
- **Routing**: Wouter for lightweight client-side routing with tourist safety pages

### Backend Architecture
- **Node.js/Express Server**: TypeScript-based Express.js server with modern ES modules
- **Database Layer**: PostgreSQL with PostGIS for geospatial data and Drizzle ORM
- **Real-time Communication**: WebSocket server for live data streaming and alerts
- **API Design**: RESTful endpoints with structured error handling and validation
- **File Processing**: Multi-modal data ingestion for images, sensor data, and geospatial information

### Smart Safety Features
- **Emergency Alert System**: Quick emergency reporting with location sharing and real-time dispatch
- **Live Tourist Tracking**: Real-time location monitoring with route deviation detection
- **Location Search & Risk Assessment**: Search any location for live weather data and safety risk analysis
- **Geo-fence Alerts**: Configurable safety zones with entry/exit notifications
- **Multi-Modal Data Processing**: Weather, crime statistics, medical facility proximity, and connectivity analysis

### Data Storage Strategy
- **PostgreSQL + PostGIS**: Primary database for structured data and geospatial features
- **Object Storage**: MinIO (S3-compatible) for images, DEM tiles, and large files
- **Time Series Data**: Optimized sensor reading storage with proper indexing
- **Caching Layer**: Redis for session management and real-time data caching

### Alert and Notification System
- **Multi-Channel Alerts**: Push notifications, SMS, email, and emergency contact integration
- **Severity-Based Routing**: Different notification channels based on alert priority levels (critical/high/medium/low)
- **Real-time Broadcasting**: WebSocket-based alert distribution to connected users and authorities
- **Alert Management**: Comprehensive settings, user preferences, and emergency contact management
- **Tourist-Specific Alerts**: Location-based alerts for weather, crime, natural disasters, wildlife, accidents

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