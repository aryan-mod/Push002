# Demo Guide - AI-Based Rockfall Prediction System

This guide provides step-by-step instructions for demonstrating the complete rockfall prediction system, from basic setup to advanced features.

## Quick Demo Setup (5 minutes)

### Prerequisites
- Docker and Docker Compose installed
- 8GB RAM minimum
- 20GB free disk space

### 1. Clone and Start
```bash
# Clone repository
git clone https://github.com/your-repo/rockfall-prediction.git
cd rockfall-prediction

# Copy environment file
cp .env.example .env

# Start services
docker-compose up -d

# Wait for services to be ready (check logs)
docker-compose logs -f backend
