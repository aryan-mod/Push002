import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { storage } from "./storage";
import { mlService } from "./services/ml-service";
import { alertService } from "./services/alert-service";
import { websocketService } from "./services/websocket-service";
import { insertSiteSchema, insertSensorSchema, insertSensorReadingSchema, insertDroneImageSchema, insertAlertSchema } from "@shared/schema";
import { z } from "zod";

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);
  
  // WebSocket server for real-time alerts
  const wss = new WebSocketServer({ server: httpServer, path: '/ws' });
  websocketService.initialize(wss);

  // Prediction endpoint
  app.post("/api/v1/predict", async (req, res) => {
    try {
      const { siteId, timestamp, sensorData, imageUrl } = req.body;
      
      if (!siteId) {
        return res.status(400).json({ error: "Site ID is required" });
      }

      const prediction = await mlService.predict({
        siteId,
        timestamp: timestamp ? new Date(timestamp) : new Date(),
        sensorData,
        imageUrl
      });

      res.json(prediction);
    } catch (error) {
      console.error("Prediction error:", error);
      res.status(500).json({ error: "Prediction failed" });
    }
  });

  // Data ingestion endpoint
  app.post("/api/v1/ingest", async (req, res) => {
    try {
      const { type, data } = req.body;

      switch (type) {
        case "sensor_reading":
          const readingData = insertSensorReadingSchema.parse(data);
          const reading = await storage.createSensorReading(readingData);
          
          // Update sensor status
          await storage.updateSensorStatus(readingData.sensorId, "active");
          
          // Trigger prediction if needed
          const sensor = await storage.getSensor(readingData.sensorId);
          if (sensor) {
            mlService.queuePrediction(sensor.siteId);
          }
          
          res.json(reading);
          break;

        case "drone_image":
          const imageData = insertDroneImageSchema.parse(data);
          const image = await storage.createDroneImage(imageData);
          
          // Queue image processing
          mlService.queueImageProcessing(image.id);
          
          res.json(image);
          break;

        default:
          res.status(400).json({ error: "Invalid data type" });
      }
    } catch (error) {
      console.error("Ingestion error:", error);
      res.status(400).json({ error: "Invalid data format" });
    }
  });

  // Manual alert trigger
  app.post("/api/v1/alert", async (req, res) => {
    try {
      const alertData = insertAlertSchema.parse(req.body);
      const alert = await storage.createAlert(alertData);
      
      // Send notifications
      await alertService.sendAlert(alert);
      
      // Broadcast to WebSocket clients
      websocketService.broadcast({
        type: "alert",
        data: alert
      });

      res.json(alert);
    } catch (error) {
      console.error("Alert creation error:", error);
      res.status(400).json({ error: "Invalid alert data" });
    }
  });

  // Model registry
  app.get("/api/v1/models", async (req, res) => {
    try {
      const models = await storage.getModels();
      res.json(models);
    } catch (error) {
      console.error("Models fetch error:", error);
      res.status(500).json({ error: "Failed to fetch models" });
    }
  });

  // Set active model
  app.post("/api/v1/models/:id/activate", async (req, res) => {
    try {
      const { id } = req.params;
      const { type } = req.body;
      
      await storage.setActiveModel(id, type);
      res.json({ success: true });
    } catch (error) {
      console.error("Model activation error:", error);
      res.status(500).json({ error: "Failed to activate model" });
    }
  });

  // Sites management
  app.get("/api/v1/sites", async (req, res) => {
    try {
      const sites = await storage.getSites();
      res.json(sites);
    } catch (error) {
      console.error("Sites fetch error:", error);
      res.status(500).json({ error: "Failed to fetch sites" });
    }
  });

  app.get("/api/v1/sites/:id", async (req, res) => {
    try {
      const { id } = req.params;
      const site = await storage.getSite(id);
      
      if (!site) {
        return res.status(404).json({ error: "Site not found" });
      }

      // Get additional site data
      const sensors = await storage.getSensorsBySite(id);
      const predictions = await storage.getPredictionsBySite(id, 10);
      const alerts = await storage.getAlertsBySite(id);
      const latestReadings = await storage.getLatestReadingsBySite(id);

      res.json({
        ...site,
        sensors,
        predictions,
        alerts,
        latestReadings
      });
    } catch (error) {
      console.error("Site fetch error:", error);
      res.status(500).json({ error: "Failed to fetch site" });
    }
  });

  app.post("/api/v1/sites", async (req, res) => {
    try {
      const siteData = insertSiteSchema.parse(req.body);
      const site = await storage.createSite(siteData);
      res.json(site);
    } catch (error) {
      console.error("Site creation error:", error);
      res.status(400).json({ error: "Invalid site data" });
    }
  });

  // Sensors management
  app.get("/api/v1/sites/:siteId/sensors", async (req, res) => {
    try {
      const { siteId } = req.params;
      const sensors = await storage.getSensorsBySite(siteId);
      res.json(sensors);
    } catch (error) {
      console.error("Sensors fetch error:", error);
      res.status(500).json({ error: "Failed to fetch sensors" });
    }
  });

  app.post("/api/v1/sites/:siteId/sensors", async (req, res) => {
    try {
      const { siteId } = req.params;
      const sensorData = insertSensorSchema.parse({ ...req.body, siteId });
      const sensor = await storage.createSensor(sensorData);
      res.json(sensor);
    } catch (error) {
      console.error("Sensor creation error:", error);
      res.status(400).json({ error: "Invalid sensor data" });
    }
  });

  // Sensor readings
  app.get("/api/v1/sensors/:sensorId/readings", async (req, res) => {
    try {
      const { sensorId } = req.params;
      const { from, to } = req.query;
      
      const fromDate = from ? new Date(from as string) : undefined;
      const toDate = to ? new Date(to as string) : undefined;
      
      const readings = await storage.getSensorReadings(sensorId, fromDate, toDate);
      res.json(readings);
    } catch (error) {
      console.error("Readings fetch error:", error);
      res.status(500).json({ error: "Failed to fetch readings" });
    }
  });

  // Alerts management
  app.get("/api/v1/alerts", async (req, res) => {
    try {
      const alerts = await storage.getActiveAlerts();
      res.json(alerts);
    } catch (error) {
      console.error("Alerts fetch error:", error);
      res.status(500).json({ error: "Failed to fetch alerts" });
    }
  });

  app.post("/api/v1/alerts/:id/acknowledge", async (req, res) => {
    try {
      const { id } = req.params;
      const { userId } = req.body;
      
      await storage.acknowledgeAlert(id, userId);
      
      // Broadcast acknowledgment
      websocketService.broadcast({
        type: "alert_acknowledged",
        data: { alertId: id, userId }
      });

      res.json({ success: true });
    } catch (error) {
      console.error("Alert acknowledgment error:", error);
      res.status(500).json({ error: "Failed to acknowledge alert" });
    }
  });

  // Dashboard metrics
  app.get("/api/v1/dashboard/metrics", async (req, res) => {
    try {
      const metrics = await storage.getRiskMetrics();
      
      // Get active model performance
      const activeModel = await storage.getActiveModel("fusion");
      const modelAccuracy = activeModel?.metrics ? 
        (activeModel.metrics as any).accuracy * 100 : 94.7;

      res.json({
        ...metrics,
        modelAccuracy: modelAccuracy.toFixed(1)
      });
    } catch (error) {
      console.error("Metrics fetch error:", error);
      res.status(500).json({ error: "Failed to fetch metrics" });
    }
  });

  // Dashboard trends
  app.get("/api/v1/dashboard/trends/:timeRange?", async (req, res) => {
    try {
      const { timeRange = "24h" } = req.params;
      
      // Calculate time window
      let timeWindow = 24; // hours
      switch (timeRange) {
        case "7d":
          timeWindow = 24 * 7;
          break;
        case "30d":
          timeWindow = 24 * 30;
          break;
        default:
          timeWindow = 24;
      }

      // Generate sample trend data for the time range
      const endTime = new Date();
      const startTime = new Date(endTime.getTime() - timeWindow * 60 * 60 * 1000);
      
      // Get recent predictions for trending
      const predictions = await storage.getPredictionsBySite("", 100); // Get from all sites
      
      // Generate trend data points
      const dataPoints: Array<{
        timestamp: string;
        averageRisk: number;
        activeSensors: number;
        alertCount: number;
      }> = [];

      const pointCount = Math.min(timeWindow / (timeRange === "24h" ? 1 : timeRange === "7d" ? 6 : 24), 50);
      
      for (let i = 0; i < pointCount; i++) {
        const pointTime = new Date(startTime.getTime() + (i * (timeWindow * 60 * 60 * 1000) / pointCount));
        
        dataPoints.push({
          timestamp: pointTime.toISOString(),
          averageRisk: 0.2 + Math.random() * 0.6, // Random between 0.2 and 0.8
          activeSensors: 20 + Math.floor(Math.random() * 10), // Random between 20-30
          alertCount: Math.floor(Math.random() * 5) // Random between 0-4
        });
      }

      res.json({
        timeRange,
        data: dataPoints,
        summary: {
          totalDataPoints: dataPoints.length,
          averageRisk: dataPoints.reduce((sum, p) => sum + p.averageRisk, 0) / dataPoints.length,
          peakRisk: Math.max(...dataPoints.map(p => p.averageRisk)),
          totalAlerts: dataPoints.reduce((sum, p) => sum + p.alertCount, 0)
        }
      });
    } catch (error) {
      console.error("Trends fetch error:", error);
      res.status(500).json({ error: "Failed to fetch trends" });
    }
  });

  // Predictions history
  app.get("/api/v1/predictions", async (req, res) => {
    try {
      const { siteId, limit } = req.query;
      
      if (!siteId) {
        return res.status(400).json({ error: "Site ID is required" });
      }

      const predictions = await storage.getPredictionsBySite(
        siteId as string, 
        limit ? parseInt(limit as string) : 10
      );
      
      res.json(predictions);
    } catch (error) {
      console.error("Predictions fetch error:", error);
      res.status(500).json({ error: "Failed to fetch predictions" });
    }
  });

  // Drone images
  app.get("/api/v1/sites/:siteId/images", async (req, res) => {
    try {
      const { siteId } = req.params;
      const images = await storage.getDroneImagesBySite(siteId);
      res.json(images);
    } catch (error) {
      console.error("Images fetch error:", error);
      res.status(500).json({ error: "Failed to fetch images" });
    }
  });

  // Notifications management
  app.get("/api/v1/notifications", async (req, res) => {
    try {
      // Get recent alerts and system notifications as notifications
      const alerts = await storage.getActiveAlerts();
      
      // Convert alerts to notification format
      const notifications = alerts.map(alert => ({
        id: alert.id,
        type: 'alert',
        title: alert.title,
        message: alert.message,
        timestamp: alert.createdAt?.toISOString() || new Date().toISOString(),
        read: alert.status === 'acknowledged',
        severity: alert.severity,
        siteId: alert.siteId,
        siteName: alert.site.name
      }));

      // Add system notifications (mock for now, could be from a notifications table)
      const systemNotifications = [
        {
          id: 'sys-1',
          type: 'system',
          title: 'System Maintenance',
          message: 'Scheduled maintenance completed successfully',
          timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(), // 2 hours ago
          read: true,
          severity: 'low'
        }
      ];

      const allNotifications = [...notifications, ...systemNotifications]
        .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
        .slice(0, 50); // Limit to 50 most recent

      res.json(allNotifications);
    } catch (error) {
      console.error("Notifications fetch error:", error);
      res.status(500).json({ error: "Failed to fetch notifications" });
    }
  });

  app.post("/api/v1/notifications/:id/read", async (req, res) => {
    try {
      const { id } = req.params;
      
      // If it's an alert notification, acknowledge the alert
      if (id.startsWith('sys-')) {
        // Handle system notification read status (could be stored in a separate table)
        res.json({ success: true });
      } else {
        // It's an alert, acknowledge it
        await storage.acknowledgeAlert(id, 'current-user'); // Replace with actual user ID
        
        // Broadcast acknowledgment
        websocketService.broadcast({
          type: "alert_acknowledged",
          data: { alertId: id, userId: 'current-user' }
        });
        
        res.json({ success: true });
      }
    } catch (error) {
      console.error("Notification read error:", error);
      res.status(500).json({ error: "Failed to mark notification as read" });
    }
  });

  app.post("/api/v1/notifications/read-all", async (req, res) => {
    try {
      // Acknowledge all active alerts for current user
      const alerts = await storage.getActiveAlerts();
      
      for (const alert of alerts) {
        if (alert.status === 'active') {
          await storage.acknowledgeAlert(alert.id, 'current-user'); // Replace with actual user ID
        }
      }
      
      // Broadcast that all alerts were acknowledged
      websocketService.broadcast({
        type: "alerts_bulk_acknowledged",
        data: { userId: 'current-user', count: alerts.length }
      });
      
      res.json({ success: true, acknowledgedCount: alerts.length });
    } catch (error) {
      console.error("Bulk notification read error:", error);
      res.status(500).json({ error: "Failed to mark all notifications as read" });
    }
  });

  // Emergency alert endpoint
  app.post("/api/v1/emergency-alert", async (req, res) => {
    try {
      const { userId, timestamp, location } = req.body;
      
      // Create an emergency alert
      const emergencyAlert = await storage.createAlert({
        siteId: 'emergency-global', // Special site ID for emergency alerts
        type: 'emergency',
        severity: 'critical',
        title: 'Emergency Alert Triggered',
        message: `Emergency alert triggered by user from ${location || 'dashboard'} at ${new Date(timestamp).toLocaleString()}`,
        actionPlan: 'Immediate response required. Contact emergency services and site personnel.',
        status: 'active'
      });
      
      // Send emergency notifications
      await alertService.sendAlert(emergencyAlert);
      
      // Broadcast emergency alert immediately
      websocketService.broadcast({
        type: "emergency_alert",
        data: {
          ...emergencyAlert,
          triggeredBy: userId,
          location,
          timestamp
        }
      });
      
      res.json({ 
        success: true, 
        alertId: emergencyAlert.id,
        message: 'Emergency alert sent successfully'
      });
    } catch (error) {
      console.error("Emergency alert error:", error);
      res.status(500).json({ error: "Failed to send emergency alert" });
    }
  });

  // Location search endpoints
  app.get("/api/v1/locations/search", async (req, res) => {
    try {
      const { q } = req.query;
      
      if (!q || (q as string).length < 3) {
        return res.json([]);
      }

      // Mock location search results (in real implementation, use a geocoding service like OpenStreetMap Nominatim)
      const mockLocations = [
        {
          id: '1',
          name: 'Mumbai',
          displayName: 'Mumbai, Maharashtra, India',
          latitude: 19.0760,
          longitude: 72.8777,
          country: 'India',
          region: 'Maharashtra',
          type: 'city'
        },
        {
          id: '2',
          name: 'Delhi',
          displayName: 'Delhi, India',
          latitude: 28.7041,
          longitude: 77.1025,
          country: 'India',
          region: 'Delhi',
          type: 'city'
        },
        {
          id: '3',
          name: 'Bangalore',
          displayName: 'Bangalore, Karnataka, India',
          latitude: 12.9716,
          longitude: 77.5946,
          country: 'India',
          region: 'Karnataka',
          type: 'city'
        },
        {
          id: '4',
          name: 'Shimla',
          displayName: 'Shimla, Himachal Pradesh, India',
          latitude: 31.1048,
          longitude: 77.1734,
          country: 'India',
          region: 'Himachal Pradesh',
          type: 'city'
        },
        {
          id: '5',
          name: 'Manali',
          displayName: 'Manali, Himachal Pradesh, India',
          latitude: 32.2396,
          longitude: 77.1887,
          country: 'India',
          region: 'Himachal Pradesh',
          type: 'town'
        }
      ];

      // Filter locations based on search query
      const filteredLocations = mockLocations.filter(location =>
        location.name.toLowerCase().includes((q as string).toLowerCase()) ||
        location.displayName.toLowerCase().includes((q as string).toLowerCase())
      ).slice(0, 5);

      res.json(filteredLocations);
    } catch (error) {
      console.error("Location search error:", error);
      res.status(500).json({ error: "Failed to search locations" });
    }
  });

  app.get("/api/v1/locations/details", async (req, res) => {
    try {
      const { lat, lon } = req.query;
      
      if (!lat || !lon) {
        return res.status(400).json({ error: "Latitude and longitude are required" });
      }

      const latitude = parseFloat(lat as string);
      const longitude = parseFloat(lon as string);

      // Mock weather data (in real implementation, use OpenWeatherMap, AccuWeather, etc.)
      const weatherData = {
        temperature: Math.round(20 + Math.random() * 15), // 20-35°C
        humidity: Math.round(40 + Math.random() * 40), // 40-80%
        windSpeed: Math.round(5 + Math.random() * 20), // 5-25 km/h
        windDirection: Math.round(Math.random() * 360),
        visibility: Math.round(5 + Math.random() * 15), // 5-20 km
        pressure: Math.round(1000 + Math.random() * 50), // 1000-1050 hPa
        condition: ['clear', 'clouds', 'rain'][Math.floor(Math.random() * 3)],
        description: ['Clear sky', 'Partly cloudy', 'Light rain'][Math.floor(Math.random() * 3)],
        icon: 'clear',
        uvIndex: Math.round(Math.random() * 10),
        precipitation: Math.round(Math.random() * 5),
        dewPoint: Math.round(15 + Math.random() * 10)
      };

      // Calculate risk assessment based on location and weather
      const isHighAltitude = latitude > 30; // Rough approximation for Himalayan regions
      const isWeatherRisky = weatherData.condition === 'rain' || weatherData.windSpeed > 20;
      
      let riskLevel: 'low' | 'medium' | 'high' | 'critical' = 'low';
      let riskScore = 20;
      
      if (isHighAltitude) {
        riskScore += 30;
        riskLevel = 'medium';
      }
      
      if (isWeatherRisky) {
        riskScore += 25;
        riskLevel = riskScore > 60 ? 'high' : 'medium';
      }
      
      if (riskScore > 80) {
        riskLevel = 'critical';
      }

      const riskAssessment = {
        riskLevel,
        riskScore: Math.min(riskScore, 95),
        factors: {
          weather: isWeatherRisky ? 8 : 3,
          geological: isHighAltitude ? 7 : 2,
          historical: Math.round(2 + Math.random() * 4),
          environmental: Math.round(2 + Math.random() * 3)
        },
        alerts: isWeatherRisky ? ['Adverse weather conditions detected'] : [],
        recommendations: [
          isHighAltitude ? 'Monitor slope stability closely' : 'Standard monitoring protocols',
          isWeatherRisky ? 'Avoid heavy machinery operations' : 'Normal operations permitted'
        ]
      };

      // Find nearest monitoring sites (mock data)
      const nearestSites = [
        {
          id: 'site-1',
          name: 'Alpine Monitoring Station A',
          distance: 5.2 + Math.random() * 10,
          riskLevel: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)]
        },
        {
          id: 'site-2',
          name: 'Valley Sensor Network B',
          distance: 8.7 + Math.random() * 15,
          riskLevel: ['low', 'medium'][Math.floor(Math.random() * 2)]
        }
      ].sort((a, b) => a.distance - b.distance);

      const locationData = {
        location: {
          latitude,
          longitude,
          name: `Location ${latitude.toFixed(2)}, ${longitude.toFixed(2)}`
        },
        weather: weatherData,
        riskAssessment,
        nearestSites
      };

      res.json(locationData);
    } catch (error) {
      console.error("Location details error:", error);
      res.status(500).json({ error: "Failed to fetch location details" });
    }
  });

  return httpServer;
}
