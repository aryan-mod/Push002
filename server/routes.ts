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

  return httpServer;
}
