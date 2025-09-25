import { 
  sites, sensors, sensorReadings, droneImages, predictions, alerts, models, alertNotifications, users,
  type Site, type InsertSite, type Sensor, type InsertSensor, type SensorReading, type InsertSensorReading,
  type DroneImage, type InsertDroneImage, type Prediction, type InsertPrediction, type Alert, type InsertAlert,
  type Model, type InsertModel, type AlertNotification, type InsertAlertNotification, type User, type InsertUser
} from "@shared/schema";
import { db } from "./db";
import { eq, desc, and, gte, lte, sql } from "drizzle-orm";

export interface IStorage {
  // Users
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;

  // Sites
  getSites(): Promise<Site[]>;
  getSite(id: string): Promise<Site | undefined>;
  createSite(site: InsertSite): Promise<Site>;
  updateSiteRiskLevel(id: string, riskLevel: string): Promise<void>;

  // Sensors
  getSensorsBySite(siteId: string): Promise<Sensor[]>;
  getSensor(id: string): Promise<Sensor | undefined>;
  createSensor(sensor: InsertSensor): Promise<Sensor>;
  updateSensorStatus(id: string, status: string, batteryLevel?: number): Promise<void>;

  // Sensor Readings
  getSensorReadings(sensorId: string, from?: Date, to?: Date): Promise<SensorReading[]>;
  createSensorReading(reading: InsertSensorReading): Promise<SensorReading>;
  getLatestReadingsBySite(siteId: string): Promise<(SensorReading & { sensor: Sensor })[]>;

  // Drone Images
  getDroneImagesBySite(siteId: string): Promise<DroneImage[]>;
  createDroneImage(image: InsertDroneImage): Promise<DroneImage>;
  markImageProcessed(id: string): Promise<void>;

  // Predictions
  getPredictionsBySite(siteId: string, limit?: number): Promise<Prediction[]>;
  getLatestPrediction(siteId: string): Promise<Prediction | undefined>;
  createPrediction(prediction: InsertPrediction): Promise<Prediction>;

  // Alerts
  getActiveAlerts(): Promise<(Alert & { site: Site })[]>;
  getAlertsBySite(siteId: string): Promise<Alert[]>;
  createAlert(alert: InsertAlert): Promise<Alert>;
  acknowledgeAlert(id: string, userId: string): Promise<void>;

  // Models
  getModels(): Promise<Model[]>;
  getActiveModel(type: string): Promise<Model | undefined>;
  createModel(model: InsertModel): Promise<Model>;
  setActiveModel(id: string, type: string): Promise<void>;

  // Alert Notifications
  createAlertNotification(notification: InsertAlertNotification): Promise<AlertNotification>;
  updateNotificationStatus(id: string, status: string): Promise<void>;

  // Analytics
  getRiskMetrics(): Promise<{
    highRisk: number;
    mediumRisk: number;
    lowRisk: number;
    activeSensors: number;
  }>;
}

export class DatabaseStorage implements IStorage {
  async getUser(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user || undefined;
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.username, username));
    return user || undefined;
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const [user] = await db.insert(users).values(insertUser).returning();
    return user;
  }

  async getSites(): Promise<Site[]> {
    return await db.select().from(sites).orderBy(sites.name);
  }

  async getSite(id: string): Promise<Site | undefined> {
    const [site] = await db.select().from(sites).where(eq(sites.id, id));
    return site || undefined;
  }

  async createSite(insertSite: InsertSite): Promise<Site> {
    const [site] = await db.insert(sites).values(insertSite).returning();
    return site;
  }

  async updateSiteRiskLevel(id: string, riskLevel: string): Promise<void> {
    await db.update(sites)
      .set({ riskLevel, updatedAt: new Date() })
      .where(eq(sites.id, id));
  }

  async getSensorsBySite(siteId: string): Promise<Sensor[]> {
    return await db.select().from(sensors).where(eq(sensors.siteId, siteId));
  }

  async getSensor(id: string): Promise<Sensor | undefined> {
    const [sensor] = await db.select().from(sensors).where(eq(sensors.id, id));
    return sensor || undefined;
  }

  async createSensor(insertSensor: InsertSensor): Promise<Sensor> {
    const [sensor] = await db.insert(sensors).values(insertSensor).returning();
    return sensor;
  }

  async updateSensorStatus(id: string, status: string, batteryLevel?: number): Promise<void> {
    const updateData: any = { status, lastReading: new Date() };
    if (batteryLevel !== undefined) {
      updateData.batteryLevel = batteryLevel;
    }
    await db.update(sensors).set(updateData).where(eq(sensors.id, id));
  }

  async getSensorReadings(sensorId: string, from?: Date, to?: Date): Promise<SensorReading[]> {
    let conditions = [eq(sensorReadings.sensorId, sensorId)];
    
    if (from) {
      conditions.push(gte(sensorReadings.timestamp, from));
    }
    if (to) {
      conditions.push(lte(sensorReadings.timestamp, to));
    }
    
    return await db
      .select()
      .from(sensorReadings)
      .where(and(...conditions))
      .orderBy(desc(sensorReadings.timestamp));
  }

  async createSensorReading(insertReading: InsertSensorReading): Promise<SensorReading> {
    const [reading] = await db.insert(sensorReadings).values(insertReading).returning();
    return reading;
  }

  async getLatestReadingsBySite(siteId: string): Promise<(SensorReading & { sensor: Sensor })[]> {
    return await db
      .select({
        id: sensorReadings.id,
        sensorId: sensorReadings.sensorId,
        value: sensorReadings.value,
        unit: sensorReadings.unit,
        quality: sensorReadings.quality,
        timestamp: sensorReadings.timestamp,
        sensor: sensors
      })
      .from(sensorReadings)
      .innerJoin(sensors, eq(sensorReadings.sensorId, sensors.id))
      .where(eq(sensors.siteId, siteId))
      .orderBy(desc(sensorReadings.timestamp))
      .limit(50);
  }

  async getDroneImagesBySite(siteId: string): Promise<DroneImage[]> {
    return await db.select().from(droneImages)
      .where(eq(droneImages.siteId, siteId))
      .orderBy(desc(droneImages.captureTime));
  }

  async createDroneImage(insertImage: InsertDroneImage): Promise<DroneImage> {
    const [image] = await db.insert(droneImages).values(insertImage).returning();
    return image;
  }

  async markImageProcessed(id: string): Promise<void> {
    await db.update(droneImages).set({ processed: true }).where(eq(droneImages.id, id));
  }

  async getPredictionsBySite(siteId: string, limit = 10): Promise<Prediction[]> {
    return await db.select().from(predictions)
      .where(eq(predictions.siteId, siteId))
      .orderBy(desc(predictions.timestamp))
      .limit(limit);
  }

  async getLatestPrediction(siteId: string): Promise<Prediction | undefined> {
    const [prediction] = await db.select().from(predictions)
      .where(eq(predictions.siteId, siteId))
      .orderBy(desc(predictions.timestamp))
      .limit(1);
    return prediction || undefined;
  }

  async createPrediction(insertPrediction: InsertPrediction): Promise<Prediction> {
    const [prediction] = await db.insert(predictions).values(insertPrediction).returning();
    return prediction;
  }

  async getActiveAlerts(): Promise<(Alert & { site: Site })[]> {
    return await db
      .select({
        id: alerts.id,
        siteId: alerts.siteId,
        predictionId: alerts.predictionId,
        type: alerts.type,
        severity: alerts.severity,
        title: alerts.title,
        message: alerts.message,
        actionPlan: alerts.actionPlan,
        status: alerts.status,
        acknowledgedBy: alerts.acknowledgedBy,
        acknowledgedAt: alerts.acknowledgedAt,
        createdAt: alerts.createdAt,
        site: sites
      })
      .from(alerts)
      .innerJoin(sites, eq(alerts.siteId, sites.id))
      .where(eq(alerts.status, "active"))
      .orderBy(desc(alerts.createdAt));
  }

  async getAlertsBySite(siteId: string): Promise<Alert[]> {
    return await db.select().from(alerts)
      .where(eq(alerts.siteId, siteId))
      .orderBy(desc(alerts.createdAt));
  }

  async createAlert(insertAlert: InsertAlert): Promise<Alert> {
    const [alert] = await db.insert(alerts).values(insertAlert).returning();
    return alert;
  }

  async acknowledgeAlert(id: string, userId: string): Promise<void> {
    await db.update(alerts)
      .set({ 
        status: "acknowledged", 
        acknowledgedBy: userId, 
        acknowledgedAt: new Date() 
      })
      .where(eq(alerts.id, id));
  }

  async getModels(): Promise<Model[]> {
    return await db.select().from(models).orderBy(desc(models.trainedAt));
  }

  async getActiveModel(type: string): Promise<Model | undefined> {
    const [model] = await db.select().from(models)
      .where(and(eq(models.type, type), eq(models.isActive, true)));
    return model || undefined;
  }

  async createModel(insertModel: InsertModel): Promise<Model> {
    const [model] = await db.insert(models).values(insertModel).returning();
    return model;
  }

  async setActiveModel(id: string, type: string): Promise<void> {
    // Deactivate all models of this type
    await db.update(models)
      .set({ isActive: false })
      .where(eq(models.type, type));
    
    // Activate the selected model
    await db.update(models)
      .set({ isActive: true })
      .where(eq(models.id, id));
  }

  async createAlertNotification(insertNotification: InsertAlertNotification): Promise<AlertNotification> {
    const [notification] = await db.insert(alertNotifications).values(insertNotification).returning();
    return notification;
  }

  async updateNotificationStatus(id: string, status: string): Promise<void> {
    const updateData: any = { status };
    if (status === "sent") {
      updateData.sentAt = new Date();
    }
    await db.update(alertNotifications).set(updateData).where(eq(alertNotifications.id, id));
  }

  async getRiskMetrics(): Promise<{
    highRisk: number;
    mediumRisk: number;
    lowRisk: number;
    activeSensors: number;
  }> {
    const [riskCounts] = await db
      .select({
        highRisk: sql<number>`count(*) filter (where risk_level in ('high', 'critical'))`,
        mediumRisk: sql<number>`count(*) filter (where risk_level = 'medium')`,
        lowRisk: sql<number>`count(*) filter (where risk_level = 'low')`,
      })
      .from(sites)
      .where(eq(sites.isActive, true));

    const [sensorCount] = await db
      .select({
        activeSensors: sql<number>`count(*)`
      })
      .from(sensors)
      .where(eq(sensors.status, "active"));

    return {
      highRisk: riskCounts.highRisk,
      mediumRisk: riskCounts.mediumRisk,
      lowRisk: riskCounts.lowRisk,
      activeSensors: sensorCount.activeSensors,
    };
  }
}

export const storage = new DatabaseStorage();
