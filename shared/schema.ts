import { sql } from "drizzle-orm";
import { pgTable, text, varchar, timestamp, integer, real, boolean, jsonb, point } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";
import { relations } from "drizzle-orm";

export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
  email: text("email").notNull().unique(),
  role: text("role").notNull().default("observer"), // admin, planner, observer
  createdAt: timestamp("created_at").defaultNow(),
});

export const sites = pgTable("sites", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  description: text("description"),
  location: point("location", { mode: "xy" }).notNull(), // PostGIS point
  elevation: real("elevation"),
  slopeAngle: real("slope_angle"),
  aspectAngle: real("aspect_angle"),
  riskLevel: text("risk_level").notNull().default("low"), // low, medium, high, critical
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const sensors = pgTable("sensors", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  siteId: varchar("site_id").references(() => sites.id).notNull(),
  type: text("type").notNull(), // strain, displacement, pore_pressure, tilt, vibration
  location: point("location", { mode: "xy" }).notNull(),
  status: text("status").notNull().default("active"), // active, inactive, maintenance
  lastReading: timestamp("last_reading"),
  batteryLevel: real("battery_level"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const sensorReadings = pgTable("sensor_readings", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  sensorId: varchar("sensor_id").references(() => sensors.id).notNull(),
  value: real("value").notNull(),
  unit: text("unit").notNull(),
  quality: real("quality").default(1.0), // data quality score 0-1
  timestamp: timestamp("timestamp").defaultNow(),
});

export const droneImages = pgTable("drone_images", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  siteId: varchar("site_id").references(() => sites.id).notNull(),
  filename: text("filename").notNull(),
  fileUrl: text("file_url").notNull(),
  captureTime: timestamp("capture_time").notNull(),
  metadata: jsonb("metadata"), // camera settings, GPS, weather
  processed: boolean("processed").default(false),
  createdAt: timestamp("created_at").defaultNow(),
});

export const predictions = pgTable("predictions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  siteId: varchar("site_id").references(() => sites.id).notNull(),
  probability: real("probability").notNull(), // 0-1 risk probability
  confidence: real("confidence").notNull(), // model confidence
  riskLevel: text("risk_level").notNull(), // low, medium, high, critical
  modelVersion: text("model_version").notNull(),
  features: jsonb("features"), // input features used
  explanation: jsonb("explanation"), // SHAP values, top features
  uncertainty: real("uncertainty"), // aleatoric uncertainty
  timestamp: timestamp("timestamp").defaultNow(),
});

export const alerts = pgTable("alerts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  siteId: varchar("site_id").references(() => sites.id).notNull(),
  predictionId: varchar("prediction_id").references(() => predictions.id),
  type: text("type").notNull(), // rockfall, emergency, medical, weather, crime, natural_disaster, wildlife, accident, other
  severity: text("severity").notNull(), // low, medium, high, critical
  title: text("title").notNull(),
  message: text("message").notNull(),
  actionPlan: text("action_plan"),
  status: text("status").notNull().default("active"), // active, acknowledged, resolved
  acknowledgedBy: varchar("acknowledged_by").references(() => users.id),
  acknowledgedAt: timestamp("acknowledged_at"),
  createdAt: timestamp("created_at").defaultNow(),
});

// User settings for tourist safety preferences
export const userSettings = pgTable("user_settings", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().default("default"), // In real app, would reference users table
  geoFenceEnabled: boolean("geo_fence_enabled").notNull().default(true),
  geoFenceRadius: integer("geo_fence_radius").notNull().default(5),
  alertOnExit: boolean("alert_on_exit").notNull().default(true),
  alertOnEntry: boolean("alert_on_entry").notNull().default(false),
  alertTypes: jsonb("alert_types").notNull().default({
    medical: true,
    weather: true,
    crime: false,
    naturalDisaster: true,
    wildlife: true,
    accident: true,
    other: false
  }),
  severityFilter: jsonb("severity_filter").notNull().default({
    critical: true,
    high: true,
    medium: false,
    low: false
  }),
  notificationSettings: jsonb("notification_settings").notNull().default({
    pushNotifications: true,
    emailAlerts: false,
    smsAlerts: false,
    soundEnabled: true,
    vibrationEnabled: true
  }),
  emergencyContact1: text("emergency_contact_1"),
  emergencyContact2: text("emergency_contact_2"),
  medicalInfo: text("medical_info"),
  bloodType: text("blood_type"),
  allergies: text("allergies"),
  shareLocation: boolean("share_location").notNull().default(true),
  highAccuracyMode: boolean("high_accuracy_mode").notNull().default(false),
  locationHistory: boolean("location_history").notNull().default(true),
  batteryOptimization: boolean("battery_optimization").notNull().default(true),
  updatedAt: timestamp("updated_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const models = pgTable("models", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  version: text("version").notNull(),
  type: text("type").notNull(), // cnn, lstm, fusion, ensemble
  filePath: text("file_path").notNull(),
  metrics: jsonb("metrics"), // accuracy, precision, recall, f1
  isActive: boolean("is_active").default(false),
  trainedAt: timestamp("trained_at").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const alertNotifications = pgTable("alert_notifications", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  alertId: varchar("alert_id").references(() => alerts.id).notNull(),
  channel: text("channel").notNull(), // sms, email, whatsapp, push
  recipient: text("recipient").notNull(),
  status: text("status").notNull().default("pending"), // pending, sent, failed
  sentAt: timestamp("sent_at"),
  createdAt: timestamp("created_at").defaultNow(),
});

// Relations
export const sitesRelations = relations(sites, ({ many }) => ({
  sensors: many(sensors),
  droneImages: many(droneImages),
  predictions: many(predictions),
  alerts: many(alerts),
}));

export const sensorsRelations = relations(sensors, ({ one, many }) => ({
  site: one(sites, {
    fields: [sensors.siteId],
    references: [sites.id],
  }),
  readings: many(sensorReadings),
}));

export const sensorReadingsRelations = relations(sensorReadings, ({ one }) => ({
  sensor: one(sensors, {
    fields: [sensorReadings.sensorId],
    references: [sensors.id],
  }),
}));

export const droneImagesRelations = relations(droneImages, ({ one }) => ({
  site: one(sites, {
    fields: [droneImages.siteId],
    references: [sites.id],
  }),
}));

export const predictionsRelations = relations(predictions, ({ one }) => ({
  site: one(sites, {
    fields: [predictions.siteId],
    references: [sites.id],
  }),
}));

export const alertsRelations = relations(alerts, ({ one, many }) => ({
  site: one(sites, {
    fields: [alerts.siteId],
    references: [sites.id],
  }),
  prediction: one(predictions, {
    fields: [alerts.predictionId],
    references: [predictions.id],
  }),
  acknowledgedByUser: one(users, {
    fields: [alerts.acknowledgedBy],
    references: [users.id],
  }),
  notifications: many(alertNotifications),
}));

export const alertNotificationsRelations = relations(alertNotifications, ({ one }) => ({
  alert: one(alerts, {
    fields: [alertNotifications.alertId],
    references: [alerts.id],
  }),
}));

// Insert schemas
export const insertUserSchema = createInsertSchema(users).omit({
  id: true,
  createdAt: true,
});

export const insertSiteSchema = createInsertSchema(sites).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertSensorSchema = createInsertSchema(sensors).omit({
  id: true,
  createdAt: true,
});

export const insertSensorReadingSchema = createInsertSchema(sensorReadings).omit({
  id: true,
  timestamp: true,
});

export const insertDroneImageSchema = createInsertSchema(droneImages).omit({
  id: true,
  createdAt: true,
});

export const insertPredictionSchema = createInsertSchema(predictions).omit({
  id: true,
  timestamp: true,
});

export const insertAlertSchema = createInsertSchema(alerts).omit({
  id: true,
  createdAt: true,
});

export const insertModelSchema = createInsertSchema(models).omit({
  id: true,
  createdAt: true,
});

export const insertAlertNotificationSchema = createInsertSchema(alertNotifications).omit({
  id: true,
  createdAt: true,
});

export const insertUserSettingsSchema = createInsertSchema(userSettings).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

// Enhanced Alert schema with tourist safety validation
export const touristAlertSchema = z.object({
  siteId: z.string().min(1, "Site ID is required"),
  type: z.enum(["emergency", "medical", "weather", "crime", "natural_disaster", "wildlife", "accident", "other"]),
  severity: z.enum(["low", "medium", "high", "critical"]),
  title: z.string().min(1, "Title is required").max(100, "Title too long"),
  message: z.string().min(1, "Message is required").max(1000, "Message too long"),
  actionPlan: z.string().optional(),
  status: z.enum(["active", "acknowledged", "resolved"]).default("active"),
});

// User settings validation schema
export const userSettingsValidationSchema = z.object({
  geoFence: z.object({
    enabled: z.boolean(),
    radius: z.number().min(1).max(100),
    alertOnExit: z.boolean(),
    alertOnEntry: z.boolean(),
  }),
  alertTypes: z.object({
    medical: z.boolean(),
    weather: z.boolean(),
    crime: z.boolean(),
    naturalDisaster: z.boolean(),
    wildlife: z.boolean(),
    accident: z.boolean(),
    other: z.boolean(),
  }),
  severityFilter: z.object({
    critical: z.boolean(),
    high: z.boolean(),
    medium: z.boolean(),
    low: z.boolean(),
  }),
  notifications: z.object({
    pushNotifications: z.boolean(),
    emailAlerts: z.boolean(),
    smsAlerts: z.boolean(),
    soundEnabled: z.boolean(),
    vibrationEnabled: z.boolean(),
  }),
  contacts: z.object({
    emergencyContact1: z.string().optional(),
    emergencyContact2: z.string().optional(),
    medicalInfo: z.string().optional(),
    bloodType: z.string().optional(),
    allergies: z.string().optional(),
  }),
  location: z.object({
    shareLocation: z.boolean(),
    highAccuracyMode: z.boolean(),
    locationHistory: z.boolean(),
    batteryOptimization: z.boolean(),
  }),
});

// Types
export type User = typeof users.$inferSelect;
export type InsertUser = z.infer<typeof insertUserSchema>;
export type Site = typeof sites.$inferSelect;
export type InsertSite = z.infer<typeof insertSiteSchema>;
export type Sensor = typeof sensors.$inferSelect;
export type InsertSensor = z.infer<typeof insertSensorSchema>;
export type SensorReading = typeof sensorReadings.$inferSelect;
export type InsertSensorReading = z.infer<typeof insertSensorReadingSchema>;
export type DroneImage = typeof droneImages.$inferSelect;
export type InsertDroneImage = z.infer<typeof insertDroneImageSchema>;
export type Prediction = typeof predictions.$inferSelect;
export type InsertPrediction = z.infer<typeof insertPredictionSchema>;
export type Alert = typeof alerts.$inferSelect;
export type InsertAlert = z.infer<typeof insertAlertSchema>;
export type Model = typeof models.$inferSelect;
export type InsertModel = z.infer<typeof insertModelSchema>;
export type AlertNotification = typeof alertNotifications.$inferSelect;
export type InsertAlertNotification = z.infer<typeof insertAlertNotificationSchema>;
export type UserSettings = typeof userSettings.$inferSelect;
export type InsertUserSettings = z.infer<typeof insertUserSettingsSchema>;
export type TouristAlert = z.infer<typeof touristAlertSchema>;
export type UserSettingsValidation = z.infer<typeof userSettingsValidationSchema>;
