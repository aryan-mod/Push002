import { apiRequest } from "./queryClient";

export interface PredictionRequest {
  siteId: string;
  timestamp?: string;
  sensorData?: any[];
  imageUrl?: string;
}

export interface PredictionResponse {
  probability: number;
  category: string;
  explanation: {
    topFeatures: string[];
    confidence: number;
  };
  modelVersion: string;
  uncertainty: number;
}

export interface IngestionRequest {
  type: "sensor_reading" | "drone_image";
  data: any;
}

export interface AlertRequest {
  siteId: string;
  predictionId?: string;
  type: string;
  severity: string;
  title: string;
  message: string;
  actionPlan?: string;
}

export class ApiClient {
  // Prediction API
  static async makePrediction(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await apiRequest("POST", "/api/v1/predict", request);
    return response.json();
  }

  // Data Ingestion API
  static async ingestData(request: IngestionRequest) {
    const response = await apiRequest("POST", "/api/v1/ingest", request);
    return response.json();
  }

  // Alert Management API
  static async createAlert(request: AlertRequest) {
    const response = await apiRequest("POST", "/api/v1/alert", request);
    return response.json();
  }

  static async acknowledgeAlert(alertId: string, userId: string) {
    const response = await apiRequest("POST", `/api/v1/alerts/${alertId}/acknowledge`, { userId });
    return response.json();
  }

  // Model Management API
  static async activateModel(modelId: string, type: string) {
    const response = await apiRequest("POST", `/api/v1/models/${modelId}/activate`, { type });
    return response.json();
  }

  // Sites API
  static async createSite(siteData: any) {
    const response = await apiRequest("POST", "/api/v1/sites", siteData);
    return response.json();
  }

  static async createSensor(siteId: string, sensorData: any) {
    const response = await apiRequest("POST", `/api/v1/sites/${siteId}/sensors`, sensorData);
    return response.json();
  }

  // Sensor Readings API
  static async getSensorReadings(sensorId: string, from?: Date, to?: Date) {
    const params = new URLSearchParams();
    if (from) params.append("from", from.toISOString());
    if (to) params.append("to", to.toISOString());
    
    const url = `/api/v1/sensors/${sensorId}/readings${params.toString() ? `?${params}` : ''}`;
    const response = await apiRequest("GET", url);
    return response.json();
  }

  // Upload helpers
  static async uploadSensorReading(sensorId: string, value: number, unit: string, quality = 1.0) {
    return this.ingestData({
      type: "sensor_reading",
      data: {
        sensorId,
        value,
        unit,
        quality
      }
    });
  }

  static async uploadDroneImage(siteId: string, filename: string, fileUrl: string, captureTime: Date, metadata?: any) {
    return this.ingestData({
      type: "drone_image",
      data: {
        siteId,
        filename,
        fileUrl,
        captureTime: captureTime.toISOString(),
        metadata
      }
    });
  }

  // Utility methods
  static async healthCheck() {
    try {
      const response = await apiRequest("GET", "/api/v1/health");
      return response.ok;
    } catch {
      return false;
    }
  }

  static async getSystemStatus() {
    const response = await apiRequest("GET", "/api/v1/status");
    return response.json();
  }
}

export default ApiClient;
