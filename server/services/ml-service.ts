import { storage } from "../storage";
import type { InsertPrediction } from "@shared/schema";

interface PredictionInput {
  siteId: string;
  timestamp: Date;
  sensorData?: any[];
  imageUrl?: string;
}

interface PredictionResult {
  probability: number;
  category: string;
  explanation: {
    topFeatures: string[];
    confidence: number;
  };
  modelVersion: string;
  uncertainty: number;
}

class MLService {
  private predictionQueue: Set<string> = new Set();
  private imageProcessingQueue: Set<string> = new Set();

  async predict(input: PredictionInput): Promise<PredictionResult> {
    try {
      // Get site data
      const site = await storage.getSite(input.siteId);
      if (!site) {
        throw new Error("Site not found");
      }

      // Get latest sensor readings
      const sensorReadings = await storage.getLatestReadingsBySite(input.siteId);
      
      // Process features
      const features = await this.extractFeatures(site, sensorReadings, input.sensorData);
      
      // Run prediction (simplified simulation)
      const prediction = await this.runModel(features, input.imageUrl);
      
      // Create prediction record
      const predictionData: InsertPrediction = {
        siteId: input.siteId,
        probability: prediction.probability,
        confidence: prediction.explanation.confidence,
        riskLevel: prediction.category,
        modelVersion: prediction.modelVersion,
        features: features,
        explanation: prediction.explanation,
        uncertainty: prediction.uncertainty
      };

      const savedPrediction = await storage.createPrediction(predictionData);
      
      // Update site risk level if needed
      if (prediction.probability > 0.7) {
        const newRiskLevel = prediction.probability > 0.9 ? "critical" : "high";
        await storage.updateSiteRiskLevel(input.siteId, newRiskLevel);
        
        // Create alert if risk is high
        if (prediction.probability > 0.8) {
          await this.createHighRiskAlert(input.siteId, savedPrediction.id, prediction);
        }
      }

      return prediction;
    } catch (error) {
      console.error("Prediction error:", error);
      throw new Error("Prediction failed");
    }
  }

  async queuePrediction(siteId: string): Promise<void> {
    if (this.predictionQueue.has(siteId)) {
      return; // Already queued
    }

    this.predictionQueue.add(siteId);
    
    // Process after a short delay to batch requests
    setTimeout(async () => {
      try {
        await this.predict({
          siteId,
          timestamp: new Date()
        });
      } catch (error) {
        console.error(`Queued prediction failed for site ${siteId}:`, error);
      } finally {
        this.predictionQueue.delete(siteId);
      }
    }, 5000);
  }

  async queueImageProcessing(imageId: string): Promise<void> {
    if (this.imageProcessingQueue.has(imageId)) {
      return;
    }

    this.imageProcessingQueue.add(imageId);
    
    // Simulate image processing
    setTimeout(async () => {
      try {
        await storage.markImageProcessed(imageId);
      } catch (error) {
        console.error(`Image processing failed for ${imageId}:`, error);
      } finally {
        this.imageProcessingQueue.delete(imageId);
      }
    }, 10000);
  }

  private async extractFeatures(site: any, sensorReadings: any[], additionalData?: any[]): Promise<any> {
    const features: any = {
      siteFeatures: {
        elevation: site.elevation || 0,
        slopeAngle: site.slopeAngle || 0,
        aspectAngle: site.aspectAngle || 0
      },
      sensorFeatures: {},
      timeSeriesFeatures: {}
    };

    // Process sensor readings
    const sensorTypes = ['strain', 'displacement', 'pore_pressure', 'tilt', 'vibration'];
    
    for (const type of sensorTypes) {
      const typeReadings = sensorReadings.filter(r => r.sensor.type === type);
      if (typeReadings.length > 0) {
        const values = typeReadings.map(r => r.value);
        features.sensorFeatures[type] = {
          latest: values[0],
          mean: values.reduce((a, b) => a + b, 0) / values.length,
          std: this.calculateStd(values),
          trend: this.calculateTrend(values)
        };
      }
    }

    // Add weather data (simulated)
    features.weatherFeatures = {
      rainfall: Math.random() * 50, // mm
      temperature: 15 + Math.random() * 20, // °C
      humidity: 40 + Math.random() * 40 // %
    };

    return features;
  }

  private async runModel(features: any, imageUrl?: string): Promise<PredictionResult> {
    // Simulate ML model inference
    const slopeRisk = Math.min(features.siteFeatures.slopeAngle / 45, 1);
    const sensorRisk = this.calculateSensorRisk(features.sensorFeatures);
    const weatherRisk = features.weatherFeatures.rainfall / 100;
    
    // Weighted combination
    const probability = Math.min(
      (slopeRisk * 0.4 + sensorRisk * 0.4 + weatherRisk * 0.2) * (0.8 + Math.random() * 0.4),
      1
    );

    const category = this.getCategoryFromProbability(probability);
    const confidence = 0.85 + Math.random() * 0.15;

    // Generate explanation
    const topFeatures = this.getTopFeatures(features, probability);

    return {
      probability,
      category,
      explanation: {
        topFeatures,
        confidence
      },
      modelVersion: "v2.1.3",
      uncertainty: Math.random() * 0.1 + 0.05
    };
  }

  private calculateSensorRisk(sensorFeatures: any): number {
    let totalRisk = 0;
    let count = 0;

    for (const [type, data] of Object.entries(sensorFeatures)) {
      const typedData = data as any;
      let typeRisk = 0;

      switch (type) {
        case 'strain':
          typeRisk = Math.min(typedData.latest / 1000, 1); // Normalize to microstrains
          break;
        case 'displacement':
          typeRisk = Math.min(typedData.latest / 10, 1); // Normalize to mm
          break;
        case 'pore_pressure':
          typeRisk = Math.min(typedData.latest / 500, 1); // Normalize to kPa
          break;
        case 'tilt':
          typeRisk = Math.min(typedData.latest / 5, 1); // Normalize to degrees
          break;
        case 'vibration':
          typeRisk = Math.min(typedData.latest / 50, 1); // Normalize to Hz
          break;
      }

      // Factor in trend
      if (typedData.trend > 0) {
        typeRisk *= 1.5; // Increasing trend is more risky
      }

      totalRisk += typeRisk;
      count++;
    }

    return count > 0 ? totalRisk / count : 0;
  }

  private getCategoryFromProbability(probability: number): string {
    if (probability >= 0.8) return "critical";
    if (probability >= 0.6) return "high";
    if (probability >= 0.3) return "medium";
    return "low";
  }

  private getTopFeatures(features: any, probability: number): string[] {
    const featureImportance = [
      { name: "Slope angle", value: features.siteFeatures.slopeAngle },
      { name: "Pore pressure", value: features.sensorFeatures.pore_pressure?.latest || 0 },
      { name: "Displacement", value: features.sensorFeatures.displacement?.latest || 0 },
      { name: "Recent rainfall", value: features.weatherFeatures.rainfall },
      { name: "Strain readings", value: features.sensorFeatures.strain?.latest || 0 }
    ];

    return featureImportance
      .sort((a, b) => b.value - a.value)
      .slice(0, 3)
      .map(f => f.name);
  }

  private async createHighRiskAlert(siteId: string, predictionId: string, prediction: PredictionResult): Promise<void> {
    const severity = prediction.category === "critical" ? "critical" : "high";
    
    await storage.createAlert({
      siteId,
      predictionId,
      type: "threshold",
      severity,
      title: `${prediction.category.charAt(0).toUpperCase() + prediction.category.slice(1)} Risk Detected`,
      message: `Risk probability: ${(prediction.probability * 100).toFixed(1)}%`,
      actionPlan: this.getActionPlan(prediction.category),
      status: "active"
    });
  }

  private getActionPlan(riskLevel: string): string {
    switch (riskLevel) {
      case "critical":
        return "1. Evacuate personnel immediately\n2. Halt all operations\n3. Contact emergency services\n4. Increase monitoring to 1-minute intervals";
      case "high":
        return "1. Restrict access to danger zone\n2. Pause blasting operations\n3. Increase monitoring frequency\n4. Prepare evacuation plan";
      case "medium":
        return "1. Monitor closely\n2. Brief personnel on risk\n3. Review safety protocols\n4. Consider operational adjustments";
      default:
        return "Continue normal monitoring";
    }
  }

  private calculateStd(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
    return Math.sqrt(variance);
  }

  private calculateTrend(values: number[]): number {
    if (values.length < 2) return 0;
    
    // Simple linear trend calculation
    const n = values.length;
    const x = Array.from({length: n}, (_, i) => i);
    const meanX = x.reduce((a, b) => a + b, 0) / n;
    const meanY = values.reduce((a, b) => a + b, 0) / n;
    
    const numerator = x.reduce((acc, xi, i) => acc + (xi - meanX) * (values[i] - meanY), 0);
    const denominator = x.reduce((acc, xi) => acc + (xi - meanX) ** 2, 0);
    
    return denominator === 0 ? 0 : numerator / denominator;
  }
}

export const mlService = new MLService();
