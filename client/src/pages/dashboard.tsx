import { useQuery } from "@tanstack/react-query";
import Header from "@/components/layout/header";
import MetricsCards from "@/components/dashboard/metrics-cards";
import InteractiveMap from "@/components/dashboard/interactive-map";
import RealTimeAlerts from "@/components/dashboard/real-time-alerts";
import TimeSeriesChart from "@/components/dashboard/time-series-chart";
import ModelPerformance from "@/components/dashboard/model-performance";
import SiteDetailsPanel from "@/components/dashboard/site-details-panel";
import { useState } from "react";
import type { Site, Alert } from "@shared/schema";

interface DashboardMetrics {
  highRisk: number;
  mediumRisk: number;
  lowRisk: number;
  activeSensors: number;
  modelAccuracy: string;
}

export default function Dashboard() {
  const [selectedSiteId, setSelectedSiteId] = useState<string | null>(null);

  const { data: metrics, isLoading: metricsLoading } = useQuery<DashboardMetrics>({
    queryKey: ["/api/v1/dashboard/metrics"],
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const { data: sites = [], isLoading: sitesLoading } = useQuery<Site[]>({
    queryKey: ["/api/v1/sites"],
    refetchInterval: 60000, // Refresh every minute
  });

  const { data: alerts = [], isLoading: alertsLoading } = useQuery<(Alert & { site: Site })[]>({
    queryKey: ["/api/v1/alerts"],
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  return (
    <div className="flex flex-col">
      <Header 
        title="Risk Assessment Dashboard"
        subtitle="Real-time rockfall monitoring and prediction"
      />
      
      <div className="flex-1 p-6 space-y-6">
        {/* Key Metrics Cards */}
        <MetricsCards 
          metrics={metrics} 
          isLoading={metricsLoading} 
        />

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Interactive Map */}
          <div className="lg:col-span-2">
            <InteractiveMap
              sites={sites}
              isLoading={sitesLoading}
              onSiteSelect={setSelectedSiteId}
            />
          </div>

          {/* Real-time Alerts */}
          <RealTimeAlerts
            alerts={alerts}
            isLoading={alertsLoading}
          />
        </div>

        {/* Additional Dashboard Sections */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <TimeSeriesChart />
          <ModelPerformance />
        </div>

        {/* Site Details Panel */}
        {selectedSiteId && (
          <SiteDetailsPanel
            siteId={selectedSiteId}
            onClose={() => setSelectedSiteId(null)}
          />
        )}
      </div>
    </div>
  );
}
