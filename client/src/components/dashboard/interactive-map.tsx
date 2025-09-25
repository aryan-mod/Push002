import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Plus, Minus, AlertTriangle, TriangleAlert } from "lucide-react";
import type { Site } from "@shared/schema";

interface InteractiveMapProps {
  sites: Site[];
  isLoading: boolean;
  onSiteSelect: (siteId: string) => void;
}

export default function InteractiveMap({ sites, isLoading, onSiteSelect }: InteractiveMapProps) {
  const [mapView, setMapView] = useState<"satellite" | "terrain" | "heatmap">("satellite");

  if (isLoading) {
    return (
      <Card className="border border-border">
        <CardHeader>
          <div className="flex items-center justify-between">
            <Skeleton className="h-6 w-48" />
            <div className="flex space-x-2">
              <Skeleton className="h-8 w-20" />
              <Skeleton className="h-8 w-20" />
              <Skeleton className="h-8 w-20" />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-96 w-full" />
        </CardContent>
      </Card>
    );
  }

  const getRiskZones = () => {
    const highRiskSites = sites.filter(site => site.riskLevel === "critical" || site.riskLevel === "high");
    const mediumRiskSites = sites.filter(site => site.riskLevel === "medium");
    
    return { highRiskSites, mediumRiskSites };
  };

  const { highRiskSites, mediumRiskSites } = getRiskZones();

  return (
    <Card className="border border-border">
      <CardHeader className="border-b border-border">
        <div className="flex items-center justify-between">
          <CardTitle>Risk Assessment Map</CardTitle>
          <div className="flex items-center space-x-2">
            <Button
              variant={mapView === "satellite" ? "default" : "outline"}
              size="sm"
              onClick={() => setMapView("satellite")}
              data-testid="button-map-satellite"
            >
              Satellite
            </Button>
            <Button
              variant={mapView === "terrain" ? "default" : "outline"}
              size="sm"
              onClick={() => setMapView("terrain")}
              data-testid="button-map-terrain"
            >
              Terrain
            </Button>
            <Button
              variant={mapView === "heatmap" ? "default" : "outline"}
              size="sm"
              onClick={() => setMapView("heatmap")}
              data-testid="button-map-heatmap"
            >
              Heatmap
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-0">
        {/* Map container with risk visualization */}
        <div className="relative h-96 map-container overflow-hidden">
          {/* Mountainous terrain background */}
          <div className="absolute inset-0 bg-gradient-to-br from-green-100 via-yellow-50 to-red-50 dark:from-green-900/20 dark:via-yellow-900/20 dark:to-red-900/20"></div>
          
          {/* Risk zones visualization */}
          <div className="absolute inset-4 space-y-4">
            {/* High Risk Zones */}
            {highRiskSites.slice(0, 3).map((site, index) => (
              <div
                key={site.id}
                className={`absolute animate-pulse cursor-pointer
                  ${index === 0 ? 'top-8 left-12 w-16 h-12' : ''}
                  ${index === 1 ? 'top-20 right-16 w-12 h-8' : ''}
                  ${index === 2 ? 'bottom-16 left-8 w-10 h-10' : ''}
                  bg-red-500/30 rounded-full border-2 border-red-500`}
                onClick={() => onSiteSelect(site.id)}
                data-testid={`risk-zone-high-${site.id}`}
              >
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  <AlertTriangle className="text-red-600 h-4 w-4" />
                </div>
              </div>
            ))}
            
            {/* Medium Risk Zones */}
            {mediumRiskSites.slice(0, 2).map((site, index) => (
              <div
                key={site.id}
                className={`absolute cursor-pointer
                  ${index === 0 ? 'top-32 right-1/3 w-10 h-10' : ''}
                  ${index === 1 ? 'bottom-24 left-1/4 w-8 h-8' : ''}
                  bg-yellow-400/30 rounded-full border border-yellow-500`}
                onClick={() => onSiteSelect(site.id)}
                data-testid={`risk-zone-medium-${site.id}`}
              >
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  <TriangleAlert className="text-yellow-600 h-3 w-3" />
                </div>
              </div>
            ))}
            
            {/* Sensor Locations */}
            {sites.slice(0, 4).map((site, index) => (
              <div
                key={`sensor-${site.id}`}
                className={`absolute w-3 h-3 bg-blue-500 rounded-full pulse-dot cursor-pointer
                  ${index === 0 ? 'top-16 left-1/2' : ''}
                  ${index === 1 ? 'top-32 right-1/3' : ''}
                  ${index === 2 ? 'bottom-24 left-1/4' : ''}
                  ${index === 3 ? 'bottom-12 right-12' : ''}`}
                onClick={() => onSiteSelect(site.id)}
                data-testid={`sensor-${site.id}`}
              />
            ))}
          </div>
          
          {/* Map controls */}
          <div className="absolute bottom-4 right-4 space-y-2">
            <Button
              variant="outline"
              size="sm"
              className="w-8 h-8 p-0 bg-card border border-border shadow-lg"
              data-testid="button-map-zoom-in"
            >
              <Plus className="h-4 w-4" />
            </Button>
            <Button
              variant="outline" 
              size="sm"
              className="w-8 h-8 p-0 bg-card border border-border shadow-lg"
              data-testid="button-map-zoom-out"
            >
              <Minus className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        {/* Map Legend */}
        <div className="p-4 border-t border-border bg-muted/30">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <span>High Risk</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <span>Medium Risk</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span>Low Risk</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span>Active Sensors</span>
              </div>
            </div>
            <span className="text-muted-foreground">Last updated: 2 min ago</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
