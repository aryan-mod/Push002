import { useState, useEffect, useMemo } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertTriangle, MapPin, Zap } from "lucide-react";
import type { Site } from "@shared/schema";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { queryClient } from "@/lib/queryClient";

// Import heatmap plugin
import "leaflet.heat";

// Fix Leaflet icon issue with bundlers
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

interface InteractiveMapProps {
  sites: Site[];
  isLoading: boolean;
  onSiteSelect: (siteId: string) => void;
}

// Memoized risk icons to avoid recreation on every render
const riskIcons = {
  critical: L.divIcon({
    html: `<div style="
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background-color: #ef4444;
      border: 2px solid white;
      box-shadow: 0 2px 4px rgba(0,0,0,0.3);
      animation: pulse 2s infinite;
    "></div>`,
    className: 'custom-marker',
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  }),
  high: L.divIcon({
    html: `<div style="
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background-color: #ef4444;
      border: 2px solid white;
      box-shadow: 0 2px 4px rgba(0,0,0,0.3);
      animation: pulse 2s infinite;
    "></div>`,
    className: 'custom-marker',
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  }),
  medium: L.divIcon({
    html: `<div style="
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background-color: #f59e0b;
      border: 2px solid white;
      box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    "></div>`,
    className: 'custom-marker',
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  }),
  low: L.divIcon({
    html: `<div style="
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background-color: #10b981;
      border: 2px solid white;
      box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    "></div>`,
    className: 'custom-marker',
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  }),
} as const;

// Get risk icon by level
const getRiskIcon = (riskLevel: string): L.DivIcon => {
  return riskIcons[riskLevel as keyof typeof riskIcons] || riskIcons.low;
};

// Component to handle map view switching and heatmap
function MapController({ mapView, sites }: { mapView: string; sites: Site[] }) {
  const map = useMap();
  
  useEffect(() => {
    // Remove existing heatmap layer if it exists
    map.eachLayer((layer) => {
      if (layer instanceof (L as any).HeatLayer) {
        map.removeLayer(layer);
      }
    });

    // Add heatmap layer for heatmap view
    if (mapView === 'heatmap' && sites.length > 0) {
      // Convert sites to heatmap points with weights based on risk level
      const heatmapData = sites.map(site => {
        const riskWeights = {
          critical: 1.0,
          high: 0.8,
          medium: 0.5,
          low: 0.2
        };
        const weight = riskWeights[site.riskLevel as keyof typeof riskWeights] || 0.2;
        return [site.location.y, site.location.x, weight] as [number, number, number];
      });

      // Create and add heatmap layer
      const heatLayer = (L as any).heatLayer(heatmapData, {
        radius: 50,
        blur: 25,
        maxZoom: 18,
        gradient: {
          0.2: '#10b981', // green for low risk
          0.5: '#f59e0b', // amber for medium risk
          0.8: '#ef4444', // red for high/critical risk
          1.0: '#dc2626'  // darker red for critical
        }
      });
      
      map.addLayer(heatLayer);
    }

    // Invalidate map size when view changes
    map.invalidateSize();
  }, [mapView, map, sites]);
  
  return null;
}

export default function InteractiveMap({ sites, isLoading, onSiteSelect }: InteractiveMapProps) {
  const [mapView, setMapView] = useState<"satellite" | "terrain" | "heatmap">("satellite");
  const [isSimulating, setIsSimulating] = useState(false);
  const { toast } = useToast();

  // Default center position (India - suitable for rockfall monitoring)
  const defaultCenter: [number, number] = [28.6139, 77.2090]; // Delhi coordinates
  const defaultZoom = 6;

  // Memoize tile layer configuration
  const tileConfig = useMemo(() => {
    switch (mapView) {
      case 'satellite':
        return {
          url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
          attribution: '&copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid'
        };
      case 'terrain':
        return {
          url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
          attribution: 'Map data: &copy; OpenTopoMap contributors'
        };
      case 'heatmap':
      default:
        return {
          url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
          attribution: '&copy; OpenStreetMap contributors'
        };
    }
  }, [mapView]);

  const handleSimulateEvent = async () => {
    setIsSimulating(true);
    
    try {
      // Use random site if available, otherwise use default coordinates
      const targetSite = sites.length > 0 ? sites[Math.floor(Math.random() * sites.length)] : null;
      const lat = targetSite ? targetSite.location.y : 28.7;
      const lon = targetSite ? targetSite.location.x : 77.1;

      // Simulate prediction with location data
      const predictionData = {
        lat,
        lon,
        sensorData: {
          strain: Math.random() * 100,
          displacement: Math.random() * 50,
          vibration: Math.random() * 25
        }
      };

      // Use apiRequest instead of fetch
      const response = await apiRequest("POST", "/api/v1/predict", predictionData);
      const result = await response.json();

      toast({
        title: "Event Simulated",
        description: `Risk prediction generated: ${result.probability?.toFixed(2) || 'N/A'}% probability`,
      });

      // Invalidate relevant cache keys
      queryClient.invalidateQueries({ queryKey: ['/api/v1/alerts'] });
      queryClient.invalidateQueries({ queryKey: ['/api/v1/predictions'] });

    } catch (error) {
      console.error('Simulation error:', error);
      toast({
        title: "Simulation Error",
        description: error instanceof Error ? error.message : "Network error during simulation",
        variant: "destructive"
      });
    } finally {
      setIsSimulating(false);
    }
  };

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

  return (
    <Card className="border border-border">
      <CardHeader className="border-b border-border">
        <div className="flex items-center justify-between">
          <CardTitle>Live Risk Assessment Map</CardTitle>
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
        <div className="relative h-96">
          {/* Simulate Event Button */}
          <div className="absolute top-2 left-2 z-[1000]">
            <Button
              onClick={handleSimulateEvent}
              disabled={isSimulating}
              size="sm"
              className="bg-orange-500 hover:bg-orange-600 text-white"
              data-testid="button-simulate-event"
            >
              <Zap className="w-4 h-4 mr-2" />
              {isSimulating ? "Simulating..." : "Simulate Event"}
            </Button>
          </div>

          <MapContainer
            center={defaultCenter}
            zoom={defaultZoom}
            style={{ height: '100%', width: '100%' }}
          >
            <TileLayer
              url={tileConfig.url}
              attribution={tileConfig.attribution}
            />
            
            <MapController mapView={mapView} sites={sites} />
            
            {/* Render site markers only if not in heatmap view */}
            {mapView !== 'heatmap' && sites.map((site) => (
              <Marker
                key={site.id}
                position={[site.location.y, site.location.x] as [number, number]}
                icon={getRiskIcon(site.riskLevel)}
                eventHandlers={{
                  click: () => {
                    onSiteSelect(site.id);
                  },
                }}
              >
                <Popup>
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <MapPin className="w-4 h-4" />
                      <h3 className="font-semibold">{site.name}</h3>
                    </div>
                    <div className="space-y-1 text-sm">
                      <p><strong>Risk Level:</strong> 
                        <Badge 
                          variant={site.riskLevel === 'critical' || site.riskLevel === 'high' ? 'destructive' : 
                                 site.riskLevel === 'medium' ? 'secondary' : 'default'}
                          className="ml-2"
                        >
                          {site.riskLevel.toUpperCase()}
                        </Badge>
                      </p>
                      <p><strong>Elevation:</strong> {site.elevation}m</p>
                      <p><strong>Slope:</strong> {site.slopeAngle}°</p>
                      <p><strong>Status:</strong> {site.isActive ? 'Active' : 'Inactive'}</p>
                      <p className="text-muted-foreground">{site.description}</p>
                    </div>
                  </div>
                </Popup>
              </Marker>
            ))}
          </MapContainer>
        </div>
        
        {/* Map Legend */}
        <div className="p-4 border-t border-border bg-muted/30">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                <span>Critical/High Risk</span>
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
                <Zap className="w-3 h-3 text-orange-500" />
                <span>Simulate Event</span>
              </div>
              {mapView === 'heatmap' && (
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500"></div>
                  <span>Risk Intensity</span>
                </div>
              )}
            </div>
            <span className="text-muted-foreground">
              Live data • {sites.length} {mapView === 'heatmap' ? 'heat zones' : 'sites'} • Updated now
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}