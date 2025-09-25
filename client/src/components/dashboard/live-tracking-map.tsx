import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { 
  MapPin, 
  Navigation, 
  Search, 
  Users, 
  AlertTriangle, 
  Thermometer,
  Wind,
  Droplets,
  Eye,
  Loader2
} from "lucide-react";
// import { useWebSocket } from "@/hooks/use-websocket";

/**
 * Live Tracking Map Component
 * Features:
 * - Real-time user/tourist location tracking via WebSocket
 * - Location search with live weather and risk data
 * - Route deviation alerts
 * - Tourist cluster heatmap
 * - Live environmental data display
 */
export default function LiveTrackingMap() {
  const { toast } = useToast();
  const mapRef = useRef<HTMLDivElement>(null);
  const [map, setMap] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedLocation, setSelectedLocation] = useState<any>(null);
  const [weatherData, setWeatherData] = useState<any>(null);
  const [riskData, setRiskData] = useState<any>(null);
  const [tourists, setTourists] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Simulated tourist locations (in real implementation, this would come from WebSocket)
  useEffect(() => {
    // Simulate live tourist tracking data
    const simulatedTourists = [
      {
        id: "tourist_1",
        name: "Tourist Group A",
        lat: 32.2432,
        lng: 77.1892,
        status: "safe",
        lastUpdate: new Date(),
        route: "Manali to Leh",
        groupSize: 4
      },
      {
        id: "tourist_2", 
        name: "Solo Traveler B",
        lat: 34.0837,
        lng: 74.7973,
        status: "deviation",
        lastUpdate: new Date(),
        route: "Srinagar Valley",
        groupSize: 1
      },
      {
        id: "tourist_3",
        name: "Family Group C",
        lat: 30.0668,
        lng: 79.0193,
        status: "safe",
        lastUpdate: new Date(),
        route: "Nainital Hills",
        groupSize: 6
      }
    ];

    setTourists(simulatedTourists);

    // Simulate real-time updates
    const interval = setInterval(() => {
      setTourists(prev => prev.map(tourist => ({
        ...tourist,
        lat: tourist.lat + (Math.random() - 0.5) * 0.01,
        lng: tourist.lng + (Math.random() - 0.5) * 0.01,
        lastUpdate: new Date()
      })));
    }, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, []);

  // Initialize map (using a simple div for now, in real implementation would use Mapbox/Leaflet)
  useEffect(() => {
    if (mapRef.current && !map) {
      // Simulated map initialization
      setMap({ initialized: true });
    }
  }, [map]);

  // Handle location search
  const handleLocationSearch = async (query: string) => {
    if (!query.trim()) return;
    
    setIsLoading(true);
    try {
      // Simulate location search API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const mockLocation = {
        name: query,
        lat: 32.2432 + Math.random() * 0.1,
        lng: 77.1892 + Math.random() * 0.1,
        address: `${query}, Himachal Pradesh, India`
      };

      setSelectedLocation(mockLocation);
      
      // Fetch weather data for selected location
      await fetchWeatherData(mockLocation.lat, mockLocation.lng);
      
      // Fetch risk assessment data
      await fetchRiskData(mockLocation);
      
      toast({
        title: "Location Found",
        description: `Displaying live data for ${mockLocation.name}`,
      });
    } catch (error) {
      toast({
        title: "Search Error",
        description: "Could not find location or fetch data",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch weather data from OpenWeatherMap API (simulated)
  const fetchWeatherData = async (lat: number, lng: number) => {
    try {
      // In real implementation, use actual OpenWeatherMap API
      // const response = await fetch(`https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lng}&appid=${API_KEY}&units=metric`);
      
      // Simulated weather data
      const mockWeatherData = {
        temperature: Math.round(15 + Math.random() * 20),
        humidity: Math.round(40 + Math.random() * 40),
        windSpeed: Math.round(5 + Math.random() * 15),
        pressure: Math.round(1000 + Math.random() * 50),
        visibility: Math.round(5 + Math.random() * 15),
        conditions: ["Clear", "Cloudy", "Rainy", "Foggy"][Math.floor(Math.random() * 4)],
        airQuality: {
          index: Math.round(1 + Math.random() * 4),
          description: ["Good", "Fair", "Moderate", "Poor", "Very Poor"][Math.floor(Math.random() * 5)]
        }
      };
      
      setWeatherData(mockWeatherData);
    } catch (error) {
      console.error("Weather data fetch error:", error);
    }
  };

  // Fetch risk assessment data
  const fetchRiskData = async (location: any) => {
    try {
      // Simulate risk assessment based on location
      const riskLevel = Math.random();
      const mockRiskData = {
        overall: riskLevel > 0.7 ? "high" : riskLevel > 0.4 ? "medium" : "low",
        crime: {
          level: Math.random() > 0.8 ? "high" : Math.random() > 0.6 ? "medium" : "low",
          incidents: Math.floor(Math.random() * 5)
        },
        natural: {
          level: Math.random() > 0.7 ? "high" : Math.random() > 0.5 ? "medium" : "low",
          warnings: Math.floor(Math.random() * 3)
        },
        medical: {
          facilities: Math.floor(1 + Math.random() * 5),
          distance: Math.round(1 + Math.random() * 20)
        },
        connectivity: {
          mobile: Math.random() > 0.3 ? "good" : "poor",
          internet: Math.random() > 0.4 ? "available" : "limited"
        }
      };
      
      setRiskData(mockRiskData);
    } catch (error) {
      console.error("Risk data fetch error:", error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Location Search Bar */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="w-5 h-5" />
            Location Search & Live Data
          </CardTitle>
          <CardDescription>
            Search for any location to view real-time weather, risk alerts, and safety information
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search for a location (e.g., Manali, Delhi, Mumbai)..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleLocationSearch(searchQuery)}
                className="pl-8"
                data-testid="input-location-search"
              />
            </div>
            <Button 
              onClick={() => handleLocationSearch(searchQuery)}
              disabled={isLoading || !searchQuery.trim()}
              data-testid="button-search-location"
            >
              {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Search"}
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Map Display */}
        <div className="lg:col-span-2">
          <Card className="h-[600px]">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <MapPin className="w-5 h-5" />
                  Live Tourist Tracking Map
                </span>
                <Badge variant="secondary">
                  {tourists.length} Active Tourists
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0 h-[500px]">
              {/* Simulated Map Display */}
              <div 
                ref={mapRef}
                className="w-full h-full bg-gradient-to-br from-green-50 to-blue-50 dark:from-green-950 dark:to-blue-950 relative overflow-hidden rounded-lg"
                data-testid="map-container"
              >
                {/* Simulated map markers for tourists */}
                {tourists.map((tourist, index) => (
                  <div
                    key={tourist.id}
                    className={`absolute w-4 h-4 rounded-full border-2 border-white shadow-lg transform -translate-x-1/2 -translate-y-1/2 ${
                      tourist.status === "safe" ? "bg-green-500" :
                      tourist.status === "deviation" ? "bg-orange-500" :
                      "bg-red-500"
                    }`}
                    style={{
                      left: `${20 + index * 25}%`,
                      top: `${30 + index * 20}%`
                    }}
                    title={`${tourist.name} (${tourist.groupSize} people)`}
                    data-testid={`tourist-marker-${tourist.id}`}
                  />
                ))}

                {/* Selected location marker */}
                {selectedLocation && (
                  <div
                    className="absolute w-6 h-6 bg-blue-600 rounded-full border-2 border-white shadow-lg transform -translate-x-1/2 -translate-y-1/2 animate-pulse"
                    style={{
                      left: "50%",
                      top: "50%"
                    }}
                    title={selectedLocation.name}
                  />
                )}

                {/* Map Controls */}
                <div className="absolute bottom-4 right-4 space-y-2">
                  <Button size="sm" variant="outline" className="bg-background/80">
                    <Navigation className="w-4 h-4" />
                  </Button>
                  <Button size="sm" variant="outline" className="bg-background/80">
                    <Users className="w-4 h-4" />
                  </Button>
                </div>

                {/* Route Deviation Alert */}
                {tourists.some(t => t.status === "deviation") && (
                  <div className="absolute top-4 left-4 right-4">
                    <div className="bg-orange-100 dark:bg-orange-950 border border-orange-200 dark:border-orange-800 rounded-lg p-3">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 text-orange-600" />
                        <span className="text-sm font-medium text-orange-800 dark:text-orange-200">
                          Route Deviation Detected
                        </span>
                      </div>
                      <p className="text-xs text-orange-700 dark:text-orange-300 mt-1">
                        Tourist group has deviated from planned route. Auto-alert sent.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Live Data Panel */}
        <div className="space-y-6">
          {/* Tourist Status */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="w-5 h-5" />
                Active Tourists
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {tourists.map((tourist) => (
                <div key={tourist.id} className="flex items-center justify-between p-2 rounded-lg bg-muted/50">
                  <div>
                    <p className="text-sm font-medium">{tourist.name}</p>
                    <p className="text-xs text-muted-foreground">{tourist.route}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={
                        tourist.status === "safe" ? "secondary" :
                        tourist.status === "deviation" ? "default" :
                        "destructive"
                      }
                      className="text-xs"
                    >
                      {tourist.status}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {tourist.groupSize}
                    </span>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Selected Location Info */}
          {selectedLocation && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MapPin className="w-5 h-5" />
                  Location Details
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <h4 className="font-medium">{selectedLocation.name}</h4>
                  <p className="text-sm text-muted-foreground">{selectedLocation.address}</p>
                  <div className="text-xs text-muted-foreground">
                    {selectedLocation.lat.toFixed(4)}, {selectedLocation.lng.toFixed(4)}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Live Weather Data */}
          {weatherData && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Thermometer className="w-5 h-5" />
                  Live Weather
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold">{weatherData.temperature}°C</div>
                    <div className="text-sm text-muted-foreground">{weatherData.conditions}</div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Droplets className="w-4 h-4 text-blue-500" />
                      <span className="text-sm">{weatherData.humidity}%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Wind className="w-4 h-4 text-gray-500" />
                      <span className="text-sm">{weatherData.windSpeed} km/h</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Eye className="w-4 h-4 text-purple-500" />
                      <span className="text-sm">{weatherData.visibility} km</span>
                    </div>
                  </div>
                </div>

                <div className="pt-2 border-t">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Air Quality</span>
                    <Badge
                      variant={
                        weatherData.airQuality.index <= 2 ? "secondary" :
                        weatherData.airQuality.index <= 3 ? "default" :
                        "destructive"
                      }
                    >
                      {weatherData.airQuality.description}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Risk Assessment */}
          {riskData && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5" />
                  Risk Assessment
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Overall Risk</span>
                  <Badge
                    variant={
                      riskData.overall === "low" ? "secondary" :
                      riskData.overall === "medium" ? "default" :
                      "destructive"
                    }
                  >
                    {riskData.overall.toUpperCase()}
                  </Badge>
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Crime Safety</span>
                    <span className={`font-medium ${
                      riskData.crime.level === "low" ? "text-green-600" :
                      riskData.crime.level === "medium" ? "text-orange-600" :
                      "text-red-600"
                    }`}>
                      {riskData.crime.level}
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span>Natural Hazards</span>
                    <span className={`font-medium ${
                      riskData.natural.level === "low" ? "text-green-600" :
                      riskData.natural.level === "medium" ? "text-orange-600" :
                      "text-red-600"
                    }`}>
                      {riskData.natural.level}
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span>Medical Facilities</span>
                    <span className="text-muted-foreground">
                      {riskData.medical.facilities} within {riskData.medical.distance}km
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span>Connectivity</span>
                    <span className={`font-medium ${
                      riskData.connectivity.mobile === "good" ? "text-green-600" : "text-orange-600"
                    }`}>
                      {riskData.connectivity.mobile}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}