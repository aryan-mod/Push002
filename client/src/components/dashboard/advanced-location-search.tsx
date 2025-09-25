import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  Search, 
  MapPin, 
  Cloud, 
  Thermometer, 
  Wind, 
  Droplets, 
  Eye, 
  Gauge,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Navigation,
  Loader2,
  Target
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";

interface LocationResult {
  id: string;
  name: string;
  displayName: string;
  latitude: number;
  longitude: number;
  country: string;
  region: string;
  type: string; // city, town, village, etc.
}

interface WeatherData {
  temperature: number;
  humidity: number;
  windSpeed: number;
  windDirection: number;
  visibility: number;
  pressure: number;
  condition: string;
  description: string;
  icon: string;
  uvIndex: number;
  precipitation: number;
  dewPoint: number;
}

interface RiskAssessment {
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  riskScore: number;
  factors: {
    weather: number;
    geological: number;
    historical: number;
    environmental: number;
  };
  alerts: string[];
  recommendations: string[];
}

interface LocationData {
  location: LocationResult;
  weather: WeatherData;
  riskAssessment: RiskAssessment;
  nearestSites: Array<{
    id: string;
    name: string;
    distance: number;
    riskLevel: string;
  }>;
}

export default function AdvancedLocationSearch() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedLocation, setSelectedLocation] = useState<LocationResult | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const { toast } = useToast();
  const searchTimeoutRef = useRef<NodeJS.Timeout>();

  // Fetch location suggestions
  const { data: locationSuggestions = [], isLoading: isLoadingSuggestions } = useQuery<LocationResult[]>({
    queryKey: ["/api/v1/locations/search", searchQuery],
    queryFn: async () => {
      if (!searchQuery || searchQuery.length < 3) return [];
      
      const response = await fetch(`/api/v1/locations/search?q=${encodeURIComponent(searchQuery)}`);
      if (!response.ok) {
        throw new Error('Failed to fetch location suggestions');
      }
      return response.json();
    },
    enabled: searchQuery.length >= 3,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  // Fetch detailed location data when a location is selected
  const { 
    data: locationData, 
    isLoading: isLoadingLocationData,
    error: locationDataError 
  } = useQuery<LocationData>({
    queryKey: ["/api/v1/locations/details", selectedLocation?.latitude, selectedLocation?.longitude],
    queryFn: async () => {
      if (!selectedLocation) return null;
      
      const response = await fetch(
        `/api/v1/locations/details?lat=${selectedLocation.latitude}&lon=${selectedLocation.longitude}`
      );
      if (!response.ok) {
        throw new Error('Failed to fetch location details');
      }
      return response.json();
    },
    enabled: !!selectedLocation,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });

  // Handle search input with debouncing
  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }
    
    searchTimeoutRef.current = setTimeout(() => {
      setIsSearching(false);
    }, 500);
    
    setIsSearching(true);
  };

  // Handle location selection
  const handleLocationSelect = (location: LocationResult) => {
    setSelectedLocation(location);
    setSearchQuery(location.displayName);
    toast({
      title: "Location Selected",
      description: `Loading data for ${location.displayName}...`,
    });
  };

  // Handle current location detection
  const handleCurrentLocation = () => {
    if (!navigator.geolocation) {
      toast({
        title: "Geolocation not supported",
        description: "Your browser doesn't support location detection.",
        variant: "destructive",
      });
      return;
    }

    setIsSearching(true);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        const currentLocation: LocationResult = {
          id: "current",
          name: "Current Location",
          displayName: `Current Location (${latitude.toFixed(4)}, ${longitude.toFixed(4)})`,
          latitude,
          longitude,
          country: "",
          region: "",
          type: "current"
        };
        
        setSelectedLocation(currentLocation);
        setSearchQuery(currentLocation.displayName);
        setIsSearching(false);
        
        toast({
          title: "Location Detected",
          description: "Loading data for your current location...",
        });
      },
      (error) => {
        setIsSearching(false);
        toast({
          title: "Location Error",
          description: "Failed to detect your location. Please search manually.",
          variant: "destructive",
        });
      }
    );
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical': return 'bg-red-500 text-white';
      case 'high': return 'bg-orange-500 text-white';
      case 'medium': return 'bg-yellow-500 text-black';
      case 'low': return 'bg-green-500 text-white';
      default: return 'bg-gray-500 text-white';
    }
  };

  const getWeatherIcon = (condition: string) => {
    const iconMap: { [key: string]: any } = {
      'clear': CheckCircle,
      'clouds': Cloud,
      'rain': Droplets,
      'storm': AlertTriangle,
      'wind': Wind
    };
    
    return iconMap[condition.toLowerCase()] || Cloud;
  };

  return (
    <div className="space-y-6">
      {/* Search Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            Advanced Location Search
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search for a location (city, coordinates, address)..."
                value={searchQuery}
                onChange={(e) => handleSearchChange(e.target.value)}
                className="pl-10"
                data-testid="input-location-search"
              />
              {(isSearching || isLoadingSuggestions) && (
                <Loader2 className="absolute right-3 top-1/2 transform -translate-y-1/2 h-4 w-4 animate-spin text-muted-foreground" />
              )}
            </div>
            <Button 
              variant="outline" 
              onClick={handleCurrentLocation}
              disabled={isSearching}
              data-testid="button-current-location"
            >
              <Navigation className="h-4 w-4 mr-2" />
              Current Location
            </Button>
          </div>

          {/* Location Suggestions */}
          {locationSuggestions.length > 0 && searchQuery.length >= 3 && !selectedLocation && (
            <Card className="mt-4 max-h-64 overflow-y-auto">
              <CardContent className="p-2">
                {locationSuggestions.map((location) => (
                  <Button
                    key={location.id}
                    variant="ghost"
                    className="w-full justify-start p-3 h-auto"
                    onClick={() => handleLocationSelect(location)}
                  >
                    <div className="flex items-center gap-3">
                      <MapPin className="h-4 w-4 text-muted-foreground" />
                      <div className="text-left">
                        <div className="font-medium">{location.name}</div>
                        <div className="text-sm text-muted-foreground">
                          {location.region}, {location.country}
                        </div>
                      </div>
                    </div>
                  </Button>
                ))}
              </CardContent>
            </Card>
          )}
        </CardContent>
      </Card>

      {/* Location Details */}
      {selectedLocation && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Weather Information */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cloud className="h-5 w-5" />
                Current Weather
              </CardTitle>
            </CardHeader>
            <CardContent>
              {isLoadingLocationData ? (
                <div className="space-y-3">
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-4 w-1/2" />
                  <Skeleton className="h-20 w-full" />
                </div>
              ) : locationData?.weather ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {(() => {
                        const WeatherIcon = getWeatherIcon(locationData.weather.condition);
                        return <WeatherIcon className="h-8 w-8 text-blue-500" />;
                      })()}
                      <div>
                        <div className="text-2xl font-bold">{locationData.weather.temperature}°C</div>
                        <div className="text-sm text-muted-foreground">{locationData.weather.description}</div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex items-center gap-2">
                      <Droplets className="h-4 w-4 text-blue-500" />
                      <span className="text-sm">Humidity: {locationData.weather.humidity}%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Wind className="h-4 w-4 text-gray-500" />
                      <span className="text-sm">Wind: {locationData.weather.windSpeed} km/h</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Eye className="h-4 w-4 text-purple-500" />
                      <span className="text-sm">Visibility: {locationData.weather.visibility} km</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Gauge className="h-4 w-4 text-green-500" />
                      <span className="text-sm">Pressure: {locationData.weather.pressure} hPa</span>
                    </div>
                  </div>
                </div>
              ) : (
                <Alert>
                  <XCircle className="h-4 w-4" />
                  <AlertDescription>
                    Unable to load weather data for this location.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Risk Assessment */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5" />
                Risk Assessment
              </CardTitle>
            </CardHeader>
            <CardContent>
              {isLoadingLocationData ? (
                <div className="space-y-3">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-20 w-full" />
                </div>
              ) : locationData?.riskAssessment ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Overall Risk Level:</span>
                    <Badge className={cn("text-sm", getRiskColor(locationData.riskAssessment.riskLevel))}>
                      {locationData.riskAssessment.riskLevel.toUpperCase()}
                    </Badge>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="text-sm text-muted-foreground">Risk Score: {locationData.riskAssessment.riskScore}/100</div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={cn(
                          "h-2 rounded-full transition-all",
                          locationData.riskAssessment.riskLevel === 'critical' ? 'bg-red-500' :
                          locationData.riskAssessment.riskLevel === 'high' ? 'bg-orange-500' :
                          locationData.riskAssessment.riskLevel === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                        )}
                        style={{ width: `${locationData.riskAssessment.riskScore}%` }}
                      />
                    </div>
                  </div>

                  <Separator />

                  <div className="space-y-3">
                    <h4 className="font-medium">Risk Factors:</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>Weather: {locationData.riskAssessment.factors.weather}/10</div>
                      <div>Geological: {locationData.riskAssessment.factors.geological}/10</div>
                      <div>Historical: {locationData.riskAssessment.factors.historical}/10</div>
                      <div>Environmental: {locationData.riskAssessment.factors.environmental}/10</div>
                    </div>
                  </div>

                  {locationData.riskAssessment.alerts.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="font-medium text-red-600">Active Alerts:</h4>
                      <ul className="text-sm space-y-1">
                        {locationData.riskAssessment.alerts.map((alert, index) => (
                          <li key={index} className="flex items-start gap-2">
                            <AlertTriangle className="h-3 w-3 text-red-500 mt-0.5 flex-shrink-0" />
                            {alert}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ) : (
                <Alert>
                  <XCircle className="h-4 w-4" />
                  <AlertDescription>
                    Unable to assess risk for this location.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Nearby Sites */}
      {locationData?.nearestSites && locationData.nearestSites.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Nearby Monitoring Sites
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {locationData.nearestSites.map((site) => (
                <div key={site.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">{site.name}</div>
                    <div className="text-sm text-muted-foreground">{site.distance.toFixed(1)} km away</div>
                  </div>
                  <Badge className={getRiskColor(site.riskLevel)}>
                    {site.riskLevel}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Error State */}
      {locationDataError && (
        <Alert variant="destructive">
          <XCircle className="h-4 w-4" />
          <AlertDescription>
            Failed to load location data. Please try again or search for a different location.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}