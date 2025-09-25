import { useState } from "react";
import Header from "@/components/layout/header";
import AdvancedLocationSearch from "@/components/dashboard/advanced-location-search";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Search, 
  MapPin, 
  Cloud, 
  AlertTriangle, 
  Navigation,
  Sparkles,
  Target,
  TrendingUp
} from "lucide-react";

export default function LocationSearch() {
  // Handle quick actions
  const handleCurrentLocation = () => {
    // This would trigger the current location functionality in the search component
    console.log('Current location clicked');
  };

  const handleQuickSearch = (location: string) => {
    // This would trigger a search for the specified location
    console.log('Quick search for:', location);
  };

  const handleHighRiskAreas = () => {
    // This would show high-risk areas
    console.log('Show high-risk areas');
  };

  const handleRecentSearches = () => {
    // This would show recent search history
    console.log('Show recent searches');
  };

  return (
    <div className="min-h-screen bg-background">
      <Header 
        title="Location Search & Risk Assessment" 
        subtitle="Advanced location analysis with weather data and rockfall risk assessment"
      />
      
      <div className="flex-1 p-6 space-y-8">
        {/* Feature Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="border-blue-200 bg-blue-50 dark:bg-blue-900/20">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-blue-700 dark:text-blue-300">
                <Search className="h-5 w-5" />
                Smart Search
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-blue-600 dark:text-blue-400">
                Search for any location using city names, coordinates, or addresses. Get instant results with autocomplete suggestions.
              </p>
            </CardContent>
          </Card>
          
          <Card className="border-green-200 bg-green-50 dark:bg-green-900/20">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-green-700 dark:text-green-300">
                <Cloud className="h-5 w-5" />
                Live Weather
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-green-600 dark:text-green-400">
                Real-time weather data including temperature, humidity, wind speed, visibility, and atmospheric pressure.
              </p>
            </CardContent>
          </Card>
          
          <Card className="border-orange-200 bg-orange-50 dark:bg-orange-900/20">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-orange-700 dark:text-orange-300">
                <AlertTriangle className="h-5 w-5" />
                Risk Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-orange-600 dark:text-orange-400">
                Comprehensive risk assessment considering weather, geological, historical, and environmental factors.
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5" />
              Quick Actions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-3">
              <Button 
                variant="outline" 
                className="flex items-center gap-2"
                onClick={handleCurrentLocation}
                data-testid="button-quick-current-location"
              >
                <Navigation className="h-4 w-4" />
                Use Current Location
              </Button>
              <Button 
                variant="outline" 
                className="flex items-center gap-2"
                onClick={() => handleQuickSearch('Mumbai')}
                data-testid="button-quick-search-mumbai"
              >
                <MapPin className="h-4 w-4" />
                Search Mumbai
              </Button>
              <Button 
                variant="outline" 
                className="flex items-center gap-2"
                onClick={handleHighRiskAreas}
                data-testid="button-quick-high-risk-areas"
              >
                <Target className="h-4 w-4" />
                High-Risk Areas
              </Button>
              <Button 
                variant="outline" 
                className="flex items-center gap-2"
                onClick={handleRecentSearches}
                data-testid="button-quick-recent-searches"
              >
                <TrendingUp className="h-4 w-4" />
                Recent Searches
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Main Search Component */}
        <AdvancedLocationSearch />

        {/* Usage Tips */}
        <Card>
          <CardHeader>
            <CardTitle>Search Tips & Features</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h4 className="font-medium">Search Methods:</h4>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <Badge variant="outline" className="text-xs">City</Badge>
                    Search by city or town name (e.g., "Mumbai", "Shimla")
                  </li>
                  <li className="flex items-start gap-2">
                    <Badge variant="outline" className="text-xs">Address</Badge>
                    Use full addresses or landmarks
                  </li>
                  <li className="flex items-start gap-2">
                    <Badge variant="outline" className="text-xs">GPS</Badge>
                    Enter coordinates (latitude, longitude)
                  </li>
                  <li className="flex items-start gap-2">
                    <Badge variant="outline" className="text-xs">Current</Badge>
                    Use "Current Location" button for your position
                  </li>
                </ul>
              </div>
              
              <div className="space-y-3">
                <h4 className="font-medium">Risk Assessment Factors:</h4>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <Badge variant="outline" className="text-xs bg-blue-50">Weather</Badge>
                    Current conditions affecting stability
                  </li>
                  <li className="flex items-start gap-2">
                    <Badge variant="outline" className="text-xs bg-orange-50">Geological</Badge>
                    Terrain, slope, and rock formation analysis
                  </li>
                  <li className="flex items-start gap-2">
                    <Badge variant="outline" className="text-xs bg-purple-50">Historical</Badge>
                    Past incidents and geological activity
                  </li>
                  <li className="flex items-start gap-2">
                    <Badge variant="outline" className="text-xs bg-green-50">Environmental</Badge>
                    Vegetation, erosion, and ecosystem factors
                  </li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}