import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import Header from "@/components/layout/header";
import { MapPin, Activity, Battery, Signal } from "lucide-react";
import type { Site } from "@shared/schema";

export default function SiteMonitoring() {
  const [searchTerm, setSearchTerm] = useState("");
  const [riskFilter, setRiskFilter] = useState("all");

  const { data: sites = [], isLoading } = useQuery<Site[]>({
    queryKey: ["/api/v1/sites"],
    refetchInterval: 30000,
  });

  const filteredSites = sites.filter((site: Site) => {
    const matchesSearch = site.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRisk = riskFilter === "all" || site.riskLevel === riskFilter;
    return matchesSearch && matchesRisk;
  });

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case "critical": return "destructive";
      case "high": return "destructive";
      case "medium": return "default";
      case "low": return "secondary";
      default: return "secondary";
    }
  };

  if (isLoading) {
    return (
      <div className="flex flex-col">
        <Header title="Site Monitoring" subtitle="Monitor all rockfall detection sites" />
        <div className="flex-1 p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <Card key={i} className="animate-pulse">
                <CardHeader>
                  <div className="h-4 bg-muted rounded w-3/4"></div>
                  <div className="h-3 bg-muted rounded w-1/2"></div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="h-3 bg-muted rounded"></div>
                    <div className="h-3 bg-muted rounded w-2/3"></div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      <Header title="Site Monitoring" subtitle="Monitor all rockfall detection sites" />
      
      <div className="flex-1 p-6">
        {/* Filters */}
        <div className="flex flex-col sm:flex-row gap-4 mb-6">
          <Input
            placeholder="Search sites..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="max-w-xs"
            data-testid="input-search-sites"
          />
          <Select value={riskFilter} onValueChange={setRiskFilter}>
            <SelectTrigger className="max-w-xs" data-testid="select-risk-filter">
              <SelectValue placeholder="Filter by risk level" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Risk Levels</SelectItem>
              <SelectItem value="critical">Critical</SelectItem>
              <SelectItem value="high">High</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="low">Low</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Sites Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredSites.map((site) => (
            <Card key={site.id} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">{site.name}</CardTitle>
                  <Badge variant={getRiskColor(site.riskLevel)} data-testid={`badge-risk-${site.id}`}>
                    {site.riskLevel}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">{site.description}</p>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Location */}
                  <div className="flex items-center space-x-2">
                    <MapPin className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm">
                      {site.location?.x?.toFixed(4)}, {site.location?.y?.toFixed(4)}
                    </span>
                  </div>

                  {/* Site Details */}
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Elevation:</span>
                      <br />
                      <span className="font-medium">{site.elevation || 'N/A'} m</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Slope:</span>
                      <br />
                      <span className="font-medium">{site.slopeAngle || 'N/A'}°</span>
                    </div>
                  </div>

                  {/* Status Indicators */}
                  <div className="flex items-center justify-between pt-2 border-t">
                    <div className="flex items-center space-x-4">
                      <div className="flex items-center space-x-1">
                        <Activity className="h-4 w-4 text-green-500" />
                        <span className="text-xs text-muted-foreground">Active</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Signal className="h-4 w-4 text-blue-500" />
                        <span className="text-xs text-muted-foreground">Online</span>
                      </div>
                    </div>
                    <Button variant="outline" size="sm" data-testid={`button-view-details-${site.id}`}>
                      View Details
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {filteredSites.length === 0 && (
          <div className="text-center py-12">
            <p className="text-muted-foreground">No sites found matching your criteria.</p>
          </div>
        )}
      </div>
    </div>
  );
}
