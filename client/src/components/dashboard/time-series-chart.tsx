import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";

export default function TimeSeriesChart() {
  const [timeRange, setTimeRange] = useState("24h");

  const { data: trendData, isLoading } = useQuery({
    queryKey: ["/api/v1/dashboard/trends", timeRange],
    refetchInterval: 300000, // Refresh every 5 minutes
  });

  if (isLoading) {
    return (
      <Card className="border border-border">
        <CardHeader>
          <div className="flex items-center justify-between">
            <Skeleton className="h-6 w-40" />
            <Skeleton className="h-8 w-32" />
          </div>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-48 w-full" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border border-border">
      <CardHeader className="border-b border-border">
        <div className="flex items-center justify-between">
          <CardTitle>Risk Trend Analysis</CardTitle>
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-40" data-testid="select-time-range">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="24h">Last 24 Hours</SelectItem>
              <SelectItem value="7d">Last 7 Days</SelectItem>
              <SelectItem value="30d">Last 30 Days</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      
      <CardContent className="p-6">
        <div className="h-48 bg-muted/30 rounded-lg flex items-center justify-center relative overflow-hidden">
          {/* Simulated chart background */}
          <div className="absolute inset-0 p-4">
            <div className="h-full w-full relative">
              {/* Y-axis labels */}
              <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-muted-foreground">
                <span>High</span>
                <span>Med</span>
                <span>Low</span>
              </div>
              
              {/* Chart area */}
              <div className="ml-8 h-full relative">
                {/* Risk level visualization */}
                <div className="absolute bottom-0 left-0 w-full h-full">
                  {/* Simulated line chart */}
                  <svg className="w-full h-full" viewBox="0 0 300 120">
                    {/* High risk trend line */}
                    <polyline 
                      points="0,80 50,85 100,70 150,60 200,45 250,40 300,35" 
                      fill="none" 
                      stroke="hsl(210, 83%, 53%)" 
                      strokeWidth="2"
                    />
                    {/* Medium risk trend line */}
                    <polyline 
                      points="0,90 50,95 100,90 150,85 200,80 250,75 300,70" 
                      fill="none" 
                      stroke="hsl(38, 92%, 50%)" 
                      strokeWidth="2"
                    />
                    {/* Low risk trend line */}
                    <polyline 
                      points="0,100 50,105 100,100 150,95 200,90 250,88 300,85" 
                      fill="none" 
                      stroke="hsl(142, 71%, 45%)" 
                      strokeWidth="2"
                    />
                    
                    {/* Data points */}
                    <circle cx="300" cy="35" r="3" fill="hsl(210, 83%, 53%)" />
                    <circle cx="300" cy="70" r="3" fill="hsl(38, 92%, 50%)" />
                    <circle cx="300" cy="85" r="3" fill="hsl(142, 71%, 45%)" />
                  </svg>
                </div>
              </div>
            </div>
          </div>
          
          {/* Chart overlay info */}
          <div className="absolute bottom-4 right-4 bg-card/80 backdrop-blur-sm border border-border rounded p-2">
            <p className="text-xs text-muted-foreground">Updated every 5 minutes</p>
          </div>
        </div>
        
        {/* Chart legend */}
        <div className="mt-4 flex items-center justify-center space-x-6 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-0.5 bg-primary"></div>
            <span>High Risk Sites</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-0.5 bg-yellow-500"></div>
            <span>Medium Risk Sites</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-0.5 bg-green-500"></div>
            <span>Low Risk Sites</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
