import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertTriangle, TriangleAlert, Info, Settings } from "lucide-react";
import type { Alert, Site } from "@shared/schema";

interface RealTimeAlertsProps {
  alerts: (Alert & { site: Site })[];
  isLoading: boolean;
}

export default function RealTimeAlerts({ alerts, isLoading }: RealTimeAlertsProps) {
  if (isLoading) {
    return (
      <Card className="border border-border">
        <CardHeader>
          <div className="flex items-center justify-between">
            <Skeleton className="h-6 w-32" />
            <div className="flex items-center space-x-2">
              <Skeleton className="w-2 h-2 rounded-full" />
              <Skeleton className="h-4 w-8" />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="p-3 border rounded-lg animate-pulse">
                <div className="flex items-start space-x-3">
                  <Skeleton className="w-3 h-3 rounded-full" />
                  <div className="flex-1 space-y-2">
                    <div className="flex justify-between">
                      <Skeleton className="h-4 w-32" />
                      <Skeleton className="h-3 w-16" />
                    </div>
                    <Skeleton className="h-3 w-48" />
                    <Skeleton className="h-3 w-40" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  const getAlertIcon = (severity: string) => {
    switch (severity) {
      case "critical":
      case "high":
        return AlertTriangle;
      case "medium":
        return TriangleAlert;
      default:
        return Info;
    }
  };

  const getAlertStyles = (severity: string) => {
    switch (severity) {
      case "critical":
        return {
          bg: "bg-red-50 dark:bg-red-900/20",
          border: "border-red-200 dark:border-red-800",
          dot: "bg-red-500",
          title: "text-red-800 dark:text-red-200",
          message: "text-red-700 dark:text-red-300",
          time: "text-red-600 dark:text-red-400",
          pulse: "pulse-dot"
        };
      case "high":
        return {
          bg: "bg-red-50 dark:bg-red-900/20",
          border: "border-red-200 dark:border-red-800",
          dot: "bg-red-500",
          title: "text-red-800 dark:text-red-200",
          message: "text-red-700 dark:text-red-300",
          time: "text-red-600 dark:text-red-400",
          pulse: ""
        };
      case "medium":
        return {
          bg: "bg-yellow-50 dark:bg-yellow-900/20",
          border: "border-yellow-200 dark:border-yellow-800",
          dot: "bg-yellow-500",
          title: "text-yellow-800 dark:text-yellow-200",
          message: "text-yellow-700 dark:text-yellow-300",
          time: "text-yellow-600 dark:text-yellow-400",
          pulse: ""
        };
      default:
        return {
          bg: "bg-blue-50 dark:bg-blue-900/20",
          border: "border-blue-200 dark:border-blue-800",
          dot: "bg-blue-500",
          title: "text-blue-800 dark:text-blue-200", 
          message: "text-blue-700 dark:text-blue-300",
          time: "text-blue-600 dark:text-blue-400",
          pulse: ""
        };
    }
  };

  const formatTimeAgo = (timestamp: string) => {
    const now = new Date();
    const alertTime = new Date(timestamp);
    const diffMs = now.getTime() - alertTime.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    
    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins} min ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)} hr ago`;
    return `${Math.floor(diffMins / 1440)} day ago`;
  };

  return (
    <Card className="border border-border">
      <CardHeader className="border-b border-border">
        <div className="flex items-center justify-between">
          <CardTitle>Real-time Alerts</CardTitle>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full pulse-dot"></div>
            <span className="text-xs text-muted-foreground">Live</span>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-0">
        <div className="p-6 space-y-4 max-h-96 overflow-y-auto">
          {alerts.length === 0 ? (
            <div className="text-center py-8">
              <Info className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">No active alerts</p>
            </div>
          ) : (
            alerts.slice(0, 10).map((alert) => {
              const styles = getAlertStyles(alert.severity);
              const Icon = getAlertIcon(alert.severity);
              
              return (
                <div 
                  key={alert.id} 
                  className={`flex items-start space-x-3 p-3 ${styles.bg} border ${styles.border} rounded-lg`}
                  data-testid={`alert-${alert.id}`}
                >
                  <div className={`w-3 h-3 ${styles.dot} rounded-full mt-1 ${styles.pulse}`}></div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <p className={`text-sm font-medium ${styles.title}`}>{alert.title}</p>
                      <span className={`text-xs ${styles.time}`}>
                        {formatTimeAgo(alert.createdAt)}
                      </span>
                    </div>
                    <p className={`text-sm ${styles.message}`}>
                      {alert.site?.name} - {alert.message}
                    </p>
                    {alert.severity === "critical" && (
                      <p className={`text-xs ${styles.time} mt-1`}>
                        Action Required: {alert.actionPlan?.split('\n')[0] || 'Immediate response needed'}
                      </p>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
        
        {alerts.length > 0 && (
          <div className="p-4 border-t border-border">
            <Button 
              variant="ghost" 
              className="w-full text-center text-sm text-primary hover:text-primary/80"
              data-testid="button-view-all-alerts"
            >
              View All Alerts
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
