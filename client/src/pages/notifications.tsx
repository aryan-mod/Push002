import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Bell, CheckCircle, AlertTriangle, Info, X, Filter } from "lucide-react";
import Header from "@/components/layout/header";
import { formatDistanceToNow } from "date-fns";

/**
 * Notifications Page Component
 * Displays all alerts and notifications from the system
 * Allows filtering by type, severity, and status
 * Connected to the bell icon in the navigation header
 */
export default function Notifications() {
  const [filter, setFilter] = useState("all");
  
  // Fetch alerts from backend
  const { data: alerts = [], isLoading } = useQuery<any[]>({
    queryKey: ["/api/v1/alerts"],
    refetchInterval: 5000, // Refresh every 5 seconds for real-time updates
  });

  // Filter alerts based on selected filter
  const filteredAlerts = alerts.filter((alert: any) => {
    switch (filter) {
      case "active":
        return alert.status === "active";
      case "acknowledged":
        return alert.status === "acknowledged";
      case "critical":
        return alert.severity === "critical";
      case "emergency":
        return alert.type === "emergency";
      default:
        return true;
    }
  });

  // Group alerts by status for better organization
  const activeAlerts = alerts.filter((alert: any) => alert.status === "active");
  const acknowledgedAlerts = alerts.filter((alert: any) => alert.status === "acknowledged");
  
  return (
    <>
      <Header 
        title="Notifications & Alerts" 
        subtitle={`${activeAlerts.length} active alerts, ${acknowledgedAlerts.length} acknowledged`}
      />
      
      <div className="p-6">
        {/* Filter Controls */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Filter:</span>
            <div className="flex space-x-2">
              {["all", "active", "critical", "emergency", "acknowledged"].map((filterOption) => (
                <Button
                  key={filterOption}
                  variant={filter === filterOption ? "default" : "outline"}
                  size="sm"
                  onClick={() => setFilter(filterOption)}
                  data-testid={`filter-${filterOption}`}
                >
                  {filterOption.charAt(0).toUpperCase() + filterOption.slice(1)}
                </Button>
              ))}
            </div>
          </div>

          <div className="text-sm text-muted-foreground">
            Last updated: {new Date().toLocaleTimeString()}
          </div>
        </div>

        <Tabs defaultValue="all" className="space-y-6">
          <TabsList>
            <TabsTrigger value="all">All Notifications</TabsTrigger>
            <TabsTrigger value="active">
              Active Alerts
              {activeAlerts.length > 0 && (
                <Badge variant="destructive" className="ml-2">
                  {activeAlerts.length}
                </Badge>
              )}
            </TabsTrigger>
            <TabsTrigger value="acknowledged">Acknowledged</TabsTrigger>
          </TabsList>

          <TabsContent value="all" className="space-y-4">
            <NotificationList 
              alerts={filteredAlerts} 
              isLoading={isLoading}
              showAll={true}
            />
          </TabsContent>

          <TabsContent value="active" className="space-y-4">
            <NotificationList 
              alerts={activeAlerts} 
              isLoading={isLoading}
              showAll={false}
            />
          </TabsContent>

          <TabsContent value="acknowledged" className="space-y-4">
            <NotificationList 
              alerts={acknowledgedAlerts} 
              isLoading={isLoading}
              showAll={false}
            />
          </TabsContent>
        </Tabs>
      </div>
    </>
  );
}

/**
 * NotificationList Component
 * Renders a list of notification cards with alert details
 */
interface NotificationListProps {
  alerts: any[];
  isLoading: boolean;
  showAll: boolean;
}

function NotificationList({ alerts, isLoading, showAll }: NotificationListProps) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <Card key={i} className="animate-pulse">
            <CardHeader className="pb-3">
              <div className="h-4 bg-muted rounded w-3/4"></div>
              <div className="h-3 bg-muted rounded w-1/2"></div>
            </CardHeader>
            <CardContent>
              <div className="h-3 bg-muted rounded w-full mb-2"></div>
              <div className="h-3 bg-muted rounded w-2/3"></div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (alerts.length === 0) {
    return (
      <Card className="text-center py-12">
        <CardContent>
          <Bell className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <CardTitle className="text-muted-foreground mb-2">No notifications</CardTitle>
          <CardDescription>
            {showAll ? "All caught up! No new notifications." : "No notifications in this category."}
          </CardDescription>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {alerts.map((alert: any) => (
        <NotificationCard key={alert.id} alert={alert} />
      ))}
    </div>
  );
}

/**
 * Individual Notification Card Component
 * Displays alert details with appropriate styling based on severity
 */
interface NotificationCardProps {
  alert: any;
}

function NotificationCard({ alert }: NotificationCardProps) {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical":
        return "border-red-500 bg-red-50 dark:bg-red-950/20";
      case "high":
        return "border-orange-500 bg-orange-50 dark:bg-orange-950/20";
      case "medium":
        return "border-yellow-500 bg-yellow-50 dark:bg-yellow-950/20";
      default:
        return "border-blue-500 bg-blue-50 dark:bg-blue-950/20";
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case "critical":
        return <AlertTriangle className="w-5 h-5 text-red-600" />;
      case "high":
        return <AlertTriangle className="w-5 h-5 text-orange-600" />;
      case "medium":
        return <Info className="w-5 h-5 text-yellow-600" />;
      default:
        return <Info className="w-5 h-5 text-blue-600" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "active":
        return <Badge variant="destructive">Active</Badge>;
      case "acknowledged":
        return <Badge variant="secondary">Acknowledged</Badge>;
      case "resolved":
        return <Badge variant="outline">Resolved</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  return (
    <Card 
      className={`${getSeverityColor(alert.severity)} transition-all hover:shadow-md`}
      data-testid={`alert-card-${alert.id}`}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-3">
            {getSeverityIcon(alert.severity)}
            <div className="flex-1">
              <CardTitle className="text-base font-medium">
                {alert.title}
              </CardTitle>
              <CardDescription className="mt-1">
                {alert.site?.name && `Site: ${alert.site.name} • `}
                {formatDistanceToNow(new Date(alert.createdAt), { addSuffix: true })}
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {getStatusBadge(alert.status)}
            <Badge variant="outline" className="text-xs">
              {alert.severity.toUpperCase()}
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        <div className="space-y-3">
          {/* Alert Message */}
          <p className="text-sm text-foreground">
            {alert.message}
          </p>

          {/* Action Plan - if available */}
          {alert.actionPlan && (
            <div className="bg-background/50 rounded-lg p-3">
              <h4 className="text-xs font-medium text-muted-foreground mb-1">
                RECOMMENDED ACTIONS:
              </h4>
              <pre className="text-xs text-foreground whitespace-pre-wrap font-sans">
                {alert.actionPlan}
              </pre>
            </div>
          )}

          {/* Acknowledgment Info */}
          {alert.status === "acknowledged" && alert.acknowledgedBy && (
            <div className="flex items-center space-x-2 text-xs text-muted-foreground">
              <CheckCircle className="w-3 h-3" />
              <span>
                Acknowledged by {alert.acknowledgedBy}
                {alert.acknowledgedAt && ` on ${new Date(alert.acknowledgedAt).toLocaleString()}`}
              </span>
            </div>
          )}

          {/* Action Buttons */}
          {alert.status === "active" && (
            <div className="flex space-x-2 pt-2 border-t">
              <Button 
                size="sm" 
                variant="outline"
                data-testid={`acknowledge-alert-${alert.id}`}
                onClick={() => {
                  // TODO: Implement acknowledge functionality
                  console.log("Acknowledge alert:", alert.id);
                }}
              >
                <CheckCircle className="w-3 h-3 mr-1" />
                Acknowledge
              </Button>
              <Button 
                size="sm" 
                variant="ghost"
                data-testid={`dismiss-alert-${alert.id}`}
                onClick={() => {
                  // TODO: Implement dismiss functionality
                  console.log("Dismiss alert:", alert.id);
                }}
              >
                <X className="w-3 h-3 mr-1" />
                Dismiss
              </Button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}