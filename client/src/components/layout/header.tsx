import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Bell, AlertTriangle } from "lucide-react";

interface HeaderProps {
  title: string;
  subtitle?: string;
}

/**
 * Header Component with Tourist Safety Navigation
 * Features functional notification bell and emergency alert button
 * Shows real-time alert counts and system status
 */
export default function Header({ title, subtitle }: HeaderProps) {
  const [, setLocation] = useLocation();
  
  // Fetch active alerts count for notification badge
  const { data: alerts = [] } = useQuery<any[]>({
    queryKey: ["/api/v1/alerts"],
    refetchInterval: 5000, // Update every 5 seconds
  });

  const activeAlerts = alerts.filter((alert: any) => alert.status === "active");
  const alertCount = activeAlerts.length;

  return (
    <header className="bg-card border-b border-border px-6 py-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold">{title}</h2>
          {subtitle && (
            <p className="text-muted-foreground">{subtitle}</p>
          )}
        </div>
        
        {/* Alert Status & Controls */}
        <div className="flex items-center space-x-4">
          {/* System Status */}
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full pulse-dot"></div>
            <span className="text-sm text-muted-foreground">Safety System Online</span>
          </div>

          {/* Notification Bell - Links to Notifications Page */}
          <div className="relative">
            <Button 
              variant="ghost" 
              size="sm" 
              className="relative p-2 text-muted-foreground hover:text-foreground"
              onClick={() => setLocation("/notifications")}
              data-testid="button-notifications"
            >
              <Bell className="h-5 w-5" />
              {alertCount > 0 && (
                <Badge 
                  variant="destructive" 
                  className="absolute -top-1 -right-1 h-5 w-5 p-0 flex items-center justify-center text-xs"
                >
                  {alertCount > 99 ? "99+" : alertCount}
                </Badge>
              )}
            </Button>
          </div>

          {/* Emergency Alert Button - Links to Emergency Page */}
          <Button 
            variant="destructive" 
            className="flex items-center space-x-2"
            onClick={() => setLocation("/emergency")}
            data-testid="button-emergency"
          >
            <AlertTriangle className="h-4 w-4" />
            <span>Emergency Alert</span>
          </Button>
        </div>
      </div>
    </header>
  );
}
