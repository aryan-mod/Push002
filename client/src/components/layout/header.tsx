import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Bell, AlertTriangle } from "lucide-react";

interface HeaderProps {
  title: string;
  subtitle?: string;
}

export default function Header({ title, subtitle }: HeaderProps) {
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
            <span className="text-sm text-muted-foreground">System Online</span>
          </div>

          {/* Real-time Alert Badge */}
          <div className="relative">
            <Button 
              variant="ghost" 
              size="sm" 
              className="relative p-2 text-muted-foreground hover:text-foreground"
              data-testid="button-alerts"
            >
              <Bell className="h-5 w-5" />
              <Badge 
                variant="destructive" 
                className="absolute -top-1 -right-1 h-5 w-5 p-0 flex items-center justify-center text-xs"
              >
                3
              </Badge>
            </Button>
          </div>

          {/* Emergency Button */}
          <Button 
            variant="destructive" 
            className="flex items-center space-x-2"
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
