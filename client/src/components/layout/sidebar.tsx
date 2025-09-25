import { useState } from "react";
import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { 
  LayoutDashboard, 
  MapPin, 
  Brain, 
  Upload, 
  Bell, 
  BarChart3, 
  Settings, 
  AlertTriangle,
  Shield
} from "lucide-react";

const navigation = [
  { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { name: "Live Map Tracking", href: "/sites", icon: MapPin },
  { name: "Emergency Alert", href: "/emergency", icon: AlertTriangle },
  { name: "Notifications", href: "/notifications", icon: Bell },
  { name: "Safety Reports", href: "/reports", icon: BarChart3 },
  { name: "Alert Settings", href: "/settings", icon: Settings },
  { name: "Model Management", href: "/models", icon: Brain },
  { name: "Data Upload", href: "/upload", icon: Upload },
];

export default function Sidebar() {
  const [location] = useLocation();

  return (
    <aside className="w-64 bg-card border-r border-border flex flex-col">
      {/* Logo & Title */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center">
            <Shield className="h-6 w-6 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-lg font-semibold">GeoMindFlow</h1>
            <p className="text-sm text-muted-foreground">Tourist Safety System</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navigation.map((item) => {
          const isActive = location === item.href || (item.href === "/dashboard" && location === "/");
          return (
            <Link key={item.name} href={item.href}>
              <Button
                variant={isActive ? "default" : "ghost"}
                className={cn(
                  "w-full justify-start",
                  isActive && "bg-primary text-primary-foreground"
                )}
                data-testid={`nav-${item.name.toLowerCase().replace(" ", "-")}`}
              >
                <item.icon className="mr-3 h-4 w-4" />
                {item.name}
              </Button>
            </Link>
          );
        })}
      </nav>

      {/* User Profile */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center space-x-3">
          <Avatar className="h-8 w-8">
            <AvatarFallback className="bg-muted">AS</AvatarFallback>
          </Avatar>
          <div className="flex-1">
            <p className="text-sm font-medium">Alex Singh</p>
            <p className="text-xs text-muted-foreground">Safety Coordinator</p>
          </div>
          <Button variant="ghost" size="sm" data-testid="button-user-settings">
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </aside>
  );
}
