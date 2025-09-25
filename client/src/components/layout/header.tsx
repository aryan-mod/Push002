import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Bell, AlertTriangle, X, Check, Clock, Settings, Sun, Moon } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { formatDistanceToNow } from "date-fns";
import { cn } from "@/lib/utils";

interface HeaderProps {
  title: string;
  subtitle?: string;
}

interface Notification {
  id: string;
  type: 'alert' | 'system' | 'info';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  severity?: 'critical' | 'high' | 'medium' | 'low';
  siteId?: string;
  siteName?: string;
}

interface EmergencyAlertDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void;
}

function EmergencyAlertDialog({ open, onOpenChange, onConfirm }: EmergencyAlertDialogProps) {
  const { toast } = useToast();
  
  const handleConfirm = () => {
    onConfirm();
    onOpenChange(false);
    toast({
      title: "🚨 Emergency Alert Sent",
      description: "Emergency services and relevant teams have been notified.",
      variant: "destructive",
      duration: 10000,
    });
  };
  
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="top" className="mx-auto max-w-md">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2 text-red-600">
            <AlertTriangle className="h-5 w-5" />
            Emergency Alert
          </SheetTitle>
        </SheetHeader>
        <div className="space-y-4 pt-4">
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              This will immediately notify emergency services and all relevant teams. 
              Only use this for genuine emergencies requiring immediate response.
            </AlertDescription>
          </Alert>
          <div className="flex gap-2">
            <Button
              variant="destructive"
              onClick={handleConfirm}
              className="flex-1"
            >
              <AlertTriangle className="mr-2 h-4 w-4" />
              Send Emergency Alert
            </Button>
            <Button
              variant="outline"
              onClick={() => onOpenChange(false)}
              className="flex-1"
            >
              Cancel
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}

export default function Header({ title, subtitle }: HeaderProps) {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [showEmergencyDialog, setShowEmergencyDialog] = useState(false);
  const { toast } = useToast();

  // Fetch notifications from the real API
  const { data: notifications = [], refetch: refetchNotifications } = useQuery<Notification[]>({
    queryKey: ["/api/v1/notifications"],
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  const unreadCount = notifications.filter(n => !n.read).length;
  const criticalCount = notifications.filter(n => n.severity === 'critical' && !n.read).length;

  const markAsRead = async (notificationId: string) => {
    try {
      const response = await fetch(`/api/v1/notifications/${notificationId}/read`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error('Failed to mark notification as read');
      }
      
      refetchNotifications();
    } catch (error) {
      console.error('Failed to mark notification as read:', error);
      toast({
        title: "Error",
        description: "Failed to mark notification as read",
        variant: "destructive",
      });
    }
  };

  const markAllAsRead = async () => {
    try {
      const response = await fetch('/api/v1/notifications/read-all', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error('Failed to mark all notifications as read');
      }
      
      const result = await response.json();
      refetchNotifications();
      toast({
        title: "All notifications marked as read",
        description: `${result.acknowledgedCount || 0} notifications acknowledged`,
        duration: 3000,
      });
    } catch (error) {
      console.error('Failed to mark all notifications as read:', error);
      toast({
        title: "Error",
        description: "Failed to mark all notifications as read",
        variant: "destructive",
      });
    }
  };

  const handleEmergencyAlert = async () => {
    try {
      const response = await fetch('/api/v1/emergency-alert', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: 'current-user', // Replace with actual user ID
          timestamp: new Date().toISOString(),
          location: 'Dashboard',
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to send emergency alert');
      }
      
      const result = await response.json();
      console.log('Emergency alert sent successfully:', result.alertId);
    } catch (error) {
      console.error('Failed to send emergency alert:', error);
      toast({
        title: "Failed to send emergency alert",
        description: "Please try again or contact support.",
        variant: "destructive",
      });
    }
  };

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
    document.documentElement.classList.toggle('dark');
  };

  const getNotificationIcon = (type: string, severity?: string) => {
    if (type === 'alert' && (severity === 'critical' || severity === 'high')) {
      return <AlertTriangle className="h-4 w-4 text-red-500" />;
    }
    return <Bell className="h-4 w-4 text-blue-500" />;
  };

  const getSeverityColor = (severity?: string) => {
    switch (severity) {
      case 'critical': return 'border-l-red-500 bg-red-50 dark:bg-red-900/20';
      case 'high': return 'border-l-orange-500 bg-orange-50 dark:bg-orange-900/20';
      case 'medium': return 'border-l-yellow-500 bg-yellow-50 dark:bg-yellow-900/20';
      case 'low': return 'border-l-green-500 bg-green-50 dark:bg-green-900/20';
      default: return 'border-l-blue-500 bg-blue-50 dark:bg-blue-900/20';
    }
  };

  return (
    <>
      <header className="bg-card border-b border-border px-6 py-4 sticky top-0 z-50 backdrop-blur supports-[backdrop-filter]:bg-card/60">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight">{title}</h2>
            {subtitle && (
              <p className="text-muted-foreground mt-1">{subtitle}</p>
            )}
          </div>
          
          {/* Alert Status & Controls */}
          <div className="flex items-center space-x-4">
            {/* System Status */}
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-muted-foreground">System Online</span>
            </div>

            {/* Theme Toggle */}
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleTheme}
              className="p-2 hover:bg-accent transition-colors"
              title="Toggle theme"
            >
              {isDarkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>

            {/* Notifications */}
            <Sheet>
              <SheetTrigger asChild>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  className="relative p-2 text-muted-foreground hover:text-foreground transition-colors"
                  data-testid="button-notifications"
                  title="View notifications"
                >
                  <Bell className="h-5 w-5" />
                  {unreadCount > 0 && (
                    <Badge 
                      variant={criticalCount > 0 ? "destructive" : "default"}
                      className={cn(
                        "absolute -top-1 -right-1 h-5 w-5 p-0 flex items-center justify-center text-xs",
                        criticalCount > 0 && "animate-pulse"
                      )}
                    >
                      {unreadCount > 99 ? '99+' : unreadCount}
                    </Badge>
                  )}
                </Button>
              </SheetTrigger>
              <SheetContent className="w-96">
                <SheetHeader>
                  <div className="flex items-center justify-between">
                    <SheetTitle>Notifications</SheetTitle>
                    {unreadCount > 0 && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={markAllAsRead}
                        className="text-xs"
                      >
                        <Check className="mr-1 h-3 w-3" />
                        Mark all read
                      </Button>
                    )}
                  </div>
                </SheetHeader>
                <ScrollArea className="h-full pt-4">
                  <div className="space-y-3">
                    {notifications.length === 0 ? (
                      <div className="text-center py-8 text-muted-foreground">
                        <Bell className="mx-auto h-8 w-8 mb-2 opacity-50" />
                        No notifications
                      </div>
                    ) : (
                      notifications.map((notification) => (
                        <div
                          key={notification.id}
                          className={cn(
                            "p-3 rounded-lg border-l-4 cursor-pointer transition-all hover:shadow-sm",
                            getSeverityColor(notification.severity),
                            !notification.read && "ring-2 ring-blue-200 dark:ring-blue-800"
                          )}
                          onClick={() => !notification.read && markAsRead(notification.id)}
                        >
                          <div className="flex items-start gap-3">
                            {getNotificationIcon(notification.type, notification.severity)}
                            <div className="flex-1 space-y-1">
                              <div className="flex items-center justify-between">
                                <h4 className={cn(
                                  "text-sm font-medium",
                                  !notification.read && "font-semibold"
                                )}>
                                  {notification.title}
                                </h4>
                                <div className="flex items-center gap-1">
                                  <Clock className="h-3 w-3 text-muted-foreground" />
                                  <span className="text-xs text-muted-foreground">
                                    {formatDistanceToNow(new Date(notification.timestamp), { addSuffix: true })}
                                  </span>
                                </div>
                              </div>
                              <p className="text-sm text-muted-foreground">
                                {notification.message}
                              </p>
                              {notification.siteName && (
                                <p className="text-xs text-blue-600 dark:text-blue-400">
                                  📍 {notification.siteName}
                                </p>
                              )}
                            </div>
                            {!notification.read && (
                              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                            )}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </ScrollArea>
              </SheetContent>
            </Sheet>

            {/* Emergency Button */}
            <Button 
              variant="destructive" 
              className="flex items-center space-x-2 shadow-lg hover:shadow-xl transition-shadow"
              onClick={() => setShowEmergencyDialog(true)}
              data-testid="button-emergency"
            >
              <AlertTriangle className="h-4 w-4" />
              <span className="hidden sm:inline">Emergency Alert</span>
              <span className="sm:hidden">Emergency</span>
            </Button>
          </div>
        </div>
      </header>

      <EmergencyAlertDialog
        open={showEmergencyDialog}
        onOpenChange={setShowEmergencyDialog}
        onConfirm={handleEmergencyAlert}
      />
    </>
  );
}