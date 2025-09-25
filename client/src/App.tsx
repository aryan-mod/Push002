import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import Dashboard from "@/pages/dashboard";
import SiteMonitoring from "@/pages/site-monitoring";
import ModelManagement from "@/pages/model-management";
import DataUpload from "@/pages/data-upload";
import EmergencyAlert from "@/pages/emergency-alert";
import Notifications from "@/pages/notifications";
import AlertSettings from "@/pages/alert-settings";
import Reports from "@/pages/reports";
import NotFound from "@/pages/not-found";
import Sidebar from "@/components/layout/sidebar";
import { useWebSocket } from "@/hooks/use-websocket";

function Router() {
  return (
    <div className="min-h-screen flex">
      <Sidebar />
      <main className="flex-1">
        <Switch>
          <Route path="/" component={Dashboard} />
          <Route path="/dashboard" component={Dashboard} />
          <Route path="/sites" component={SiteMonitoring} />
          <Route path="/models" component={ModelManagement} />
          <Route path="/upload" component={DataUpload} />
          <Route path="/emergency" component={EmergencyAlert} />
          <Route path="/notifications" component={Notifications} />
          <Route path="/settings" component={AlertSettings} />
          <Route path="/reports" component={Reports} />
          <Route component={NotFound} />
        </Switch>
      </main>
    </div>
  );
}

function App() {
  // Initialize WebSocket connection
  useWebSocket();

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Router />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
