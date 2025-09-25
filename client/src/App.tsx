import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import Dashboard from "@/pages/dashboard";
import SiteMonitoring from "@/pages/site-monitoring";
import LocationSearch from "@/pages/location-search";
import ModelManagement from "@/pages/model-management";
import DataUpload from "@/pages/data-upload";
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
          <Route path="/location-search" component={LocationSearch} />
          <Route path="/models" component={ModelManagement} />
          <Route path="/upload" component={DataUpload} />
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
