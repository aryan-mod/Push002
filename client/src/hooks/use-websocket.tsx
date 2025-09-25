import { useEffect, useRef } from "react";
import { useToast } from "./use-toast";

interface WebSocketMessage {
  type: string;
  data: any;
}

export function useWebSocket() {
  const { toast } = useToast();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = () => {
    try {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${protocol}//${window.location.host}/ws`;
      
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log("WebSocket connected");
        reconnectAttempts.current = 0;
        
        // Send authentication if needed
        wsRef.current?.send(JSON.stringify({
          type: "authenticate",
          data: { userId: "current-user" } // Replace with actual user ID
        }));

        // Subscribe to all sites
        wsRef.current?.send(JSON.stringify({
          type: "subscribe_sites",
          data: { siteIds: ["all"] }
        }));
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          handleMessage(message);
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error);
        }
      };

      wsRef.current.onclose = (event) => {
        console.log("WebSocket disconnected:", event.code, event.reason);
        wsRef.current = null;

        // Attempt to reconnect if not a clean close
        if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
          reconnectAttempts.current++;
          
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log(`Attempting to reconnect (${reconnectAttempts.current}/${maxReconnectAttempts})...`);
            connect();
          }, delay);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error("WebSocket error:", error);
      };

    } catch (error) {
      console.error("Failed to create WebSocket connection:", error);
    }
  };

  const handleMessage = (message: WebSocketMessage) => {
    switch (message.type) {
      case "connected":
        console.log("WebSocket connection established:", message.data.message);
        break;

      case "alert":
        const alert = message.data;
        const severity = alert.severity;
        
        toast({
          title: `🚨 ${alert.title}`,
          description: `${alert.site?.name || 'Unknown Site'} - ${alert.message}`,
          variant: severity === "critical" || severity === "high" ? "destructive" : "default",
          duration: severity === "critical" ? 0 : 5000, // Critical alerts don't auto-dismiss
        });
        break;

      case "alert_acknowledged":
        toast({
          title: "Alert Acknowledged",
          description: "Alert has been acknowledged by team member.",
        });
        break;

      case "system_status":
        // Handle system status updates
        console.log("System status:", message.data);
        break;

      case "prediction_update":
        const prediction = message.data;
        if (prediction.probability > 0.8) {
          toast({
            title: "High Risk Detected",
            description: `Site ${prediction.siteId} shows elevated risk levels.`,
            variant: "destructive",
          });
        }
        break;

      case "sensor_offline":
        toast({
          title: "Sensor Offline",
          description: `Sensor ${message.data.sensorId} has gone offline.`,
          variant: "destructive",
        });
        break;

      case "pong":
        // Response to ping, used for connection health check
        break;

      default:
        console.log("Unknown WebSocket message type:", message.type);
    }
  };

  const sendMessage = (message: WebSocketMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn("WebSocket is not connected");
    }
  };

  const disconnect = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, "Component unmounting");
      wsRef.current = null;
    }
  };

  useEffect(() => {
    connect();

    // Ping interval to keep connection alive
    const pingInterval = setInterval(() => {
      sendMessage({ type: "ping", data: {} });
    }, 30000);

    return () => {
      clearInterval(pingInterval);
      disconnect();
    };
  }, []);

  return {
    isConnected: wsRef.current?.readyState === WebSocket.OPEN,
    sendMessage,
    disconnect,
  };
}
