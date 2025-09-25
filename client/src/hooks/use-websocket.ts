import { useEffect, useRef } from "react";

/**
 * WebSocket hook for real-time updates
 * Connects to the WebSocket server for live alerts and data updates
 */
export function useWebSocket() {
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Connect to WebSocket server
    const connectWebSocket = () => {
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const wsUrl = `${protocol}//${host}/ws`;
        
        ws.current = new WebSocket(wsUrl);
        
        ws.current.onopen = () => {
          console.log('WebSocket connected');
        };
        
        ws.current.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('WebSocket message received:', data);
            
            // Handle different types of real-time updates
            if (data.type === 'alert') {
              // Handle real-time alerts
              console.log('New alert received:', data.payload);
            } else if (data.type === 'location_update') {
              // Handle tourist location updates
              console.log('Location update:', data.payload);
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };
        
        ws.current.onclose = () => {
          console.log('WebSocket disconnected');
          // Attempt to reconnect after a delay
          setTimeout(connectWebSocket, 5000);
        };
        
        ws.current.onerror = (error) => {
          console.error('WebSocket error:', error);
        };
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
      }
    };

    connectWebSocket();

    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  return ws.current;
}