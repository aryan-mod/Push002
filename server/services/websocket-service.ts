import { WebSocketServer, WebSocket } from "ws";

interface WebSocketMessage {
  type: string;
  data: any;
}

interface ConnectedClient {
  ws: WebSocket;
  userId?: string;
  subscribedSites?: string[];
}

class WebSocketService {
  private wss?: WebSocketServer;
  private clients: Map<WebSocket, ConnectedClient> = new Map();

  initialize(wss: WebSocketServer): void {
    this.wss = wss;

    wss.on('connection', (ws: WebSocket) => {
      console.log('New WebSocket connection');
      
      this.clients.set(ws, { ws });

      ws.on('message', (data: Buffer) => {
        try {
          const message = JSON.parse(data.toString()) as WebSocketMessage;
          this.handleMessage(ws, message);
        } catch (error) {
          console.error('Invalid WebSocket message:', error);
        }
      });

      ws.on('close', () => {
        console.log('WebSocket connection closed');
        this.clients.delete(ws);
      });

      ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        this.clients.delete(ws);
      });

      // Send welcome message
      this.sendToClient(ws, {
        type: 'connected',
        data: { message: 'Connected to GeoMindFlow Tourist Safety System' }
      });
    });
  }

  private handleMessage(ws: WebSocket, message: WebSocketMessage): void {
    const client = this.clients.get(ws);
    if (!client) return;

    switch (message.type) {
      case 'authenticate':
        client.userId = message.data.userId;
        this.sendToClient(ws, {
          type: 'authenticated',
          data: { userId: client.userId }
        });
        break;

      case 'subscribe_sites':
        client.subscribedSites = message.data.siteIds;
        this.sendToClient(ws, {
          type: 'subscribed',
          data: { sites: client.subscribedSites }
        });
        break;

      case 'ping':
        this.sendToClient(ws, {
          type: 'pong',
          data: { timestamp: new Date().toISOString() }
        });
        break;

      default:
        console.log('Unknown message type:', message.type);
    }
  }

  broadcast(message: WebSocketMessage): void {
    this.clients.forEach((client) => {
      if (client.ws.readyState === WebSocket.OPEN) {
        this.sendToClient(client.ws, message);
      }
    });
  }

  broadcastToSite(siteId: string, message: WebSocketMessage): void {
    this.clients.forEach((client) => {
      if (client.ws.readyState === WebSocket.OPEN && 
          client.subscribedSites?.includes(siteId)) {
        this.sendToClient(client.ws, message);
      }
    });
  }

  broadcastToUser(userId: string, message: WebSocketMessage): void {
    this.clients.forEach((client) => {
      if (client.ws.readyState === WebSocket.OPEN && 
          client.userId === userId) {
        this.sendToClient(client.ws, message);
      }
    });
  }

  private sendToClient(ws: WebSocket, message: WebSocketMessage): void {
    try {
      ws.send(JSON.stringify(message));
    } catch (error) {
      console.error('Failed to send WebSocket message:', error);
    }
  }

  // Periodic status updates
  startPeriodicUpdates(): void {
    setInterval(() => {
      this.broadcast({
        type: 'system_status',
        data: {
          timestamp: new Date().toISOString(),
          connectedClients: this.clients.size,
          status: 'online'
        }
      });
    }, 30000); // Every 30 seconds
  }

  getConnectionCount(): number {
    return this.clients.size;
  }

  getConnectedUsers(): string[] {
    const users: string[] = [];
    this.clients.forEach((client) => {
      if (client.userId) {
        users.push(client.userId);
      }
    });
    return users;
  }
}

export const websocketService = new WebSocketService();
