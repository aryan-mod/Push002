"""
WebSocket connection manager for real-time communications.

This module manages WebSocket connections and handles broadcasting
of alerts and updates to connected clients.
"""

from typing import List, Dict, Any
from fastapi import WebSocket
import json
import asyncio

class WebSocketManager:
    """
    Manager class for handling WebSocket connections and broadcasting messages.
    
    This class maintains a list of active connections and provides methods
    to connect, disconnect, and broadcast messages to all connected clients.
    """
    
    def __init__(self):
        # List of active WebSocket connections
        self.active_connections: List[WebSocket] = []
        # Dictionary to store connection metadata (optional)
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str = None):
        """
        Accept a new WebSocket connection and add it to the active connections list.
        
        Args:
            websocket: WebSocket connection object
            user_id: Optional user identifier for the connection
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Store connection metadata if provided
        if user_id:
            self.connection_info[websocket] = {"user_id": user_id}
        
        print(f"📱 New WebSocket connection established. Total connections: {len(self.active_connections)}")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connected",
            "message": "Successfully connected to GeoMindFlow alerts",
            "total_connections": len(self.active_connections)
        }, websocket)
    
    def disconnect(self, websocket: WebSocket):
        """
        Remove a WebSocket connection from the active connections list.
        
        Args:
            websocket: WebSocket connection to remove
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
        if websocket in self.connection_info:
            del self.connection_info[websocket]
            
        print(f"📱 WebSocket connection closed. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """
        Send a message to a specific WebSocket connection.
        
        Args:
            message: Dictionary containing the message data
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"❌ Error sending personal message: {e}")
            # Remove dead connection
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast a message to all active WebSocket connections.
        
        Args:
            message: Dictionary containing the message data to broadcast
        """
        if not self.active_connections:
            print("📡 No active WebSocket connections to broadcast to")
            return
        
        # Create list of coroutines for concurrent sending
        send_tasks = []
        dead_connections = []
        
        message_str = json.dumps(message)
        
        for connection in self.active_connections:
            try:
                send_tasks.append(connection.send_text(message_str))
            except Exception as e:
                print(f"❌ Error preparing broadcast for connection: {e}")
                dead_connections.append(connection)
        
        # Remove dead connections
        for connection in dead_connections:
            self.disconnect(connection)
        
        # Send messages concurrently
        if send_tasks:
            try:
                await asyncio.gather(*send_tasks, return_exceptions=True)
                print(f"📡 Broadcasted message to {len(send_tasks)} connections")
            except Exception as e:
                print(f"❌ Error during broadcast: {e}")
    
    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """
        Broadcast an alert message to all connected clients.
        
        Args:
            alert_data: Alert information to broadcast
        """
        alert_message = {
            "type": "alert",
            "data": alert_data,
            "timestamp": alert_data.get("timestamp")
        }
        await self.broadcast(alert_message)
    
    async def broadcast_prediction(self, prediction_data: Dict[str, Any]):
        """
        Broadcast a prediction update to all connected clients.
        
        Args:
            prediction_data: Prediction information to broadcast
        """
        prediction_message = {
            "type": "prediction_update", 
            "data": prediction_data
        }
        await self.broadcast(prediction_message)
    
    def get_connection_count(self) -> int:
        """
        Get the number of active WebSocket connections.
        
        Returns:
            int: Number of active connections
        """
        return len(self.active_connections)
    
    def get_connections_by_user(self, user_id: str) -> List[WebSocket]:
        """
        Get all connections for a specific user.
        
        Args:
            user_id: User identifier to search for
            
        Returns:
            List[WebSocket]: List of connections for the user
        """
        connections = []
        for websocket, info in self.connection_info.items():
            if info.get("user_id") == user_id:
                connections.append(websocket)
        return connections

# Global WebSocket manager instance
websocket_manager = WebSocketManager()