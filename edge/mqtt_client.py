#!/usr/bin/env python3
"""
MQTT Client for Edge Agent Communication
Handles pub/sub messaging between edge devices and central system.
"""

import json
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable
import ssl

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("WARNING: paho-mqtt not available. MQTT functionality disabled.")

logger = logging.getLogger(__name__)

class MQTTClient:
    """
    MQTT client wrapper for rockfall prediction system communication.
    Handles connection management, message publishing, and subscription.
    """
    
    def __init__(
        self,
        broker: str,
        port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: str = "rockfall_edge",
        use_ssl: bool = False,
        keepalive: int = 60
    ):
        """Initialize MQTT client."""
        if not MQTT_AVAILABLE:
            raise ImportError("paho-mqtt package is required for MQTT functionality")
        
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.client_id = client_id
        self.use_ssl = use_ssl
        self.keepalive = keepalive
        
        # Connection state
        self.connected = False
        self.connection_lock = threading.Lock()
        
        # Message callback
        self.message_callback: Optional[Callable[[str, str], None]] = None
        
        # Initialize MQTT client
        self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        
        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_publish = self._on_publish
        self.client.on_subscribe = self._on_subscribe
        self.client.on_log = self._on_log
        
        # Set credentials if provided
        if username and password:
            self.client.username_pw_set(username, password)
        
        # SSL configuration
        if use_ssl:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            self.client.tls_set_context(context)
        
        logger.info(f"MQTT client initialized for broker {broker}:{port}")
    
    def connect(self, retry_attempts: int = 5, retry_delay: int = 5) -> bool:
        """
        Connect to MQTT broker with retry logic.
        
        Args:
            retry_attempts: Number of connection attempts
            retry_delay: Delay between attempts in seconds
            
        Returns:
            True if connected successfully, False otherwise
        """
        with self.connection_lock:
            for attempt in range(retry_attempts):
                try:
                    logger.info(f"Connecting to MQTT broker {self.broker}:{self.port} (attempt {attempt + 1})")
                    
                    result = self.client.connect(self.broker, self.port, self.keepalive)
                    
                    if result == mqtt.MQTT_ERR_SUCCESS:
                        # Start network loop in background thread
                        self.client.loop_start()
                        
                        # Wait for connection confirmation
                        wait_time = 0
                        while not self.connected and wait_time < 10:
                            time.sleep(0.5)
                            wait_time += 0.5
                        
                        if self.connected:
                            logger.info("MQTT connection established successfully")
                            return True
                        else:
                            logger.warning("MQTT connection timeout")
                    else:
                        logger.error(f"MQTT connection failed with code: {result}")
                
                except Exception as e:
                    logger.error(f"MQTT connection attempt {attempt + 1} failed: {e}")
                
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
            
            logger.error(f"Failed to connect to MQTT broker after {retry_attempts} attempts")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        try:
            with self.connection_lock:
                if self.connected:
                    self.client.loop_stop()
                    self.client.disconnect()
                    self.connected = False
                    logger.info("MQTT client disconnected")
        
        except Exception as e:
            logger.error(f"MQTT disconnect failed: {e}")
    
    def publish(
        self,
        topic: str,
        payload: str,
        qos: int = 1,
        retain: bool = False
    ) -> bool:
        """
        Publish message to MQTT topic.
        
        Args:
            topic: MQTT topic
            payload: Message payload
            qos: Quality of Service level (0, 1, or 2)
            retain: Whether to retain the message
            
        Returns:
            True if message was queued successfully, False otherwise
        """
        try:
            if not self.connected:
                logger.warning("Cannot publish - MQTT client not connected")
                return False
            
            result = self.client.publish(topic, payload, qos=qos, retain=retain)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Published to {topic}: {payload[:100]}...")
                return True
            else:
                logger.error(f"Failed to publish to {topic}, error code: {result.rc}")
                return False
        
        except Exception as e:
            logger.error(f"MQTT publish failed: {e}")
            return False
    
    def subscribe(self, topic: str, qos: int = 1) -> bool:
        """
        Subscribe to MQTT topic.
        
        Args:
            topic: MQTT topic pattern
            qos: Quality of Service level
            
        Returns:
            True if subscription was successful, False otherwise
        """
        try:
            if not self.connected:
                logger.warning("Cannot subscribe - MQTT client not connected")
                return False
            
            result, mid = self.client.subscribe(topic, qos=qos)
            
            if result == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Subscribed to topic: {topic}")
                return True
            else:
                logger.error(f"Failed to subscribe to {topic}, error code: {result}")
                return False
        
        except Exception as e:
            logger.error(f"MQTT subscribe failed: {e}")
            return False
    
    def unsubscribe(self, topic: str) -> bool:
        """Unsubscribe from MQTT topic."""
        try:
            if not self.connected:
                return False
            
            result, mid = self.client.unsubscribe(topic)
            
            if result == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Unsubscribed from topic: {topic}")
                return True
            else:
                logger.error(f"Failed to unsubscribe from {topic}, error code: {result}")
                return False
        
        except Exception as e:
            logger.error(f"MQTT unsubscribe failed: {e}")
            return False
    
    def set_message_callback(self, callback: Callable[[str, str], None]):
        """Set callback function for received messages."""
        self.message_callback = callback
    
    def is_connected(self) -> bool:
        """Check if client is connected to broker."""
        return self.connected
    
    def publish_json(self, topic: str, data: Dict[str, Any], **kwargs) -> bool:
        """Publish JSON data to topic."""
        try:
            payload = json.dumps(data, default=str)
            return self.publish(topic, payload, **kwargs)
        except Exception as e:
            logger.error(f"JSON publish failed: {e}")
            return False
    
    def publish_sensor_data(self, sensor_data: Dict[str, Any]) -> bool:
        """Publish sensor data with standard format."""
        topic = f"rockfall/sensor/{sensor_data.get('site_id', 'unknown')}"
        return self.publish_json(topic, sensor_data)
    
    def publish_prediction(self, prediction: Dict[str, Any]) -> bool:
        """Publish prediction with standard format."""
        topic = f"rockfall/prediction/{prediction.get('site_id', 'unknown')}"
        return self.publish_json(topic, prediction)
    
    def publish_alert(self, alert: Dict[str, Any]) -> bool:
        """Publish alert with high priority."""
        topic = f"rockfall/alert/{alert.get('site_id', 'unknown')}"
        return self.publish_json(topic, alert, qos=2, retain=True)
    
    # MQTT event callbacks
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for when client connects to broker."""
        if rc == 0:
            self.connected = True
            logger.info("MQTT client connected successfully")
        else:
            self.connected = False
            error_messages = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier",
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorised"
            }
            error_msg = error_messages.get(rc, f"Connection refused - unknown error ({rc})")
            logger.error(f"MQTT connection failed: {error_msg}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for when client disconnects from broker."""
        self.connected = False
        if rc != 0:
            logger.warning(f"MQTT client disconnected unexpectedly (code: {rc})")
        else:
            logger.info("MQTT client disconnected")
    
    def _on_message(self, client, userdata, msg):
        """Callback for when message is received."""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            logger.debug(f"Received message on {topic}: {payload[:100]}...")
            
            # Call user-defined callback if set
            if self.message_callback:
                self.message_callback(topic, payload)
        
        except Exception as e:
            logger.error(f"Message handling failed: {e}")
    
    def _on_publish(self, client, userdata, mid):
        """Callback for when message is published."""
        logger.debug(f"Message published (mid: {mid})")
    
    def _on_subscribe(self, client, userdata, mid, granted_qos):
        """Callback for when subscription is acknowledged."""
        logger.debug(f"Subscription confirmed (mid: {mid}, QoS: {granted_qos})")
    
    def _on_log(self, client, userdata, level, buf):
        """Callback for MQTT client logging."""
        if level == mqtt.MQTT_LOG_DEBUG:
            logger.debug(f"MQTT: {buf}")
        elif level == mqtt.MQTT_LOG_INFO:
            logger.info(f"MQTT: {buf}")
        elif level == mqtt.MQTT_LOG_WARNING:
            logger.warning(f"MQTT: {buf}")
        elif level == mqtt.MQTT_LOG_ERR:
            logger.error(f"MQTT: {buf}")


class MQTTTopicManager:
    """Helper class for managing MQTT topics with consistent naming."""
    
    def __init__(self, base_topic: str = "rockfall"):
        self.base_topic = base_topic.rstrip('/')
    
    def sensor_data(self, site_id: str, sensor_type: str = None) -> str:
        """Generate sensor data topic."""
        if sensor_type:
            return f"{self.base_topic}/sensor/{site_id}/{sensor_type}"
        return f"{self.base_topic}/sensor/{site_id}"
    
    def prediction(self, site_id: str) -> str:
        """Generate prediction topic."""
        return f"{self.base_topic}/prediction/{site_id}"
    
    def alert(self, site_id: str, severity: str = None) -> str:
        """Generate alert topic."""
        if severity:
            return f"{self.base_topic}/alert/{site_id}/{severity}"
        return f"{self.base_topic}/alert/{site_id}"
    
    def status(self, site_id: str) -> str:
        """Generate status topic."""
        return f"{self.base_topic}/status/{site_id}"
    
    def control(self, site_id: str) -> str:
        """Generate control topic."""
        return f"{self.base_topic}/control/{site_id}"
    
    def system(self, message_type: str) -> str:
        """Generate system topic."""
        return f"{self.base_topic}/system/{message_type}"


# Example usage and testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MQTT Client Test')
    parser.add_argument('--broker', default='localhost', help='MQTT broker host')
    parser.add_argument('--port', type=int, default=1883, help='MQTT broker port')
    parser.add_argument('--username', help='MQTT username')
    parser.add_argument('--password', help='MQTT password')
    parser.add_argument('--test-publish', action='store_true', help='Test publishing')
    parser.add_argument('--test-subscribe', action='store_true', help='Test subscription')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    def message_handler(topic: str, message: str):
        print(f"Received on {topic}: {message}")
    
    # Create client
    client = MQTTClient(
        broker=args.broker,
        port=args.port,
        username=args.username,
        password=args.password,
        client_id="test_client"
    )
    
    client.set_message_callback(message_handler)
    
    # Connect
    if client.connect():
        print("Connected successfully")
        
        if args.test_subscribe:
            client.subscribe("rockfall/+/+")
            print("Subscribed to rockfall topics")
            
            # Keep alive for receiving messages
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Stopping...")
        
        if args.test_publish:
            # Test publishing
            test_data = {
                "site_id": "TEST_001",
                "sensor_type": "strain",
                "value": 150.5,
                "timestamp": "2024-01-01T12:00:00Z"
            }
            
            client.publish_sensor_data(test_data)
            print("Test message published")
            time.sleep(2)
        
        client.disconnect()
    else:
        print("Connection failed")
