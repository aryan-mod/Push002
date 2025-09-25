#!/usr/bin/env python3
"""
Edge Agent for Rockfall Prediction System
Runs on Raspberry Pi or Jetson devices for local inference and data collection.
"""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import signal
import atexit

import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import serial
import sqlite3
import requests
from dataclasses import dataclass, asdict

# Local imports
from mqtt_client import MQTTClient

# GPIO imports (handle gracefully if not on Pi)
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("WARNING: RPi.GPIO not available. LED/Buzzer controls disabled.")

# Camera imports
try:
    from picamera2 import Picamera2, Preview
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("WARNING: PiCamera2 not available. Using OpenCV camera.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/rockfall_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """Data structure for sensor readings."""
    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    timestamp: str
    quality: float = 1.0

@dataclass
class PredictionResult:
    """Data structure for prediction results."""
    site_id: str
    probability: float
    risk_level: str
    confidence: float
    timestamp: str
    model_version: str
    uncertainty: float = 0.0

class EdgeAgent:
    """
    Main Edge Agent class for rockfall prediction system.
    Handles sensor data collection, local inference, and communication.
    """
    
    def __init__(self, config_path: str = '/etc/rockfall/config.json'):
        """Initialize the edge agent."""
        self.config = self.load_config(config_path)
        self.running = False
        self.models = {}
        self.sensor_connections = {}
        self.camera = None
        self.mqtt_client = None
        self.local_db = None
        
        # Initialize components
        self.setup_database()
        self.setup_gpio()
        self.setup_camera()
        self.setup_models()
        self.setup_mqtt()
        self.setup_sensors()
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        logger.info("Edge Agent initialized successfully")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        default_config = {
            "site_id": "EDGE_001",
            "sampling_interval": 60,  # seconds
            "prediction_interval": 300,  # seconds
            "models": {
                "fusion_model_path": "/opt/rockfall/models/fusion_model.onnx",
                "lightweight_model_path": "/opt/rockfall/models/lightweight_model.onnx"
            },
            "sensors": {
                "strain": {"port": "/dev/ttyUSB0", "baud": 9600},
                "displacement": {"port": "/dev/ttyUSB1", "baud": 9600},
                "pore_pressure": {"port": "/dev/ttyUSB2", "baud": 9600},
                "tilt": {"port": "/dev/ttyUSB3", "baud": 9600},
                "vibration": {"port": "/dev/ttyUSB4", "baud": 9600}
            },
            "camera": {
                "enabled": True,
                "resolution": [640, 480],
                "capture_interval": 3600  # seconds
            },
            "mqtt": {
                "broker": "localhost",
                "port": 1883,
                "username": None,
                "password": None,
                "topics": {
                    "sensor_data": "rockfall/sensor_data",
                    "predictions": "rockfall/predictions",
                    "alerts": "rockfall/alerts",
                    "status": "rockfall/status"
                }
            },
            "gpio": {
                "led_pin": 18,
                "buzzer_pin": 19,
                "button_pin": 20
            },
            "communication": {
                "backend_url": "http://backend:8000",
                "sync_interval": 3600,  # seconds
                "offline_threshold": 7200  # seconds
            },
            "thresholds": {
                "high_risk": 0.7,
                "critical_risk": 0.9
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            else:
                logger.warning(f"Config file {config_path} not found, using defaults")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
        
        return default_config
    
    def setup_database(self):
        """Setup local SQLite database for offline storage."""
        try:
            db_path = '/var/lib/rockfall/edge_data.db'
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            self.local_db = sqlite3.connect(db_path, check_same_thread=False)
            self.local_db.execute('''
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sensor_id TEXT,
                    sensor_type TEXT,
                    value REAL,
                    unit TEXT,
                    timestamp TEXT,
                    quality REAL,
                    synced INTEGER DEFAULT 0
                )
            ''')
            
            self.local_db.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_id TEXT,
                    probability REAL,
                    risk_level TEXT,
                    confidence REAL,
                    timestamp TEXT,
                    model_version TEXT,
                    uncertainty REAL,
                    synced INTEGER DEFAULT 0
                )
            ''')
            
            self.local_db.commit()
            logger.info("Local database setup completed")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def setup_gpio(self):
        """Setup GPIO pins for LED and buzzer."""
        if not GPIO_AVAILABLE:
            return
        
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup LED
            self.led_pin = self.config['gpio']['led_pin']
            GPIO.setup(self.led_pin, GPIO.OUT)
            GPIO.output(self.led_pin, GPIO.LOW)
            
            # Setup buzzer
            self.buzzer_pin = self.config['gpio']['buzzer_pin']
            GPIO.setup(self.buzzer_pin, GPIO.OUT)
            GPIO.output(self.buzzer_pin, GPIO.LOW)
            
            # Setup button (for manual alerts)
            self.button_pin = self.config['gpio']['button_pin']
            GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(self.button_pin, GPIO.FALLING, 
                                callback=self.button_callback, bouncetime=200)
            
            logger.info("GPIO setup completed")
            
        except Exception as e:
            logger.error(f"GPIO setup failed: {e}")
    
    def setup_camera(self):
        """Setup camera for image capture."""
        if not self.config['camera']['enabled']:
            return
        
        try:
            if PICAMERA_AVAILABLE:
                # Use PiCamera2
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": tuple(self.config['camera']['resolution'])}
                )
                self.camera.configure(config)
                logger.info("PiCamera2 setup completed")
            else:
                # Use OpenCV
                self.camera = cv2.VideoCapture(0)
                if self.camera.isOpened():
                    logger.info("OpenCV camera setup completed")
                else:
                    logger.error("Failed to open camera")
                    self.camera = None
                    
        except Exception as e:
            logger.error(f"Camera setup failed: {e}")
            self.camera = None
    
    def setup_models(self):
        """Load ONNX models for inference."""
        try:
            # Load fusion model if available
            fusion_model_path = self.config['models']['fusion_model_path']
            if os.path.exists(fusion_model_path):
                self.models['fusion'] = ort.InferenceSession(fusion_model_path)
                logger.info("Fusion model loaded successfully")
            
            # Load lightweight model
            lightweight_model_path = self.config['models']['lightweight_model_path']
            if os.path.exists(lightweight_model_path):
                self.models['lightweight'] = ort.InferenceSession(lightweight_model_path)
                logger.info("Lightweight model loaded successfully")
            else:
                logger.error("No models available for inference")
                
        except Exception as e:
            logger.error(f"Model setup failed: {e}")
    
    def setup_mqtt(self):
        """Setup MQTT client for communication."""
        try:
            mqtt_config = self.config['mqtt']
            self.mqtt_client = MQTTClient(
                broker=mqtt_config['broker'],
                port=mqtt_config['port'],
                username=mqtt_config.get('username'),
                password=mqtt_config.get('password'),
                client_id=f"edge_agent_{self.config['site_id']}"
            )
            
            # Connect and subscribe to control topics
            self.mqtt_client.connect()
            self.mqtt_client.subscribe(f"rockfall/control/{self.config['site_id']}")
            self.mqtt_client.set_message_callback(self.mqtt_message_callback)
            
            logger.info("MQTT client setup completed")
            
        except Exception as e:
            logger.error(f"MQTT setup failed: {e}")
            self.mqtt_client = None
    
    def setup_sensors(self):
        """Setup serial connections for sensors."""
        for sensor_type, config in self.config['sensors'].items():
            try:
                if os.path.exists(config['port']):
                    connection = serial.Serial(
                        port=config['port'],
                        baudrate=config['baud'],
                        timeout=1
                    )
                    self.sensor_connections[sensor_type] = connection
                    logger.info(f"Sensor {sensor_type} connected on {config['port']}")
                else:
                    logger.warning(f"Sensor port {config['port']} not found")
                    
            except Exception as e:
                logger.error(f"Failed to setup sensor {sensor_type}: {e}")
    
    def start(self):
        """Start the edge agent main loop."""
        logger.info("Starting Edge Agent...")
        self.running = True
        
        # Start background threads
        sensor_thread = threading.Thread(target=self.sensor_collection_loop)
        prediction_thread = threading.Thread(target=self.prediction_loop)
        sync_thread = threading.Thread(target=self.sync_loop)
        camera_thread = threading.Thread(target=self.camera_capture_loop)
        
        sensor_thread.daemon = True
        prediction_thread.daemon = True
        sync_thread.daemon = True
        camera_thread.daemon = True
        
        sensor_thread.start()
        prediction_thread.start()
        sync_thread.start()
        
        if self.camera:
            camera_thread.start()
        
        # Main status loop
        try:
            while self.running:
                self.publish_status()
                time.sleep(30)  # Publish status every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("Shutting down Edge Agent...")
            self.stop()
    
    def stop(self):
        """Stop the edge agent."""
        self.running = False
        logger.info("Edge Agent stopped")
    
    def sensor_collection_loop(self):
        """Main loop for sensor data collection."""
        while self.running:
            try:
                for sensor_type, connection in self.sensor_connections.items():
                    reading = self.read_sensor(sensor_type, connection)
                    if reading:
                        self.store_sensor_reading(reading)
                        self.publish_sensor_data(reading)
                
                time.sleep(self.config['sampling_interval'])
                
            except Exception as e:
                logger.error(f"Sensor collection error: {e}")
                time.sleep(5)
    
    def read_sensor(self, sensor_type: str, connection: serial.Serial) -> Optional[SensorReading]:
        """Read data from a specific sensor."""
        try:
            # Send request command
            connection.write(b'READ\n')
            time.sleep(0.1)
            
            # Read response
            response = connection.readline().decode().strip()
            if response:
                # Parse sensor response (format: "VALUE:123.45")
                if ':' in response:
                    value_str = response.split(':')[1]
                    value = float(value_str)
                    
                    # Sensor-specific units
                    units = {
                        'strain': 'μɛ',
                        'displacement': 'mm',
                        'pore_pressure': 'kPa',
                        'tilt': 'degrees',
                        'vibration': 'Hz'
                    }
                    
                    return SensorReading(
                        sensor_id=f"{self.config['site_id']}_{sensor_type}",
                        sensor_type=sensor_type,
                        value=value,
                        unit=units.get(sensor_type, 'units'),
                        timestamp=datetime.now().isoformat(),
                        quality=1.0
                    )
                    
        except Exception as e:
            logger.error(f"Failed to read sensor {sensor_type}: {e}")
        
        return None
    
    def store_sensor_reading(self, reading: SensorReading):
        """Store sensor reading in local database."""
        try:
            self.local_db.execute(
                '''INSERT INTO sensor_readings 
                   (sensor_id, sensor_type, value, unit, timestamp, quality)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (reading.sensor_id, reading.sensor_type, reading.value,
                 reading.unit, reading.timestamp, reading.quality)
            )
            self.local_db.commit()
            
        except Exception as e:
            logger.error(f"Failed to store sensor reading: {e}")
    
    def publish_sensor_data(self, reading: SensorReading):
        """Publish sensor data via MQTT."""
        if self.mqtt_client:
            topic = self.config['mqtt']['topics']['sensor_data']
            message = json.dumps(asdict(reading))
            self.mqtt_client.publish(topic, message)
    
    def prediction_loop(self):
        """Main loop for making predictions."""
        while self.running:
            try:
                prediction = self.make_prediction()
                if prediction:
                    self.store_prediction(prediction)
                    self.publish_prediction(prediction)
                    self.handle_alerts(prediction)
                
                time.sleep(self.config['prediction_interval'])
                
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
                time.sleep(10)
    
    def make_prediction(self) -> Optional[PredictionResult]:
        """Make risk prediction using local models."""
        try:
            # Get recent sensor data
            sensor_features = self.get_recent_sensor_features()
            
            # Use lightweight model if available
            if 'lightweight' in self.models:
                model = self.models['lightweight']
                
                # Prepare input features
                input_features = np.array([sensor_features], dtype=np.float32)
                
                # Run inference
                input_name = model.get_inputs()[0].name
                outputs = model.run(None, {input_name: input_features})
                
                # Process outputs
                probabilities = outputs[0][0]  # Assuming softmax output
                predicted_class = np.argmax(probabilities)
                confidence = float(probabilities[predicted_class])
                
                # Map to risk levels
                risk_levels = ['low', 'medium', 'high', 'critical']
                risk_level = risk_levels[predicted_class]
                
                return PredictionResult(
                    site_id=self.config['site_id'],
                    probability=float(probabilities[predicted_class]),
                    risk_level=risk_level,
                    confidence=confidence,
                    timestamp=datetime.now().isoformat(),
                    model_version='lightweight_v1.0',
                    uncertainty=0.1  # Default uncertainty
                )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
        
        return None
    
    def get_recent_sensor_features(self) -> List[float]:
        """Get recent sensor data as feature vector."""
        try:
            # Get last 48 hours of data for each sensor type
            cursor = self.local_db.cursor()
            
            features = []
            sensor_types = ['strain', 'displacement', 'pore_pressure', 'tilt', 'vibration']
            
            for sensor_type in sensor_types:
                cursor.execute('''
                    SELECT value FROM sensor_readings 
                    WHERE sensor_type = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 48
                ''', (sensor_type,))
                
                readings = cursor.fetchall()
                if readings:
                    values = [r[0] for r in readings]
                    
                    # Calculate statistical features
                    features.extend([
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values),
                        np.median(values)
                    ])
                else:
                    # Default values if no data
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            
            # Add some basic site features
            features.extend([
                1000.0,  # elevation
                30.0,    # slope_angle
                180.0,   # aspect_angle
                0.0,     # rainfall
                15.0     # temperature
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return [0.0] * 30  # Default feature vector
    
    def store_prediction(self, prediction: PredictionResult):
        """Store prediction in local database."""
        try:
            self.local_db.execute(
                '''INSERT INTO predictions 
                   (site_id, probability, risk_level, confidence, timestamp, model_version, uncertainty)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (prediction.site_id, prediction.probability, prediction.risk_level,
                 prediction.confidence, prediction.timestamp, prediction.model_version,
                 prediction.uncertainty)
            )
            self.local_db.commit()
            
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
    
    def publish_prediction(self, prediction: PredictionResult):
        """Publish prediction via MQTT."""
        if self.mqtt_client:
            topic = self.config['mqtt']['topics']['predictions']
            message = json.dumps(asdict(prediction))
            self.mqtt_client.publish(topic, message)
    
    def handle_alerts(self, prediction: PredictionResult):
        """Handle alerts based on prediction risk level."""
        try:
            risk_probability = prediction.probability
            
            if risk_probability >= self.config['thresholds']['critical_risk']:
                self.trigger_critical_alert(prediction)
            elif risk_probability >= self.config['thresholds']['high_risk']:
                self.trigger_high_risk_alert(prediction)
            
        except Exception as e:
            logger.error(f"Alert handling failed: {e}")
    
    def trigger_critical_alert(self, prediction: PredictionResult):
        """Trigger critical risk alert."""
        logger.critical(f"CRITICAL RISK DETECTED: {prediction.probability:.2%}")
        
        # Visual/audio alerts
        self.activate_buzzer(duration=10, pattern='continuous')
        self.activate_led(duration=60, pattern='fast_blink')
        
        # Send alert via MQTT
        alert_message = {
            'type': 'critical_alert',
            'site_id': self.config['site_id'],
            'prediction': asdict(prediction),
            'timestamp': datetime.now().isoformat(),
            'message': 'Critical rockfall risk detected - immediate action required'
        }
        
        if self.mqtt_client:
            topic = self.config['mqtt']['topics']['alerts']
            self.mqtt_client.publish(topic, json.dumps(alert_message))
        
        # Try to send SMS if GSM modem available
        self.send_emergency_sms(alert_message)
    
    def trigger_high_risk_alert(self, prediction: PredictionResult):
        """Trigger high risk alert."""
        logger.warning(f"HIGH RISK DETECTED: {prediction.probability:.2%}")
        
        # Visual alert
        self.activate_led(duration=30, pattern='slow_blink')
        
        # Send alert via MQTT
        alert_message = {
            'type': 'high_risk_alert',
            'site_id': self.config['site_id'],
            'prediction': asdict(prediction),
            'timestamp': datetime.now().isoformat(),
            'message': 'Elevated rockfall risk detected - monitor closely'
        }
        
        if self.mqtt_client:
            topic = self.config['mqtt']['topics']['alerts']
            self.mqtt_client.publish(topic, json.dumps(alert_message))
    
    def activate_buzzer(self, duration: int = 5, pattern: str = 'continuous'):
        """Activate buzzer with specified pattern."""
        if not GPIO_AVAILABLE:
            return
        
        def buzzer_thread():
            try:
                if pattern == 'continuous':
                    GPIO.output(self.buzzer_pin, GPIO.HIGH)
                    time.sleep(duration)
                    GPIO.output(self.buzzer_pin, GPIO.LOW)
                elif pattern == 'beeps':
                    for _ in range(duration):
                        GPIO.output(self.buzzer_pin, GPIO.HIGH)
                        time.sleep(0.5)
                        GPIO.output(self.buzzer_pin, GPIO.LOW)
                        time.sleep(0.5)
            except Exception as e:
                logger.error(f"Buzzer activation failed: {e}")
        
        threading.Thread(target=buzzer_thread, daemon=True).start()
    
    def activate_led(self, duration: int = 10, pattern: str = 'solid'):
        """Activate LED with specified pattern."""
        if not GPIO_AVAILABLE:
            return
        
        def led_thread():
            try:
                if pattern == 'solid':
                    GPIO.output(self.led_pin, GPIO.HIGH)
                    time.sleep(duration)
                    GPIO.output(self.led_pin, GPIO.LOW)
                elif pattern == 'fast_blink':
                    end_time = time.time() + duration
                    while time.time() < end_time:
                        GPIO.output(self.led_pin, GPIO.HIGH)
                        time.sleep(0.2)
                        GPIO.output(self.led_pin, GPIO.LOW)
                        time.sleep(0.2)
                elif pattern == 'slow_blink':
                    end_time = time.time() + duration
                    while time.time() < end_time:
                        GPIO.output(self.led_pin, GPIO.HIGH)
                        time.sleep(1.0)
                        GPIO.output(self.led_pin, GPIO.LOW)
                        time.sleep(1.0)
            except Exception as e:
                logger.error(f"LED activation failed: {e}")
        
        threading.Thread(target=led_thread, daemon=True).start()
    
    def send_emergency_sms(self, alert_message: Dict[str, Any]):
        """Send emergency SMS if GSM modem is available."""
        try:
            # Check for GSM modem (e.g., on /dev/ttyUSB5)
            gsm_port = '/dev/ttyUSB5'
            if os.path.exists(gsm_port):
                gsm_connection = serial.Serial(gsm_port, 9600, timeout=1)
                
                # Emergency contact number (from config)
                emergency_number = self.config.get('emergency_contact', '+1234567890')
                
                # SMS content
                sms_text = (f"ROCKFALL ALERT - Site {self.config['site_id']}: "
                           f"{alert_message['message']} at {alert_message['timestamp']}")
                
                # Send SMS using AT commands
                gsm_connection.write(b'AT+CMGF=1\r')  # Set SMS text mode
                time.sleep(1)
                gsm_connection.write(f'AT+CMGS="{emergency_number}"\r'.encode())
                time.sleep(1)
                gsm_connection.write(sms_text.encode() + b'\x1A')  # Send SMS
                
                logger.info(f"Emergency SMS sent to {emergency_number}")
                gsm_connection.close()
                
        except Exception as e:
            logger.error(f"Emergency SMS failed: {e}")
    
    def camera_capture_loop(self):
        """Loop for periodic image capture."""
        while self.running and self.camera:
            try:
                image = self.capture_image()
                if image:
                    # Save image locally
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = f"/var/lib/rockfall/images/{timestamp}.jpg"
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    
                    if isinstance(image, np.ndarray):
                        cv2.imwrite(image_path, image)
                    else:
                        image.save(image_path)
                    
                    logger.info(f"Image captured: {image_path}")
                
                time.sleep(self.config['camera']['capture_interval'])
                
            except Exception as e:
                logger.error(f"Camera capture error: {e}")
                time.sleep(60)
    
    def capture_image(self):
        """Capture image from camera."""
        try:
            if PICAMERA_AVAILABLE and hasattr(self.camera, 'capture_array'):
                # PiCamera2
                image = self.camera.capture_array()
                return image
            elif self.camera and hasattr(self.camera, 'read'):
                # OpenCV
                ret, frame = self.camera.read()
                if ret:
                    return frame
        except Exception as e:
            logger.error(f"Image capture failed: {e}")
        
        return None
    
    def sync_loop(self):
        """Loop for syncing data with backend."""
        while self.running:
            try:
                self.sync_with_backend()
                time.sleep(self.config['communication']['sync_interval'])
                
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def sync_with_backend(self):
        """Sync local data with backend server."""
        try:
            backend_url = self.config['communication']['backend_url']
            
            # Sync sensor readings
            cursor = self.local_db.cursor()
            cursor.execute('SELECT * FROM sensor_readings WHERE synced = 0 LIMIT 100')
            readings = cursor.fetchall()
            
            for reading in readings:
                reading_data = {
                    'type': 'sensor_reading',
                    'data': {
                        'sensorId': reading[1],
                        'value': reading[3],
                        'unit': reading[4],
                        'quality': reading[6]
                    }
                }
                
                response = requests.post(
                    f"{backend_url}/api/v1/ingest",
                    json=reading_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    # Mark as synced
                    cursor.execute('UPDATE sensor_readings SET synced = 1 WHERE id = ?', (reading[0],))
            
            self.local_db.commit()
            logger.info(f"Synced {len(readings)} sensor readings")
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Backend sync failed - operating in offline mode: {e}")
        except Exception as e:
            logger.error(f"Sync failed: {e}")
    
    def publish_status(self):
        """Publish agent status."""
        try:
            status = {
                'site_id': self.config['site_id'],
                'timestamp': datetime.now().isoformat(),
                'status': 'online',
                'sensors_connected': len(self.sensor_connections),
                'models_loaded': len(self.models),
                'camera_available': self.camera is not None,
                'mqtt_connected': self.mqtt_client.is_connected() if self.mqtt_client else False,
                'uptime': time.time() - getattr(self, 'start_time', time.time())
            }
            
            if self.mqtt_client:
                topic = self.config['mqtt']['topics']['status']
                self.mqtt_client.publish(topic, json.dumps(status))
                
        except Exception as e:
            logger.error(f"Status publish failed: {e}")
    
    def mqtt_message_callback(self, topic: str, message: str):
        """Handle incoming MQTT messages."""
        try:
            data = json.loads(message)
            command = data.get('command')
            
            if command == 'trigger_alert':
                self.trigger_manual_alert()
            elif command == 'capture_image':
                self.capture_and_send_image()
            elif command == 'restart':
                self.restart_agent()
            elif command == 'update_config':
                self.update_config(data.get('config', {}))
                
        except Exception as e:
            logger.error(f"MQTT message handling failed: {e}")
    
    def button_callback(self, channel):
        """Handle manual button press."""
        logger.info("Manual alert button pressed")
        self.trigger_manual_alert()
    
    def trigger_manual_alert(self):
        """Trigger manual alert."""
        alert_message = {
            'type': 'manual_alert',
            'site_id': self.config['site_id'],
            'timestamp': datetime.now().isoformat(),
            'message': 'Manual alert triggered by on-site personnel'
        }
        
        if self.mqtt_client:
            topic = self.config['mqtt']['topics']['alerts']
            self.mqtt_client.publish(topic, json.dumps(alert_message))
        
        self.activate_led(duration=5, pattern='fast_blink')
    
    def signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}")
        self.stop()
    
    def cleanup(self):
        """Cleanup resources on shutdown."""
        try:
            logger.info("Cleaning up resources...")
            
            # Close sensor connections
            for connection in self.sensor_connections.values():
                if connection.is_open:
                    connection.close()
            
            # Cleanup GPIO
            if GPIO_AVAILABLE:
                GPIO.cleanup()
            
            # Close camera
            if self.camera:
                if hasattr(self.camera, 'close'):
                    self.camera.close()
                elif hasattr(self.camera, 'release'):
                    self.camera.release()
            
            # Close database
            if self.local_db:
                self.local_db.close()
            
            # Close MQTT
            if self.mqtt_client:
                self.mqtt_client.disconnect()
                
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def main():
    """Main entry point."""
    try:
        # Check if running as root (required for GPIO)
        if GPIO_AVAILABLE and os.geteuid() != 0:
            print("WARNING: Not running as root. GPIO features may not work.")
        
        # Initialize and start agent
        agent = EdgeAgent()
        agent.start_time = time.time()
        agent.start()
        
    except Exception as e:
        logger.error(f"Agent startup failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
