import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any, Optional
import os
import json
import random
from datetime import datetime, timedelta
import cv2
from scipy.ndimage import gaussian_filter
import rasterio
from rasterio.transform import from_bounds
import logging

from .data_processor import RockfallDataset, collate_fn
from ml.models.cnn_model import ImageProcessor
from ml.models.lstm_model import SensorDataProcessor

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """
    Generate synthetic data for training and testing rockfall prediction models.
    Creates realistic drone images, DEM data, sensor readings, and tabular features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_size = config.get('image_size', (224, 224))
        self.sequence_length = config.get('sequence_length', 48)
        
        # Risk level probabilities for data generation
        self.risk_probabilities = {
            'low': 0.5,
            'medium': 0.3,
            'high': 0.15,
            'critical': 0.05
        }
        
        # Sensor types and their characteristics
        self.sensor_configs = {
            'strain': {'range': (0, 2000), 'noise': 50, 'unit': 'μɛ'},
            'displacement': {'range': (0, 20), 'noise': 0.5, 'unit': 'mm'},
            'pore_pressure': {'range': (0, 500), 'noise': 10, 'unit': 'kPa'},
            'tilt': {'range': (0, 10), 'noise': 0.1, 'unit': 'degrees'},
            'vibration': {'range': (0, 100), 'noise': 5, 'unit': 'Hz'}
        }
        
        # Initialize random seed for reproducibility
        self.set_random_seed(config.get('random_seed', 42))
    
    def set_random_seed(self, seed: int):
        """Set random seed for reproducible generation."""
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_training_data(
        self,
        num_samples: int = 1000,
        val_split: float = 0.2,
        test_split: float = 0.1,
        output_dir: str = 'data/synthetic'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate complete synthetic dataset for training.
        
        Args:
            num_samples: Total number of samples to generate
            val_split: Validation split ratio
            test_split: Test split ratio
            output_dir: Directory to save synthetic data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/dem", exist_ok=True)
        
        # Generate samples
        samples = []
        sensor_data_dict = {}
        
        for i in range(num_samples):
            if i % 100 == 0:
                logger.info(f"Generated {i}/{num_samples} samples")
            
            sample = self.generate_single_sample(i, output_dir)
            samples.append(sample['metadata'])
            
            if sample['site_id'] not in sensor_data_dict:
                sensor_data_dict[sample['site_id']] = sample['sensor_data']
        
        # Create DataFrame
        df = pd.DataFrame(samples)
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        train_df, temp_df = train_test_split(
            df, test_size=(val_split + test_split), 
            stratify=df['risk_level'], random_state=42
        )
        
        val_size = val_split / (val_split + test_split)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_size),
            stratify=temp_df['risk_level'], random_state=42
        )
        
        # Save datasets
        train_df.to_csv(f"{output_dir}/train.csv", index=False)
        val_df.to_csv(f"{output_dir}/val.csv", index=False)
        test_df.to_csv(f"{output_dir}/test.csv", index=False)
        
        # Save sensor data
        with open(f"{output_dir}/sensor_data.json", 'w') as f:
            json.dump(sensor_data_dict, f, indent=2)
        
        logger.info(f"Synthetic data generation completed!")
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def generate_single_sample(self, sample_id: int, output_dir: str) -> Dict[str, Any]:
        """Generate a single complete sample with all modalities."""
        
        # Generate site characteristics
        site_id = f"SYNTH_{sample_id:04d}"
        risk_level = self.sample_risk_level()
        
        # Generate location
        latitude = np.random.uniform(28.0, 35.0)  # Himalayan region
        longitude = np.random.uniform(77.0, 85.0)
        elevation = np.random.uniform(1000, 5000)
        
        # Generate terrain characteristics based on risk level
        slope_angle, aspect_angle, geological_features = self.generate_terrain_features(risk_level)
        
        # Generate weather conditions
        weather_features = self.generate_weather_features(risk_level)
        
        # Generate DEM data
        dem_data = self.generate_dem_data(slope_angle, aspect_angle, elevation)
        dem_filename = f"{site_id}_dem.tif"
        dem_path = f"{output_dir}/dem/{dem_filename}"
        self.save_dem_data(dem_data, dem_path)
        
        # Generate drone image
        image = self.generate_drone_image(
            terrain_type=geological_features['rock_type'],
            slope_angle=slope_angle,
            risk_level=risk_level,
            weather=weather_features
        )
        image_filename = f"{site_id}_drone.jpg"
        image_path = f"{output_dir}/images/{image_filename}"
        image.save(image_path, quality=90)
        
        # Generate sensor time series
        sensor_data = self.generate_sensor_time_series(
            risk_level=risk_level,
            weather_influence=weather_features,
            geological_stability=geological_features['stability_index']
        )
        
        # Create metadata entry
        metadata = {
            'site_id': site_id,
            'image_filename': image_filename,
            'dem_filename': dem_filename,
            'latitude': latitude,
            'longitude': longitude,
            'elevation': elevation,
            'slope_angle': slope_angle,
            'aspect_angle': aspect_angle,
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat(),
            
            # Weather features
            'rainfall': weather_features['rainfall'],
            'temperature': weather_features['temperature'],
            'humidity': weather_features['humidity'],
            'wind_speed': weather_features['wind_speed'],
            
            # Geological features
            'rock_type_encoded': geological_features['rock_type_encoded'],
            'vegetation_cover': geological_features['vegetation_cover'],
            'geological_stability': geological_features['stability_index']
        }
        
        return {
            'metadata': metadata,
            'sensor_data': sensor_data,
            'site_id': site_id
        }
    
    def sample_risk_level(self) -> str:
        """Sample risk level based on defined probabilities."""
        return np.random.choice(
            list(self.risk_probabilities.keys()),
            p=list(self.risk_probabilities.values())
        )
    
    def generate_terrain_features(self, risk_level: str) -> Tuple[float, float, Dict[str, Any]]:
        """Generate terrain characteristics based on risk level."""
        
        # Risk-dependent slope angle ranges
        slope_ranges = {
            'low': (5, 25),
            'medium': (20, 40),
            'high': (35, 60),
            'critical': (50, 80)
        }
        
        slope_angle = np.random.uniform(*slope_ranges[risk_level])
        aspect_angle = np.random.uniform(0, 360)
        
        # Geological features
        rock_types = ['granite', 'limestone', 'sandstone', 'shale', 'schist']
        rock_type = np.random.choice(rock_types)
        
        # Rock stability (lower values = higher risk)
        stability_ranges = {
            'low': (0.8, 1.0),
            'medium': (0.6, 0.8),
            'high': (0.4, 0.6),
            'critical': (0.2, 0.4)
        }
        stability_index = np.random.uniform(*stability_ranges[risk_level])
        
        # Vegetation cover (higher cover = lower risk)
        vegetation_ranges = {
            'low': (0.6, 0.9),
            'medium': (0.4, 0.7),
            'high': (0.2, 0.5),
            'critical': (0.0, 0.3)
        }
        vegetation_cover = np.random.uniform(*vegetation_ranges[risk_level])
        
        geological_features = {
            'rock_type': rock_type,
            'rock_type_encoded': rock_types.index(rock_type),
            'stability_index': stability_index,
            'vegetation_cover': vegetation_cover
        }
        
        return slope_angle, aspect_angle, geological_features
    
    def generate_weather_features(self, risk_level: str) -> Dict[str, float]:
        """Generate weather conditions that influence risk."""
        
        # Base weather conditions
        temperature = np.random.uniform(5, 25)  # Celsius
        humidity = np.random.uniform(30, 80)    # Percentage
        wind_speed = np.random.uniform(0, 20)   # km/h
        
        # Risk-influenced rainfall
        rainfall_ranges = {
            'low': (0, 10),      # mm in 24h
            'medium': (5, 25),
            'high': (20, 60),
            'critical': (50, 150)
        }
        rainfall = np.random.uniform(*rainfall_ranges[risk_level])
        
        return {
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'rainfall': rainfall
        }
    
    def generate_dem_data(
        self, 
        slope_angle: float, 
        aspect_angle: float, 
        elevation: float
    ) -> np.ndarray:
        """Generate synthetic DEM data."""
        
        # Create base elevation grid
        size = 200  # 200x200 grid
        dem = np.ones((size, size)) * elevation
        
        # Add slope trend
        y_gradient = np.tan(np.radians(slope_angle)) * np.linspace(-100, 100, size)
        dem = dem + y_gradient[:, np.newaxis]
        
        # Add random terrain variation
        noise = np.random.normal(0, 10, (size, size))
        dem = dem + gaussian_filter(noise, sigma=5)
        
        # Add some realistic terrain features
        # Random hills and valleys
        for _ in range(np.random.randint(3, 8)):
            center_x = np.random.randint(20, size-20)
            center_y = np.random.randint(20, size-20)
            radius = np.random.randint(10, 30)
            amplitude = np.random.uniform(-50, 100)
            
            y, x = np.ogrid[:size, :size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            dem[mask] += amplitude * np.exp(-(((x - center_x)**2 + (y - center_y)**2) / (2 * (radius/3)**2)))[mask]
        
        # Add ridge lines for high-risk areas
        if slope_angle > 40:
            ridge_y = np.random.randint(10, size-10)
            ridge_amplitude = np.random.uniform(20, 60)
            ridge_width = np.random.randint(5, 15)
            
            for x in range(size):
                for y in range(max(0, ridge_y - ridge_width), min(size, ridge_y + ridge_width)):
                    distance = abs(y - ridge_y)
                    if distance <= ridge_width:
                        dem[y, x] += ridge_amplitude * (1 - distance / ridge_width)
        
        return dem
    
    def save_dem_data(self, dem_data: np.ndarray, output_path: str):
        """Save DEM data as GeoTIFF file."""
        try:
            # Define spatial reference (dummy coordinates)
            bounds = (77.0, 28.0, 77.01, 28.01)  # Example bounds
            transform = from_bounds(*bounds, dem_data.shape[1], dem_data.shape[0])
            
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=dem_data.shape[0],
                width=dem_data.shape[1],
                count=1,
                dtype=dem_data.dtype,
                crs='+proj=latlong',
                transform=transform,
            ) as dst:
                dst.write(dem_data, 1)
                
        except Exception as e:
            logger.warning(f"Failed to save DEM data: {e}")
            # Save as numpy array instead
            np.save(output_path.replace('.tif', '.npy'), dem_data)
    
    def generate_drone_image(
        self,
        terrain_type: str,
        slope_angle: float,
        risk_level: str,
        weather: Dict[str, float]
    ) -> Image.Image:
        """Generate synthetic drone image of terrain."""
        
        width, height = self.image_size
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        
        # Base terrain colors
        terrain_colors = {
            'granite': [(120, 120, 120), (140, 140, 140), (100, 100, 100)],
            'limestone': [(200, 190, 170), (220, 210, 190), (180, 170, 150)],
            'sandstone': [(210, 180, 140), (230, 200, 160), (190, 160, 120)],
            'shale': [(80, 70, 60), (100, 90, 80), (60, 50, 40)],
            'schist': [(90, 85, 80), (110, 105, 100), (70, 65, 60)]
        }
        
        base_colors = terrain_colors.get(terrain_type, terrain_colors['granite'])
        
        # Create base terrain texture
        for y in range(height):
            for x in range(width):
                # Add some randomness to color
                color_idx = np.random.randint(0, len(base_colors))
                base_color = base_colors[color_idx]
                
                # Add noise
                noise = np.random.randint(-20, 20)
                color = tuple(max(0, min(255, c + noise)) for c in base_color)
                
                image.putpixel((x, y), color)
        
        # Apply Gaussian blur for natural texture
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Add risk-related features
        if risk_level in ['high', 'critical']:
            # Add cracks and unstable areas
            self.add_cracks_to_image(image, draw, risk_level)
            
        if slope_angle > 30:
            # Add steep terrain features
            self.add_steep_terrain_features(image, draw, slope_angle)
        
        # Add vegetation based on cover
        vegetation_cover = weather.get('vegetation_cover', 0.5)
        if vegetation_cover > 0.3:
            self.add_vegetation_to_image(image, draw, vegetation_cover)
        
        # Weather effects
        if weather['rainfall'] > 20:
            # Add water effects
            self.add_water_effects(image, draw)
        
        # Convert to numpy for additional processing
        img_array = np.array(image)
        
        # Add shadows and lighting based on slope
        img_array = self.add_lighting_effects(img_array, slope_angle)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def add_cracks_to_image(self, image: Image.Image, draw: ImageDraw.Draw, risk_level: str):
        """Add crack patterns to indicate instability."""
        width, height = image.size
        
        num_cracks = 3 if risk_level == 'high' else 6
        
        for _ in range(num_cracks):
            # Random crack path
            start_x = np.random.randint(0, width)
            start_y = np.random.randint(0, height)
            
            crack_length = np.random.randint(20, 80)
            angle = np.random.uniform(0, 2 * np.pi)
            
            points = []
            x, y = start_x, start_y
            
            for step in range(crack_length):
                points.append((int(x), int(y)))
                
                # Add some randomness to crack path
                angle += np.random.normal(0, 0.2)
                x += np.cos(angle) * 1.5
                y += np.sin(angle) * 1.5
                
                if x < 0 or x >= width or y < 0 or y >= height:
                    break
            
            # Draw crack
            if len(points) > 1:
                draw.line(points, fill=(40, 30, 20), width=2)
    
    def add_steep_terrain_features(self, image: Image.Image, draw: ImageDraw.Draw, slope_angle: float):
        """Add features typical of steep terrain."""
        width, height = image.size
        
        # Add rock face lines
        intensity = min(slope_angle / 90.0, 1.0)
        num_lines = int(5 * intensity)
        
        for _ in range(num_lines):
            start_y = np.random.randint(0, height)
            end_y = start_y + np.random.randint(-30, 30)
            
            x = np.random.randint(0, width)
            
            draw.line([(x, start_y), (x, min(max(end_y, 0), height-1))], 
                     fill=(60, 50, 40), width=1)
    
    def add_vegetation_to_image(self, image: Image.Image, draw: ImageDraw.Draw, vegetation_cover: float):
        """Add vegetation patches to the image."""
        width, height = image.size
        
        num_patches = int(vegetation_cover * 20)
        
        for _ in range(num_patches):
            patch_size = np.random.randint(5, 15)
            center_x = np.random.randint(patch_size, width - patch_size)
            center_y = np.random.randint(patch_size, height - patch_size)
            
            # Green vegetation color with variation
            green_base = np.random.randint(60, 120)
            vegetation_color = (
                np.random.randint(20, 60),
                green_base,
                np.random.randint(20, 50)
            )
            
            # Draw irregular vegetation patch
            points = []
            for angle in np.linspace(0, 2*np.pi, 8):
                radius = patch_size + np.random.randint(-3, 3)
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                points.append((x, y))
            
            draw.polygon(points, fill=vegetation_color)
    
    def add_water_effects(self, image: Image.Image, draw: ImageDraw.Draw):
        """Add water effects for wet conditions."""
        width, height = image.size
        
        # Add small water puddles
        num_puddles = np.random.randint(2, 6)
        
        for _ in range(num_puddles):
            puddle_size = np.random.randint(3, 10)
            center_x = np.random.randint(puddle_size, width - puddle_size)
            center_y = np.random.randint(puddle_size, height - puddle_size)
            
            # Dark blue-grey water color
            water_color = (50, 70, 90)
            
            draw.ellipse([
                center_x - puddle_size, center_y - puddle_size,
                center_x + puddle_size, center_y + puddle_size
            ], fill=water_color)
    
    def add_lighting_effects(self, img_array: np.ndarray, slope_angle: float) -> np.ndarray:
        """Add realistic lighting and shadow effects."""
        
        # Simulate sun angle effect
        light_angle = 45  # degrees
        shadow_intensity = min(slope_angle / 90.0, 0.3)
        
        # Create gradient shadow effect
        height, width = img_array.shape[:2]
        
        # Vertical shadow gradient
        shadow_gradient = np.linspace(1.0, 1.0 - shadow_intensity, height)
        shadow_gradient = shadow_gradient[:, np.newaxis, np.newaxis]
        
        # Apply shadow
        img_array = img_array * shadow_gradient
        
        # Add some random lighting variation
        light_variation = np.random.normal(1.0, 0.05, img_array.shape)
        img_array = img_array * light_variation
        
        return np.clip(img_array, 0, 255)
    
    def generate_sensor_time_series(
        self,
        risk_level: str,
        weather_influence: Dict[str, float],
        geological_stability: float
    ) -> Dict[str, List[float]]:
        """Generate synthetic sensor time series data."""
        
        sensor_data = {}
        
        for sensor_type, config in self.sensor_configs.items():
            # Base value based on risk level
            risk_multipliers = {
                'low': 0.2,
                'medium': 0.5,
                'high': 0.8,
                'critical': 1.0
            }
            
            base_multiplier = risk_multipliers[risk_level]
            base_value = config['range'][1] * base_multiplier * (1 - geological_stability)
            
            # Generate time series
            time_series = []
            current_value = base_value
            
            for t in range(self.sequence_length):
                # Add trend based on risk level
                if risk_level in ['high', 'critical']:
                    trend = np.random.normal(0.05, 0.02)  # Slight upward trend
                else:
                    trend = np.random.normal(0, 0.01)     # Stable
                
                # Weather influence
                weather_effect = 0
                if sensor_type == 'pore_pressure':
                    weather_effect = weather_influence['rainfall'] * 0.1
                elif sensor_type == 'displacement':
                    weather_effect = weather_influence['rainfall'] * 0.05
                
                # Random noise
                noise = np.random.normal(0, config['noise'] * 0.1)
                
                # Update value
                current_value += trend * current_value + weather_effect + noise
                
                # Clamp to realistic range
                current_value = max(config['range'][0], 
                                  min(config['range'][1], current_value))
                
                time_series.append(float(current_value))
            
            sensor_data[sensor_type] = time_series
        
        return sensor_data
    
    def create_data_loaders(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch DataLoaders from synthetic data."""
        
        try:
            # Load sensor data
            sensor_data = {}
            sensor_data_path = 'data/synthetic/sensor_data.json'
            if os.path.exists(sensor_data_path):
                with open(sensor_data_path, 'r') as f:
                    sensor_data_raw = json.load(f)
                    # Convert to numpy arrays
                    for site_id, data in sensor_data_raw.items():
                        sensor_data[site_id] = {
                            sensor_type: np.array(values)
                            for sensor_type, values in data.items()
                        }
            
            # Initialize processors
            image_processor = ImageProcessor()
            sensor_processor = SensorDataProcessor()
            
            # Create datasets
            train_dataset = RockfallDataset(
                train_df,
                image_dir='data/synthetic/images',
                sensor_data=sensor_data,
                image_transform=image_processor.train_transform,
                sensor_processor=sensor_processor
            )
            
            val_dataset = RockfallDataset(
                val_df,
                image_dir='data/synthetic/images',
                sensor_data=sensor_data,
                image_transform=image_processor.val_transform,
                sensor_processor=sensor_processor
            )
            
            test_dataset = RockfallDataset(
                test_df,
                image_dir='data/synthetic/images',
                sensor_data=sensor_data,
                image_transform=image_processor.val_transform,
                sensor_processor=sensor_processor
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=collate_fn
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=collate_fn
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=collate_fn
            )
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            logger.error(f"Failed to create data loaders: {e}")
            raise
    
    def generate_quick_demo_data(self, num_samples: int = 50) -> Dict[str, Any]:
        """Generate a small dataset for quick demonstration."""
        logger.info(f"Generating {num_samples} demo samples...")
        
        demo_data = {
            'samples': [],
            'sensor_data': {},
            'images': []
        }
        
        for i in range(num_samples):
            sample = self.generate_single_sample(i, 'data/demo')
            demo_data['samples'].append(sample['metadata'])
            demo_data['sensor_data'][sample['site_id']] = sample['sensor_data']
        
        return demo_data


if __name__ == '__main__':
    # Example usage
    config = {
        'image_size': (224, 224),
        'sequence_length': 48,
        'random_seed': 42
    }
    
    generator = SyntheticDataGenerator(config)
    train_df, val_df, test_df = generator.generate_training_data(
        num_samples=500,
        output_dir='data/synthetic'
    )
    
    print("Synthetic data generation completed!")
    print(f"Generated {len(train_df) + len(val_df) + len(test_df)} total samples")
