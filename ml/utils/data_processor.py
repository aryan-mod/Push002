import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RockfallDataset(Dataset):
    """
    PyTorch Dataset for multi-modal rockfall prediction data.
    Handles images, sensor time series, DEM features, and tabular data.
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        image_dir: Optional[str] = None,
        dem_dir: Optional[str] = None,
        sensor_data: Optional[Dict[str, np.ndarray]] = None,
        image_transform = None,
        sensor_processor = None,
        sequence_length: int = 48
    ):
        self.data_df = data_df.reset_index(drop=True)
        self.image_dir = Path(image_dir) if image_dir else None
        self.dem_dir = Path(dem_dir) if dem_dir else None
        self.sensor_data = sensor_data or {}
        self.image_transform = image_transform
        self.sensor_processor = sensor_processor
        self.sequence_length = sequence_length
        
        # Risk level mapping
        self.risk_mapping = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data_df.iloc[idx]
        sample = {}
        
        # Load image if available
        if self.image_dir and 'image_filename' in row and pd.notna(row['image_filename']):
            try:
                image_path = self.image_dir / row['image_filename']
                image = Image.open(image_path).convert('RGB')
                
                if self.image_transform:
                    image = self.image_transform(image)
                else:
                    # Default transform
                    import torchvision.transforms as transforms
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
                    image = transform(image)
                
                sample['images'] = image
            except Exception as e:
                logger.warning(f"Failed to load image {row.get('image_filename', 'unknown')}: {e}")
                # Create dummy image
                sample['images'] = torch.zeros(3, 224, 224)
        
        # Load sensor data if available
        site_id = row.get('site_id')
        if site_id and site_id in self.sensor_data:
            sensor_sequences, lengths = self.sensor_processor.create_sequences(
                self.sensor_data[site_id]
            )
            
            if len(sensor_sequences) > 0:
                # Use the most recent sequence
                sample['sensor_data'] = sensor_sequences[-1]
                sample['sensor_lengths'] = lengths[-1]
        
        # Extract tabular features
        tabular_features = []
        
        # DEM features
        dem_features = [
            row.get('elevation', 0.0),
            row.get('slope_angle', 0.0),
            row.get('aspect_angle', 0.0),
        ]
        tabular_features.extend(dem_features)
        
        # Weather features
        weather_features = [
            row.get('rainfall', 0.0),
            row.get('temperature', 15.0),
            row.get('humidity', 50.0),
            row.get('wind_speed', 0.0),
        ]
        tabular_features.extend(weather_features)
        
        # Site characteristics
        site_features = [
            row.get('rock_type_encoded', 0.0),
            row.get('vegetation_cover', 0.0),
            row.get('geological_stability', 0.0),
        ]
        tabular_features.extend(site_features)
        
        sample['tabular_features'] = torch.FloatTensor(tabular_features)
        
        # Target (risk level)
        risk_level = row.get('risk_level', 'low')
        target = self.risk_mapping.get(risk_level, 0)
        sample['targets'] = torch.LongTensor([target])[0]
        
        # Additional metadata
        sample['site_id'] = site_id
        sample['timestamp'] = row.get('timestamp', '')
        
        return sample


class DEMProcessor:
    """Process Digital Elevation Model (DEM) files to extract slope, aspect, and other features."""
    
    def __init__(self):
        self.cache = {}
    
    def process_dem_file(self, dem_path: str) -> Dict[str, np.ndarray]:
        """
        Process DEM file to extract topographic features.
        
        Args:
            dem_path: Path to DEM file (GeoTIFF)
            
        Returns:
            Dictionary containing elevation, slope, aspect arrays
        """
        if dem_path in self.cache:
            return self.cache[dem_path]
        
        try:
            with rasterio.open(dem_path) as src:
                elevation = src.read(1)
                
                # Calculate slope using gradient
                dy, dx = np.gradient(elevation)
                slope = np.arctan(np.sqrt(dx*dx + dy*dy)) * 180.0 / np.pi
                
                # Calculate aspect
                aspect = np.arctan2(-dx, dy) * 180.0 / np.pi
                aspect = np.where(aspect < 0, aspect + 360, aspect)
                
                # Calculate curvature (second derivatives)
                dxx, dxy = np.gradient(dx)
                dyx, dyy = np.gradient(dy)
                
                # Profile curvature
                profile_curvature = (dx*dx*dyy - 2*dx*dy*dxy + dy*dy*dxx) / ((dx*dx + dy*dy)**1.5 + 1e-8)
                
                # Plan curvature  
                plan_curvature = (dx*dx*dxx + 2*dx*dy*dxy + dy*dy*dyy) / ((dx*dx + dy*dy) + 1e-8)
                
                results = {
                    'elevation': elevation,
                    'slope': slope,
                    'aspect': aspect,
                    'profile_curvature': profile_curvature,
                    'plan_curvature': plan_curvature
                }
                
                self.cache[dem_path] = results
                return results
                
        except Exception as e:
            logger.error(f"Failed to process DEM file {dem_path}: {e}")
            # Return dummy data
            dummy_shape = (100, 100)
            return {
                'elevation': np.zeros(dummy_shape),
                'slope': np.zeros(dummy_shape),
                'aspect': np.zeros(dummy_shape),
                'profile_curvature': np.zeros(dummy_shape),
                'plan_curvature': np.zeros(dummy_shape)
            }
    
    def extract_point_features(
        self, 
        dem_data: Dict[str, np.ndarray], 
        x: float, 
        y: float,
        window_size: int = 5
    ) -> Dict[str, float]:
        """
        Extract features at a specific point with neighborhood statistics.
        
        Args:
            dem_data: Processed DEM data
            x, y: Point coordinates (pixel coordinates)
            window_size: Size of neighborhood window
            
        Returns:
            Dictionary of extracted features
        """
        try:
            h, w = dem_data['elevation'].shape
            x, y = int(x), int(y)
            
            # Ensure coordinates are within bounds
            x = max(window_size//2, min(w - window_size//2 - 1, x))
            y = max(window_size//2, min(h - window_size//2 - 1, y))
            
            features = {}
            
            for feature_name, data in dem_data.items():
                # Point value
                features[f'{feature_name}_point'] = float(data[y, x])
                
                # Neighborhood statistics
                window = data[y-window_size//2:y+window_size//2+1, 
                             x-window_size//2:x+window_size//2+1]
                
                features[f'{feature_name}_mean'] = float(np.mean(window))
                features[f'{feature_name}_std'] = float(np.std(window))
                features[f'{feature_name}_min'] = float(np.min(window))
                features[f'{feature_name}_max'] = float(np.max(window))
                
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract point features: {e}")
            return {key: 0.0 for key in [
                'elevation_point', 'slope_point', 'aspect_point',
                'elevation_mean', 'slope_mean', 'aspect_mean',
                'elevation_std', 'slope_std', 'aspect_std'
            ]}


class SensorDataAggregator:
    """Aggregate and process sensor data for ML training."""
    
    def __init__(self, sensor_types: List[str] = None):
        self.sensor_types = sensor_types or [
            'strain', 'displacement', 'pore_pressure', 'tilt', 'vibration'
        ]
    
    def aggregate_sensor_readings(
        self,
        readings_df: pd.DataFrame,
        time_window: str = '1H'  # 1 hour aggregation
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate sensor readings into time series.
        
        Args:
            readings_df: DataFrame with columns [timestamp, sensor_id, sensor_type, value]
            time_window: Aggregation window (pandas frequency string)
            
        Returns:
            Dictionary mapping sensor types to time series arrays
        """
        try:
            readings_df['timestamp'] = pd.to_datetime(readings_df['timestamp'])
            
            aggregated_data = {}
            
            for sensor_type in self.sensor_types:
                # Filter readings for this sensor type
                type_readings = readings_df[readings_df['sensor_type'] == sensor_type]
                
                if len(type_readings) == 0:
                    # Create dummy time series if no data
                    aggregated_data[sensor_type] = np.zeros(168)  # 1 week of hourly data
                    continue
                
                # Group by time window and calculate statistics
                grouped = type_readings.groupby(pd.Grouper(key='timestamp', freq=time_window))
                
                time_series = []
                for timestamp, group in grouped:
                    if len(group) > 0:
                        # Use mean value for the time window
                        time_series.append(group['value'].mean())
                    else:
                        time_series.append(0.0)
                
                aggregated_data[sensor_type] = np.array(time_series)
            
            return aggregated_data
            
        except Exception as e:
            logger.error(f"Failed to aggregate sensor readings: {e}")
            # Return dummy data
            return {sensor_type: np.zeros(168) for sensor_type in self.sensor_types}
    
    def detect_anomalies(self, time_series: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Detect anomalies in time series data.
        
        Args:
            time_series: Input time series
            method: Anomaly detection method ('zscore', 'iqr')
            
        Returns:
            Boolean array indicating anomalies
        """
        try:
            if method == 'zscore':
                z_scores = np.abs((time_series - np.mean(time_series)) / (np.std(time_series) + 1e-8))
                return z_scores > 3
            
            elif method == 'iqr':
                q25, q75 = np.percentile(time_series, [25, 75])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                return (time_series < lower_bound) | (time_series > upper_bound)
            
            else:
                return np.zeros(len(time_series), dtype=bool)
                
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return np.zeros(len(time_series), dtype=bool)


def create_data_loaders(
    config: Dict[str, Any],
    data_path: str = 'data/processed',
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        config: Configuration dictionary
        data_path: Path to processed data directory
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    try:
        # Load processed data
        train_df = pd.read_csv(f'{data_path}/train.csv')
        val_df = pd.read_csv(f'{data_path}/val.csv')
        test_df = pd.read_csv(f'{data_path}/test.csv')
        
        # Load sensor data if available
        sensor_data = {}
        sensor_data_path = f'{data_path}/sensor_data.json'
        if os.path.exists(sensor_data_path):
            with open(sensor_data_path, 'r') as f:
                sensor_data = json.load(f)
                # Convert lists back to numpy arrays
                for site_id, data in sensor_data.items():
                    for sensor_type, values in data.items():
                        sensor_data[site_id][sensor_type] = np.array(values)
        
        # Initialize processors
        from ml.models.cnn_model import ImageProcessor
        from ml.models.lstm_model import SensorDataProcessor
        
        image_processor = ImageProcessor()
        sensor_processor = SensorDataProcessor()
        
        # Create datasets
        train_dataset = RockfallDataset(
            train_df,
            image_dir=f'{data_path}/images',
            sensor_data=sensor_data,
            image_transform=image_processor.train_transform,
            sensor_processor=sensor_processor
        )
        
        val_dataset = RockfallDataset(
            val_df,
            image_dir=f'{data_path}/images',
            sensor_data=sensor_data,
            image_transform=image_processor.val_transform,
            sensor_processor=sensor_processor
        )
        
        test_dataset = RockfallDataset(
            test_df,
            image_dir=f'{data_path}/images',
            sensor_data=sensor_data,
            image_transform=image_processor.val_transform,
            sensor_processor=sensor_processor
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        raise


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-length sequences and missing modalities.
    """
    collated = {}
    
    # Handle images
    if 'images' in batch[0]:
        images = [sample['images'] for sample in batch if 'images' in sample]
        if images:
            collated['images'] = torch.stack(images)
    
    # Handle sensor data
    if 'sensor_data' in batch[0]:
        sensor_data = [sample['sensor_data'] for sample in batch if 'sensor_data' in sample]
        sensor_lengths = [sample.get('sensor_lengths', torch.tensor(48)) for sample in batch if 'sensor_data' in sample]
        
        if sensor_data:
            # Pad sequences to same length
            max_length = max(seq.size(0) for seq in sensor_data)
            padded_sequences = []
            
            for seq in sensor_data:
                if seq.size(0) < max_length:
                    padding = torch.zeros(max_length - seq.size(0), seq.size(1))
                    padded_seq = torch.cat([seq, padding], dim=0)
                else:
                    padded_seq = seq
                padded_sequences.append(padded_seq)
            
            collated['sensor_data'] = torch.stack(padded_sequences)
            collated['sensor_lengths'] = torch.stack(sensor_lengths)
    
    # Handle tabular features
    if 'tabular_features' in batch[0]:
        tabular_features = [sample['tabular_features'] for sample in batch]
        collated['tabular_features'] = torch.stack(tabular_features)
    
    # Handle targets
    targets = [sample['targets'] for sample in batch]
    collated['targets'] = torch.stack(targets)
    
    # Handle metadata
    collated['site_ids'] = [sample.get('site_id', '') for sample in batch]
    collated['timestamps'] = [sample.get('timestamp', '') for sample in batch]
    
    return collated


def preprocess_raw_data(
    sites_csv: str,
    sensors_csv: str,
    readings_csv: str,
    images_dir: str,
    dem_dir: str,
    output_dir: str = 'data/processed'
) -> None:
    """
    Preprocess raw data files into ML-ready format.
    
    Args:
        sites_csv: Path to sites CSV file
        sensors_csv: Path to sensors CSV file  
        readings_csv: Path to sensor readings CSV file
        images_dir: Directory containing drone images
        dem_dir: Directory containing DEM files
        output_dir: Output directory for processed data
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Load raw data
        sites_df = pd.read_csv(sites_csv)
        sensors_df = pd.read_csv(sensors_csv)
        readings_df = pd.read_csv(readings_csv)
        
        # Initialize processors
        dem_processor = DEMProcessor()
        sensor_aggregator = SensorDataAggregator()
        
        # Process each site
        processed_data = []
        sensor_data_dict = {}
        
        for _, site in sites_df.iterrows():
            site_id = site['id']
            logger.info(f"Processing site {site_id}")
            
            # Get site sensors
            site_sensors = sensors_df[sensors_df['site_id'] == site_id]
            
            # Get sensor readings
            sensor_readings = readings_df[
                readings_df['sensor_id'].isin(site_sensors['id'])
            ]
            
            # Aggregate sensor data
            site_sensor_data = sensor_aggregator.aggregate_sensor_readings(sensor_readings)
            sensor_data_dict[site_id] = {
                sensor_type: data.tolist()  # Convert to list for JSON serialization
                for sensor_type, data in site_sensor_data.items()
            }
            
            # Process DEM data if available
            dem_features = {}
            dem_file = f"{dem_dir}/{site_id}_dem.tif"
            if os.path.exists(dem_file):
                dem_data = dem_processor.process_dem_file(dem_file)
                # Extract features at site location (using image center for now)
                dem_features = dem_processor.extract_point_features(
                    dem_data, 
                    x=dem_data['elevation'].shape[1]//2,
                    y=dem_data['elevation'].shape[0]//2
                )
            
            # Find associated images
            site_images = []
            for img_file in os.listdir(images_dir):
                if img_file.startswith(site_id):
                    site_images.append(img_file)
            
            # Create data entry for each image (or one entry if no images)
            if site_images:
                for img_file in site_images:
                    entry = {
                        'site_id': site_id,
                        'image_filename': img_file,
                        'elevation': site.get('elevation', 0),
                        'slope_angle': site.get('slope_angle', 0),
                        'aspect_angle': site.get('aspect_angle', 0),
                        'risk_level': site.get('risk_level', 'low'),
                        'timestamp': pd.Timestamp.now().isoformat(),
                        **dem_features
                    }
                    processed_data.append(entry)
            else:
                # Entry without image
                entry = {
                    'site_id': site_id,
                    'elevation': site.get('elevation', 0),
                    'slope_angle': site.get('slope_angle', 0),
                    'aspect_angle': site.get('aspect_angle', 0),
                    'risk_level': site.get('risk_level', 'low'),
                    'timestamp': pd.Timestamp.now().isoformat(),
                    **dem_features
                }
                processed_data.append(entry)
        
        # Create DataFrame
        processed_df = pd.DataFrame(processed_data)
        
        # Split into train/val/test
        from sklearn.model_selection import train_test_split
        
        train_df, temp_df = train_test_split(
            processed_df, test_size=0.3, stratify=processed_df['risk_level'], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['risk_level'], random_state=42
        )
        
        # Save processed data
        train_df.to_csv(f'{output_dir}/train.csv', index=False)
        val_df.to_csv(f'{output_dir}/val.csv', index=False)
        test_df.to_csv(f'{output_dir}/test.csv', index=False)
        
        # Save sensor data
        with open(f'{output_dir}/sensor_data.json', 'w') as f:
            json.dump(sensor_data_dict, f, indent=2)
        
        logger.info(f"Data preprocessing completed. Processed {len(processed_data)} samples.")
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise


if __name__ == '__main__':
    # Example usage
    preprocess_raw_data(
        sites_csv='data/raw/sites.csv',
        sensors_csv='data/raw/sensors.csv',
        readings_csv='data/raw/readings.csv',
        images_dir='data/raw/images',
        dem_dir='data/raw/dem',
        output_dir='data/processed'
    )
