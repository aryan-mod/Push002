import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, List
import torch.nn.functional as F

class SensorLSTM(nn.Module):
    """
    LSTM model for processing time-series sensor data.
    Handles multiple sensor types with attention mechanism.
    """
    
    def __init__(
        self,
        input_size: int = 5,  # Number of sensor types
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 64,  # Feature output size for fusion
        dropout_rate: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        super(SensorLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, output_size)
        )
        
        # Classification head (for standalone use)
        self.classifier = nn.Sequential(
            nn.Linear(output_size, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(32, 4)  # 4 risk levels
        )
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            lengths: Actual sequence lengths for each sample
            
        Returns:
            Tuple of (classification_output, extracted_features)
        """
        batch_size = x.size(0)
        
        # Pack sequences if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Unpack sequences if they were packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        
        # Apply attention mechanism
        if self.use_attention:
            # Self-attention over sequence
            attended_out, attention_weights = self.attention(
                lstm_out, lstm_out, lstm_out
            )
            
            # Global average pooling over sequence dimension
            if lengths is not None:
                # Mask out padded positions
                mask = torch.arange(attended_out.size(1)).expand(
                    batch_size, attended_out.size(1)
                ).to(attended_out.device) < lengths.unsqueeze(1)
                
                attended_out = attended_out * mask.unsqueeze(-1).float()
                sequence_output = attended_out.sum(dim=1) / lengths.unsqueeze(-1).float()
            else:
                sequence_output = attended_out.mean(dim=1)
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward final states
                sequence_output = torch.cat([
                    hidden[-2], hidden[-1]
                ], dim=-1)
            else:
                sequence_output = hidden[-1]
        
        # Extract features
        features = self.feature_extractor(sequence_output)
        
        # Classification
        classification = self.classifier(features)
        
        return classification, features
    
    def extract_features(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """Extract features without classification."""
        _, features = self.forward(x, lengths)
        return features


class TemporalCNN(nn.Module):
    """
    Temporal CNN for sensor time-series processing.
    Alternative to LSTM with better parallelization.
    """
    
    def __init__(
        self,
        input_size: int = 5,
        num_filters: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [3, 5, 7],
        output_size: int = 64,
        dropout_rate: float = 0.3
    ):
        super(TemporalCNN, self).__init__()
        
        self.input_size = input_size
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        
        in_channels = input_size
        for i, (out_channels, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.MaxPool1d(2)
            )
            self.conv_layers.append(conv_block)
            in_channels = out_channels
        
        # Global pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(sum(num_filters), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, output_size)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(output_size, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Temporal CNN.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Tuple of (classification_output, extracted_features)
        """
        # Transpose for Conv1d: (batch, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        conv_outputs = []
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            # Global pool each conv output
            pooled = self.global_pool(x).squeeze(-1)
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs
        features_raw = torch.cat(conv_outputs, dim=1)
        
        # Extract final features
        features = self.feature_extractor(features_raw)
        
        # Classification
        classification = self.classifier(features)
        
        return classification, features


class SensorDataProcessor:
    """Preprocessing and feature engineering for sensor data."""
    
    def __init__(
        self,
        sensor_types: List[str] = ['strain', 'displacement', 'pore_pressure', 'tilt', 'vibration'],
        sequence_length: int = 48,  # 48 hours of hourly data
        normalization_method: str = 'zscore'
    ):
        self.sensor_types = sensor_types
        self.sequence_length = sequence_length
        self.normalization_method = normalization_method
        
        # Statistics for normalization
        self.mean_values = {}
        self.std_values = {}
        self.min_values = {}
        self.max_values = {}
    
    def fit_normalization(self, data: Dict[str, np.ndarray]):
        """Fit normalization parameters from training data."""
        for sensor_type in self.sensor_types:
            if sensor_type in data:
                values = data[sensor_type]
                self.mean_values[sensor_type] = np.mean(values)
                self.std_values[sensor_type] = np.std(values)
                self.min_values[sensor_type] = np.min(values)
                self.max_values[sensor_type] = np.max(values)
    
    def normalize_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize sensor data."""
        normalized = {}
        
        for sensor_type in self.sensor_types:
            if sensor_type in data:
                values = data[sensor_type]
                
                if self.normalization_method == 'zscore':
                    mean = self.mean_values.get(sensor_type, 0)
                    std = self.std_values.get(sensor_type, 1)
                    normalized[sensor_type] = (values - mean) / (std + 1e-8)
                
                elif self.normalization_method == 'minmax':
                    min_val = self.min_values.get(sensor_type, 0)
                    max_val = self.max_values.get(sensor_type, 1)
                    normalized[sensor_type] = (values - min_val) / (max_val - min_val + 1e-8)
                
                else:
                    normalized[sensor_type] = values
            else:
                # Fill missing sensor types with zeros
                normalized[sensor_type] = np.zeros_like(
                    list(data.values())[0] if data else np.array([0])
                )
        
        return normalized
    
    def create_sequences(self, data: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sequences for time-series prediction.
        
        Args:
            data: Dictionary of sensor data arrays
            
        Returns:
            Tuple of (sequences, lengths)
        """
        # Normalize data
        normalized_data = self.normalize_data(data)
        
        # Stack sensor data
        sensor_arrays = []
        for sensor_type in self.sensor_types:
            sensor_arrays.append(normalized_data[sensor_type])
        
        # Create time-series matrix
        time_series = np.stack(sensor_arrays, axis=-1)  # (time_steps, num_sensors)
        
        # Create sliding window sequences
        sequences = []
        for i in range(len(time_series) - self.sequence_length + 1):
            sequences.append(time_series[i:i + self.sequence_length])
        
        if not sequences:
            # Handle short sequences
            padded = np.zeros((self.sequence_length, len(self.sensor_types)))
            if len(time_series) > 0:
                padded[:len(time_series)] = time_series
            sequences = [padded]
        
        sequences = np.array(sequences)
        lengths = torch.tensor([self.sequence_length] * len(sequences))
        
        return torch.FloatTensor(sequences), lengths


def create_lstm_model(config: Dict[str, Any]) -> SensorLSTM:
    """Factory function to create LSTM model."""
    model = SensorLSTM(
        input_size=config.get('input_size', 5),
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2),
        output_size=config.get('output_size', 64),
        dropout_rate=config.get('dropout_rate', 0.3),
        bidirectional=config.get('bidirectional', True),
        use_attention=config.get('use_attention', True)
    )
    
    if 'pretrained_path' in config:
        checkpoint = torch.load(config['pretrained_path'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def create_temporal_cnn_model(config: Dict[str, Any]) -> TemporalCNN:
    """Factory function to create Temporal CNN model."""
    model = TemporalCNN(
        input_size=config.get('input_size', 5),
        num_filters=config.get('num_filters', [64, 128, 256]),
        kernel_sizes=config.get('kernel_sizes', [3, 5, 7]),
        output_size=config.get('output_size', 64),
        dropout_rate=config.get('dropout_rate', 0.3)
    )
    
    if 'pretrained_path' in config:
        checkpoint = torch.load(config['pretrained_path'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model
