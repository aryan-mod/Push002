import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .cnn_model import DroneImageCNN
from .lstm_model import SensorLSTM
import json

class MultiModalFusionModel(nn.Module):
    """
    Multi-modal fusion model combining CNN (images), LSTM (sensors), and tabular features.
    Implements uncertainty estimation and explainability features.
    """
    
    def __init__(
        self,
        cnn_config: Dict[str, Any],
        lstm_config: Dict[str, Any],
        tabular_input_size: int = 10,  # DEM + weather features
        fusion_hidden_size: int = 256,
        num_classes: int = 4,
        dropout_rate: float = 0.3,
        use_uncertainty: bool = True
    ):
        super(MultiModalFusionModel, self).__init__()
        
        self.use_uncertainty = use_uncertainty
        
        # Individual modality models
        self.cnn_model = DroneImageCNN(**cnn_config)
        self.lstm_model = SensorLSTM(**lstm_config)
        
        # Tabular feature processor
        self.tabular_processor = nn.Sequential(
            nn.Linear(tabular_input_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        
        # Feature dimensions
        cnn_feature_dim = cnn_config.get('output_size', 128)
        lstm_feature_dim = lstm_config.get('output_size', 64)
        tabular_feature_dim = 32
        
        total_feature_dim = cnn_feature_dim + lstm_feature_dim + tabular_feature_dim
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=total_feature_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_feature_dim, fusion_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2)
        )
        
        # Classification head
        self.classifier = nn.Linear(fusion_hidden_size // 2, num_classes)
        
        # Uncertainty estimation (aleatoric)
        if use_uncertainty:
            self.uncertainty_head = nn.Linear(fusion_hidden_size // 2, 1)
        
        # Feature importance for explainability
        self.feature_importance = nn.Linear(fusion_hidden_size // 2, 3)  # importance for each modality
        
    def forward(
        self, 
        images: Optional[torch.Tensor] = None,
        sensor_data: Optional[torch.Tensor] = None,
        sensor_lengths: Optional[torch.Tensor] = None,
        tabular_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through fusion model.
        
        Args:
            images: Drone images tensor (batch_size, 3, H, W)
            sensor_data: Sensor time series (batch_size, seq_len, num_sensors)
            sensor_lengths: Actual lengths of sensor sequences
            tabular_features: DEM and weather features (batch_size, feature_dim)
            
        Returns:
            Dictionary containing predictions, uncertainty, and feature importance
        """
        features = []
        modality_masks = []
        
        # Process images
        if images is not None:
            _, cnn_features = self.cnn_model(images)
            features.append(cnn_features)
            modality_masks.append(torch.ones(cnn_features.size(0), 1, device=images.device))
        else:
            # Zero padding for missing modality
            batch_size = sensor_data.size(0) if sensor_data is not None else tabular_features.size(0)
            cnn_features = torch.zeros(batch_size, self.cnn_model.feature_dim, device=self.get_device())
            features.append(cnn_features)
            modality_masks.append(torch.zeros(batch_size, 1, device=self.get_device()))
        
        # Process sensor data
        if sensor_data is not None:
            _, lstm_features = self.lstm_model(sensor_data, sensor_lengths)
            features.append(lstm_features)
            modality_masks.append(torch.ones(lstm_features.size(0), 1, device=sensor_data.device))
        else:
            batch_size = images.size(0) if images is not None else tabular_features.size(0)
            lstm_features = torch.zeros(batch_size, self.lstm_model.output_size, device=self.get_device())
            features.append(lstm_features)
            modality_masks.append(torch.zeros(batch_size, 1, device=self.get_device()))
        
        # Process tabular features
        if tabular_features is not None:
            tab_features = self.tabular_processor(tabular_features)
            features.append(tab_features)
            modality_masks.append(torch.ones(tab_features.size(0), 1, device=tabular_features.device))
        else:
            batch_size = images.size(0) if images is not None else sensor_data.size(0)
            tab_features = torch.zeros(batch_size, 32, device=self.get_device())
            features.append(tab_features)
            modality_masks.append(torch.zeros(batch_size, 1, device=self.get_device()))
        
        # Concatenate all features
        fused_features = torch.cat(features, dim=1)
        
        # Apply cross-modal attention
        attended_features, attention_weights = self.cross_attention(
            fused_features.unsqueeze(1),
            fused_features.unsqueeze(1),
            fused_features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # Fusion processing
        fusion_output = self.fusion_layers(attended_features)
        
        # Classification
        logits = self.classifier(fusion_output)
        probabilities = F.softmax(logits, dim=1)
        
        # Uncertainty estimation
        uncertainty = None
        if self.use_uncertainty:
            log_variance = self.uncertainty_head(fusion_output)
            uncertainty = torch.exp(log_variance)
        
        # Feature importance
        modality_importance = F.softmax(self.feature_importance(fusion_output), dim=1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'uncertainty': uncertainty,
            'modality_importance': modality_importance,
            'attention_weights': attention_weights,
            'modality_masks': torch.cat(modality_masks, dim=1)
        }
    
    def get_device(self) -> torch.device:
        """Get device of the model."""
        return next(self.parameters()).device
    
    def predict_with_uncertainty(
        self,
        images: Optional[torch.Tensor] = None,
        sensor_data: Optional[torch.Tensor] = None,
        sensor_lengths: Optional[torch.Tensor] = None,
        tabular_features: Optional[torch.Tensor] = None,
        num_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Prediction with Monte Carlo dropout for epistemic uncertainty.
        
        Args:
            num_samples: Number of MC samples for uncertainty estimation
            
        Returns:
            Dictionary with mean predictions and uncertainty estimates
        """
        self.train()  # Enable dropout
        
        predictions = []
        uncertainties = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                output = self.forward(images, sensor_data, sensor_lengths, tabular_features)
                predictions.append(output['probabilities'])
                if output['uncertainty'] is not None:
                    uncertainties.append(output['uncertainty'])
        
        self.eval()  # Disable dropout
        
        # Calculate statistics
        predictions_stack = torch.stack(predictions)
        mean_prediction = torch.mean(predictions_stack, dim=0)
        epistemic_uncertainty = torch.var(predictions_stack, dim=0)
        
        aleatoric_uncertainty = None
        if uncertainties:
            aleatoric_uncertainty = torch.mean(torch.stack(uncertainties), dim=0)
        
        return {
            'mean_prediction': mean_prediction,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'prediction_std': torch.std(predictions_stack, dim=0)
        }


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple fusion models for improved performance and uncertainty estimation.
    """
    
    def __init__(self, models: list, weights: Optional[list] = None):
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            self.weights = torch.tensor(weights)
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Ensemble forward pass."""
        outputs = []
        
        for model in self.models:
            output = model(*args, **kwargs)
            outputs.append(output)
        
        # Weighted average of predictions
        weighted_probs = torch.zeros_like(outputs[0]['probabilities'])
        weighted_uncertainties = torch.zeros_like(outputs[0]['uncertainty']) if outputs[0]['uncertainty'] is not None else None
        
        for i, output in enumerate(outputs):
            weight = self.weights[i]
            weighted_probs += weight * output['probabilities']
            
            if weighted_uncertainties is not None and output['uncertainty'] is not None:
                weighted_uncertainties += weight * output['uncertainty']
        
        # Ensemble uncertainty (variance across models)
        prob_stack = torch.stack([out['probabilities'] for out in outputs])
        ensemble_uncertainty = torch.var(prob_stack, dim=0)
        
        return {
            'probabilities': weighted_probs,
            'uncertainty': weighted_uncertainties,
            'ensemble_uncertainty': ensemble_uncertainty,
            'individual_predictions': [out['probabilities'] for out in outputs]
        }


class LightweightModel(nn.Module):
    """
    Lightweight model for edge deployment (scikit-learn compatible features).
    """
    
    def __init__(self, input_size: int = 50, hidden_size: int = 64, num_classes: int = 4):
        super(LightweightModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_fusion_model(config: Dict[str, Any]) -> MultiModalFusionModel:
    """Factory function to create fusion model."""
    model = MultiModalFusionModel(
        cnn_config=config.get('cnn_config', {}),
        lstm_config=config.get('lstm_config', {}),
        tabular_input_size=config.get('tabular_input_size', 10),
        fusion_hidden_size=config.get('fusion_hidden_size', 256),
        num_classes=config.get('num_classes', 4),
        dropout_rate=config.get('dropout_rate', 0.3),
        use_uncertainty=config.get('use_uncertainty', True)
    )
    
    if 'pretrained_path' in config:
        checkpoint = torch.load(config['pretrained_path'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def export_to_onnx(
    model: MultiModalFusionModel,
    export_path: str,
    example_inputs: Dict[str, torch.Tensor],
    opset_version: int = 11
):
    """Export model to ONNX format for deployment."""
    model.eval()
    
    # Create dummy inputs
    dummy_inputs = []
    input_names = []
    
    if 'images' in example_inputs:
        dummy_inputs.append(example_inputs['images'])
        input_names.append('images')
    
    if 'sensor_data' in example_inputs:
        dummy_inputs.append(example_inputs['sensor_data'])
        input_names.append('sensor_data')
    
    if 'tabular_features' in example_inputs:
        dummy_inputs.append(example_inputs['tabular_features'])
        input_names.append('tabular_features')
    
    # Export to ONNX
    torch.onnx.export(
        model,
        tuple(dummy_inputs),
        export_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=['probabilities', 'uncertainty'],
        dynamic_axes={
            'images': {0: 'batch_size'} if 'images' in input_names else {},
            'sensor_data': {0: 'batch_size'} if 'sensor_data' in input_names else {},
            'tabular_features': {0: 'batch_size'} if 'tabular_features' in input_names else {},
            'probabilities': {0: 'batch_size'},
            'uncertainty': {0: 'batch_size'}
        }
    )


def export_to_torchscript(model: MultiModalFusionModel, export_path: str):
    """Export model to TorchScript for deployment."""
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(export_path)


def convert_to_lightweight_features(
    images: Optional[torch.Tensor] = None,
    sensor_data: Optional[np.ndarray] = None,
    tabular_features: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert multi-modal inputs to lightweight feature vector for scikit-learn models.
    """
    features = []
    
    # Image features (simplified)
    if images is not None:
        # Use basic statistical features instead of CNN
        img_features = []
        for img in images:
            # Convert to grayscale and calculate statistics
            gray = torch.mean(img, dim=0)
            img_features.extend([
                torch.mean(gray).item(),
                torch.std(gray).item(),
                torch.min(gray).item(),
                torch.max(gray).item(),
                torch.median(gray).item()
            ])
        features.extend(img_features)
    else:
        features.extend([0] * 5)  # Placeholder features
    
    # Sensor features
    if sensor_data is not None:
        # Statistical features from time series
        sensor_features = []
        for sensor_type in range(sensor_data.shape[-1]):
            data = sensor_data[:, sensor_type]
            sensor_features.extend([
                np.mean(data),
                np.std(data),
                np.min(data),
                np.max(data),
                np.median(data),
                np.percentile(data, 25),
                np.percentile(data, 75)
            ])
        features.extend(sensor_features)
    else:
        features.extend([0] * 35)  # 5 sensors * 7 features
    
    # Tabular features
    if tabular_features is not None:
        features.extend(tabular_features.tolist())
    else:
        features.extend([0] * 10)  # Default tabular features
    
    return np.array(features)
