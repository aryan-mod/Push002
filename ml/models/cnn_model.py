import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from typing import Tuple, Dict, Any
import numpy as np

class DroneImageCNN(nn.Module):
    """
    CNN model for drone imagery analysis using pretrained ResNet50 backbone.
    Supports GradCAM for explainability.
    """
    
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.5):
        super(DroneImageCNN, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature extraction layers
        self.feature_dim = 2048
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
        # GradCAM hook storage
        self.gradients = None
        self.activations = None
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with feature extraction for fusion model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Tuple of (classification_output, extracted_features)
        """
        # Feature extraction
        features = self.features(x)
        
        # Store activations for GradCAM
        if self.training:
            features.register_hook(self.save_gradients)
        self.activations = features
        
        # Global pooling and classification
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        
        # Classification head
        output = self.classifier(flattened)
        
        return output, flattened
    
    def save_gradients(self, grad):
        """Hook function to save gradients for GradCAM."""
        self.gradients = grad
    
    def get_gradcam(self, class_idx: int = None) -> np.ndarray:
        """
        Generate GradCAM heatmap for explainability.
        
        Args:
            class_idx: Target class index for GradCAM
            
        Returns:
            GradCAM heatmap as numpy array
        """
        if self.gradients is None or self.activations is None:
            raise ValueError("No gradients or activations available. Run forward pass first.")
        
        # Get gradients and activations
        gradients = self.gradients[0]  # Take first sample in batch
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps
        gradcam = torch.zeros(activations.shape[1:])
        for i, w in enumerate(weights):
            gradcam += w * activations[i]
        
        # ReLU and normalization
        gradcam = F.relu(gradcam)
        gradcam = gradcam / torch.max(gradcam)
        
        return gradcam.detach().cpu().numpy()
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        features = self.features(x)
        pooled = self.global_pool(features)
        return pooled.view(pooled.size(0), -1)


class ImageProcessor:
    """Preprocessing and augmentation for drone images."""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        
        # Training augmentations
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Validation/inference transforms
        self.val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_training(self, image):
        """Apply training augmentations."""
        return self.train_transform(image)
    
    def preprocess_inference(self, image):
        """Apply inference preprocessing."""
        return self.val_transform(image)


def create_cnn_model(config: Dict[str, Any]) -> DroneImageCNN:
    """
    Factory function to create CNN model with configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured CNN model
    """
    model = DroneImageCNN(
        num_classes=config.get('num_classes', 4),
        dropout_rate=config.get('dropout_rate', 0.5)
    )
    
    # Load pretrained weights if specified
    if 'pretrained_path' in config:
        checkpoint = torch.load(config['pretrained_path'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def calculate_model_complexity(model: DroneImageCNN) -> Dict[str, int]:
    """Calculate model complexity metrics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }
