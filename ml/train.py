import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List, Optional
import json
import os
from datetime import datetime
import logging
import wandb

from models.fusion_model import create_fusion_model, MultiModalFusionModel
from models.cnn_model import create_cnn_model, ImageProcessor
from models.lstm_model import create_lstm_model, SensorDataProcessor
from utils.data_processor import RockfallDataset, create_data_loaders
from utils.explainability import ExplainabilityAnalyzer
from utils.synthetic_data import SyntheticDataGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RockfallTrainer:
    """
    Main training class for rockfall prediction models.
    Supports single modality and multi-modal fusion training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Loss function
        if config.get('weighted_loss', False):
            class_weights = torch.tensor(config.get('class_weights', [1.0, 1.0, 1.0, 1.0]))
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Explainability analyzer
        self.explainer = ExplainabilityAnalyzer(self.model)
        
    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        model_type = self.config.get('model_type', 'fusion')
        
        if model_type == 'fusion':
            return create_fusion_model(self.config)
        elif model_type == 'cnn':
            return create_cnn_model(self.config)
        elif model_type == 'lstm':
            return create_lstm_model(self.config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', None)
        
        if scheduler_type == 'step':
            step_size = self.config.get('step_size', 10)
            gamma = self.config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'cosine':
            T_max = self.config.get('T_max', 50)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_type == 'reduce_on_plateau':
            patience = self.config.get('patience', 5)
            factor = self.config.get('factor', 0.5)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=patience, factor=factor
            )
        
        return None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            images = batch.get('images', None)
            sensor_data = batch.get('sensor_data', None)
            sensor_lengths = batch.get('sensor_lengths', None)
            tabular_features = batch.get('tabular_features', None)
            targets = batch['targets'].to(self.device)
            
            if images is not None:
                images = images.to(self.device)
            if sensor_data is not None:
                sensor_data = sensor_data.to(self.device)
            if sensor_lengths is not None:
                sensor_lengths = sensor_lengths.to(self.device)
            if tabular_features is not None:
                tabular_features = tabular_features.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, MultiModalFusionModel):
                outputs = self.model(images, sensor_data, sensor_lengths, tabular_features)
                logits = outputs['logits']
            else:
                logits, _ = self.model(images if images is not None else sensor_data)
            
            # Calculate loss
            loss = self.criterion(logits, targets)
            
            # Add uncertainty loss if available
            if isinstance(self.model, MultiModalFusionModel) and outputs.get('uncertainty') is not None:
                uncertainty_loss = torch.mean(outputs['uncertainty'])
                loss += 0.1 * uncertainty_loss  # Weight uncertainty loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Log progress
            if batch_idx % self.config.get('log_interval', 100) == 0:
                logger.info(f'Train Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                images = batch.get('images', None)
                sensor_data = batch.get('sensor_data', None)
                sensor_lengths = batch.get('sensor_lengths', None)
                tabular_features = batch.get('tabular_features', None)
                targets = batch['targets'].to(self.device)
                
                if images is not None:
                    images = images.to(self.device)
                if sensor_data is not None:
                    sensor_data = sensor_data.to(self.device)
                if sensor_lengths is not None:
                    sensor_lengths = sensor_lengths.to(self.device)
                if tabular_features is not None:
                    tabular_features = tabular_features.to(self.device)
                
                # Forward pass
                if isinstance(self.model, MultiModalFusionModel):
                    outputs = self.model(images, sensor_data, sensor_lengths, tabular_features)
                    logits = outputs['logits']
                    probabilities = outputs['probabilities']
                else:
                    logits, _ = self.model(images if images is not None else sensor_data)
                    probabilities = torch.softmax(logits, dim=1)
                
                # Calculate loss
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                
                # Track predictions
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Calculate additional metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision_score(all_targets, all_predictions, average='weighted', zero_division=0),
            'recall': recall_score(all_targets, all_predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        }
        
        # Calculate AUC for binary classification (high risk vs others)
        try:
            binary_targets = [1 if t >= 2 else 0 for t in all_targets]  # High/Critical risk vs Low/Medium
            binary_probs = [sum(p[2:]) for p in all_probabilities]  # Sum of high/critical probabilities
            metrics['auc_roc'] = roc_auc_score(binary_targets, binary_probs)
        except:
            metrics['auc_roc'] = 0.0
        
        return avg_loss, accuracy, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Main training loop."""
        num_epochs = self.config.get('num_epochs', 100)
        early_stopping_patience = self.config.get('early_stopping_patience', 10)
        
        # Initialize tracking
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f'Epoch {epoch+1}/{num_epochs}')
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Log metrics
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            logger.info(f'Val Metrics: {val_metrics}')
            
            # Log to wandb if enabled
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping and best model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                
                # Save best model
                if self.config.get('save_path'):
                    self.save_model(self.config['save_path'], epoch, val_metrics)
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f'Early stopping after {epoch+1} epochs')
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'best_val_loss': best_val_loss,
            'final_metrics': val_metrics,
            'num_epochs': epoch + 1
        }
    
    def save_model(self, save_path: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, save_path)
        logger.info(f'Model saved to {save_path}')
    
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f'Model loaded from {checkpoint_path}')
        return checkpoint.get('metrics', {})
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Train Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate (if scheduler is used)
        if self.scheduler is not None and hasattr(self.scheduler, 'get_last_lr'):
            lrs = [self.scheduler.get_last_lr()[0] for _ in epochs]
            ax3.plot(epochs, lrs, 'g-')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True)
        
        # Model complexity info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        ax4.text(0.1, 0.8, f'Total Parameters: {total_params:,}', transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f'Trainable Parameters: {trainable_params:,}', transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f'Model Size: {total_params * 4 / (1024**2):.2f} MB', transform=ax4.transAxes)
        ax4.text(0.1, 0.5, f'Device: {self.device}', transform=ax4.transAxes)
        ax4.set_title('Model Information')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """Main training function."""
    # Configuration
    config = {
        'model_type': 'fusion',
        'num_epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'T_max': 100,
        'early_stopping_patience': 15,
        'grad_clip': 1.0,
        'weighted_loss': True,
        'class_weights': [1.0, 1.2, 1.5, 2.0],  # Higher weight for critical risk
        'use_wandb': False,
        'save_path': 'models/checkpoints/fusion_model_best.pth',
        'log_interval': 50,
        
        # Model-specific configs
        'cnn_config': {
            'num_classes': 4,
            'dropout_rate': 0.5,
            'output_size': 128
        },
        'lstm_config': {
            'input_size': 5,
            'hidden_size': 128,
            'num_layers': 2,
            'output_size': 64,
            'dropout_rate': 0.3,
            'bidirectional': True,
            'use_attention': True
        },
        'tabular_input_size': 10,
        'fusion_hidden_size': 256,
        'num_classes': 4,
        'dropout_rate': 0.3,
        'use_uncertainty': True
    }
    
    # Generate synthetic data if no real data available
    data_generator = SyntheticDataGenerator(config)
    
    # Check for real data first
    if os.path.exists('data/train.csv'):
        logger.info("Using real data")
        train_loader, val_loader, test_loader = create_data_loaders(config)
    else:
        logger.info("Generating synthetic data for training")
        train_data, val_data, test_data = data_generator.generate_training_data(
            num_samples=1000,
            val_split=0.2,
            test_split=0.1
        )
        
        # Create data loaders from synthetic data
        train_loader, val_loader, test_loader = data_generator.create_data_loaders(
            train_data, val_data, test_data, config['batch_size']
        )
    
    # Initialize wandb if enabled
    if config.get('use_wandb', False):
        wandb.init(project='rockfall-prediction', config=config)
    
    # Initialize trainer
    trainer = RockfallTrainer(config)
    
    # Train model
    logger.info("Starting training...")
    results = trainer.train(train_loader, val_loader)
    
    # Plot training curves
    trainer.plot_training_curves('plots/training_curves.png')
    
    # Evaluate on test set
    if test_loader is not None:
        logger.info("Evaluating on test set...")
        test_loss, test_acc, test_metrics = trainer.validate_epoch(test_loader)
        logger.info(f'Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}')
        logger.info(f'Test Metrics: {test_metrics}')
    
    # Export models
    if isinstance(trainer.model, MultiModalFusionModel):
        # Export to ONNX
        logger.info("Exporting to ONNX...")
        example_batch = next(iter(val_loader))
        example_inputs = {
            'images': example_batch.get('images'),
            'sensor_data': example_batch.get('sensor_data'),
            'tabular_features': example_batch.get('tabular_features')
        }
        
        from models.fusion_model import export_to_onnx, export_to_torchscript
        export_to_onnx(trainer.model, 'models/exports/fusion_model.onnx', example_inputs)
        export_to_torchscript(trainer.model, 'models/exports/fusion_model_script.pt')
    
    # Generate explainability report
    logger.info("Generating explainability analysis...")
    explainer = ExplainabilityAnalyzer(trainer.model)
    explanation_results = explainer.analyze_batch(next(iter(val_loader)))
    
    # Save results
    with open('results/training_results.json', 'w') as f:
        json.dump({
            'config': config,
            'results': results,
            'test_metrics': test_metrics if 'test_metrics' in locals() else None,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()
