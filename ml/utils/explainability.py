import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import json
import shap
from lime.lime_tabular import LimeTabularExplainer
from captum.attr import IntegratedGradients, GradCam, LayerConductance
import logging

logger = logging.getLogger(__name__)

class ExplainabilityAnalyzer:
    """
    Comprehensive explainability analysis for multi-modal rockfall prediction models.
    Supports SHAP, LIME, GradCAM, and integrated gradients.
    """
    
    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.integrated_gradients = None
        self.gradcam = None
        
        # Feature names
        self.tabular_feature_names = [
            'elevation', 'slope_angle', 'aspect_angle',
            'rainfall', 'temperature', 'humidity', 'wind_speed',
            'rock_type', 'vegetation_cover', 'geological_stability'
        ]
        
        self.sensor_feature_names = [
            'strain', 'displacement', 'pore_pressure', 'tilt', 'vibration'
        ]
        
        self.risk_labels = ['Low', 'Medium', 'High', 'Critical']
    
    def setup_shap_explainer(self, background_data: torch.Tensor):
        """Setup SHAP explainer with background data."""
        try:
            # Convert model to SHAP-compatible function
            def model_wrapper(x):
                with torch.no_grad():
                    if len(x.shape) == 2:  # Tabular data
                        x_tensor = torch.FloatTensor(x).to(self.device)
                        # Assuming tabular-only prediction for SHAP
                        output = self.model(tabular_features=x_tensor)
                        if isinstance(output, dict):
                            return output['probabilities'].cpu().numpy()
                        else:
                            return F.softmax(output[0], dim=1).cpu().numpy()
            
            # Create SHAP explainer
            background_np = background_data.cpu().numpy()
            self.shap_explainer = shap.KernelExplainer(model_wrapper, background_np)
            
            logger.info("SHAP explainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
    
    def setup_lime_explainer(self, training_data: np.ndarray):
        """Setup LIME explainer with training data."""
        try:
            self.lime_explainer = LimeTabularExplainer(
                training_data,
                feature_names=self.tabular_feature_names,
                class_names=self.risk_labels,
                mode='classification',
                discretize_continuous=True
            )
            
            logger.info("LIME explainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LIME explainer: {e}")
    
    def setup_gradcam(self, target_layer: str = 'features'):
        """Setup GradCAM for image explainability."""
        try:
            if hasattr(self.model, 'cnn_model'):
                # For fusion model
                target_layer_module = getattr(self.model.cnn_model, target_layer)
            else:
                # For standalone CNN model
                target_layer_module = getattr(self.model, target_layer)
            
            self.gradcam = GradCam(self.model, target_layer_module)
            self.integrated_gradients = IntegratedGradients(self.model)
            
            logger.info("GradCAM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GradCAM: {e}")
    
    def explain_prediction(
        self,
        images: Optional[torch.Tensor] = None,
        sensor_data: Optional[torch.Tensor] = None,
        tabular_features: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanations for a single prediction.
        
        Args:
            images: Input images
            sensor_data: Input sensor time series
            tabular_features: Input tabular features
            target_class: Target class for explanation (if None, uses predicted class)
            
        Returns:
            Dictionary containing various explanations
        """
        explanations = {}
        
        try:
            # Get model prediction
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    output = self.model(images, sensor_data, None, tabular_features)
                    if isinstance(output, dict):
                        probabilities = output['probabilities']
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                    else:
                        logits = output[0]
                        probabilities = F.softmax(logits, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                else:
                    raise ValueError("Model does not have forward method")
            
            if target_class is None:
                target_class = predicted_class
            
            explanations['prediction'] = {
                'predicted_class': predicted_class,
                'predicted_class_name': self.risk_labels[predicted_class],
                'probabilities': probabilities[0].cpu().numpy().tolist(),
                'confidence': float(probabilities[0, predicted_class])
            }
            
            # Image explanations (GradCAM)
            if images is not None and self.gradcam is not None:
                try:
                    gradcam_explanation = self.explain_image(images, target_class)
                    explanations['image_explanation'] = gradcam_explanation
                except Exception as e:
                    logger.error(f"Image explanation failed: {e}")
            
            # Tabular explanations (SHAP)
            if tabular_features is not None and self.shap_explainer is not None:
                try:
                    shap_explanation = self.explain_tabular_shap(tabular_features)
                    explanations['tabular_shap'] = shap_explanation
                except Exception as e:
                    logger.error(f"SHAP explanation failed: {e}")
            
            # Tabular explanations (LIME)  
            if tabular_features is not None and self.lime_explainer is not None:
                try:
                    lime_explanation = self.explain_tabular_lime(tabular_features, target_class)
                    explanations['tabular_lime'] = lime_explanation
                except Exception as e:
                    logger.error(f"LIME explanation failed: {e}")
            
            # Sensor data explanation
            if sensor_data is not None:
                try:
                    sensor_explanation = self.explain_sensor_data(sensor_data, target_class)
                    explanations['sensor_explanation'] = sensor_explanation
                except Exception as e:
                    logger.error(f"Sensor explanation failed: {e}")
            
            # Generate summary
            explanations['summary'] = self.generate_explanation_summary(explanations)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return {'error': str(e)}
    
    def explain_image(self, images: torch.Tensor, target_class: int) -> Dict[str, Any]:
        """Generate GradCAM explanation for images."""
        try:
            images = images.to(self.device)
            
            # Generate GradCAM
            attribution = self.gradcam.attribute(
                images, 
                target=target_class,
                relu_attributions=True
            )
            
            # Convert to numpy and normalize
            gradcam_map = attribution[0].cpu().numpy()
            gradcam_map = np.transpose(gradcam_map, (1, 2, 0))
            
            if gradcam_map.shape[-1] == 1:
                gradcam_map = gradcam_map[:, :, 0]
            else:
                gradcam_map = np.mean(gradcam_map, axis=-1)
            
            # Normalize to 0-1
            gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-8)
            
            return {
                'gradcam_map': gradcam_map.tolist(),
                'explanation': 'Red areas indicate regions most important for the prediction',
                'max_activation': float(gradcam_map.max()),
                'activation_center': self.find_activation_center(gradcam_map)
            }
            
        except Exception as e:
            logger.error(f"GradCAM explanation failed: {e}")
            return {'error': str(e)}
    
    def explain_tabular_shap(self, tabular_features: torch.Tensor) -> Dict[str, Any]:
        """Generate SHAP explanation for tabular features."""
        try:
            features_np = tabular_features.cpu().numpy()
            
            # Generate SHAP values
            shap_values = self.shap_explainer.shap_values(features_np)
            
            if isinstance(shap_values, list):
                # Multi-class case
                shap_values_dict = {
                    self.risk_labels[i]: shap_values[i][0].tolist()
                    for i in range(len(shap_values))
                }
            else:
                # Binary case
                shap_values_dict = {'values': shap_values[0].tolist()}
            
            # Feature importance ranking
            total_importance = np.abs(shap_values[0] if isinstance(shap_values, list) else shap_values).sum(axis=0)
            feature_importance = [
                {
                    'feature': self.tabular_feature_names[i],
                    'importance': float(total_importance[i]),
                    'value': float(features_np[0, i])
                }
                for i in range(len(self.tabular_feature_names))
            ]
            
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                'shap_values': shap_values_dict,
                'feature_importance': feature_importance,
                'top_features': feature_importance[:3]
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'error': str(e)}
    
    def explain_tabular_lime(self, tabular_features: torch.Tensor, target_class: int) -> Dict[str, Any]:
        """Generate LIME explanation for tabular features."""
        try:
            features_np = tabular_features.cpu().numpy()[0]  # Single sample
            
            # Model prediction function for LIME
            def predict_fn(x):
                x_tensor = torch.FloatTensor(x).to(self.device)
                with torch.no_grad():
                    output = self.model(tabular_features=x_tensor)
                    if isinstance(output, dict):
                        return output['probabilities'].cpu().numpy()
                    else:
                        return F.softmax(output[0], dim=1).cpu().numpy()
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                features_np,
                predict_fn,
                num_features=len(self.tabular_feature_names),
                top_labels=len(self.risk_labels)
            )
            
            # Extract explanation for target class
            lime_values = explanation.as_list()
            
            return {
                'lime_explanation': lime_values,
                'intercept': explanation.intercept[target_class],
                'prediction_local': explanation.local_pred[target_class],
                'score': explanation.score
            }
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {'error': str(e)}
    
    def explain_sensor_data(self, sensor_data: torch.Tensor, target_class: int) -> Dict[str, Any]:
        """Generate explanation for sensor time series data."""
        try:
            # Use integrated gradients for time series
            if self.integrated_gradients is None:
                # Fallback to simple sensitivity analysis
                return self.sensitivity_analysis_sensors(sensor_data, target_class)
            
            sensor_data = sensor_data.to(self.device)
            
            # Generate integrated gradients
            baseline = torch.zeros_like(sensor_data)
            attribution = self.integrated_gradients.attribute(
                sensor_data,
                baseline,
                target=target_class,
                n_steps=50
            )
            
            # Aggregate attributions by sensor type
            attribution_np = attribution[0].cpu().numpy()  # Shape: (seq_len, num_sensors)
            
            sensor_importance = []
            for i, sensor_name in enumerate(self.sensor_feature_names):
                if i < attribution_np.shape[1]:
                    importance = np.abs(attribution_np[:, i]).sum()
                    recent_trend = self.calculate_trend(sensor_data[0, :, i].cpu().numpy())
                    
                    sensor_importance.append({
                        'sensor': sensor_name,
                        'importance': float(importance),
                        'recent_trend': recent_trend,
                        'current_value': float(sensor_data[0, -1, i])
                    })
            
            sensor_importance.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                'sensor_importance': sensor_importance,
                'top_sensors': sensor_importance[:3],
                'attribution_map': attribution_np.tolist(),
                'temporal_pattern': self.analyze_temporal_patterns(attribution_np)
            }
            
        except Exception as e:
            logger.error(f"Sensor explanation failed: {e}")
            return {'error': str(e)}
    
    def sensitivity_analysis_sensors(self, sensor_data: torch.Tensor, target_class: int) -> Dict[str, Any]:
        """Fallback sensitivity analysis for sensor data."""
        try:
            sensor_data = sensor_data.to(self.device)
            original_output = self.model(sensor_data=sensor_data)
            
            if isinstance(original_output, dict):
                original_prob = original_output['probabilities'][0, target_class]
            else:
                original_prob = F.softmax(original_output[0], dim=1)[0, target_class]
            
            sensor_importance = []
            
            for i, sensor_name in enumerate(self.sensor_feature_names):
                if i < sensor_data.shape[2]:
                    # Zero out sensor and measure change
                    modified_data = sensor_data.clone()
                    modified_data[:, :, i] = 0
                    
                    modified_output = self.model(sensor_data=modified_data)
                    if isinstance(modified_output, dict):
                        modified_prob = modified_output['probabilities'][0, target_class]
                    else:
                        modified_prob = F.softmax(modified_output[0], dim=1)[0, target_class]
                    
                    importance = float(abs(original_prob - modified_prob))
                    
                    sensor_importance.append({
                        'sensor': sensor_name,
                        'importance': importance,
                        'current_value': float(sensor_data[0, -1, i])
                    })
            
            sensor_importance.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                'sensor_importance': sensor_importance,
                'top_sensors': sensor_importance[:3]
            }
            
        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {e}")
            return {'error': str(e)}
    
    def generate_explanation_summary(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable explanation summary."""
        try:
            summary = {
                'prediction_summary': '',
                'key_factors': [],
                'risk_explanation': '',
                'confidence_level': 'medium'
            }
            
            # Prediction summary
            if 'prediction' in explanations:
                pred = explanations['prediction']
                summary['prediction_summary'] = (
                    f"Predicted risk level: {pred['predicted_class_name']} "
                    f"(Confidence: {pred['confidence']:.1%})"
                )
                
                # Confidence level
                if pred['confidence'] > 0.8:
                    summary['confidence_level'] = 'high'
                elif pred['confidence'] < 0.6:
                    summary['confidence_level'] = 'low'
            
            # Key factors from different modalities
            key_factors = []
            
            # From tabular features
            if 'tabular_shap' in explanations and 'top_features' in explanations['tabular_shap']:
                for feature in explanations['tabular_shap']['top_features']:
                    key_factors.append({
                        'factor': feature['feature'],
                        'impact': 'high' if feature['importance'] > 0.1 else 'medium',
                        'modality': 'site_characteristics'
                    })
            
            # From sensor data
            if 'sensor_explanation' in explanations and 'top_sensors' in explanations['sensor_explanation']:
                for sensor in explanations['sensor_explanation']['top_sensors']:
                    key_factors.append({
                        'factor': f"{sensor['sensor']} readings",
                        'impact': 'high' if sensor['importance'] > 0.1 else 'medium',
                        'modality': 'sensor_data'
                    })
            
            # From image analysis
            if 'image_explanation' in explanations:
                key_factors.append({
                    'factor': 'visual terrain analysis',
                    'impact': 'medium',
                    'modality': 'imagery'
                })
            
            summary['key_factors'] = key_factors
            
            # Risk explanation based on prediction
            if 'prediction' in explanations:
                risk_level = explanations['prediction']['predicted_class']
                
                risk_explanations = {
                    0: "Low risk conditions detected. Current monitoring parameters are within normal ranges.",
                    1: "Medium risk identified. Some elevated readings detected, continued monitoring recommended.",
                    2: "High risk conditions present. Multiple factors indicate potential instability.",
                    3: "Critical risk detected. Immediate attention and potential evacuation measures required."
                }
                
                summary['risk_explanation'] = risk_explanations.get(risk_level, "Unknown risk level.")
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {'error': str(e)}
    
    def analyze_batch(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze a batch of data for explainability."""
        try:
            batch_size = data_batch['targets'].size(0)
            batch_explanations = []
            
            for i in range(min(batch_size, 5)):  # Limit to 5 samples for efficiency
                sample_explanation = self.explain_prediction(
                    images=data_batch.get('images', [None])[i:i+1] if data_batch.get('images') is not None else None,
                    sensor_data=data_batch.get('sensor_data', [None])[i:i+1] if data_batch.get('sensor_data') is not None else None,
                    tabular_features=data_batch.get('tabular_features', [None])[i:i+1] if data_batch.get('tabular_features') is not None else None
                )
                
                batch_explanations.append(sample_explanation)
            
            # Aggregate insights
            aggregate_insights = self.aggregate_batch_insights(batch_explanations)
            
            return {
                'individual_explanations': batch_explanations,
                'aggregate_insights': aggregate_insights
            }
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return {'error': str(e)}
    
    def aggregate_batch_insights(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate insights from multiple explanations."""
        try:
            insights = {
                'common_factors': {},
                'prediction_distribution': {},
                'average_confidence': 0.0
            }
            
            # Count predictions
            for exp in explanations:
                if 'prediction' in exp:
                    pred_class = exp['prediction']['predicted_class_name']
                    insights['prediction_distribution'][pred_class] = \
                        insights['prediction_distribution'].get(pred_class, 0) + 1
                    
                    insights['average_confidence'] += exp['prediction']['confidence']
            
            insights['average_confidence'] /= max(len(explanations), 1)
            
            # Count common factors
            for exp in explanations:
                if 'summary' in exp and 'key_factors' in exp['summary']:
                    for factor in exp['summary']['key_factors']:
                        factor_name = factor['factor']
                        insights['common_factors'][factor_name] = \
                            insights['common_factors'].get(factor_name, 0) + 1
            
            return insights
            
        except Exception as e:
            logger.error(f"Insight aggregation failed: {e}")
            return {'error': str(e)}
    
    def visualize_explanation(
        self, 
        explanation: Dict[str, Any], 
        save_path: str = None
    ) -> None:
        """Create visualization of explanation results."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Prediction probabilities
            if 'prediction' in explanation:
                pred = explanation['prediction']
                axes[0, 0].bar(self.risk_labels, pred['probabilities'])
                axes[0, 0].set_title('Prediction Probabilities')
                axes[0, 0].set_ylabel('Probability')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Feature importance
            if 'tabular_shap' in explanation and 'feature_importance' in explanation['tabular_shap']:
                features = explanation['tabular_shap']['feature_importance'][:8]  # Top 8
                feature_names = [f['feature'] for f in features]
                importances = [f['importance'] for f in features]
                
                axes[0, 1].barh(feature_names, importances)
                axes[0, 1].set_title('Feature Importance (SHAP)')
                axes[0, 1].set_xlabel('Importance')
            
            # Plot 3: Sensor importance
            if 'sensor_explanation' in explanation and 'sensor_importance' in explanation['sensor_explanation']:
                sensors = explanation['sensor_explanation']['sensor_importance']
                sensor_names = [s['sensor'] for s in sensors]
                sensor_importances = [s['importance'] for s in sensors]
                
                axes[1, 0].bar(sensor_names, sensor_importances)
                axes[1, 0].set_title('Sensor Importance')
                axes[1, 0].set_ylabel('Importance')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Summary text
            if 'summary' in explanation:
                summary_text = explanation['summary']['prediction_summary']
                risk_text = explanation['summary']['risk_explanation']
                
                axes[1, 1].text(0.1, 0.8, 'Prediction:', fontweight='bold', transform=axes[1, 1].transAxes)
                axes[1, 1].text(0.1, 0.7, summary_text, wrap=True, transform=axes[1, 1].transAxes)
                axes[1, 1].text(0.1, 0.5, 'Risk Assessment:', fontweight='bold', transform=axes[1, 1].transAxes)
                axes[1, 1].text(0.1, 0.1, risk_text, wrap=True, transform=axes[1, 1].transAxes)
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    # Helper methods
    def find_activation_center(self, gradcam_map: np.ndarray) -> Tuple[int, int]:
        """Find the center of highest activation in GradCAM."""
        try:
            # Find coordinates of maximum activation
            max_coords = np.unravel_index(np.argmax(gradcam_map), gradcam_map.shape)
            return (int(max_coords[0]), int(max_coords[1]))
        except:
            return (0, 0)
    
    def calculate_trend(self, time_series: np.ndarray) -> str:
        """Calculate trend in time series."""
        try:
            if len(time_series) < 2:
                return 'stable'
            
            # Simple linear trend
            x = np.arange(len(time_series))
            coeffs = np.polyfit(x, time_series, 1)
            slope = coeffs[0]
            
            if slope > 0.1:
                return 'increasing'
            elif slope < -0.1:
                return 'decreasing'
            else:
                return 'stable'
                
        except:
            return 'unknown'
    
    def analyze_temporal_patterns(self, attribution_map: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal patterns in attributions."""
        try:
            # Find time points with highest attribution
            temporal_importance = np.sum(np.abs(attribution_map), axis=1)
            
            # Find peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(temporal_importance, height=np.mean(temporal_importance))
            
            return {
                'peak_times': peaks.tolist(),
                'overall_trend': self.calculate_trend(temporal_importance),
                'max_importance_time': int(np.argmax(temporal_importance))
            }
            
        except Exception as e:
            logger.error(f"Temporal pattern analysis failed: {e}")
            return {'error': str(e)}


def generate_explanation_report(
    model,
    test_data: Dict[str, torch.Tensor],
    output_path: str = 'explanations/report.json'
) -> None:
    """Generate comprehensive explanation report for model predictions."""
    try:
        analyzer = ExplainabilityAnalyzer(model)
        
        # Analyze test batch
        results = analyzer.analyze_batch(test_data)
        
        # Generate report
        report = {
            'model_analysis': {
                'total_samples_analyzed': len(results.get('individual_explanations', [])),
                'aggregate_insights': results.get('aggregate_insights', {}),
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'individual_explanations': results.get('individual_explanations', [])
        }
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Explanation report saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
