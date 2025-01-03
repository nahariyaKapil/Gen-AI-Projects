import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

class ActivityRecognizer:
    """
    Human activity recognition using pre-trained video models
    """
    
    def __init__(self):
        """Initialize the activity recognizer"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.class_labels = self._load_kinetics_labels()
        self.model_configs = {
            "I3D": {
                "input_size": (224, 224),
                "num_frames": 16,
                "model_name": "i3d_r50",
                "pretrained": True
            },
            "SlowFast": {
                "input_size": (224, 224),
                "num_frames": 32,
                "model_name": "slowfast_r50",
                "pretrained": True
            },
            "ViViT": {
                "input_size": (224, 224),
                "num_frames": 16,
                "model_name": "vivit_base",
                "pretrained": True
            }
        }
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Activity recognizer initialized on device: {self.device}")
    
    def _load_kinetics_labels(self) -> List[str]:
        """Load Kinetics-400 class labels"""
        try:
            # Common activity labels for demonstration
            # In production, this would be loaded from a file
            labels = [
                "walking", "running", "jumping", "sitting", "standing",
                "dancing", "cooking", "eating", "drinking", "reading",
                "writing", "typing", "talking", "singing", "exercising",
                "stretching", "yoga", "playing_guitar", "playing_piano",
                "clapping", "waving", "pointing", "shaking_hands",
                "hugging", "kissing", "sleeping", "lying_down",
                "getting_up", "falling_down", "climbing_stairs",
                "opening_door", "closing_door", "washing_hands",
                "brushing_teeth", "combing_hair", "putting_on_clothes",
                "taking_off_clothes", "wearing_shoes", "tying_shoelaces",
                "carrying_baby", "playing_with_kids", "feeding_pet",
                "walking_dog", "riding_bike", "driving_car",
                "waiting_in_line", "shopping", "paying_bills",
                "using_phone", "texting", "taking_photo"
            ]
            
            return labels
            
        except Exception as e:
            logger.error(f"Error loading class labels: {str(e)}")
            return ["unknown_activity"]
    
    def load_model(self, model_type: str) -> torch.nn.Module:
        """
        Load and cache activity recognition model
        
        Args:
            model_type: Type of model ("I3D", "SlowFast", "ViViT")
            
        Returns:
            Loaded model
        """
        try:
            if model_type in self.models:
                return self.models[model_type]
            
            logger.info(f"Loading {model_type} model...")
            
            if model_type == "I3D":
                model = self._load_i3d_model()
            elif model_type == "SlowFast":
                model = self._load_slowfast_model()
            elif model_type == "ViViT":
                model = self._load_vivit_model()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Move to device and set to evaluation mode
            model = model.to(self.device)
            model.eval()
            
            # Cache the model
            self.models[model_type] = model
            
            logger.info(f"{model_type} model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {str(e)}")
            # Return a dummy model for demonstration
            return self._create_dummy_model()
    
    def _load_i3d_model(self) -> torch.nn.Module:
        """Load I3D model"""
        try:
            # For demonstration, create a simplified I3D-like model
            # In production, you would load from torchvision or pytorchvideo
            model = I3DModel(num_classes=len(self.class_labels))
            return model
            
        except Exception as e:
            logger.error(f"Error loading I3D model: {str(e)}")
            return self._create_dummy_model()
    
    def _load_slowfast_model(self) -> torch.nn.Module:
        """Load SlowFast model"""
        try:
            # For demonstration, create a simplified SlowFast-like model
            model = SlowFastModel(num_classes=len(self.class_labels))
            return model
            
        except Exception as e:
            logger.error(f"Error loading SlowFast model: {str(e)}")
            return self._create_dummy_model()
    
    def _load_vivit_model(self) -> torch.nn.Module:
        """Load ViViT model"""
        try:
            # For demonstration, create a simplified ViViT-like model
            model = ViViTModel(num_classes=len(self.class_labels))
            return model
            
        except Exception as e:
            logger.error(f"Error loading ViViT model: {str(e)}")
            return self._create_dummy_model()
    
    def _create_dummy_model(self) -> torch.nn.Module:
        """Create a dummy model for demonstration"""
        class DummyModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.classifier = nn.Linear(1024, num_classes)
                
            def forward(self, x):
                # Simple processing for demonstration
                batch_size = x.size(0)
                features = torch.randn(batch_size, 1024, device=x.device)
                return self.classifier(features)
        
        return DummyModel(len(self.class_labels))
    
    def get_model(self, model_type: str) -> torch.nn.Module:
        """Get cached model or load if not cached"""
        return self.load_model(model_type)
    
    def predict_activity(self, frame: np.ndarray, model: torch.nn.Module, 
                        model_type: str, use_onnx: bool = False) -> Dict:
        """
        Predict activity from a single frame or video clip
        
        Args:
            frame: Input frame or video clip
            model: Model to use for prediction
            model_type: Type of model
            use_onnx: Whether using ONNX optimization
            
        Returns:
            Prediction results
        """
        try:
            start_time = time.time()
            
            # Preprocess input
            if isinstance(frame, np.ndarray):
                if len(frame.shape) == 3:  # Single frame
                    # Convert to video clip format
                    frames = [frame] * self.model_configs[model_type]["num_frames"]
                    input_tensor = self._preprocess_video_clip(frames)
                else:
                    input_tensor = self._preprocess_frame(frame)
            else:
                input_tensor = frame
            
            # Move to device
            input_tensor = input_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                if use_onnx:
                    # ONNX inference would be handled differently
                    outputs = model(input_tensor)
                else:
                    outputs = model(input_tensor)
            
            # Process outputs
            probabilities = torch.softmax(outputs, dim=1)
            top_k_values, top_k_indices = torch.topk(probabilities, k=5, dim=1)
            
            # Create predictions
            predictions = []
            for i in range(5):
                idx = top_k_indices[0][i].item()
                confidence = top_k_values[0][i].item()
                activity = self.class_labels[idx] if idx < len(self.class_labels) else "unknown"
                
                predictions.append({
                    "activity": activity,
                    "confidence": confidence,
                    "class_idx": idx
                })
            
            inference_time = time.time() - start_time
            
            return {
                "predictions": predictions,
                "model_type": model_type,
                "inference_time": inference_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error during activity prediction: {str(e)}")
            return {
                "predictions": [],
                "error": str(e),
                "success": False
            }
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess single frame"""
        try:
            # Convert to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = frame[:, :, ::-1]  # BGR to RGB
            else:
                frame_rgb = frame
            
            # Convert to tensor
            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
            
            # Apply transforms
            frame_tensor = self.transform(frame_tensor)
            
            # Add batch and temporal dimensions
            frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(2)  # [B, C, T, H, W]
            
            return frame_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {str(e)}")
            raise
    
    def _preprocess_video_clip(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess video clip"""
        try:
            processed_frames = []
            
            for frame in frames:
                # Convert to RGB if needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_rgb = frame[:, :, ::-1]  # BGR to RGB
                else:
                    frame_rgb = frame
                
                # Convert to tensor
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
                
                # Apply transforms
                frame_tensor = self.transform(frame_tensor)
                processed_frames.append(frame_tensor)
            
            # Stack frames
            video_tensor = torch.stack(processed_frames, dim=1)  # [C, T, H, W]
            video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension [B, C, T, H, W]
            
            return video_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing video clip: {str(e)}")
            raise
    
    def batch_predict(self, frames: List[np.ndarray], model_type: str) -> List[Dict]:
        """
        Predict activities for batch of frames
        
        Args:
            frames: List of frames
            model_type: Type of model to use
            
        Returns:
            List of prediction results
        """
        try:
            model = self.get_model(model_type)
            results = []
            
            for frame in frames:
                result = self.predict_activity(frame, model, model_type)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            return []
    
    def get_model_info(self, model_type: str) -> Dict:
        """
        Get information about a model
        
        Args:
            model_type: Type of model
            
        Returns:
            Model information
        """
        if model_type not in self.model_configs:
            return {"error": f"Unknown model type: {model_type}"}
        
        config = self.model_configs[model_type]
        is_loaded = model_type in self.models
        
        return {
            "model_type": model_type,
            "input_size": config["input_size"],
            "num_frames": config["num_frames"],
            "num_classes": len(self.class_labels),
            "is_loaded": is_loaded,
            "device": str(self.device),
            "pretrained": config["pretrained"]
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types"""
        return list(self.model_configs.keys())
    
    def unload_model(self, model_type: str):
        """Unload model from memory"""
        if model_type in self.models:
            del self.models[model_type]
            logger.info(f"{model_type} model unloaded")
    
    def unload_all_models(self):
        """Unload all models from memory"""
        self.models.clear()
        logger.info("All models unloaded")


# Simplified model architectures for demonstration
class I3DModel(nn.Module):
    """Simplified I3D model for demonstration"""
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 192, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Linear(192, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SlowFastModel(nn.Module):
    """Simplified SlowFast model for demonstration"""
    
    def __init__(self, num_classes: int):
        super().__init__()
        # Slow pathway
        self.slow_pathway = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Fast pathway
        self.fast_pathway = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Linear(96, num_classes)  # 64 + 32 features
        
    def forward(self, x):
        # For simplicity, use same input for both pathways
        slow_features = self.slow_pathway(x)
        fast_features = self.fast_pathway(x)
        
        # Flatten and concatenate
        slow_features = slow_features.view(slow_features.size(0), -1)
        fast_features = fast_features.view(fast_features.size(0), -1)
        
        combined_features = torch.cat([slow_features, fast_features], dim=1)
        output = self.classifier(combined_features)
        
        return output


class ViViTModel(nn.Module):
    """Simplified ViViT model for demonstration"""
    
    def __init__(self, num_classes: int, embed_dim: int = 768):
        super().__init__()
        self.patch_embed = nn.Conv3d(3, embed_dim, kernel_size=(2, 16, 16), stride=(2, 16, 16))
        
        # Simplified transformer (single layer for demonstration)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True),
            num_layers=1
        )
        
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, T', H', W']
        
        # Flatten spatial and temporal dimensions
        B, C, T, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)  # [B, seq_len, embed_dim]
        
        # Add positional encoding (simplified)
        seq_len = x.size(1)
        pos_encoding = torch.zeros_like(x)
        x = x + pos_encoding
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        output = self.classifier(x)
        
        return output 