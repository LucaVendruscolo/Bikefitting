"""
Bike Angle Predictor - Uses the trained ConvNeXt model.

IMPORTANT: This must use the EXACT same model architecture and inference
pipeline as training to get correct predictions.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from typing import Tuple


class AngleClassifier(nn.Module):
    """
    Angle classifier with circular soft labels.
    
    Architecture (must match training exactly):
    - Backbone: ConvNeXt, ResNet50, or EfficientNet
    - Head: LayerNorm → Linear → GELU → Dropout → Linear → GELU → Dropout → Linear
    """
    
    def __init__(self, backbone_name: str, num_bins: int):
        super().__init__()
        
        if backbone_name == 'convnext_tiny':
            self.backbone = models.convnext_tiny(weights=None)
            num_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name == 'convnext_small':
            self.backbone = models.convnext_small(weights=None)
            num_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights=None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        self.head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_bins),
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.flatten(1)
        return self.head(features)


class BikeAnglePredictor:
    """
    Predicts bike tilt angle from masked bike images.
    
    Uses the trained ConvNeXt classification model with circular soft labels.
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Load the trained model.
        
        Args:
            model_path: Path to best_model.pt checkpoint
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # #region agent log
        import json as _json
        checkpoint_keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else ['NOT_A_DICT']
        with open(r'c:\Users\lucav\Downloads\Bikefitting2\.cursor\debug.log', 'a') as _f:
            _f.write(_json.dumps({"location":"angle_predictor.py:BikeAnglePredictor.__init__:checkpoint_loaded","message":"Checkpoint loaded","data":{"checkpoint_keys":checkpoint_keys},"hypothesisId":"C","sessionId":"debug","runId":"run1"})+'\n')
        # #endregion
        
        # Validate checkpoint
        required_keys = ['model_state_dict', 'num_bins', 'backbone']
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(
                    f"Invalid model checkpoint: missing '{key}'. "
                    f"Expected keys: {required_keys}"
                )
        
        self.num_bins = checkpoint['num_bins']
        backbone = checkpoint['backbone']
        
        # #region agent log
        state_dict_keys = list(checkpoint['model_state_dict'].keys())[:10]
        with open(r'c:\Users\lucav\Downloads\Bikefitting2\.cursor\debug.log', 'a') as _f:
            _f.write(_json.dumps({"location":"angle_predictor.py:BikeAnglePredictor.__init__:state_dict_keys","message":"First 10 keys from checkpoint model_state_dict","data":{"num_bins":self.num_bins,"backbone":backbone,"first_10_checkpoint_keys":state_dict_keys},"hypothesisId":"D","sessionId":"debug","runId":"run1"})+'\n')
        # #endregion
        
        # Create model
        self.model = AngleClassifier(backbone, self.num_bins)
        
        # #region agent log
        model_state_keys = list(self.model.state_dict().keys())[:10]
        with open(r'c:\Users\lucav\Downloads\Bikefitting2\.cursor\debug.log', 'a') as _f:
            _f.write(_json.dumps({"location":"angle_predictor.py:BikeAnglePredictor.__init__:pre_load","message":"Model state_dict keys before load","data":{"first_10_model_keys":model_state_keys},"hypothesisId":"E","sessionId":"debug","runId":"run1"})+'\n')
        # #endregion
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Pre-compute bin centers for circular mean
        bin_size = 360.0 / self.num_bins
        self.bin_centers = torch.linspace(
            -180 + bin_size/2, 
            180 - bin_size/2, 
            self.num_bins
        ).to(self.device)
        self.bin_centers_rad = torch.deg2rad(self.bin_centers)
        
        # Transform (MUST match training exactly)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, masked_image: np.ndarray) -> Tuple[float, float]:
        """
        Predict bike angle from masked bike image.
        
        Args:
            masked_image: 224x224 BGR image with bike pixels only (black background)
            
        Returns:
            (angle_deg, confidence) - angle in degrees [-180, 180], confidence as percentage
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        
        # Apply transform and add batch dimension
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get logits
            logits = self.model(img_tensor)
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=1)
            
            # Circular mean to get angle
            sin_sum = (probs * torch.sin(self.bin_centers_rad)).sum(dim=-1)
            cos_sum = (probs * torch.cos(self.bin_centers_rad)).sum(dim=-1)
            angle_deg = torch.rad2deg(torch.atan2(sin_sum, cos_sum)).item()
            
            # Confidence is max probability
            confidence = probs.max().item() * 100
        
        return angle_deg, confidence
    
    def predict_batch(self, masked_images: list) -> list:
        """
        Predict angles for a batch of images.
        
        Args:
            masked_images: List of 224x224 BGR images
            
        Returns:
            List of (angle_deg, confidence) tuples
        """
        if not masked_images:
            return []
        
        # Stack images into batch
        tensors = []
        for img in masked_images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(img_rgb))
        
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            
            sin_sum = (probs * torch.sin(self.bin_centers_rad)).sum(dim=-1)
            cos_sum = (probs * torch.cos(self.bin_centers_rad)).sum(dim=-1)
            angles = torch.rad2deg(torch.atan2(sin_sum, cos_sum))
            
            confidences = probs.max(dim=1).values * 100
        
        return list(zip(angles.cpu().tolist(), confidences.cpu().tolist()))

