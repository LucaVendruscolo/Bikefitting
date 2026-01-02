"""
Convert PyTorch models to ONNX for web deployment.
Run this script from the web folder.
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import shutil

# Paths
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "public" / "models"
BIKE_MODEL_DIR = SCRIPT_DIR.parent / "bike_angle_detection_model" / "models" / "optuna_best"

def convert_bike_angle_model():
    """Convert the bike angle classifier to ONNX."""
    print("Converting bike angle model...")
    
    # Load checkpoint
    checkpoint_path = BIKE_MODEL_DIR / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"  Model not found: {checkpoint_path}")
        return False
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    num_bins = checkpoint['num_bins']
    backbone_name = checkpoint['backbone']
    
    print(f"  Backbone: {backbone_name}, Bins: {num_bins}")
    
    # Recreate model architecture
    class AngleClassifier(nn.Module):
        def __init__(self, backbone_name, num_bins):
            super().__init__()
            if backbone_name == 'convnext_tiny':
                self.backbone = models.convnext_tiny(weights=None)
                num_features = self.backbone.classifier[2].in_features
                self.backbone.classifier = nn.Identity()
            elif backbone_name == 'convnext_small':
                self.backbone = models.convnext_small(weights=None)
                num_features = self.backbone.classifier[2].in_features
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
    
    model = AngleClassifier(backbone_name, num_bins)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    output_path = MODELS_DIR / "bike_angle.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=14,
    )
    
    print(f"  Saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Save config
    import json
    config = {'num_bins': num_bins, 'backbone': backbone_name, 'input_size': 224}
    with open(MODELS_DIR / "bike_angle_config.json", 'w') as f:
        json.dump(config, f)
    
    return True

def convert_yolo_models():
    """Convert YOLO models to ONNX."""
    from ultralytics import YOLO
    
    # Use nano pose model for faster web inference (6MB vs 52MB for medium)
    print("Converting YOLOv8n-pose (nano - fast)...")
    pose_model = YOLO("yolov8n-pose.pt")
    pose_model.export(format="onnx", imgsz=640, simplify=True)
    shutil.move("yolov8n-pose.onnx", MODELS_DIR / "yolov8n-pose.onnx")
    print(f"  Saved: {MODELS_DIR / 'yolov8n-pose.onnx'}")
    
    print("Converting YOLOv8n-seg...")
    seg_model = YOLO("yolov8n-seg.pt")
    seg_model.export(format="onnx", imgsz=640, simplify=True)
    shutil.move("yolov8n-seg.onnx", MODELS_DIR / "yolov8n-seg.onnx")
    print(f"  Saved: {MODELS_DIR / 'yolov8n-seg.onnx'}")

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("Converting models for web deployment")
    print("=" * 50)
    
    convert_bike_angle_model()
    convert_yolo_models()
    
    print("\nDone! Models saved to:", MODELS_DIR)
    print("\nNext: git add, commit, and push the models.")

