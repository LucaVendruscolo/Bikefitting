"""
Convert PyTorch models to ONNX format for browser deployment.
Run this script once to generate the ONNX models.

Usage:
    python convert_models_to_onnx.py
"""

import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models
from ultralytics import YOLO
import shutil


class AngleClassifier(nn.Module):
    """Same model architecture as training."""
    
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


def convert_bike_angle_model():
    """Convert the bike angle classifier to ONNX."""
    print("Converting bike angle model to ONNX...")
    
    model_path = Path(__file__).parent.parent / "bike_angle_detection_model" / "models" / "optuna_best" / "best_model.pt"
    output_path = Path(__file__).parent / "public" / "models" / "bike_angle.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    num_bins = checkpoint['num_bins']
    backbone = checkpoint['backbone']
    
    print(f"  Backbone: {backbone}, Bins: {num_bins}")
    
    # Create model
    model = AngleClassifier(backbone, num_bins)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Save config
    import json
    config = {
        'num_bins': num_bins,
        'backbone': backbone,
        'input_size': 224
    }
    config_path = output_path.parent / "bike_angle_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  Saved to: {output_path}")
    print(f"  Config saved to: {config_path}")
    return output_path


def convert_yolo_models():
    """Convert YOLO models to ONNX."""
    output_dir = Path(__file__).parent / "public" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert YOLOv8m-pose
    print("\nConverting YOLOv8m-pose to ONNX...")
    pose_model = YOLO("yolov8m-pose.pt")
    pose_onnx_path = output_dir / "yolov8m-pose.onnx"
    pose_model.export(format="onnx", imgsz=640, simplify=True)
    shutil.move("yolov8m-pose.onnx", pose_onnx_path)
    print(f"  Saved to: {pose_onnx_path}")
    
    # Convert YOLOv8n-seg for bike masking
    print("\nConverting YOLOv8n-seg to ONNX...")
    seg_model = YOLO("yolov8n-seg.pt")
    seg_onnx_path = output_dir / "yolov8n-seg.onnx"
    seg_model.export(format="onnx", imgsz=640, simplify=True)
    shutil.move("yolov8n-seg.onnx", seg_onnx_path)
    print(f"  Saved to: {seg_onnx_path}")
    
    return pose_onnx_path, seg_onnx_path


def main():
    print("=" * 50)
    print("Converting models to ONNX for browser deployment")
    print("=" * 50)
    
    # Convert bike angle model
    bike_angle_path = convert_bike_angle_model()
    
    # Convert YOLO models
    pose_path, seg_path = convert_yolo_models()
    
    print("\n" + "=" * 50)
    print("Conversion complete!")
    print("=" * 50)
    print(f"\nModels saved to: {Path(__file__).parent / 'public' / 'models'}")
    print("\nNext steps:")
    print("1. Run 'npm install' in the web folder")
    print("2. Run 'npm run dev' to test locally")
    print("3. Push to GitHub and deploy to Vercel")


if __name__ == "__main__":
    main()

