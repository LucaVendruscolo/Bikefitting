"""
Convert PyTorch models to ONNX format for browser inference.

This script converts:
1. Bike angle classifier (ConvNeXT) -> ONNX
2. YOLOv8 pose model -> ONNX  
3. YOLOv8 segmentation model -> ONNX

Run from the web folder:
    python convert_models.py

Models will be saved to public/models/
"""

import sys
import json
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torchvision import models

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_OUTPUT = Path(__file__).parent / "public" / "models"
MODELS_OUTPUT.mkdir(parents=True, exist_ok=True)


class AngleClassifier(nn.Module):
    """Same model architecture as training."""
    
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
        logits = self.head(features)
        # Return softmax probabilities for easier JS processing
        return torch.softmax(logits, dim=1)


def convert_bike_angle_model():
    """Convert the bike angle classifier to ONNX."""
    print("Converting bike angle model...")
    
    model_path = PROJECT_ROOT / "bike_angle_detection_model" / "models" / "optuna_best" / "best_model.pt"
    
    if not model_path.exists():
        print(f"  ERROR: Model not found at {model_path}")
        return False
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    num_bins = checkpoint['num_bins']
    backbone = checkpoint['backbone']
    
    print(f"  Backbone: {backbone}, Bins: {num_bins}")
    
    # Create model and load weights
    model = AngleClassifier(backbone, num_bins)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input (batch=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    output_path = MODELS_OUTPUT / "bike_angle.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['probabilities'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'probabilities': {0: 'batch_size'}
        }
    )
    
    print(f"  Saved to {output_path}")
    
    # Save config for JS
    config = {
        "num_bins": num_bins,
        "backbone": backbone,
        "input_size": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
    config_path = MODELS_OUTPUT / "bike_angle_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to {config_path}")
    
    return True


def quantize_onnx_model(model_path: Path, output_path: Path):
    """Quantize ONNX model to int8 for smaller size and faster inference."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        print(f"  Quantizing {model_path.name}...")
        quantize_dynamic(
            str(model_path),
            str(output_path),
            weight_type=QuantType.QUInt8,
            optimize_model=True
        )
        
        original_size = model_path.stat().st_size / (1024 * 1024)
        quantized_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  Original: {original_size:.1f}MB -> Quantized: {quantized_size:.1f}MB ({100*quantized_size/original_size:.0f}%)")
        return True
    except ImportError:
        print("  WARNING: onnxruntime not installed, skipping quantization")
        return False
    except Exception as e:
        print(f"  WARNING: Quantization failed: {e}")
        return False


def convert_yolo_models(quantize=True):
    """Convert YOLO models to ONNX using ultralytics export."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ERROR: ultralytics not installed")
        return False
    
    # Pose model - use smaller input size (480) for faster inference
    print("Converting YOLOv8 pose model (optimized for browser)...")
    pose_path = PROJECT_ROOT / "joint_angle_detection" / "models" / "yolov8n-pose.pt"
    if not pose_path.exists():
        pose_path = PROJECT_ROOT / "bike_angle_detection_model" / "yolov8m-pose.pt"
    
    if pose_path.exists():
        model = YOLO(str(pose_path))
        # Use imgsz=480 for faster browser inference (vs 640)
        model.export(format='onnx', imgsz=480, simplify=True, dynamic=False)
        exported = pose_path.with_suffix('.onnx')
        if exported.exists():
            target = MODELS_OUTPUT / "yolov8-pose.onnx"
            exported.rename(target)
            print(f"  Saved to {target} (size: {target.stat().st_size / (1024*1024):.1f}MB)")
            
            # Quantize for smaller size
            if quantize:
                quantized_path = MODELS_OUTPUT / "yolov8-pose-quantized.onnx"
                if quantize_onnx_model(target, quantized_path):
                    # Replace original with quantized
                    target.unlink()
                    quantized_path.rename(target)
    else:
        print("  Downloading and converting yolov8n-pose...")
        model = YOLO("yolov8n-pose.pt")
        model.export(format='onnx', imgsz=480, simplify=True, dynamic=False)
        exported = Path("yolov8n-pose.onnx")
        if exported.exists():
            target = MODELS_OUTPUT / "yolov8-pose.onnx"
            exported.rename(target)
            print(f"  Saved to {target}")
            
            if quantize:
                quantized_path = MODELS_OUTPUT / "yolov8-pose-quantized.onnx"
                if quantize_onnx_model(target, quantized_path):
                    target.unlink()
                    quantized_path.rename(target)
    
    # Segmentation model - skip for now as it's not used in current inference
    print("Skipping segmentation model (not used in current inference pipeline)")
    
    return True


def main():
    print("=" * 50)
    print("Model Conversion for Browser Inference")
    print("=" * 50)
    print()
    
    success = True
    
    # Convert bike angle model
    if not convert_bike_angle_model():
        success = False
    
    print()
    
    # Convert YOLO models
    if not convert_yolo_models():
        success = False
    
    print()
    print("=" * 50)
    if success:
        print("Conversion complete! Models saved to public/models/")
    else:
        print("Some conversions failed. Check errors above.")
    print("=" * 50)


if __name__ == "__main__":
    main()
