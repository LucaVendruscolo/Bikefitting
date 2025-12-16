"""
Bike Angle Detection - Inference Script

Use trained models to predict bike angles from images.
Supports both Method 1 (YOLO sin/cos) and Method 2 (EfficientNet wrapped).

Usage:
    # Single image
    python 3_inference.py --model models/best_model/best_model.pt --image path/to/bike.jpg
    
    # Batch inference on CSV
    python 3_inference.py --model models/best_model/best_model.pt --csv data/dataset.csv --output predictions.csv
"""

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with trained bike angle models")
    
    # Single model inference
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, choices=["yolo_sincos", "efficientnet_wrapped", "auto"],
                        default="auto", help="Model type (auto-detected from checkpoint)")
    
    # Input options
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--csv", type=str, help="Path to CSV with frame_path column")
    parser.add_argument("--base_dir", type=str, help="Base directory for resolving frame paths in CSV")
    
    # Output
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Output CSV for batch predictions")
    
    # Compare mode
    parser.add_argument("--compare", action="store_true", help="Compare two models")
    parser.add_argument("--model1", type=str, help="First model for comparison")
    parser.add_argument("--model2", type=str, help="Second model for comparison")
    
    # Other options
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visualize", action="store_true", help="Show visualization for single image")
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str = "cpu"):
    """
    Load a trained model from checkpoint.
    Auto-detects model type from checkpoint contents.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Determine model type
    state_dict = checkpoint['model_state_dict']
    
    # Check if it's EfficientNet (has 'backbone' key with efficientnet layers)
    is_efficientnet = any('backbone.features' in k for k in state_dict.keys())
    
    if is_efficientnet or checkpoint.get('backbone', None):
        # Method 2: EfficientNet
        from train_efficientnet_wrapped import EfficientNetAngleModel
        backbone = checkpoint.get('backbone', 'efficientnet_b0')
        model = EfficientNetAngleModel(backbone_name=backbone, freeze_backbone=False)
        model_type = "efficientnet_wrapped"
    else:
        # Method 1: YOLO
        from train_yolo_sincos import YOLOAngleModel
        model = YOLOAngleModel(pretrained_path="yolov8n-cls.pt", freeze_backbone=False)
        model_type = "yolo_sincos"
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {model_type} model from {checkpoint_path}")
    print(f"  Validation error at save: {checkpoint.get('val_error', 'N/A'):.2f}°")
    
    return model, model_type


def preprocess_image(image_path: str, img_size: int = 224):
    """Load and preprocess a single image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)


def predict_angle(model, image_tensor, model_type: str, device: str = "cpu"):
    """
    Predict bike angle from preprocessed image tensor.
    Returns angle in degrees.
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        if model_type == "yolo_sincos":
            output = model(image_tensor)  # Shape: (B, 2) [sin, cos]
            angle_rad = torch.atan2(output[:, 0], output[:, 1])
        else:  # efficientnet_wrapped
            angle_rad, _ = model(image_tensor)  # Returns (angle, sincos)
            # Wrap to [-π, π]
            angle_rad = torch.atan2(torch.sin(angle_rad), torch.cos(angle_rad))
        
        angle_deg = torch.rad2deg(angle_rad)
    
    return angle_deg.cpu().numpy()


def predict_single_image(model, model_type: str, image_path: str, 
                         img_size: int = 224, device: str = "cpu", visualize: bool = False):
    """Predict angle for a single image."""
    image_tensor = preprocess_image(image_path, img_size)
    angle = predict_angle(model, image_tensor, model_type, device)[0]
    
    print(f"\nPredicted angle: {angle:.1f}°")
    print(f"  Interpretation:")
    print(f"    0° = facing camera")
    print(f"    90° = facing right")
    print(f"    -90° = facing left")
    print(f"    ±180° = facing away")
    
    if visualize:
        import matplotlib.pyplot as plt
        
        # Load original image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        ax1.imshow(image)
        ax1.set_title(f"Input Image\nPredicted: {angle:.1f}°")
        ax1.axis('off')
        
        # Show angle on polar plot
        ax2 = plt.subplot(122, projection='polar')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.annotate('', xy=(math.radians(angle), 1), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='red', lw=3))
        ax2.set_rticks([])
        ax2.set_title("Bike Orientation\n(0° = facing camera)")
        
        # Mark cardinal directions
        angles = [0, 90, -90, 180]
        labels = ['0° (facing)', '90° (right)', '-90° (left)', '180° (away)']
        for a, l in zip(angles, labels):
            ax2.annotate(l, xy=(math.radians(a), 1.15), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    return angle


def predict_batch(model, model_type: str, csv_path: str, base_dir: str = None,
                  img_size: int = 224, device: str = "cpu"):
    """Predict angles for all images in a CSV."""
    df = pd.read_csv(csv_path)
    base_dir = Path(base_dir) if base_dir else Path(csv_path).parent
    
    predictions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        img_path = base_dir / row["frame_path"]
        
        try:
            image_tensor = preprocess_image(str(img_path), img_size)
            angle = predict_angle(model, image_tensor, model_type, device)[0]
            predictions.append(angle)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            predictions.append(np.nan)
    
    df['predicted_angle'] = predictions
    
    # If ground truth exists, compute error
    if 'bike_angle_deg' in df.columns:
        true_angles = df['bike_angle_deg'].values
        pred_angles = np.array(predictions)
        
        # Compute angular error (handling wrap-around)
        diff = pred_angles - true_angles
        diff_wrapped = np.degrees(np.arctan2(np.sin(np.radians(diff)), np.cos(np.radians(diff))))
        df['error_deg'] = np.abs(diff_wrapped)
        
        mean_error = np.nanmean(df['error_deg'])
        median_error = np.nanmedian(df['error_deg'])
        print(f"\n=== Results ===")
        print(f"Mean angular error: {mean_error:.2f}°")
        print(f"Median angular error: {median_error:.2f}°")
    
    return df


def compare_models(model1_path: str, model2_path: str, csv_path: str,
                   base_dir: str = None, img_size: int = 224, device: str = "cpu"):
    """Compare predictions from two models."""
    
    # Load both models
    model1, type1 = load_model(model1_path, device)
    model2, type2 = load_model(model2_path, device)
    
    # Load data
    df = pd.read_csv(csv_path)
    base_dir = Path(base_dir) if base_dir else Path(csv_path).parent
    
    preds1, preds2 = [], []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Comparing models"):
        img_path = base_dir / row["frame_path"]
        
        try:
            image_tensor = preprocess_image(str(img_path), img_size)
            
            angle1 = predict_angle(model1, image_tensor, type1, device)[0]
            angle2 = predict_angle(model2, image_tensor, type2, device)[0]
            
            preds1.append(angle1)
            preds2.append(angle2)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            preds1.append(np.nan)
            preds2.append(np.nan)
    
    df[f'pred_{type1}'] = preds1
    df[f'pred_{type2}'] = preds2
    
    # Compute errors if ground truth exists
    if 'bike_angle_deg' in df.columns:
        true_angles = df['bike_angle_deg'].values
        
        for name, preds in [(type1, preds1), (type2, preds2)]:
            pred_angles = np.array(preds)
            diff = pred_angles - true_angles
            diff_wrapped = np.degrees(np.arctan2(np.sin(np.radians(diff)), np.cos(np.radians(diff))))
            df[f'error_{name}'] = np.abs(diff_wrapped)
        
        print(f"\n=== Model Comparison ===")
        print(f"\n{type1}:")
        print(f"  Mean error: {np.nanmean(df[f'error_{type1}']):.2f}°")
        print(f"  Median error: {np.nanmedian(df[f'error_{type1}']):.2f}°")
        print(f"  Std: {np.nanstd(df[f'error_{type1}']):.2f}°")
        
        print(f"\n{type2}:")
        print(f"  Mean error: {np.nanmean(df[f'error_{type2}']):.2f}°")
        print(f"  Median error: {np.nanmedian(df[f'error_{type2}']):.2f}°")
        print(f"  Std: {np.nanstd(df[f'error_{type2}']):.2f}°")
        
        # Statistical comparison
        error1 = df[f'error_{type1}'].dropna()
        error2 = df[f'error_{type2}'].dropna()
        
        better_count1 = (error1 < error2).sum()
        better_count2 = (error2 < error1).sum()
        tie_count = (error1 == error2).sum()
        
        print(f"\n{type1} better: {better_count1} images ({100*better_count1/len(error1):.1f}%)")
        print(f"{type2} better: {better_count2} images ({100*better_count2/len(error2):.1f}%)")
        print(f"Tied: {tie_count} images")
    
    return df


def main():
    args = parse_args()
    
    if args.compare:
        # Compare two models
        if not args.model1 or not args.model2:
            raise ValueError("--compare requires --model1 and --model2")
        if not args.csv:
            raise ValueError("--compare requires --csv")
        
        df = compare_models(
            args.model1, args.model2, args.csv,
            base_dir=args.base_dir,
            img_size=args.img_size,
            device=args.device
        )
        df.to_csv(args.output, index=False)
        print(f"\nComparison results saved to {args.output}")
        
    elif args.model:
        # Single model inference
        model, model_type = load_model(args.model, args.device)
        
        if args.image:
            # Single image
            predict_single_image(
                model, model_type, args.image,
                img_size=args.img_size,
                device=args.device,
                visualize=args.visualize
            )
        elif args.csv:
            # Batch
            df = predict_batch(
                model, model_type, args.csv,
                base_dir=args.base_dir,
                img_size=args.img_size,
                device=args.device
            )
            df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to {args.output}")
        else:
            raise ValueError("Must provide --image or --csv")
    else:
        raise ValueError("Must provide --model or use --compare mode")


if __name__ == "__main__":
    main()

