"""
Bike Angle Detection - Method 3: Classification with Circular Soft Labels

This approach treats angle prediction as a CLASSIFICATION problem:
1. Discretize angles into N bins (e.g., 72 bins = 5° each)
2. Use soft Gaussian labels that wrap around the circle
3. Cross-entropy loss naturally handles uncertainty
4. Final angle = weighted average of bin centers

Key advantages over regression:
- Classification is often more stable and converges faster
- Soft labels capture uncertainty and handle circular wrap-around
- Can detect ambiguous orientations (multi-modal predictions)
- Networks are very good at classification tasks

Uses ConvNeXt-Tiny backbone (modern CNN that rivals Vision Transformers).

Usage:
    python 2_train.py --data_dir data
"""

import argparse
import os
import math
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

# Use non-interactive backend to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Train angle classifier with circular soft labels")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing preprocessed masked bike images and dataset.csv")
    parser.add_argument("--train_csv", type=str, default=None,
                        help="Path to training CSV (overrides data_dir/dataset.csv)")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="Path to validation CSV")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--output_dir", type=str, default="models/classification_bins",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--num_bins", type=int, default=72,
                        help="Number of angle bins (72 = 5° per bin)")
    parser.add_argument("--label_smoothing", type=float, default=18.0,
                        help="Gaussian smoothing sigma in degrees for soft labels")
    parser.add_argument("--backbone", type=str, default="convnext_tiny",
                        choices=["convnext_tiny", "convnext_small", "resnet50", "efficientnet_b0"],
                        help="Backbone architecture")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    return parser.parse_args()


def angle_to_soft_label(angle_deg: float, num_bins: int, sigma_deg: float = 15.0) -> np.ndarray:
    """
    Convert angle to soft circular label distribution.
    
    Creates a Gaussian distribution centered on the angle that wraps around
    the circle, so -180° and +180° are adjacent.
    
    Args:
        angle_deg: Angle in degrees (-180 to 180)
        num_bins: Number of bins
        sigma_deg: Standard deviation of Gaussian in degrees
        
    Returns:
        Soft label array of shape (num_bins,) that sums to 1
    """
    bin_size = 360.0 / num_bins
    bin_centers = np.linspace(-180 + bin_size/2, 180 - bin_size/2, num_bins)
    
    # Compute circular distance to each bin center
    diff = bin_centers - angle_deg
    # Wrap to [-180, 180]
    diff = np.arctan2(np.sin(np.radians(diff)), np.cos(np.radians(diff)))
    diff = np.degrees(diff)
    
    # Gaussian distribution
    soft_label = np.exp(-0.5 * (diff / sigma_deg) ** 2)
    
    # Normalize to sum to 1
    soft_label = soft_label / soft_label.sum()
    
    return soft_label.astype(np.float32)


def soft_label_to_angle(soft_label: torch.Tensor, num_bins: int) -> torch.Tensor:
    """
    Convert soft label distribution back to angle using circular mean.
    
    Uses the circular mean formula:
    angle = atan2(sum(sin(bin_angles) * probs), sum(cos(bin_angles) * probs))
    
    Args:
        soft_label: Probability distribution over bins, shape (B, num_bins)
        num_bins: Number of bins
        
    Returns:
        Angles in degrees, shape (B,)
    """
    bin_size = 360.0 / num_bins
    bin_centers = torch.linspace(-180 + bin_size/2, 180 - bin_size/2, num_bins)
    bin_centers = bin_centers.to(soft_label.device)
    bin_centers_rad = torch.deg2rad(bin_centers)
    
    # Circular mean
    sin_sum = (soft_label * torch.sin(bin_centers_rad)).sum(dim=-1)
    cos_sum = (soft_label * torch.cos(bin_centers_rad)).sum(dim=-1)
    
    angle_rad = torch.atan2(sin_sum, cos_sum)
    angle_deg = torch.rad2deg(angle_rad)
    
    return angle_deg


class BikeAngleDataset(Dataset):
    """Dataset for bike angle classification with soft labels."""
    
    def __init__(self, csv_path: str, base_dir: str = None, img_size: int = 224,
                 num_bins: int = 72, label_smoothing: float = 15.0, augment: bool = False):
        self.df = pd.read_csv(csv_path)
        self.base_dir = Path(base_dir) if base_dir else Path(csv_path).parent
        self.img_size = img_size
        self.num_bins = num_bins
        self.label_smoothing = label_smoothing
        self.augment = augment
        
        # Define transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                normalize
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.base_dir / row["frame_path"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Get angle
        angle_deg = row["bike_angle_deg"]
        
        # Random horizontal flip (reverses angle sign)
        if self.augment and np.random.random() < 0.5:
            image = np.fliplr(image).copy()
            angle_deg = -angle_deg
        
        # Apply transforms
        image = self.transform(image)
        
        # Create soft label
        soft_label = angle_to_soft_label(angle_deg, self.num_bins, self.label_smoothing)
        soft_label = torch.from_numpy(soft_label)
        
        return image, soft_label, angle_deg


class AngleClassifier(nn.Module):
    """
    Angle classifier with circular soft labels.
    Outputs probability distribution over angle bins.
    """
    
    def __init__(self, num_bins: int = 72, backbone_name: str = "convnext_tiny"):
        super().__init__()
        self.num_bins = num_bins
        
        # Load pretrained backbone
        if backbone_name == "convnext_tiny":
            self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            feature_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name == "convnext_small":
            self.backbone = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
            feature_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_bins)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        # Flatten if needed (ConvNeXt outputs [B, C, 1, 1])
        if len(features.shape) == 4:
            features = features.flatten(1)
        logits = self.head(features)
        return logits
    
    def predict_angle(self, x):
        """Predict angle in degrees."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            angle_deg = soft_label_to_angle(probs, self.num_bins)
        return angle_deg


class SoftCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for soft labels.
    Equivalent to KL divergence when target is a probability distribution.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, soft_targets):
        """
        Args:
            logits: Raw model output, shape (B, num_bins)
            soft_targets: Target probability distribution, shape (B, num_bins)
        """
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return loss


def angular_error_deg(pred_probs: torch.Tensor, target_angles: torch.Tensor, 
                      num_bins: int) -> torch.Tensor:
    """Compute angular error in degrees."""
    pred_angles = soft_label_to_angle(pred_probs, num_bins)
    
    # Circular difference
    diff = pred_angles - target_angles
    diff_rad = torch.deg2rad(diff)
    diff_wrapped = torch.rad2deg(torch.atan2(torch.sin(diff_rad), torch.cos(diff_rad)))
    
    return torch.abs(diff_wrapped)


def train_epoch(model, dataloader, criterion, optimizer, device, num_bins):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_error = 0
    
    for images, soft_labels, angles in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        soft_labels = soft_labels.to(device)
        angles = angles.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        
        loss = criterion(logits, soft_labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
        # Compute angular error
        probs = F.softmax(logits, dim=-1)
        errors = angular_error_deg(probs, angles, num_bins)
        total_error += errors.sum().item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_error = total_error / len(dataloader.dataset)
    
    return avg_loss, avg_error


def validate(model, dataloader, criterion, device, num_bins):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_error = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, soft_labels, angles in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            soft_labels = soft_labels.to(device)
            angles = angles.to(device)
            
            logits = model(images)
            loss = criterion(logits, soft_labels)
            
            total_loss += loss.item() * images.size(0)
            
            probs = F.softmax(logits, dim=-1)
            pred_angles = soft_label_to_angle(probs, num_bins)
            errors = angular_error_deg(probs, angles, num_bins)
            total_error += errors.sum().item()
            
            all_preds.extend(pred_angles.cpu().numpy())
            all_targets.extend(angles.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_error = total_error / len(dataloader.dataset)
    
    return avg_loss, avg_error, all_preds, all_targets, np.concatenate(all_probs)


def plot_training_curves(train_losses, val_losses, train_errors, val_errors, output_path):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_errors, label="Train")
    ax2.plot(val_errors, label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Angular Error (degrees)")
    ax2.set_title("Mean Angular Error")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_predictions(targets, predictions, output_path):
    """Plot predicted vs actual angles."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.scatter(targets, predictions, alpha=0.5, s=10)
    ax1.plot([-180, 180], [-180, 180], 'r--', label="Perfect prediction")
    ax1.set_xlabel("True Angle (degrees)")
    ax1.set_ylabel("Predicted Angle (degrees)")
    ax1.set_title("Predicted vs True Bike Angle")
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-180, 180)
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect('equal')
    
    errors = np.array(predictions) - np.array(targets)
    errors = np.degrees(np.arctan2(np.sin(np.radians(errors)), np.cos(np.radians(errors))))
    
    ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_xlabel("Prediction Error (degrees)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Error Distribution (Mean: {np.mean(np.abs(errors)):.1f}°)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_probability_examples(images_batch, probs_batch, targets_batch, preds_batch, 
                              num_bins, output_path, num_examples=6):
    """Plot example probability distributions."""
    fig, axes = plt.subplots(2, num_examples, figsize=(3*num_examples, 6))
    
    bin_size = 360.0 / num_bins
    bin_centers = np.linspace(-180 + bin_size/2, 180 - bin_size/2, num_bins)
    
    for i in range(min(num_examples, len(probs_batch))):
        # Probability distribution
        axes[0, i].bar(bin_centers, probs_batch[i], width=bin_size*0.8, alpha=0.7)
        axes[0, i].axvline(x=targets_batch[i], color='g', linestyle='--', 
                          label=f'True: {targets_batch[i]:.0f}°')
        axes[0, i].axvline(x=preds_batch[i], color='r', linestyle='-', 
                          label=f'Pred: {preds_batch[i]:.0f}°')
        axes[0, i].set_xlim(-180, 180)
        axes[0, i].set_xlabel("Angle (°)")
        axes[0, i].set_ylabel("Probability")
        axes[0, i].legend(fontsize=8)
        axes[0, i].set_title(f"Sample {i+1}")
        
        # Polar plot
        ax_polar = fig.add_subplot(2, num_examples, num_examples + i + 1, projection='polar')
        theta = np.radians(bin_centers)
        ax_polar.bar(theta, probs_batch[i], width=np.radians(bin_size)*0.8, alpha=0.7)
        ax_polar.set_theta_zero_location('N')
        ax_polar.set_theta_direction(-1)
        
        # Mark true and predicted
        ax_polar.annotate('', xy=(np.radians(targets_batch[i]), 0.8*probs_batch[i].max()), 
                         xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax_polar.annotate('', xy=(np.radians(preds_batch[i]), probs_batch[i].max()), 
                         xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        axes[1, i].axis('off')  # Hide the cartesian subplot
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data - flexible to find the dataset
    data_dir = Path(args.data_dir)
    
    # Try to find the dataset CSV
    if args.train_csv:
        train_csv = Path(args.train_csv)
    elif (data_dir / "dataset.csv").exists():
        train_csv = data_dir / "dataset.csv"
    else:
        raise FileNotFoundError(
            f"No dataset found. Please specify --train_csv or ensure "
            f"{data_dir / 'dataset.csv'} exists."
        )
    
    val_csv = args.val_csv
    
    # If no validation CSV, split train data
    if val_csv is None:
        print(f"Loading dataset from: {train_csv}")
        df = pd.read_csv(train_csv)
        
        # Create train/val split (85/15)
        val_size = int(0.15 * len(df))
        np.random.seed(42)  # Fixed seed for reproducibility
        indices = np.random.permutation(len(df))
        train_df = df.iloc[indices[val_size:]]
        val_df = df.iloc[indices[:val_size]]
        
        # Save splits to output dir
        train_csv = output_dir / "train_split.csv"
        val_csv = output_dir / "val_split.csv"
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        print(f"Created train/val split: {len(train_df)}/{len(val_df)} samples")
        print(f"  Train CSV: {train_csv}")
        print(f"  Val CSV: {val_csv}")
    
    # Create datasets
    train_dataset = BikeAngleDataset(
        str(train_csv), 
        base_dir=str(data_dir),
        img_size=args.img_size,
        num_bins=args.num_bins,
        label_smoothing=args.label_smoothing,
        augment=True
    )
    val_dataset = BikeAngleDataset(
        str(val_csv),
        base_dir=str(data_dir),
        img_size=args.img_size,
        num_bins=args.num_bins,
        label_smoothing=args.label_smoothing,
        augment=False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of angle bins: {args.num_bins} ({360/args.num_bins:.1f}° per bin)")
    print(f"Label smoothing sigma: {args.label_smoothing}°")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print(f"\nInitializing model with backbone: {args.backbone}")
    model = AngleClassifier(
        num_bins=args.num_bins,
        backbone_name=args.backbone
    )
    model = model.to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = SoftCrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print(f"\nTraining on {args.device} for {args.epochs} epochs...")
    best_val_error = float('inf')
    train_losses, val_losses = [], []
    train_errors, val_errors = [], []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs} (lr: {scheduler.get_last_lr()[0]:.2e})")
        
        train_loss, train_error = train_epoch(
            model, train_loader, criterion, optimizer, args.device, args.num_bins)
        train_losses.append(train_loss)
        train_errors.append(train_error)
        
        val_loss, val_error, preds, targets, probs = validate(
            model, val_loader, criterion, args.device, args.num_bins)
        val_losses.append(val_loss)
        val_errors.append(val_error)
        
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}, Train Error: {train_error:.2f}°")
        print(f"  Val Loss: {val_loss:.4f}, Val Error: {val_error:.2f}°")
        
        # Save best model
        if val_error < best_val_error:
            best_val_error = val_error
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_error': val_error,
                'num_bins': args.num_bins,
                'backbone': args.backbone,
            }, output_dir / "best_model.pt")
            print(f"  ✓ New best model saved (error: {val_error:.2f}°)")
            
            # Save plots
            plot_predictions(targets, preds, output_dir / "best_predictions.png")
            plot_probability_examples(None, probs[:6], targets[:6], preds[:6],
                                     args.num_bins, output_dir / "probability_examples.png")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_error': val_error,
        'num_bins': args.num_bins,
        'backbone': args.backbone,
    }, output_dir / "final_model.pt")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_errors, val_errors,
                         output_dir / "training_curves.png")
    
    # Save history
    history_df = pd.DataFrame({
        'epoch': range(1, args.epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_error_deg': train_errors,
        'val_error_deg': val_errors
    })
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    
    print(f"\n=== Training Complete ===")
    print(f"Best validation error: {best_val_error:.2f}°")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

