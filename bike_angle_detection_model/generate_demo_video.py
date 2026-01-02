"""
Generate a demo video showing:
- Left: Original video frame with skeleton overlay
- Right top: Masked bike pixels (same as training)
- Right middle: Joint angles (knee, hip, elbow)
- Right bottom: Bike angle (True vs Predicted)

Uses the EXACT same preprocessing pipeline as training.

Run with --gui for interactive section picker, or use command line args.
"""

import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models, transforms
from ultralytics import YOLO
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from typing import Dict, Tuple, Optional
import sys

# Add parent directory to find joint_angle_detection
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============= Joint Angle Detection (from joint_angle_detection/core.py) =============

COCO_KPTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
IDX = {name: i for i, name in enumerate(COCO_KPTS)}

# Skeleton connections for drawing - only the side facing camera
SKELETON_LEFT = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_shoulder", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
]

SKELETON_RIGHT = [
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_shoulder", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


class KeypointSmoother:
    """Exponential moving average smoother for keypoint positions."""
    
    def __init__(self, alpha: float = 0.6):
        """
        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive, lower = smoother.
                   0.6 gives slight smoothing while staying responsive.
        """
        self.alpha = alpha
        self.prev_xy = None
        self.prev_conf = None
    
    def smooth(self, kpts_xy: np.ndarray, kpts_conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply exponential moving average to keypoints."""
        if self.prev_xy is None:
            # First frame, no smoothing
            self.prev_xy = kpts_xy.copy()
            self.prev_conf = kpts_conf.copy()
            return kpts_xy, kpts_conf
        
        # Apply EMA: smoothed = alpha * current + (1 - alpha) * previous
        smoothed_xy = self.alpha * kpts_xy + (1 - self.alpha) * self.prev_xy
        smoothed_conf = self.alpha * kpts_conf + (1 - self.alpha) * self.prev_conf
        
        # Update previous
        self.prev_xy = smoothed_xy.copy()
        self.prev_conf = smoothed_conf.copy()
        
        return smoothed_xy, smoothed_conf
    
    def reset(self):
        """Reset smoother state."""
        self.prev_xy = None
        self.prev_conf = None


def detect_joints(
    image: np.ndarray,
    model: YOLO,
    side: str = "auto",
    min_conf: float = 0.5,
) -> Tuple[Optional[Dict[str, Tuple[float, float, float]]], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Detect joints and return simplified joint dict plus raw keypoints for skeleton drawing.
    Returns: (joints_dict, raw_xy, raw_conf, detected_side)
    """
    side = side.lower()
    
    def side_map(which: str) -> Dict[str, str]:
        if which == "right":
            return {
                "shoulder": "right_shoulder", "elbow": "right_elbow",
                "hand": "right_wrist", "hip": "right_hip",
                "knee": "right_knee", "foot": "right_ankle",
            }
        else:
            return {
                "shoulder": "left_shoulder", "elbow": "left_elbow",
                "hand": "left_wrist", "hip": "left_hip",
                "knee": "left_knee", "foot": "left_ankle",
            }

    results = model.predict(image, verbose=False)
    if not results or results[0].keypoints is None:
        return None, None, None, None

    r = results[0]
    kpts_xy = r.keypoints.xy.cpu().numpy()
    kpts_cf = r.keypoints.conf.cpu().numpy()

    if kpts_cf.size == 0 or kpts_cf.shape[0] == 0:
        return None, None, None, None

    person_idx = int(np.argmax(kpts_cf.mean(axis=1)))
    
    # Get raw keypoints for this person
    raw_xy = kpts_xy[person_idx]
    raw_conf = kpts_cf[person_idx]

    def joints_for_side(which: str):
        mapping = side_map(which)
        joints_side = {}
        confidences = []
        for simple_name, coco_name in mapping.items():
            idx = IDX[coco_name]
            x, y = raw_xy[idx]
            c = raw_conf[idx]
            if c < min_conf:
                continue
            joints_side[simple_name] = (float(x), float(y), float(c))
            confidences.append(float(c))
        count = len(confidences)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        return joints_side, count, mean_conf

    if side in {"right", "left"}:
        joints, count, _ = joints_for_side(side)
        if count == 0:
            return None, raw_xy, raw_conf, None
        return joints, raw_xy, raw_conf, side

    # Auto-detect: compare left vs right and pick the side facing camera (more visible)
    joints_right, count_r, mean_r = joints_for_side("right")
    joints_left, count_l, mean_l = joints_for_side("left")

    if count_r == 0 and count_l == 0:
        return None, raw_xy, raw_conf, None

    if (count_r > count_l) or (count_r == count_l and mean_r >= mean_l):
        return joints_right if joints_right else None, raw_xy, raw_conf, "right"
    else:
        return joints_left if joints_left else None, raw_xy, raw_conf, "left"


def _angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Return angle ABC in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def compute_angles(joints: Dict[str, Tuple[float, float, float]]) -> Dict[str, float]:
    """Compute knee, hip, and elbow angles from joints."""
    def pt(name: str):
        if name not in joints:
            return None
        x, y, _ = joints[name]
        return (x, y)

    def maybe_angle(a: str, b: str, c: str):
        pa, pb, pc = pt(a), pt(b), pt(c)
        if pa is None or pb is None or pc is None:
            return None
        return _angle(pa, pb, pc)

    angles = {}
    k = maybe_angle("hip", "knee", "foot")
    if k is not None:
        angles["knee_angle"] = k
    h = maybe_angle("shoulder", "hip", "knee")
    if h is not None:
        angles["hip_angle"] = h
    e = maybe_angle("shoulder", "elbow", "hand")
    if e is not None:
        angles["elbow_angle"] = e
    return angles


def draw_skeleton(img: np.ndarray, kpts_xy: np.ndarray, kpts_conf: np.ndarray, 
                  detected_side: Optional[str] = None, min_conf: float = 0.3, 
                  scale: float = 1.0) -> np.ndarray:
    """Draw skeleton on image - only the side facing the camera (1080p optimized)."""
    vis = img.copy()
    
    # Choose which skeleton connections to draw based on detected side
    if detected_side == "right":
        main_connections = SKELETON_RIGHT
        side_joints = ["right_shoulder", "right_elbow", "right_wrist", 
                       "right_hip", "right_knee", "right_ankle"]
    elif detected_side == "left":
        main_connections = SKELETON_LEFT
        side_joints = ["left_shoulder", "left_elbow", "left_wrist",
                       "left_hip", "left_knee", "left_ankle"]
    else:
        # No side detected, don't draw
        return vis
    
    # Draw main side connections (bright yellow, thick for 1080p)
    for start_name, end_name in main_connections:
        start_idx = IDX[start_name]
        end_idx = IDX[end_name]
        
        if kpts_conf[start_idx] < min_conf or kpts_conf[end_idx] < min_conf:
            continue
            
        start_pt = (int(kpts_xy[start_idx][0] * scale), int(kpts_xy[start_idx][1] * scale))
        end_pt = (int(kpts_xy[end_idx][0] * scale), int(kpts_xy[end_idx][1] * scale))
        
        # Draw thick outline then main line for visibility
        cv2.line(vis, start_pt, end_pt, (0, 0, 0), 7)
        cv2.line(vis, start_pt, end_pt, (0, 255, 255), 4)
    
    # Draw only the joints for the detected side (larger for 1080p)
    for joint_name in side_joints:
        idx = IDX[joint_name]
        if kpts_conf[idx] < min_conf:
            continue
        x, y = kpts_xy[idx]
        pt = (int(x * scale), int(y * scale))
        cv2.circle(vis, pt, 12, (0, 0, 0), -1)  # Black outline
        cv2.circle(vis, pt, 10, (255, 0, 255), -1)  # Magenta fill
        cv2.circle(vis, pt, 10, (255, 255, 255), 2)  # White border
    
    return vis


# ============= Bike Angle Detection =============

class BikeSegmenter:
    """Identical to 1_preprocess.py - masks bike pixels only."""
    
    BIKE_CLASSES = [1, 3]
    
    def __init__(self, model_name="yolov8n-seg.pt", conf_threshold=0.3):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        
    def get_bike_mask(self, image, dilate_pixels=5):
        h, w = image.shape[:2]
        results = self.model(image, verbose=False, conf=self.conf_threshold)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for result in results:
            if result.masks is not None:
                for i, cls in enumerate(result.boxes.cls):
                    if int(cls.item()) in self.BIKE_CLASSES:
                        seg_mask = result.masks.data[i].cpu().numpy()
                        seg_mask = cv2.resize(seg_mask, (w, h))
                        mask = np.maximum(mask, (seg_mask > 0.5).astype(np.uint8) * 255)
        
        if mask.max() > 0 and dilate_pixels > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1))
            mask = cv2.dilate(mask, kernel, iterations=1)
            return mask, True
        
        return mask, False
    
    def mask_bike(self, image, target_size=224, dilate_pixels=5):
        mask, success = self.get_bike_mask(image, dilate_pixels)
        
        if not success:
            return np.zeros((target_size, target_size, 3), dtype=np.uint8), mask, False
        
        masked = cv2.bitwise_and(image, image, mask=mask)
        
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, bw, bh = cv2.boundingRect(coords)
            pad = 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            bw = min(image.shape[1] - x, bw + 2 * pad)
            bh = min(image.shape[0] - y, bh + 2 * pad)
            
            if bw > bh:
                diff = bw - bh
                y = max(0, y - diff // 2)
                bh = bw
            else:
                diff = bh - bw
                x = max(0, x - diff // 2)
                bw = bh
            
            if y + bh > image.shape[0]:
                bh = image.shape[0] - y
            if x + bw > image.shape[1]:
                bw = image.shape[1] - x
            
            masked = masked[y:y+bh, x:x+bw]
        
        if masked.shape[0] > 0 and masked.shape[1] > 0:
            masked = cv2.resize(masked, (target_size, target_size))
        else:
            masked = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            success = False
        
        return masked, mask, success


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


def load_model(model_path, device):
    """Load trained model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'num_bins' not in checkpoint:
        raise ValueError(
            f"Invalid model file: {model_path}\n"
            "This doesn't appear to be a trained angle classifier.\n"
            "Expected: models/optuna_best/best_model.pt\n"
            "You may have selected a YOLO model by mistake."
        )
    
    num_bins = checkpoint['num_bins']
    backbone = checkpoint['backbone']
    
    model = AngleClassifier(backbone, num_bins)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    return model, num_bins


def predict_angle(model, masked_img, num_bins, device, transform):
    """Run inference - same as training."""
    img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
    img_t = transform(img_rgb).unsqueeze(0).to(device)
    
    bin_size = 360.0 / num_bins
    bin_centers = torch.linspace(-180 + bin_size/2, 180 - bin_size/2, num_bins).to(device)
    bin_centers_rad = torch.deg2rad(bin_centers)
    
    with torch.no_grad():
        out = model(img_t)
        probs = torch.softmax(out, dim=1)
        
        sin_sum = (probs * torch.sin(bin_centers_rad)).sum(dim=-1)
        cos_sum = (probs * torch.cos(bin_centers_rad)).sum(dim=-1)
        pred = torch.rad2deg(torch.atan2(sin_sum, cos_sum)).item()
        
        confidence = probs.max().item() * 100
    
    return pred, confidence


def draw_angle_indicator(img, angle, label, color, center, radius):
    """Draw a circular angle indicator."""
    # Outer circle
    cv2.circle(img, center, radius, (100, 100, 100), 2)
    
    # Angle line
    angle_rad = np.radians(-angle + 90)
    end_x = int(center[0] + radius * 0.8 * np.cos(angle_rad))
    end_y = int(center[1] - radius * 0.8 * np.sin(angle_rad))
    cv2.line(img, center, (end_x, end_y), color, 4)
    
    # Center dot
    cv2.circle(img, center, 6, color, -1)
    
    # Label
    cv2.putText(img, label, (center[0] - 25, center[1] + radius + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # Value
    cv2.putText(img, f"{angle:.1f}", (center[0] - 30, center[1] + radius + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_joint_angles_panel(img, angles: Dict[str, float], detected_side: Optional[str],
                            x: int, y: int, width: int):
    """Draw a panel showing joint angles for the detected side."""
    panel_height = 160
    
    # Background
    cv2.rectangle(img, (x, y), (x + width, y + panel_height), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + width, y + panel_height), (80, 80, 80), 2)
    
    # Title with side indicator
    if detected_side:
        side_text = f"Joint Angles ({detected_side.upper()} side)"
        title_color = (255, 255, 255)
    else:
        side_text = "Joint Angles (no detection)"
        title_color = (100, 100, 100)
    
    cv2.putText(img, side_text, (x + 15, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2)
    
    # Angles
    angle_y = y + 70
    angle_names = [("knee_angle", "Knee", (50, 255, 255)),
                   ("hip_angle", "Hip", (255, 150, 50)),
                   ("elbow_angle", "Elbow", (150, 255, 150))]
    
    for angle_key, label, color in angle_names:
        if angle_key in angles:
            value = angles[angle_key]
            text = f"{label}: {value:.1f} deg"
        else:
            text = f"{label}: --"
            color = (100, 100, 100)
        
        cv2.putText(img, text, (x + 20, angle_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        angle_y += 32
    
    return panel_height


def generate_demo(video_path, labels_csv, model_path, output_path, 
                  start_frame=0, duration_sec=30, output_fps=10, progress_callback=None):
    """Generate demo video with bike angle and joint angle detection."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load bike angle model
    print("Loading bike angle model...")
    model, num_bins = load_model(model_path, device)
    
    # Load YOLO models
    print("Loading YOLO segmenter...")
    segmenter = BikeSegmenter()
    
    print("Loading pose model (yolov8m-pose for better accuracy)...")
    pose_model_path = Path(__file__).parent.parent / "joint_angle_detection" / "models" / "yolov8m-pose.pt"
    if not pose_model_path.exists():
        pose_model_path = "yolov8m-pose.pt"  # Will download if needed
    pose_model = YOLO(str(pose_model_path))
    
    # Transform (identical to training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load labels
    print("Loading labels...")
    df = pd.read_csv(labels_csv)
    video_name = Path(video_path).name
    df_video = df[df['source_video'] == video_name].copy()
    df_video = df_video.sort_values('frame_number')
    
    if len(df_video) == 0:
        print(f"No labels found for video: {video_name}")
        return None
    
    print(f"Found {len(df_video)} labeled frames for {video_name}")
    
    frame_to_angle = dict(zip(df_video['frame_number'], df_video['bike_angle_deg']))
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_fps:.1f} FPS, {total_frames} frames, {orig_width}x{orig_height}")
    
    frame_skip = max(1, int(video_fps / output_fps))
    frames_to_process = min(int(duration_sec * output_fps), 
                            (total_frames - start_frame) // frame_skip)
    
    print(f"Processing {frames_to_process} frames (skip={frame_skip})")
    
    # Output video setup - 1080p main video with side panel
    # Main video display: 1920x1080 (or scaled to fit 1080p height maintaining aspect ratio)
    main_display_height = 1080
    main_display_width = int(orig_width * (main_display_height / orig_height))
    
    # Ensure we don't exceed reasonable width
    if main_display_width > 1920:
        main_display_width = 1920
        main_display_height = int(orig_height * (main_display_width / orig_width))
    
    # Side panel for controls
    panel_width = 420
    
    out_width = main_display_width + panel_width
    out_height = max(main_display_height, 1080)
    
    print(f"Output resolution: {out_width}x{out_height} (main video: {main_display_width}x{main_display_height})")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (out_width, out_height))
    
    errors = []
    
    # Scale factor for skeleton drawing (from original to display size)
    scale = main_display_height / orig_height
    
    # Keypoint smoother for reducing jitter (alpha=0.6 for slight smoothing)
    smoother = KeypointSmoother(alpha=0.6)
    
    for i in tqdm(range(frames_to_process), desc="Generating video"):
        frame_num = start_frame + i * frame_skip
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Get ground truth bike angle
        true_angle = frame_to_angle.get(frame_num, None)
        
        # Detect joints (only the side facing the camera)
        joints, kpts_xy, kpts_conf, detected_side = detect_joints(frame, pose_model)
        
        # Apply temporal smoothing to reduce jitter
        if kpts_xy is not None and kpts_conf is not None:
            kpts_xy, kpts_conf = smoother.smooth(kpts_xy, kpts_conf)
            
            # Recompute joints dict with smoothed values for angle calculation
            if detected_side and joints:
                side_map = {
                    "right": {"shoulder": "right_shoulder", "elbow": "right_elbow",
                              "hand": "right_wrist", "hip": "right_hip",
                              "knee": "right_knee", "foot": "right_ankle"},
                    "left": {"shoulder": "left_shoulder", "elbow": "left_elbow",
                             "hand": "left_wrist", "hip": "left_hip",
                             "knee": "left_knee", "foot": "left_ankle"}
                }
                mapping = side_map[detected_side]
                for simple_name, coco_name in mapping.items():
                    idx = IDX[coco_name]
                    if kpts_conf[idx] >= 0.5:
                        joints[simple_name] = (float(kpts_xy[idx][0]), float(kpts_xy[idx][1]), float(kpts_conf[idx]))
        joint_angles = compute_angles(joints) if joints else {}
        
        # Apply bike masking
        masked, mask, success = segmenter.mask_bike(frame, target_size=224)
        
        # Predict bike angle
        if success:
            pred_angle, confidence = predict_angle(model, masked, num_bins, device, transform)
        else:
            pred_angle, confidence = 0, 0
        
        # Track error
        if true_angle is not None:
            error = abs(pred_angle - true_angle)
            if error > 180:
                error = 360 - error
            errors.append(error)
        
        # Create output frame
        output = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        output[:] = (25, 25, 25)
        
        # Resize original for 1080p display (joint detection uses full resolution)
        orig_display = cv2.resize(frame, (main_display_width, main_display_height))
        
        # Draw skeleton on original (only the side facing camera)
        # Note: skeleton is drawn at display scale, joints detected at full resolution
        if kpts_xy is not None and kpts_conf is not None and detected_side:
            orig_display = draw_skeleton(orig_display, kpts_xy, kpts_conf, 
                                         detected_side=detected_side, scale=scale)
        
        # Draw bike mask outline at 1080p
        if mask is not None and mask.max() > 0:
            mask_resized = cv2.resize(mask, (main_display_width, main_display_height))
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(orig_display, contours, -1, (0, 255, 0), 3)
        
        # Place original frame (left side, full height)
        output[0:main_display_height, 0:main_display_width] = orig_display
        
        # Title overlay on main video
        cv2.putText(output, "1080p Video + Skeleton + Bike Mask", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(output, "1080p Video + Skeleton + Bike Mask", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(output, "1080p Video + Skeleton + Bike Mask", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Right panel start
        panel_x = main_display_width + 20
        panel_content_width = panel_width - 40
        
        # Masked bike image (top right) - kept at 224x224 display (actual model input)
        masked_display_size = 280
        masked_display = cv2.resize(masked, (masked_display_size, masked_display_size))
        output[50:50+masked_display_size, panel_x:panel_x+masked_display_size] = masked_display
        cv2.putText(output, "Masked Bike (224x224 model input)", (panel_x, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        # Joint angles panel (middle right) - shows which side is detected
        joint_panel_y = 50 + masked_display_size + 30
        draw_joint_angles_panel(output, joint_angles, detected_side, panel_x, joint_panel_y, 300)
        
        # Bike angle indicators (bottom right)
        indicator_y = joint_panel_y + 180
        
        cv2.putText(output, "Bike Angle", (panel_x + 80, indicator_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        if true_angle is not None:
            draw_angle_indicator(output, true_angle, "True", (255, 150, 50), 
                               (panel_x + 70, indicator_y + 80), 55)
        
        draw_angle_indicator(output, pred_angle, "Pred", (50, 255, 100), 
                           (panel_x + 210, indicator_y + 80), 55)
        
        # Frame info at bottom of panel
        cv2.putText(output, f"Frame: {frame_num}", (panel_x, out_height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
        
        if confidence > 0:
            cv2.putText(output, f"Bike Conf: {confidence:.0f}%", (panel_x, out_height - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        
        out.write(output)
        
        if progress_callback:
            progress_callback(i + 1, frames_to_process)
    
    cap.release()
    out.release()
    
    if errors:
        errors = np.array(errors)
        print(f"\n=== Bike Angle Results ===")
        print(f"Mean Absolute Error: {np.mean(errors):.2f} deg")
        print(f"Median Error: {np.median(errors):.2f} deg")
        print(f"95th percentile: {np.percentile(errors, 95):.2f} deg")
    
    print(f"\nDemo video saved to: {output_path}")
    return errors


class DemoVideoGUI:
    """GUI for selecting video section and generating demo."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Demo Video Generator")
        self.root.geometry("900x700")
        self.root.configure(bg='#2b2b2b')
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white', font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 10))
        style.configure('TScale', background='#2b2b2b')
        
        self.video_path = None
        self.labels_csv = None
        self.model_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 30
        self.current_frame = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Video:").grid(row=0, column=0, sticky='w', padx=5)
        self.video_label = ttk.Label(file_frame, text="No video selected", width=60)
        self.video_label.grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_video).grid(row=0, column=2, padx=5)
        
        ttk.Label(file_frame, text="Labels:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.labels_label = ttk.Label(file_frame, text="No labels selected", width=60)
        self.labels_label.grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_labels).grid(row=1, column=2, padx=5)
        
        ttk.Label(file_frame, text="Model:").grid(row=2, column=0, sticky='w', padx=5)
        self.model_label = ttk.Label(file_frame, text="models/optuna_best/best_model.pt", width=60)
        self.model_label.grid(row=2, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_model).grid(row=2, column=2, padx=5)
        self.model_path = "models/optuna_best/best_model.pt"
        
        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.canvas = tk.Canvas(preview_frame, width=640, height=360, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack()
        
        slider_frame = ttk.Frame(main_frame)
        slider_frame.pack(fill=tk.X, pady=5)
        
        self.timeline_var = tk.IntVar(value=0)
        self.timeline = ttk.Scale(slider_frame, from_=0, to=100, variable=self.timeline_var,
                                  orient=tk.HORIZONTAL, command=self.on_timeline_change)
        self.timeline.pack(fill=tk.X, padx=10)
        
        time_frame = ttk.Frame(main_frame)
        time_frame.pack(fill=tk.X)
        
        self.time_label = ttk.Label(time_frame, text="00:00 / 00:00  |  Frame: 0 / 0")
        self.time_label.pack()
        
        range_frame = ttk.Frame(main_frame)
        range_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(range_frame, text="Start Time (sec):").grid(row=0, column=0, padx=5)
        self.start_var = tk.StringVar(value="0")
        self.start_entry = ttk.Entry(range_frame, textvariable=self.start_var, width=10)
        self.start_entry.grid(row=0, column=1, padx=5)
        ttk.Button(range_frame, text="Set to Current", command=self.set_start_current).grid(row=0, column=2, padx=5)
        
        ttk.Label(range_frame, text="Duration (sec):").grid(row=0, column=3, padx=20)
        self.duration_var = tk.StringVar(value="30")
        self.duration_entry = ttk.Entry(range_frame, textvariable=self.duration_var, width=10)
        self.duration_entry.grid(row=0, column=4, padx=5)
        
        ttk.Label(range_frame, text="Output FPS:").grid(row=0, column=5, padx=20)
        self.fps_var = tk.StringVar(value="10")
        self.fps_entry = ttk.Entry(range_frame, textvariable=self.fps_var, width=10)
        self.fps_entry.grid(row=0, column=6, padx=5)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=10, padx=10)
        
        self.generate_btn = ttk.Button(main_frame, text="Generate Demo Video", command=self.generate)
        self.generate_btn.pack(pady=10)
        
        self.status_label = ttk.Label(main_frame, text="Select a video to begin")
        self.status_label.pack()
        
    def browse_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.MOV"), ("All files", "*.*")]
        )
        if path:
            self.video_path = path
            self.video_label.config(text=Path(path).name)
            self.load_video()
            
    def browse_labels(self):
        path = filedialog.askopenfilename(
            title="Select Labels CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.labels_csv = path
            self.labels_label.config(text=Path(path).name)
            
    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[("PyTorch models", "*.pt *.pth"), ("All files", "*.*")]
        )
        if path:
            self.model_path = path
            self.model_label.config(text=Path(path).name)
            
    def load_video(self):
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.timeline.config(to=self.total_frames - 1)
        self.timeline_var.set(0)
        
        self.show_frame(0)
        self.update_time_label()
        self.status_label.config(text=f"Loaded: {self.total_frames} frames @ {self.fps:.1f} FPS")
        
    def show_frame(self, frame_num):
        if not self.cap:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame_num
            frame = cv2.resize(frame, (640, 360))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            img = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
    def on_timeline_change(self, value):
        frame_num = int(float(value))
        self.show_frame(frame_num)
        self.update_time_label()
        
    def update_time_label(self):
        if self.fps > 0:
            current_time = self.current_frame / self.fps
            total_time = self.total_frames / self.fps
            self.time_label.config(
                text=f"{int(current_time//60):02d}:{int(current_time%60):02d} / "
                     f"{int(total_time//60):02d}:{int(total_time%60):02d}  |  "
                     f"Frame: {self.current_frame} / {self.total_frames}"
            )
            
    def set_start_current(self):
        if self.fps > 0:
            self.start_var.set(f"{self.current_frame / self.fps:.1f}")
            
    def update_progress(self, current, total):
        self.progress_var.set((current / total) * 100)
        self.root.update_idletasks()
        
    def generate(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video first")
            return
        if not self.labels_csv:
            messagebox.showerror("Error", "Please select a labels CSV")
            return
        if not self.model_path:
            messagebox.showerror("Error", "Please select a model")
            return
            
        try:
            start_sec = float(self.start_var.get())
            duration = float(self.duration_var.get())
            output_fps = int(self.fps_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid start time, duration, or FPS")
            return
            
        start_frame = int(start_sec * self.fps)
        
        output_path = filedialog.asksaveasfilename(
            title="Save Demo Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")],
            initialfile="demo_video.mp4"
        )
        
        if not output_path:
            return
            
        self.generate_btn.config(state='disabled')
        self.status_label.config(text="Generating...")
        self.progress_var.set(0)
        self.root.update()
        
        try:
            errors = generate_demo(
                video_path=self.video_path,
                labels_csv=self.labels_csv,
                model_path=self.model_path,
                output_path=output_path,
                start_frame=start_frame,
                duration_sec=duration,
                output_fps=output_fps,
                progress_callback=self.update_progress
            )
            
            if errors is not None and len(errors) > 0:
                mae = np.mean(errors)
                self.status_label.config(text=f"Done! MAE: {mae:.2f} deg. Saved to {Path(output_path).name}")
            else:
                self.status_label.config(text=f"Done! Saved to {Path(output_path).name}")
                
            messagebox.showinfo("Complete", f"Demo video saved to:\n{output_path}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text=f"Error: {str(e)}")
            
        finally:
            self.generate_btn.config(state='normal')
            self.progress_var.set(0)
            
    def run(self):
        self.root.mainloop()
        if self.cap:
            self.cap.release()


def main():
    parser = argparse.ArgumentParser(description="Generate demo video")
    parser.add_argument("--gui", action="store_true", help="Launch GUI for section selection")
    parser.add_argument("--video", type=str, help="Input video path")
    parser.add_argument("--labels", type=str, help="Labels CSV path")
    parser.add_argument("--model", type=str, default="models/optuna_best/best_model.pt")
    parser.add_argument("--output", type=str, default="demo_video.mp4")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--duration", type=float, default=30, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=10, help="Output FPS")
    args = parser.parse_args()
    
    if args.gui:
        app = DemoVideoGUI()
        app.run()
    else:
        if not args.video or not args.labels:
            print("Error: --video and --labels are required when not using --gui")
            print("Use --gui for interactive mode, or provide all required arguments")
            return
            
        generate_demo(
            video_path=args.video,
            labels_csv=args.labels,
            model_path=args.model,
            output_path=args.output,
            start_frame=args.start,
            duration_sec=args.duration,
            output_fps=args.fps
        )


if __name__ == "__main__":
    main()
