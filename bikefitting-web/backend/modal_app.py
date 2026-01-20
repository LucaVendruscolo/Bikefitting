"""
Modal Backend for Bikefitting Web App

Provides serverless GPU endpoints for video processing with real-time progress streaming.

Endpoints:
    POST /process_video_stream - Process video with SSE progress updates
    GET  /download/{job_id}    - Download processed video
    GET  /health               - Health check

Deploy: modal deploy modal_app.py
Test:   modal serve modal_app.py
"""

import modal
import os
import uuid
from pathlib import Path
from collections import defaultdict
from typing import Optional

# ============= APP CONFIGURATION =============

app = modal.App("bikefitting")

# Security limits
MAX_DURATION_SEC = 120      # Max video duration (2 min)
MAX_FILE_SIZE_MB = 200      # Max upload size
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_OUTPUT_FPS = 5
MAX_OUTPUT_FPS = 15
RATE_LIMIT_REQUESTS = 10    # Requests per hour per client
RATE_LIMIT_WINDOW_SEC = 3600

# Container image with ML dependencies including BoTorch for Bayesian Optimization
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libx264-dev", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.0.0", "torchvision>=0.15.0",
        "opencv-python-headless>=4.8.0", "numpy>=1.24.0",
        "ultralytics>=8.0.0", "Pillow>=10.0.0",
        "tqdm>=4.65.0", "fastapi",
        # BoTorch/GPyTorch for Gaussian Process-based Active Learning
        "botorch>=0.9.0", "gpytorch>=1.11.0", "scikit-learn>=1.3.0"
    )
)

# Persistent volumes
model_volume = modal.Volume.from_name("bikefitting-models", create_if_missing=True)
temp_volume = modal.Volume.from_name("bikefitting-temp", create_if_missing=True)

# Simple rate limiter (resets on container restart)
rate_limit_store = defaultdict(list)


# ============= HELPER FUNCTIONS =============

def check_rate_limit(client_id: str) -> tuple[bool, str]:
    """Check if client has exceeded rate limit."""
    import time
    now = time.time()
    # Clean old entries
    rate_limit_store[client_id] = [
        t for t in rate_limit_store[client_id] 
        if now - t < RATE_LIMIT_WINDOW_SEC
    ]
    if len(rate_limit_store[client_id]) >= RATE_LIMIT_REQUESTS:
        return False, f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per hour."
    rate_limit_store[client_id].append(now)
    return True, "OK"


def validate_api_key(api_key: Optional[str]) -> tuple[bool, str]:
    """Validate API key from environment. Optional for direct browser access."""
    # API key is optional - allows direct frontend calls without exposing secrets
    # For production, consider adding rate limiting or other protections
    expected_key = os.environ.get("BIKEFITTING_API_KEY", "")
    if not expected_key:
        return True, "No API key configured"
    if api_key and api_key == expected_key:
        return True, "API key valid"
    # Allow requests without API key (for frontend direct calls)
    return True, "No API key provided (allowed)"


def setup_processing_modules():
    """Create the processing module files in /root/processing/."""
    proc_dir = Path("/root/processing")
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    # __init__.py
    (proc_dir / "__init__.py").write_text("# Processing modules\n")
    
    # bike_segmenter.py - Segments bike from frame using YOLO
    (proc_dir / "bike_segmenter.py").write_text('''
import cv2
import numpy as np
from ultralytics import YOLO

class BikeSegmenter:
    """Segments bike from video frames using YOLOv8 segmentation."""
    
    BIKE_CLASS_ID = 1  # COCO class for bicycle
    
    def __init__(self, model_path="yolov8n-seg.pt"):
        self.model = YOLO(model_path)
    
    def mask_bike(self, frame):
        """
        Extract bike region from frame.
        Returns: (masked_224x224, mask, success)
        """
        results = self.model(frame, verbose=False)
        
        for result in results:
            if result.masks is None:
                continue
            for i, cls in enumerate(result.boxes.cls):
                if int(cls) == self.BIKE_CLASS_ID:
                    mask = result.masks.data[i].cpu().numpy()
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    
                    # Dilate mask
                    kernel = np.ones((15, 15), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=2)
                    
                    # Find bounding box and crop to square
                    coords = cv2.findNonZero(mask)
                    if coords is None:
                        continue
                    x, y, w, h = cv2.boundingRect(coords)
                    
                    # Square crop centered on bike
                    size = max(w, h)
                    cx, cy = x + w // 2, y + h // 2
                    x1 = max(0, cx - size // 2)
                    y1 = max(0, cy - size // 2)
                    x2 = min(frame.shape[1], x1 + size)
                    y2 = min(frame.shape[0], y1 + size)
                    
                    # Crop and apply mask
                    cropped = frame[y1:y2, x1:x2].copy()
                    mask_crop = mask[y1:y2, x1:x2]
                    masked = cv2.bitwise_and(cropped, cropped, mask=mask_crop)
                    
                    # Resize to 224x224 (model input size)
                    masked_224 = cv2.resize(masked, (224, 224))
                    return masked_224, mask, True
        
        # No bike found - return black image
        return np.zeros((224, 224, 3), dtype=np.uint8), None, False
''')
    
    # angle_predictor.py - Predicts bike angle using ConvNeXt
    (proc_dir / "angle_predictor.py").write_text('''
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
from pathlib import Path
from torchvision import models

class AngleClassifier(nn.Module):
    """Angle classifier with custom head (matches training code)."""
    
    def __init__(self, backbone_name="convnext_tiny", num_bins=120):
        super().__init__()
        self.num_bins = num_bins
        
        # Load backbone and replace classifier with Identity
        if backbone_name == "convnext_tiny":
            self.backbone = models.convnext_tiny(weights=None)
            feature_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name == "convnext_small":
            self.backbone = models.convnext_small(weights=None)
            feature_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=None)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=None)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Custom classification head (must match training)
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
        if len(features.shape) == 4:
            features = features.flatten(1)
        return self.head(features)


class BikeAnglePredictor:
    """Predicts bike tilt angle from masked bike image."""
    
    def __init__(self, model_path, device="cuda"):
        self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get config from checkpoint or config.json
        if "num_bins" in checkpoint:
            num_bins = checkpoint["num_bins"]
            backbone = checkpoint.get("backbone", "convnext_tiny")
        else:
            config_path = Path(model_path).parent / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                num_bins = config.get("num_bins", 120)
                backbone = config.get("backbone", "convnext_tiny")
            else:
                num_bins, backbone = 120, "convnext_tiny"
        
        self.num_bins = num_bins
        self.model = AngleClassifier(backbone, num_bins)
        
        # Load weights
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()
        
        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    def predict(self, masked_image):
        """Predict angle from 224x224 masked bike image."""
        # Convert BGR to RGB, normalize
        img = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device) / 255.0
        tensor = (tensor - self.mean) / self.std
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        # Convert bin probabilities to angle
        bin_size = 360.0 / self.num_bins
        angles = np.arange(self.num_bins) * bin_size + bin_size / 2
        
        # Circular mean calculation
        angles_rad = np.radians(angles)
        sin_sum = np.sum(probs * np.sin(angles_rad))
        cos_sum = np.sum(probs * np.cos(angles_rad))
        angle = np.degrees(np.arctan2(sin_sum, cos_sum))  # Returns -180 to 180
        
        confidence = float(np.max(probs) * 100)
        return float(angle), confidence
''')
    
    # pose_detector.py - Detects human pose using YOLO
    (proc_dir / "pose_detector.py").write_text('''
import cv2
import numpy as np
from ultralytics import YOLO

# COCO keypoint indices
KPT_IDX = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}

SKELETON_LEFT = [("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                 ("left_shoulder", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle")]
SKELETON_RIGHT = [("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
                  ("right_shoulder", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle")]


class KeypointSmoother:
    """Smooths keypoints over time using exponential moving average."""
    
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_xy = None
        self.prev_conf = None
    
    def smooth(self, kpts_xy, kpts_conf):
        if self.prev_xy is None:
            self.prev_xy, self.prev_conf = kpts_xy.copy(), kpts_conf.copy()
            return kpts_xy, kpts_conf
        smoothed_xy = self.alpha * kpts_xy + (1 - self.alpha) * self.prev_xy
        smoothed_conf = self.alpha * kpts_conf + (1 - self.alpha) * self.prev_conf
        self.prev_xy, self.prev_conf = smoothed_xy.copy(), smoothed_conf.copy()
        return smoothed_xy, smoothed_conf
    
    def reset(self):
        self.prev_xy = self.prev_conf = None


class PoseDetector:
    """Detects human pose and calculates joint angles."""
    
    def __init__(self, model_path="yolov8m-pose.pt", min_conf=0.5):
        self.model = YOLO(model_path)
        self.min_conf = min_conf
        self.smoother = KeypointSmoother()
    
    def reset_smoother(self):
        self.smoother.reset()
    
    def _angle(self, a, b, c):
        """Calculate angle ABC in degrees."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom == 0:
            return float("nan")
        return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1))))
    
    def detect(self, image):
        """Detect pose and return keypoints + angles."""
        empty = {"keypoints_xy": None, "keypoints_conf": None, "angles": {}, 
                 "detected_side": None, "detected": False}
        
        results = self.model.predict(image, verbose=False)
        if not results or results[0].keypoints is None:
            return empty
        
        kpts_xy = results[0].keypoints.xy.cpu().numpy()
        kpts_conf = results[0].keypoints.conf.cpu().numpy()
        
        if kpts_conf.size == 0:
            return empty
        
        # Select person with highest confidence
        person_idx = int(np.argmax(kpts_conf.mean(axis=1)))
        raw_xy, raw_conf = kpts_xy[person_idx], kpts_conf[person_idx]
        raw_xy, raw_conf = self.smoother.smooth(raw_xy, raw_conf)
        
        # Determine which side is facing camera
        def count_side(side):
            names = [f"{side}_{j}" for j in ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]]
            return sum(1 for n in names if raw_conf[KPT_IDX[n]] >= self.min_conf)
        
        cnt_r, cnt_l = count_side("right"), count_side("left")
        if cnt_r == 0 and cnt_l == 0:
            return empty
        
        detected_side = "right" if cnt_r >= cnt_l else "left"
        
        # Calculate joint angles
        angles = {}
        s = detected_side
        hip, knee, ankle = KPT_IDX[f"{s}_hip"], KPT_IDX[f"{s}_knee"], KPT_IDX[f"{s}_ankle"]
        shoulder, elbow, wrist = KPT_IDX[f"{s}_shoulder"], KPT_IDX[f"{s}_elbow"], KPT_IDX[f"{s}_wrist"]
        
        if all(raw_conf[i] > 0.5 for i in [hip, knee, ankle]):
            angles["knee_angle"] = self._angle(raw_xy[hip], raw_xy[knee], raw_xy[ankle])
        if all(raw_conf[i] > 0.5 for i in [shoulder, hip, knee]):
            angles["hip_angle"] = self._angle(raw_xy[shoulder], raw_xy[hip], raw_xy[knee])
        if all(raw_conf[i] > 0.5 for i in [shoulder, elbow, wrist]):
            angles["elbow_angle"] = self._angle(raw_xy[shoulder], raw_xy[elbow], raw_xy[wrist])
        
        return {"keypoints_xy": raw_xy, "keypoints_conf": raw_conf, "angles": angles,
                "detected_side": detected_side, "detected": True}
    
    def draw_skeleton(self, image, kpts_xy, kpts_conf, detected_side, scale=1.0):
        """Draw skeleton overlay on image."""
        if detected_side is None:
            return image
        
        vis = image.copy()
        conns = SKELETON_RIGHT if detected_side == "right" else SKELETON_LEFT
        side_joints = [f"{detected_side}_{j}" for j in ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]]
        
        # Higher confidence threshold - only show if YOLO is confident
        min_conf = 0.6
        
        # Draw skeleton lines (yellow with black outline)
        for start_name, end_name in conns:
            si, ei = KPT_IDX[start_name], KPT_IDX[end_name]
            if kpts_conf[si] < min_conf or kpts_conf[ei] < min_conf:
                continue
            sp = (int(kpts_xy[si][0] * scale), int(kpts_xy[si][1] * scale))
            ep = (int(kpts_xy[ei][0] * scale), int(kpts_xy[ei][1] * scale))
            cv2.line(vis, sp, ep, (0, 0, 0), 3)
            cv2.line(vis, sp, ep, (0, 255, 255), 2)
        
        # Draw smaller joint circles (magenta with white border)
        for jn in side_joints:
            idx = KPT_IDX[jn]
            if kpts_conf[idx] < min_conf:
                continue
            pt = (int(kpts_xy[idx][0] * scale), int(kpts_xy[idx][1] * scale))
            cv2.circle(vis, pt, 5, (0, 0, 0), -1)
            cv2.circle(vis, pt, 4, (255, 0, 255), -1)
            cv2.circle(vis, pt, 4, (255, 255, 255), 1)
        
        return vis
''')
    
    # video_processor.py - Uses Fivos's SmartBikeFitter with Gaussian Process Active Learning
    (proc_dir / "video_processor.py").write_text('''
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior

from .bike_segmenter import BikeSegmenter
from .angle_predictor import BikeAnglePredictor
from .pose_detector import PoseDetector


class ALSimExperiment:
    """
    Active Learning Experiment using Gaussian Processes.
    Uses uncertainty-based acquisition to select which frames to sample.
    """
    def __init__(self, timestamps, kernel_type="rbf", acq_strategy="joint_uncertainty"):
        self.timestamps = timestamps
        self.total_frames = len(timestamps)
        self.y_values = np.full(self.total_frames, np.nan)
        self.visited_indices = []
        self.train_indices = []
        self.wasted_indices = []
        self.kernel_type = kernel_type
        self.acq_strategy = acq_strategy
        self.model = None
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.X_all = torch.tensor(timestamps.reshape(-1, 1), dtype=torch.double)
        self.X_all_norm = torch.tensor(
            self.x_scaler.fit_transform(timestamps.reshape(-1, 1)),
            dtype=torch.double
        )

    def update_model(self, fit=False):
        if len(self.train_indices) == 0:
            return
        X_train = self.timestamps[self.train_indices].reshape(-1, 1)
        Y_train = self.y_values[self.train_indices].reshape(-1, 1)
        X_train_norm = torch.tensor(self.x_scaler.transform(X_train), dtype=torch.double)
        Y_train_norm = torch.tensor(self.y_scaler.fit_transform(Y_train), dtype=torch.double)
        time_std = self.x_scaler.scale_[0]
        
        old_state_dict = None
        if self.model is not None and not fit:
            old_state_dict = {k: v for k, v in self.model.state_dict().items()
                            if "train_inputs" not in k and "train_targets" not in k}
        
        fast_ls_target = 0.40 / time_std
        fast_ls_prior = GammaPrior(concentration=2.0, rate=2.0/fast_ls_target)
        
        if self.kernel_type == "rbf":
            covar = ScaleKernel(RBFKernel(lengthscale_prior=fast_ls_prior))
        else:
            covar = ScaleKernel(MaternKernel(nu=2.5, lengthscale_prior=fast_ls_prior))
        
        self.model = SingleTaskGP(X_train_norm, Y_train_norm, covar_module=covar)
        
        if fit:
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            try:
                fit_gpytorch_mll(mll)
            except:
                pass
        elif old_state_dict:
            self.model.load_state_dict(old_state_dict, strict=False)

    def _get_model_variance(self, candidate_indices):
        if self.model is None:
            return torch.ones(len(candidate_indices))
        X_candidates = self.X_all_norm[candidate_indices]
        self.model.eval()
        with torch.no_grad():
            return self.model(X_candidates).variance

    def select_next_point(self, other_experiments=None):
        visited_mask = np.zeros(self.total_frames, dtype=bool)
        visited_mask[self.visited_indices] = True
        candidate_indices = np.where(~visited_mask)[0]
        if len(candidate_indices) == 0:
            return None
        if self.acq_strategy == "random":
            return np.random.choice(candidate_indices)
        
        my_variance = self._get_model_variance(candidate_indices)
        if self.acq_strategy == "joint_uncertainty" and other_experiments:
            variances = [my_variance] + [e._get_model_variance(candidate_indices) for e in other_experiments]
            stacked = torch.stack(variances)
            if stacked.shape[0] == 2:
                weights = torch.tensor([0.8, 0.2], dtype=torch.double)
                final_variance = torch.sum(stacked * weights.view(-1, 1), dim=0)
            else:
                final_variance, _ = torch.max(stacked, dim=0)
        else:
            final_variance = my_variance
        
        # Spatial suppression for wasted indices
        if self.wasted_indices:
            candidate_times = torch.tensor(self.timestamps[candidate_indices], dtype=torch.float32)
            wasted_times = torch.tensor(self.timestamps[self.wasted_indices], dtype=torch.float32)
            dists = torch.abs(candidate_times.unsqueeze(1) - wasted_times.unsqueeze(0))
            min_dists, _ = torch.min(dists, dim=1)
            too_close = min_dists < 1.0
            final_variance[too_close] = -1.0
            if torch.max(final_variance) == -1.0:
                final_variance[too_close] = 0.0
        
        return candidate_indices[torch.argmax(final_variance).item()]

    def add_observation(self, idx, value, fit=False):
        if idx in self.visited_indices:
            return False
        self.visited_indices.append(idx)
        if np.isnan(value):
            self.wasted_indices.append(idx)
            return False
        self.train_indices.append(idx)
        self.y_values[idx] = value
        self.update_model(fit=fit)
        return True

    def predict_curve(self):
        if self.model is None:
            return None
        self.model.eval()
        with torch.no_grad():
            pred_y_norm = self.model(self.X_all_norm).mean
        return self.y_scaler.inverse_transform(pred_y_norm.numpy().reshape(-1, 1)).flatten()


def generate_recommendations(results):
    """Generate bike fit recommendations from GP-predicted metrics (matches Fivos's output)."""
    k_max = results.get("max_knee_ext", 0)
    k_min = results.get("min_knee_flex", 70)
    h_min = results.get("min_hip_angle", 60)
    e_avg = results.get("avg_elbow_angle", 155)
    
    rec = {
        "saddle_height": {"status": "ok", "action": None, "adjustment_mm": 0, "details": ""},
        "saddle_fore_aft": {"status": "ok", "action": None, "adjustment_mm": 0, "details": ""},
        "crank_length": {"status": "ok", "action": None, "details": ""},
        "cockpit": {"status": "ok", "reach_action": None, "adjustment_mm": 0, "details": ""},
        "summary": [],
        "metrics": {
            "knee_max_extension": round(k_max, 1) if k_max else None,
            "knee_min_flexion": round(k_min, 1) if k_min else None,
            "min_hip_angle": round(h_min, 1) if h_min else None,
            "avg_elbow_angle": round(e_avg, 1) if e_avg else None
        }
    }
    
    if not k_max:
        rec["summary"].append("Not enough data")
        return rec
    
    # 1. Saddle Height (target: 140-150 deg knee extension)
    if k_max < 140:
        mm = (140 - k_max) * 2.0
        rec["saddle_height"] = {"status": "low", "action": "raise", "adjustment_mm": round(mm),
                                "details": f"Knee ext {k_max:.0f}° below optimal. Raise ~{mm:.0f}mm."}
        rec["summary"].append(f"Raise saddle ~{mm:.0f}mm")
    elif k_max > 150:
        mm = (k_max - 150) * 2.0
        rec["saddle_height"] = {"status": "high", "action": "lower", "adjustment_mm": round(mm),
                                "details": f"Knee ext {k_max:.0f}° - overextension risk. Lower ~{mm:.0f}mm."}
        rec["summary"].append(f"Lower saddle ~{mm:.0f}mm")
    else:
        rec["saddle_height"]["details"] = f"Knee ext {k_max:.0f}° is optimal (140-150°)."
        rec["summary"].append("Saddle height optimal")
    
    # 2. Saddle Fore/Aft (target: knee flexion >70 at top of stroke)
    if k_min < 70:
        rec["saddle_fore_aft"] = {"status": "forward", "action": "move_back", "adjustment_mm": 10,
                                  "details": f"Knee closed at top ({k_min:.0f}°). Move back 5-10mm."}
        rec["summary"].append("Move saddle back 5-10mm")
    else:
        rec["saddle_fore_aft"]["details"] = f"Knee clearance at top ({k_min:.0f}°) is good."
    
    # 3. Crank Length (impingement check: hip <48 or knee <68)
    if h_min < 48 or k_min < 68:
        rec["crank_length"] = {"status": "issue", "action": "consider_shorter",
                               "details": f"Hip {h_min:.0f}° / Knee {k_min:.0f}° indicates impingement. Consider shorter cranks (-5mm)."}
        rec["summary"].append("Consider shorter cranks")
    else:
        rec["crank_length"]["details"] = f"Hip clearance ({h_min:.0f}°) is adequate. No impingement."
    
    # 4. Cockpit / Stem (target: elbow 150-160 deg)
    if e_avg > 165:
        mm = max(10, ((e_avg - 160) / 5) * 10)
        rec["cockpit"] = {"status": "issue", "reach_action": "shorten", "adjustment_mm": round(mm),
                          "details": f"Arms locked ({e_avg:.0f}°). Shorten stem ~{mm:.0f}mm."}
        rec["summary"].append(f"Shorten stem ~{mm:.0f}mm")
    elif e_avg < 150:
        mm = max(10, ((150 - e_avg) / 5) * 10)
        rec["cockpit"] = {"status": "issue", "reach_action": "lengthen", "adjustment_mm": round(mm),
                          "details": f"Arms bent ({e_avg:.0f}°). Lengthen stem ~{mm:.0f}mm."}
        rec["summary"].append(f"Lengthen stem ~{mm:.0f}mm")
    else:
        rec["cockpit"]["details"] = f"Elbow angle ({e_avg:.0f}°) is in optimal range (150-160°)."
    
    return rec


class VideoProcessor:
    """
    Smart Bike Fitter using Gaussian Process Active Learning.
    Only samples ~30 frames instead of processing all frames.
    """
    
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.bike_segmenter = BikeSegmenter()
        self.angle_predictor = BikeAnglePredictor(model_path, device)
        self.pose_detector = PoseDetector()
        print("VideoProcessor: All models loaded (with GP/BO support)")
    
    def process_video(self, input_path, output_path, output_fps=10,
                      max_duration_sec=None, start_time=0, end_time=None,
                      progress_callback=None, n_samples=30):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start_time * fps) if start_time > 0 else 0
        end_frame = min(int(end_time * fps), total_frames) if end_time else total_frames
        if max_duration_sec:
            end_frame = min(end_frame, start_frame + int(max_duration_sec * fps))
        
        frames_in_range = end_frame - start_frame
        skip = max(1, int(fps / 30))  # Scan at 30fps
        frames_to_scan = frames_in_range // skip
        
        # Output video setup
        out_h = min(720, h)
        out_w = int(w * (out_h / h))
        if out_w > 1280:
            out_w, out_h = 1280, int(h * (1280 / w))
        scale = out_h / h
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (out_w, out_h))
        
        # Phase 1: Scan for valid side-view frames
        print(f"Phase 1: Scanning {frames_to_scan} frames for valid bike angles...")
        valid_frames = []
        frame_data = []
        
        for i in tqdm(range(frames_to_scan), desc="Scanning"):
            frame_num = start_frame + (i * skip)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break
            
            masked, mask, bike_found = self.bike_segmenter.mask_bike(frame)
            if not bike_found:
                continue
            
            yaw, conf = self.angle_predictor.predict(masked)
            
            # Gating: only side-view frames (60-120 degrees)
            if 60 <= abs(yaw) <= 120:
                physical_time = frame_num / fps
                valid_frames.append({"idx": frame_num, "time": physical_time, "frame": frame})
            
            if progress_callback:
                progress_callback(i + 1, frames_to_scan + n_samples)
        
        if len(valid_frames) < 10:
            print("Not enough valid side-view frames")
            cap.release()
            out.release()
            return {"stats": {"frames_processed": 0, "error": "Not enough side-view frames"}, "frame_data": []}
        
        print(f"Found {len(valid_frames)} valid frames. Phase 2: Active Learning with GP...")
        
        # Phase 2: Active Learning with Gaussian Processes
        timestamps = np.array([f["time"] for f in valid_frames])
        exp_knee = ALSimExperiment(timestamps, kernel_type="rbf", acq_strategy="joint_uncertainty")
        exp_hip = ALSimExperiment(timestamps, kernel_type="rbf", acq_strategy="joint_uncertainty")
        exp_elbow = ALSimExperiment(timestamps, kernel_type="rbf", acq_strategy="joint_uncertainty")
        
        # Initialize with 5 random samples
        init_indices = np.random.choice(len(valid_frames), min(5, len(valid_frames)), replace=False)
        self.pose_detector.reset_smoother()
        
        for local_idx in init_indices:
            self._process_sample(valid_frames[local_idx], local_idx, exp_knee, exp_hip, exp_elbow, fit=False)
        
        exp_knee.update_model(fit=True)
        exp_hip.update_model(fit=True)
        exp_elbow.update_model(fit=True)
        
        # Active learning loop
        samples_taken = len(init_indices)
        while samples_taken < n_samples:
            next_idx = exp_knee.select_next_point(other_experiments=[exp_hip])
            if next_idx is None:
                break
            
            do_fit = ((samples_taken + 1) % 5 == 0)
            self._process_sample(valid_frames[next_idx], next_idx, exp_knee, exp_hip, exp_elbow, fit=do_fit)
            samples_taken += 1
            
            if progress_callback:
                progress_callback(frames_to_scan + samples_taken, frames_to_scan + n_samples)
        
        cap.release()
        
        # Predict full curves using GP
        pred_knee = exp_knee.predict_curve()
        pred_hip = exp_hip.predict_curve()
        pred_elbow = exp_elbow.predict_curve()
        
        results = {
            "max_knee_ext": float(np.max(pred_knee)) if pred_knee is not None else 0,
            "min_knee_flex": float(np.min(pred_knee)) if pred_knee is not None else 70,
            "min_hip_angle": float(np.min(pred_hip)) if pred_hip is not None else 60,
            "avg_elbow_angle": float(np.mean(pred_elbow)) if pred_elbow is not None else 155
        }
        
        # Generate output video from sampled frames
        for i, vf in enumerate(valid_frames[:min(100, len(valid_frames))]):
            output = cv2.resize(vf["frame"], (out_w, out_h))
            pose = self.pose_detector.detect(vf["frame"])
            if pose["keypoints_xy"] is not None:
                output = self.pose_detector.draw_skeleton(
                    output, pose["keypoints_xy"], pose["keypoints_conf"],
                    pose.get("detected_side"), scale
                )
            out.write(output)
            
            frame_data.append({
                "frame": i,
                "time": vf["time"],
                "knee_angle": float(pred_knee[i]) if pred_knee is not None and i < len(pred_knee) else None,
                "hip_angle": float(pred_hip[i]) if pred_hip is not None and i < len(pred_hip) else None,
                "elbow_angle": float(pred_elbow[i]) if pred_elbow is not None and i < len(pred_elbow) else None,
                "is_valid": True
            })
        
        out.release()
        
        recommendations = generate_recommendations(results)
        
        stats = {
            "frames_processed": len(valid_frames),
            "samples_taken": samples_taken,
            "output_fps": output_fps,
            "valid_frames": len(valid_frames),
            "knee_max_extension": results["max_knee_ext"],
            "knee_min_flexion": results["min_knee_flex"],
            "min_hip_angle": results["min_hip_angle"],
            "avg_elbow_angle": results["avg_elbow_angle"],
            "recommendations": recommendations,
            "method": "Gaussian Process Active Learning"
        }
        
        print(f"Analysis complete. Sampled {samples_taken} frames using GP/BO.")
        return {"stats": stats, "frame_data": frame_data}
    
    def _process_sample(self, frame_info, idx, exp_knee, exp_hip, exp_elbow, fit):
        """Process a single frame and update GP experiments."""
        self.pose_detector.reset_smoother()
        pose = self.pose_detector.detect(frame_info["frame"])
        angles = pose.get("angles", {})
        
        k = angles.get("knee_angle", np.nan)
        h = angles.get("hip_angle", np.nan)
        e = angles.get("elbow_angle", np.nan)
        
        exp_knee.add_observation(idx, k, fit=fit)
        exp_hip.add_observation(idx, h, fit=fit)
        exp_elbow.add_observation(idx, e, fit=fit)
''')


# ============= API ENDPOINTS =============

@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "bikefitting",
        "limits": {
            "max_duration_sec": MAX_DURATION_SEC,
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "max_output_fps": MAX_OUTPUT_FPS,
        }
    }


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    volumes={"/models": model_volume, "/temp": temp_volume},
    secrets=[modal.Secret.from_name("bikefitting-api-key", required_keys=[])],
)
@modal.fastapi_endpoint(method="POST")
def process_video_stream(request: dict):
    """
    Process video with real-time progress streaming via Server-Sent Events.
    
    Request body:
        video_base64: Base64 encoded video
        api_key: API key for authentication
        output_fps: Output frame rate (5-15)
        start_time: Start time in seconds
        end_time: End time in seconds
    
    Returns SSE stream with progress updates, then final result with frame_data.
    """
    from fastapi.responses import StreamingResponse
    import base64
    import json
    import subprocess
    
    def generate():
        # Validate request
        video_base64 = request.get("video_base64", "")
        api_key = request.get("api_key", "")
        
        is_valid, msg = validate_api_key(api_key)
        if not is_valid:
            yield f"data: {json.dumps({'type': 'error', 'error': msg})}\n\n"
            return
        
        if not video_base64:
            yield f"data: {json.dumps({'type': 'error', 'error': 'video_base64 required'})}\n\n"
            return
        
        # Decode video
        try:
            video_bytes = base64.b64decode(video_base64)
        except Exception:
            yield f"data: {json.dumps({'type': 'error', 'error': 'Invalid base64'})}\n\n"
            return
        
        if len(video_bytes) > MAX_FILE_SIZE_BYTES:
            yield f"data: {json.dumps({'type': 'error', 'error': 'File too large'})}\n\n"
            return
        
        job_id = str(uuid.uuid4())[:8]
        yield f"data: {json.dumps({'type': 'progress', 'stage': 'upload', 'message': 'Video received', 'percent': 5})}\n\n"
        
        # Parse parameters
        output_fps = max(MIN_OUTPUT_FPS, min(MAX_OUTPUT_FPS, int(request.get("output_fps", 10))))
        max_duration = min(float(request.get("max_duration_sec", MAX_DURATION_SEC)), MAX_DURATION_SEC)
        start_time = max(0, float(request.get("start_time", 0)))
        end_time = float(request.get("end_time", 0))
        
        input_path = f"/temp/input_{job_id}.mp4"
        output_path = f"/temp/output_{job_id}.mp4"
        temp_output = f"/temp/temp_{job_id}.mp4"
        
        with open(input_path, "wb") as f:
            f.write(video_bytes)
        
        yield f"data: {json.dumps({'type': 'progress', 'stage': 'setup', 'message': 'Loading AI models...', 'percent': 10})}\n\n"
        
        try:
            # Setup and import processing modules
            import sys
            if "/root" not in sys.path:
                sys.path.insert(0, "/root")
            
            setup_processing_modules()
            
            # Clear module cache for fresh import
            for mod_name in list(sys.modules.keys()):
                if mod_name.startswith('processing'):
                    del sys.modules[mod_name]
            
            from processing.video_processor import VideoProcessor
            
            model_path = "/models/best_model.pt"
            if not Path(model_path).exists():
                yield f"data: {json.dumps({'type': 'error', 'error': 'Model not found'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'models', 'message': 'Models loaded', 'percent': 15})}\n\n"
            
            processor = VideoProcessor(model_path, device="cuda")
            
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'processing', 'message': 'Processing frames...', 'percent': 20})}\n\n"
            
            # Process video
            result = processor.process_video(
                input_path=input_path,
                output_path=temp_output,
                output_fps=output_fps,
                max_duration_sec=max_duration,
                start_time=start_time,
                end_time=end_time if end_time > 0 else None
            )
            
            stats = result.get("stats", {})
            frame_data = result.get("frame_data", [])
            
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'encoding', 'message': 'Encoding video...', 'percent': 85})}\n\n"
            
            # Convert to H.264 for browser compatibility
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_output,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                output_path
            ], capture_output=True)
            
            if os.path.exists(temp_output):
                os.remove(temp_output)
            
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'finalizing', 'message': 'Saving...', 'percent': 95})}\n\n"
            
            temp_volume.commit()
            os.remove(input_path)
            
            # Convert numpy types for JSON serialization
            def to_python(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: to_python(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [to_python(v) for v in obj]
                return obj
            
            yield f"data: {json.dumps({'type': 'complete', 'job_id': job_id, 'stats': to_python(stats), 'frame_data': to_python(frame_data), 'percent': 100})}\n\n"
            
        except Exception as e:
            for path in [input_path, output_path, temp_output]:
                if os.path.exists(path):
                    os.remove(path)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"}
    )


@app.function(image=image, volumes={"/temp": temp_volume})
@modal.fastapi_endpoint(method="GET")
def download(job_id: str):
    """Download processed video by job ID."""
    from fastapi.responses import Response
    
    temp_volume.reload()
    
    # Sanitize job_id
    safe_job_id = "".join(c for c in job_id if c.isalnum())
    if len(safe_job_id) != 8:
        return Response(content='{"error": "Invalid job ID"}', status_code=400,
                       media_type="application/json", headers={"Access-Control-Allow-Origin": "*"})
    
    output_path = f"/temp/output_{safe_job_id}.mp4"
    
    if not os.path.exists(output_path):
        return Response(content='{"error": "Video not found or expired"}', status_code=404,
                       media_type="application/json", headers={"Access-Control-Allow-Origin": "*"})
    
    with open(output_path, "rb") as f:
        video_bytes = f.read()
    
    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'attachment; filename="bikefitting_{safe_job_id}.mp4"',
            "Access-Control-Allow-Origin": "*",
        }
    )


# ============= LOCAL ENTRY POINT =============

@app.local_entrypoint()
def main():
    print("Bikefitting API")
    print("=" * 40)
    print("Deploy: modal deploy modal_app.py")
    print("Test:   modal serve modal_app.py")
