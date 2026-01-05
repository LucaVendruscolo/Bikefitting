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

# Container image with ML dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libx264-dev", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.0.0", "torchvision>=0.15.0",
        "opencv-python-headless>=4.8.0", "numpy>=1.24.0",
        "ultralytics>=8.0.0", "Pillow>=10.0.0",
        "tqdm>=4.65.0", "fastapi"
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
        
        # Draw skeleton lines (yellow with black outline)
        for start_name, end_name in conns:
            si, ei = KPT_IDX[start_name], KPT_IDX[end_name]
            if kpts_conf[si] < 0.3 or kpts_conf[ei] < 0.3:
                continue
            sp = (int(kpts_xy[si][0] * scale), int(kpts_xy[si][1] * scale))
            ep = (int(kpts_xy[ei][0] * scale), int(kpts_xy[ei][1] * scale))
            cv2.line(vis, sp, ep, (0, 0, 0), 7)
            cv2.line(vis, sp, ep, (0, 255, 255), 4)
        
        # Draw joint circles (magenta with white border)
        for jn in side_joints:
            idx = KPT_IDX[jn]
            if kpts_conf[idx] < 0.3:
                continue
            pt = (int(kpts_xy[idx][0] * scale), int(kpts_xy[idx][1] * scale))
            cv2.circle(vis, pt, 12, (0, 0, 0), -1)
            cv2.circle(vis, pt, 10, (255, 0, 255), -1)
            cv2.circle(vis, pt, 10, (255, 255, 255), 2)
        
        return vis
''')
    
    # video_processor.py - Main processing pipeline
    (proc_dir / "video_processor.py").write_text('''
import cv2
import numpy as np
from tqdm import tqdm
from .bike_segmenter import BikeSegmenter
from .angle_predictor import BikeAnglePredictor
from .pose_detector import PoseDetector


class VideoProcessor:
    """
    Processes cycling videos to detect pose, bike angle, and generate annotated output.
    
    Output video contains ONLY visual overlays (skeleton + bike mask).
    All angle data is returned separately for frontend display.
    """
    
    def __init__(self, model_path, device="cuda"):
        self.bike_segmenter = BikeSegmenter()
        self.angle_predictor = BikeAnglePredictor(model_path, device)
        self.pose_detector = PoseDetector()
        print("VideoProcessor: All models loaded")
    
    def process_video(self, input_path, output_path, output_fps=10,
                      max_duration_sec=None, start_time=0, end_time=None,
                      progress_callback=None):
        """
        Process video and return frame-by-frame data.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video (clean, no text overlays)
            output_fps: Output frame rate (lower = faster processing)
            max_duration_sec: Maximum duration to process
            start_time: Start time in seconds
            end_time: End time in seconds
            progress_callback: Optional callback(current, total) for progress updates
        
        Returns:
            dict with 'stats' and 'frame_data' for frontend display
        """
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame range
        start_frame = int(start_time * fps) if start_time > 0 else 0
        end_frame = min(int(end_time * fps), total_frames) if end_time else total_frames
        if max_duration_sec:
            end_frame = min(end_frame, start_frame + int(max_duration_sec * fps))
        
        frames_in_range = end_frame - start_frame
        skip = max(1, int(fps / output_fps))
        frames_to_process = frames_in_range // skip
        
        # Output size (720p max, maintain aspect ratio)
        out_h = min(720, h)
        out_w = int(w * (out_h / h))
        if out_w > 1280:
            out_w, out_h = 1280, int(h * (1280 / w))
        scale = out_h / h
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (out_w, out_h))
        self.pose_detector.reset_smoother()
        
        frame_data = []
        
        for i in tqdm(range(frames_to_process), desc="Processing"):
            frame_num = start_frame + (i * skip)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            pose = self.pose_detector.detect(frame)
            masked, mask, bike_found = self.bike_segmenter.mask_bike(frame)
            bike_angle, conf = self.angle_predictor.predict(masked) if bike_found else (0, 0)
            
            # Store frame data for frontend
            frame_data.append({
                "frame": i,
                "time": float(i / output_fps),
                "bike_angle": float(bike_angle) if bike_found and conf > 0.3 else None,
                "knee_angle": float(pose["angles"]["knee_angle"]) if "knee_angle" in pose["angles"] else None,
                "hip_angle": float(pose["angles"]["hip_angle"]) if "hip_angle" in pose["angles"] else None,
                "elbow_angle": float(pose["angles"]["elbow_angle"]) if "elbow_angle" in pose["angles"] else None,
                "detected_side": pose.get("detected_side")
            })
            
            # Create clean output frame (skeleton + mask only, NO text)
            output = cv2.resize(frame, (out_w, out_h))
            
            if pose["keypoints_xy"] is not None and pose["detected_side"]:
                output = self.pose_detector.draw_skeleton(
                    output, pose["keypoints_xy"], pose["keypoints_conf"],
                    pose["detected_side"], scale
                )
            
            if mask is not None and mask.max() > 0:
                m = cv2.resize(mask, (out_w, out_h))
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
            
            out.write(output)
            if progress_callback:
                progress_callback(i + 1, frames_to_process)
        
        cap.release()
        out.release()
        
        # Calculate stats
        bike_angles = [f["bike_angle"] for f in frame_data if f["bike_angle"] is not None]
        knee_angles = [f["knee_angle"] for f in frame_data if f["knee_angle"] is not None]
        hip_angles = [f["hip_angle"] for f in frame_data if f["hip_angle"] is not None]
        elbow_angles = [f["elbow_angle"] for f in frame_data if f["elbow_angle"] is not None]
        
        stats = {"frames_processed": len(frame_data), "output_fps": output_fps}
        if bike_angles:
            stats["avg_bike_angle"] = float(np.mean(bike_angles))
        if knee_angles:
            stats["avg_knee_angle"] = float(np.mean(knee_angles))
        if hip_angles:
            stats["avg_hip_angle"] = float(np.mean(hip_angles))
        if elbow_angles:
            stats["avg_elbow_angle"] = float(np.mean(elbow_angles))
        
        return {"stats": stats, "frame_data": frame_data}
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
