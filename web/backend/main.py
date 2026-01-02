"""
BikeFit Pro - FastAPI Backend

This backend processes cycling videos to detect:
1. Joint angles (knee, hip, elbow) using YOLO pose estimation
2. Bike angle using a trained ConvNeXT classifier

The processing pipeline is identical to generate_demo_video.py to ensure
consistent predictions with the training data.

Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import time
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, Tuple, Optional, AsyncGenerator
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure we can import from parent directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Run: pip install ultralytics")
    YOLO = None


# ============= Constants (from generate_demo_video.py) =============

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


# ============= Model Classes (exact copies from generate_demo_video.py) =============

class KeypointSmoother:
    """Exponential moving average smoother for keypoint positions."""
    
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self.prev_xy = None
        self.prev_conf = None
    
    def smooth(self, kpts_xy: np.ndarray, kpts_conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.prev_xy is None:
            self.prev_xy = kpts_xy.copy()
            self.prev_conf = kpts_conf.copy()
            return kpts_xy, kpts_conf
        
        smoothed_xy = self.alpha * kpts_xy + (1 - self.alpha) * self.prev_xy
        smoothed_conf = self.alpha * kpts_conf + (1 - self.alpha) * self.prev_conf
        
        self.prev_xy = smoothed_xy.copy()
        self.prev_conf = smoothed_conf.copy()
        
        return smoothed_xy, smoothed_conf
    
    def reset(self):
        self.prev_xy = None
        self.prev_conf = None


class BikeSegmenter:
    """Identical to generate_demo_video.py - masks bike pixels only."""
    
    BIKE_CLASSES = [1, 3]  # bicycle, motorcycle in COCO
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.3):
        if model_path is None:
            model_path = str(PROJECT_ROOT / "bike_angle_detection_model" / "yolov8n-seg.pt")
        
        if not Path(model_path).exists():
            model_path = "yolov8n-seg.pt"  # Will download if needed
            
        self.model = YOLO(model_path) if YOLO else None
        self.conf_threshold = conf_threshold
        
    def get_bike_mask(self, image: np.ndarray, dilate_pixels: int = 5) -> Tuple[np.ndarray, bool]:
        if self.model is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), False
            
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
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)
            return mask, True
        
        return mask, False
    
    def mask_bike(self, image: np.ndarray, target_size: int = 224, dilate_pixels: int = 5) -> Tuple[np.ndarray, np.ndarray, bool]:
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
            
            # Make square
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
        return self.head(features)


# ============= Model Loader =============

class ModelManager:
    """Manages loading and caching of ML models."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bike_angle_model = None
        self.num_bins = None
        self.pose_model = None
        self.segmenter = None
        self.transform = None
        self._loaded = False
    
    def load_models(self):
        if self._loaded:
            return
            
        print(f"Loading models on device: {self.device}")
        
        # Load bike angle classifier
        model_path = PROJECT_ROOT / "bike_angle_detection_model" / "models" / "optuna_best" / "best_model.pt"
        if model_path.exists():
            print(f"Loading bike angle model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.num_bins = checkpoint['num_bins']
            backbone = checkpoint['backbone']
            
            self.bike_angle_model = AngleClassifier(backbone, self.num_bins)
            self.bike_angle_model.load_state_dict(checkpoint['model_state_dict'])
            self.bike_angle_model.eval()
            self.bike_angle_model = self.bike_angle_model.to(self.device)
            print(f"  Loaded: {backbone} with {self.num_bins} bins")
        else:
            print(f"Warning: Bike angle model not found at {model_path}")
        
        # Load pose model
        pose_model_path = PROJECT_ROOT / "joint_angle_detection" / "models" / "yolov8m-pose.pt"
        if not pose_model_path.exists():
            pose_model_path = PROJECT_ROOT / "bike_angle_detection_model" / "yolov8m-pose.pt"
        
        if pose_model_path.exists() and YOLO:
            print(f"Loading pose model from {pose_model_path}")
            self.pose_model = YOLO(str(pose_model_path))
        elif YOLO:
            print("Downloading yolov8m-pose.pt...")
            self.pose_model = YOLO("yolov8m-pose.pt")
        
        # Load segmenter
        print("Loading bike segmenter...")
        self.segmenter = BikeSegmenter()
        
        # Setup transform (identical to training)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self._loaded = True
        print("All models loaded successfully!")
    
    def is_loaded(self) -> bool:
        return self._loaded


# Global model manager
model_manager = ModelManager()


# ============= Processing Functions =============

def detect_joints(
    image: np.ndarray,
    model,
    side: str = "auto",
    min_conf: float = 0.5,
) -> Tuple[Optional[Dict], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Detect joints and return joint dict plus raw keypoints."""
    
    if model is None:
        return None, None, None, None
    
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

    # Auto-detect
    joints_right, count_r, mean_r = joints_for_side("right")
    joints_left, count_l, mean_l = joints_for_side("left")

    if count_r == 0 and count_l == 0:
        return None, raw_xy, raw_conf, None

    if (count_r > count_l) or (count_r == count_l and mean_r >= mean_l):
        return joints_right if joints_right else None, raw_xy, raw_conf, "right"
    else:
        return joints_left if joints_left else None, raw_xy, raw_conf, "left"


def compute_angles(joints: Dict) -> Dict[str, float]:
    """Compute knee, hip, and elbow angles from joints."""
    
    def pt(name: str):
        if name not in joints:
            return None
        x, y, _ = joints[name]
        return (x, y)
    
    def angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom == 0:
            return None
        cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))

    angles = {}
    
    # Knee: hip-knee-foot
    hip_pt, knee_pt, foot_pt = pt("hip"), pt("knee"), pt("foot")
    if all([hip_pt, knee_pt, foot_pt]):
        angles["knee"] = angle(hip_pt, knee_pt, foot_pt)
    
    # Hip: shoulder-hip-knee
    shoulder_pt = pt("shoulder")
    if all([shoulder_pt, hip_pt, knee_pt]):
        angles["hip"] = angle(shoulder_pt, hip_pt, knee_pt)
    
    # Elbow: shoulder-elbow-hand
    elbow_pt, hand_pt = pt("elbow"), pt("hand")
    if all([shoulder_pt, elbow_pt, hand_pt]):
        angles["elbow"] = angle(shoulder_pt, elbow_pt, hand_pt)
    
    return angles


def predict_bike_angle(model, masked_img: np.ndarray, num_bins: int, device, transform) -> Tuple[float, float]:
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


# ============= FastAPI App =============

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    model_manager.load_models()
    yield
    # Cleanup on shutdown
    pass

app = FastAPI(
    title="BikeFit Pro API",
    description="AI-powered bike fitting analysis API",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": model_manager.is_loaded(),
        "device": str(model_manager.device)
    }


@app.post("/api/process")
async def process_video(
    video: UploadFile = File(...),
    start_time: float = Form(0.0),
    end_time: float = Form(30.0),
    output_fps: int = Form(10)
):
    """
    Process a video file and return frame-by-frame analysis.
    
    Streams progress updates as JSON lines, then returns the final result.
    """
    
    async def generate_response() -> AsyncGenerator[str, None]:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(await video.read())
            video_path = tmp.name
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame range
            start_frame = int(start_time * video_fps)
            end_frame = int(end_time * video_fps)
            end_frame = min(end_frame, total_frames)
            
            frame_skip = max(1, int(video_fps / output_fps))
            frames_to_process = (end_frame - start_frame) // frame_skip
            
            yield json.dumps({
                "type": "progress",
                "progress": 0,
                "currentFrame": 0,
                "status": f"Starting processing: {frames_to_process} frames"
            }) + "\n"
            
            # Processing metrics
            metrics = {
                "totalFrames": frames_to_process,
                "processedFrames": 0,
                "segmentationTime": 0,
                "poseDetectionTime": 0,
                "bikeAngleTime": 0,
                "totalTime": 0
            }
            
            # Initialize smoother
            smoother = KeypointSmoother(alpha=0.6)
            
            # Output video writer
            output_path = video_path.replace('.mp4', '_processed.mp4')
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create output with side panel (similar to demo video)
            panel_width = 300
            out_width = orig_width + panel_width
            out_height = orig_height
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (out_width, out_height))
            
            # Frame data collection
            frame_data_list = []
            
            total_start = time.time()
            
            for i in range(frames_to_process):
                frame_num = start_frame + i * frame_skip
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_start = time.time()
                
                # 1. Pose detection
                pose_start = time.time()
                joints, kpts_xy, kpts_conf, detected_side = detect_joints(
                    frame, model_manager.pose_model
                )
                
                # Apply smoothing
                if kpts_xy is not None and kpts_conf is not None:
                    kpts_xy, kpts_conf = smoother.smooth(kpts_xy, kpts_conf)
                    
                    # Recompute joints with smoothed values
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
                                joints[simple_name] = (
                                    float(kpts_xy[idx][0]),
                                    float(kpts_xy[idx][1]),
                                    float(kpts_conf[idx])
                                )
                
                joint_angles = compute_angles(joints) if joints else {}
                metrics["poseDetectionTime"] += (time.time() - pose_start) * 1000
                
                # 2. Bike segmentation
                seg_start = time.time()
                masked, mask, success = model_manager.segmenter.mask_bike(frame, target_size=224)
                metrics["segmentationTime"] += (time.time() - seg_start) * 1000
                
                # 3. Bike angle prediction
                angle_start = time.time()
                bike_angle, confidence = None, None
                if success and model_manager.bike_angle_model is not None:
                    bike_angle, confidence = predict_bike_angle(
                        model_manager.bike_angle_model,
                        masked,
                        model_manager.num_bins,
                        model_manager.device,
                        model_manager.transform
                    )
                metrics["bikeAngleTime"] += (time.time() - angle_start) * 1000
                
                # Store frame data
                frame_data = {
                    "frameIndex": i,
                    "timestamp": i / output_fps,
                    "jointAngles": {
                        "knee": joint_angles.get("knee"),
                        "hip": joint_angles.get("hip"),
                        "elbow": joint_angles.get("elbow")
                    },
                    "bikeAngle": bike_angle,
                    "bikeConfidence": confidence,
                    "detectedSide": detected_side
                }
                frame_data_list.append(frame_data)
                
                # Create output frame with overlays
                output_frame = create_output_frame(
                    frame, masked, joint_angles, bike_angle, confidence,
                    detected_side, kpts_xy, kpts_conf, panel_width
                )
                out.write(output_frame)
                
                metrics["processedFrames"] = i + 1
                
                # Send progress update every 5 frames
                if i % 5 == 0 or i == frames_to_process - 1:
                    progress = ((i + 1) / frames_to_process) * 100
                    yield json.dumps({
                        "type": "progress",
                        "progress": progress,
                        "currentFrame": i + 1,
                        "status": f"Processing frame {i + 1}/{frames_to_process}"
                    }) + "\n"
                    await asyncio.sleep(0)  # Allow other tasks to run
            
            cap.release()
            out.release()
            
            metrics["totalTime"] = (time.time() - total_start) * 1000
            metrics["avgTimePerFrame"] = metrics["totalTime"] / max(1, metrics["processedFrames"])
            
            # Average the timing metrics
            n = metrics["processedFrames"]
            metrics["segmentationTime"] /= max(1, n)
            metrics["poseDetectionTime"] /= max(1, n)
            metrics["bikeAngleTime"] /= max(1, n)
            
            # Return final result
            yield json.dumps({
                "type": "complete",
                "result": {
                    "videoUrl": f"/api/video/{Path(output_path).name}",
                    "frames": frame_data_list,
                    "startFrame": start_frame,
                    "endFrame": end_frame,
                    "fps": output_fps,
                    "metrics": metrics
                }
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "message": str(e)
            }) + "\n"
        finally:
            # Cleanup temp files
            if os.path.exists(video_path):
                os.remove(video_path)
    
    return StreamingResponse(
        generate_response(),
        media_type="application/x-ndjson"
    )


def create_output_frame(
    frame: np.ndarray,
    masked: np.ndarray,
    joint_angles: Dict,
    bike_angle: Optional[float],
    confidence: Optional[float],
    detected_side: Optional[str],
    kpts_xy: Optional[np.ndarray],
    kpts_conf: Optional[np.ndarray],
    panel_width: int
) -> np.ndarray:
    """Create output frame with skeleton overlay and side panel."""
    
    h, w = frame.shape[:2]
    output = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    output[:, :w] = frame
    
    # Draw skeleton on frame
    if kpts_xy is not None and kpts_conf is not None and detected_side:
        draw_skeleton(output[:, :w], kpts_xy, kpts_conf, detected_side)
    
    # Side panel background
    output[:, w:] = (25, 25, 30)
    
    # Panel content
    panel_x = w + 15
    y = 30
    
    # Title
    cv2.putText(output, "BikeFit Analysis", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 40
    
    # Masked bike preview
    preview_size = 120
    masked_preview = cv2.resize(masked, (preview_size, preview_size))
    output[y:y+preview_size, panel_x:panel_x+preview_size] = masked_preview
    cv2.putText(output, "Bike Mask", (panel_x, y + preview_size + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y += preview_size + 50
    
    # Joint angles
    cv2.putText(output, "Joint Angles", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y += 30
    
    angle_colors = {
        "knee": (255, 217, 0),    # Cyan
        "hip": (50, 158, 245),    # Orange
        "elbow": (129, 211, 16)   # Green
    }
    
    for angle_name in ["knee", "hip", "elbow"]:
        color = angle_colors[angle_name]
        value = joint_angles.get(angle_name)
        text = f"{angle_name.title()}: {value:.1f}°" if value else f"{angle_name.title()}: --"
        cv2.putText(output, text, (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y += 25
    
    if detected_side:
        cv2.putText(output, f"Side: {detected_side.upper()}", (panel_x, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    y += 40
    
    # Bike angle
    cv2.putText(output, "Bike Angle", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y += 30
    
    if bike_angle is not None:
        cv2.putText(output, f"{bike_angle:.1f}°", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (99, 102, 241), 2)
        y += 30
        cv2.putText(output, f"Conf: {confidence:.0f}%", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    else:
        cv2.putText(output, "No detection", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    return output


def draw_skeleton(
    img: np.ndarray,
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    detected_side: str,
    min_conf: float = 0.3
) -> None:
    """Draw skeleton overlay on image."""
    
    if detected_side == "right":
        connections = [
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("right_shoulder", "right_hip"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        ]
        joints = ["right_shoulder", "right_elbow", "right_wrist",
                  "right_hip", "right_knee", "right_ankle"]
    else:
        connections = [
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("left_shoulder", "left_hip"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
        ]
        joints = ["left_shoulder", "left_elbow", "left_wrist",
                  "left_hip", "left_knee", "left_ankle"]
    
    # Draw connections
    for start_name, end_name in connections:
        start_idx = IDX[start_name]
        end_idx = IDX[end_name]
        
        if kpts_conf[start_idx] < min_conf or kpts_conf[end_idx] < min_conf:
            continue
        
        start_pt = (int(kpts_xy[start_idx][0]), int(kpts_xy[start_idx][1]))
        end_pt = (int(kpts_xy[end_idx][0]), int(kpts_xy[end_idx][1]))
        
        cv2.line(img, start_pt, end_pt, (0, 0, 0), 5)
        cv2.line(img, start_pt, end_pt, (0, 255, 255), 3)
    
    # Draw joints
    for joint_name in joints:
        idx = IDX[joint_name]
        if kpts_conf[idx] < min_conf:
            continue
        
        pt = (int(kpts_xy[idx][0]), int(kpts_xy[idx][1]))
        cv2.circle(img, pt, 8, (0, 0, 0), -1)
        cv2.circle(img, pt, 6, (255, 0, 255), -1)
        cv2.circle(img, pt, 6, (255, 255, 255), 1)


# Video file serving
TEMP_VIDEO_DIR = tempfile.gettempdir()

@app.get("/api/video/{filename}")
async def serve_video(filename: str):
    video_path = os.path.join(TEMP_VIDEO_DIR, filename)
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    return {"error": "Video not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

