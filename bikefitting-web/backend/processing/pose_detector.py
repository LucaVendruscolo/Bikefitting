"""
Pose Detector - Joint angle detection using YOLOv8-pose.

Detects cyclist pose and computes knee, hip, and elbow angles.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Tuple, Optional, List


# COCO keypoint names
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
KPT_IDX = {name: i for i, name in enumerate(COCO_KEYPOINTS)}

# Skeleton connections for drawing
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
    """Exponential moving average smoother for reducing jitter."""
    
    def __init__(self, alpha: float = 0.6):
        """
        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive.
        """
        self.alpha = alpha
        self.prev_xy = None
        self.prev_conf = None
    
    def smooth(self, kpts_xy: np.ndarray, kpts_conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply EMA smoothing to keypoints."""
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
        """Reset smoother state (call when switching videos)."""
        self.prev_xy = None
        self.prev_conf = None


class PoseDetector:
    """
    Detects cyclist pose and computes joint angles.
    
    Uses YOLOv8-pose for keypoint detection, then computes:
    - Knee angle (hip-knee-ankle)
    - Hip angle (shoulder-hip-knee)
    - Elbow angle (shoulder-elbow-wrist)
    """
    
    def __init__(self, model_path: str = "yolov8m-pose.pt", min_conf: float = 0.5):
        """
        Initialize pose detector.
        
        Args:
            model_path: Path to YOLOv8-pose model (m or l recommended)
            min_conf: Minimum keypoint confidence threshold
        """
        self.model = YOLO(model_path)
        self.min_conf = min_conf
        self.smoother = KeypointSmoother(alpha=0.6)
    
    def _get_side_mapping(self, side: str) -> Dict[str, str]:
        """Get joint name mapping for a side."""
        if side == "right":
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
    
    def _angle_between(self, a: Tuple[float, float], b: Tuple[float, float], 
                       c: Tuple[float, float]) -> float:
        """Compute angle ABC in degrees."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom == 0:
            return float("nan")
        cos_angle = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_angle)))
    
    def detect(self, image: np.ndarray, side: str = "auto", 
               apply_smoothing: bool = True) -> Dict:
        """
        Detect pose and compute joint angles.
        
        Args:
            image: Input image (BGR)
            side: "left", "right", or "auto" (detect which side faces camera)
            apply_smoothing: Whether to apply temporal smoothing
            
        Returns:
            Dict with:
                - joints: Dict of joint positions {name: (x, y, conf)}
                - angles: Dict of computed angles {knee_angle, hip_angle, elbow_angle}
                - keypoints_xy: Raw keypoint positions (17, 2)
                - keypoints_conf: Keypoint confidences (17,)
                - detected_side: Which side was detected ("left" or "right")
        """
        results = self.model.predict(image, verbose=False)
        
        # Default return for no detection
        empty_result = {
            "joints": {},
            "angles": {},
            "keypoints_xy": None,
            "keypoints_conf": None,
            "detected_side": None
        }
        
        if not results or results[0].keypoints is None:
            return empty_result
        
        r = results[0]
        kpts_xy = r.keypoints.xy.cpu().numpy()
        kpts_conf = r.keypoints.conf.cpu().numpy()
        
        if kpts_conf.size == 0:
            return empty_result
        
        # Select person with highest average confidence
        person_idx = int(np.argmax(kpts_conf.mean(axis=1)))
        raw_xy = kpts_xy[person_idx]
        raw_conf = kpts_conf[person_idx]
        
        # Apply temporal smoothing
        if apply_smoothing:
            raw_xy, raw_conf = self.smoother.smooth(raw_xy, raw_conf)
        
        # Determine which side to use
        def get_joints_for_side(which: str):
            mapping = self._get_side_mapping(which)
            joints = {}
            confidences = []
            for simple_name, coco_name in mapping.items():
                idx = KPT_IDX[coco_name]
                x, y = raw_xy[idx]
                c = raw_conf[idx]
                if c >= self.min_conf:
                    joints[simple_name] = (float(x), float(y), float(c))
                    confidences.append(float(c))
            return joints, len(confidences), np.mean(confidences) if confidences else 0
        
        # Auto-detect side or use specified
        if side.lower() in ("left", "right"):
            detected_side = side.lower()
            joints, _, _ = get_joints_for_side(detected_side)
        else:
            # Compare visibility of both sides
            joints_r, count_r, conf_r = get_joints_for_side("right")
            joints_l, count_l, conf_l = get_joints_for_side("left")
            
            if count_r == 0 and count_l == 0:
                return empty_result
            
            if (count_r > count_l) or (count_r == count_l and conf_r >= conf_l):
                detected_side = "right"
                joints = joints_r
            else:
                detected_side = "left"
                joints = joints_l
        
        if not joints:
            return empty_result
        
        # Compute angles
        def maybe_angle(a_name: str, b_name: str, c_name: str):
            if a_name not in joints or b_name not in joints or c_name not in joints:
                return None
            return self._angle_between(
                joints[a_name][:2],
                joints[b_name][:2],
                joints[c_name][:2]
            )
        
        angles = {}
        knee = maybe_angle("hip", "knee", "foot")
        if knee is not None:
            angles["knee_angle"] = knee
        hip = maybe_angle("shoulder", "hip", "knee")
        if hip is not None:
            angles["hip_angle"] = hip
        elbow = maybe_angle("shoulder", "elbow", "hand")
        if elbow is not None:
            angles["elbow_angle"] = elbow
        
        return {
            "joints": joints,
            "angles": angles,
            "keypoints_xy": raw_xy,
            "keypoints_conf": raw_conf,
            "detected_side": detected_side
        }
    
    def draw_skeleton(self, image: np.ndarray, keypoints_xy: np.ndarray,
                      keypoints_conf: np.ndarray, detected_side: str,
                      scale: float = 1.0) -> np.ndarray:
        """
        Draw skeleton overlay on image.
        
        Args:
            image: Image to draw on (will be modified)
            keypoints_xy: Keypoint positions
            keypoints_conf: Keypoint confidences
            detected_side: Which side to draw
            scale: Scale factor for keypoint positions
            
        Returns:
            Image with skeleton drawn
        """
        if detected_side == "right":
            connections = SKELETON_RIGHT
            joint_names = ["right_shoulder", "right_elbow", "right_wrist",
                          "right_hip", "right_knee", "right_ankle"]
        elif detected_side == "left":
            connections = SKELETON_LEFT
            joint_names = ["left_shoulder", "left_elbow", "left_wrist",
                          "left_hip", "left_knee", "left_ankle"]
        else:
            return image
        
        # Draw connections
        for start_name, end_name in connections:
            start_idx = KPT_IDX[start_name]
            end_idx = KPT_IDX[end_name]
            
            if keypoints_conf[start_idx] < 0.3 or keypoints_conf[end_idx] < 0.3:
                continue
            
            start_pt = (int(keypoints_xy[start_idx][0] * scale), 
                       int(keypoints_xy[start_idx][1] * scale))
            end_pt = (int(keypoints_xy[end_idx][0] * scale),
                     int(keypoints_xy[end_idx][1] * scale))
            
            # Draw with outline for visibility
            cv2.line(image, start_pt, end_pt, (0, 0, 0), 7)
            cv2.line(image, start_pt, end_pt, (0, 255, 255), 4)
        
        # Draw joints
        for joint_name in joint_names:
            idx = KPT_IDX[joint_name]
            if keypoints_conf[idx] < 0.3:
                continue
            pt = (int(keypoints_xy[idx][0] * scale), 
                 int(keypoints_xy[idx][1] * scale))
            cv2.circle(image, pt, 12, (0, 0, 0), -1)
            cv2.circle(image, pt, 10, (255, 0, 255), -1)
            cv2.circle(image, pt, 10, (255, 255, 255), 2)
        
        return image
    
    def reset_smoother(self):
        """Reset temporal smoothing (call between videos)."""
        self.smoother.reset()

