"""
Video Processor - Generates annotated output video with all detections.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable
from tqdm import tqdm

from .bike_segmenter import BikeSegmenter
from .angle_predictor import BikeAnglePredictor
from .pose_detector import PoseDetector


def draw_angle_indicator(img: np.ndarray, angle: float, label: str, 
                         color: tuple, center: tuple, radius: int):
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
    cv2.putText(img, f"{angle:.1f}", (center[0] - 30, center[1] + radius + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_joint_angles_panel(img: np.ndarray, angles: Dict[str, float], 
                            detected_side: Optional[str], x: int, y: int, width: int) -> int:
    """Draw panel showing joint angles."""
    panel_height = 160
    
    # Background
    cv2.rectangle(img, (x, y), (x + width, y + panel_height), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + width, y + panel_height), (80, 80, 80), 2)
    
    # Title
    if detected_side:
        title = f"Joint Angles ({detected_side.upper()} side)"
        title_color = (255, 255, 255)
    else:
        title = "Joint Angles (no detection)"
        title_color = (100, 100, 100)
    
    cv2.putText(img, title, (x + 15, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2)
    
    # Angle values
    angle_y = y + 70
    angle_info = [
        ("knee_angle", "Knee", (50, 255, 255)),
        ("hip_angle", "Hip", (255, 150, 50)),
        ("elbow_angle", "Elbow", (150, 255, 150))
    ]
    
    for key, label, color in angle_info:
        if key in angles:
            text = f"{label}: {angles[key]:.1f} deg"
        else:
            text = f"{label}: --"
            color = (100, 100, 100)
        
        cv2.putText(img, text, (x + 20, angle_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        angle_y += 32
    
    return panel_height


class VideoProcessor:
    """
    Processes cycling videos with bike angle and pose detection.
    
    Generates annotated output video with:
    - Main video with skeleton overlay and bike mask outline
    - Side panel with masked bike image
    - Joint angles display
    - Bike angle indicator
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize all models.
        
        Args:
            model_path: Path to trained bike angle model
            device: 'cuda' or 'cpu'
        """
        self.device = device or ('cuda' if __import__('torch').cuda.is_available() else 'cpu')
        
        print(f"Initializing VideoProcessor on {self.device}")
        print("Loading bike segmenter (YOLOv8n-seg)...")
        self.segmenter = BikeSegmenter()
        
        print("Loading bike angle predictor (ConvNeXt)...")
        self.angle_predictor = BikeAnglePredictor(model_path, self.device)
        
        print("Loading pose detector (YOLOv8m-pose)...")
        self.pose_detector = PoseDetector()
        
        print("Models loaded!")
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        output_fps: int = 10,
        max_duration_sec: Optional[float] = None,
        start_time: float = 0,
        end_time: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict:
        """
        Process a video and generate annotated output.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            output_fps: Output video frame rate
            max_duration_sec: Maximum duration to process (None for full video)
            start_time: Start time in seconds
            end_time: End time in seconds (None for full video)
            progress_callback: Callback(current_frame, total_frames)
            
        Returns:
            Dict with processing stats
        """
        cap = cv2.VideoCapture(input_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_duration = total_frames / video_fps if video_fps > 0 else 0
        
        print(f"Input: {video_fps:.1f} FPS, {total_frames} frames, {orig_width}x{orig_height}, {video_duration:.1f}s")
        
        # Calculate frame range based on start/end times
        start_frame = int(start_time * video_fps) if start_time > 0 else 0
        if end_time and end_time > start_time:
            end_frame = min(int(end_time * video_fps), total_frames)
        else:
            end_frame = total_frames
        
        # Apply max_duration_sec limit
        if max_duration_sec:
            max_frames = int(max_duration_sec * video_fps)
            end_frame = min(end_frame, start_frame + max_frames)
        
        frames_in_range = end_frame - start_frame
        
        print(f"Processing range: {start_time:.1f}s to {(end_frame/video_fps):.1f}s ({frames_in_range} frames)")
        
        # Calculate frame skip for target output FPS
        frame_skip = max(1, int(video_fps / output_fps))
        
        frames_to_process = frames_in_range // frame_skip
        
        print(f"Processing {frames_to_process} frames (skip={frame_skip})")
        
        # Output video dimensions (1080p main + side panel)
        main_height = 1080
        main_width = int(orig_width * (main_height / orig_height))
        if main_width > 1920:
            main_width = 1920
            main_height = int(orig_height * (main_width / orig_width))
        
        panel_width = 420
        out_width = main_width + panel_width
        out_height = max(main_height, 1080)
        
        scale = main_height / orig_height
        
        # Setup output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (out_width, out_height))
        
        # Reset pose smoother for new video
        self.pose_detector.reset_smoother()
        
        # Processing stats
        bike_angles = []
        joint_angle_history = {"knee": [], "hip": [], "elbow": []}
        frames_with_bike = 0
        frames_with_pose = 0
        
        for i in tqdm(range(frames_to_process), desc="Processing"):
            frame_num = start_frame + (i * frame_skip)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Detect pose
            pose_result = self.pose_detector.detect(frame)
            detected_side = pose_result["detected_side"]
            joint_angles = pose_result["angles"]
            
            if detected_side:
                frames_with_pose += 1
                for key in ["knee", "hip", "elbow"]:
                    angle_key = f"{key}_angle"
                    if angle_key in joint_angles:
                        joint_angle_history[key].append(joint_angles[angle_key])
            
            # Segment and predict bike angle
            masked, mask, success = self.segmenter.mask_bike(frame)
            
            if success:
                bike_angle, confidence = self.angle_predictor.predict(masked)
                frames_with_bike += 1
                bike_angles.append(bike_angle)
            else:
                bike_angle, confidence = 0, 0
            
            # Create output frame
            output = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            output[:] = (25, 25, 25)
            
            # Resize main video
            main_display = cv2.resize(frame, (main_width, main_height))
            
            # Draw skeleton
            if pose_result["keypoints_xy"] is not None:
                main_display = self.pose_detector.draw_skeleton(
                    main_display,
                    pose_result["keypoints_xy"],
                    pose_result["keypoints_conf"],
                    detected_side,
                    scale=scale
                )
            
            # Draw bike mask outline
            if mask is not None and mask.max() > 0:
                mask_resized = cv2.resize(mask, (main_width, main_height))
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(main_display, contours, -1, (0, 255, 0), 3)
            
            # Place main video
            output[0:main_height, 0:main_width] = main_display
            
            # Title on main video
            cv2.putText(output, "Bike Fitting Analysis", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(output, "Bike Fitting Analysis", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Side panel
            panel_x = main_width + 20
            
            # Masked bike image
            masked_size = 280
            masked_display = cv2.resize(masked, (masked_size, masked_size))
            output[50:50+masked_size, panel_x:panel_x+masked_size] = masked_display
            cv2.putText(output, "Masked Bike (Model Input)", (panel_x, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            
            # Joint angles panel
            joint_panel_y = 50 + masked_size + 30
            draw_joint_angles_panel(output, joint_angles, detected_side, 
                                   panel_x, joint_panel_y, 300)
            
            # Bike angle indicator
            indicator_y = joint_panel_y + 180
            cv2.putText(output, "Bike Angle", (panel_x + 80, indicator_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            draw_angle_indicator(output, bike_angle, "Predicted", (50, 255, 100),
                               (panel_x + 140, indicator_y + 80), 55)
            
            # Frame info
            cv2.putText(output, f"Frame: {frame_num}", (panel_x, out_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
            
            if confidence > 0:
                cv2.putText(output, f"Confidence: {confidence:.0f}%", 
                           (panel_x, out_height - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
            
            out.write(output)
            
            if progress_callback:
                progress_callback(i + 1, frames_to_process)
        
        cap.release()
        out.release()
        
        # Compute stats
        stats = {
            "frames_processed": frames_to_process,
            "frames_with_bike": frames_with_bike,
            "frames_with_pose": frames_with_pose,
            "output_fps": output_fps,
            "output_resolution": f"{out_width}x{out_height}",
        }
        
        if bike_angles:
            stats["avg_bike_angle"] = float(np.mean(bike_angles))
            stats["bike_angle_std"] = float(np.std(bike_angles))
        
        for key in ["knee", "hip", "elbow"]:
            if joint_angle_history[key]:
                stats[f"avg_{key}_angle"] = float(np.mean(joint_angle_history[key]))
        
        print(f"\nProcessing complete!")
        print(f"Output saved to: {output_path}")
        print(f"Stats: {stats}")
        
        return stats

