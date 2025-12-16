"""
Bike Masking Preprocessor

Uses YOLOv8 segmentation to mask only bike pixels from images (background set to black).
Also subsamples frames from 30Hz video to 5Hz to reduce redundant similar frames.

This preprocessing step:
1. Masks background pixels (sets non-bike areas to black)
2. Reduces frame count 6x (30Hz -> 5Hz) to avoid redundant training data
3. Speeds up training and improves model focus on bike features

Usage:
    python 1_preprocess.py --input_csv ../create_labeled_dataset/output/synchronized_dataset.csv --output_dir data
    
    # Custom frame rate (e.g., 10Hz from 30Hz source)
    python 1_preprocess.py --input_csv ../create_labeled_dataset/output/synchronized_dataset.csv --output_dir data --target_fps 10 --source_fps 30
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Mask bike pixels and subsample frames")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to CSV with frame_path and bike_angle_deg columns")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for masked images and new CSV")
    parser.add_argument("--yolo_model", type=str, default="yolov8n-seg.pt",
                        help="YOLO segmentation model to use")
    parser.add_argument("--target_size", type=int, default=224,
                        help="Target size for output images (square)")
    parser.add_argument("--conf_threshold", type=float, default=0.3,
                        help="Confidence threshold for bike detection")
    parser.add_argument("--source_fps", type=float, default=30.0,
                        help="Source video frame rate (Hz)")
    parser.add_argument("--target_fps", type=float, default=5.0,
                        help="Target frame rate after subsampling (Hz)")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base directory for resolving relative frame paths")
    parser.add_argument("--dilate_mask", type=int, default=5,
                        help="Pixels to dilate mask by (helps include bike edges)")
    parser.add_argument("--fallback_to_bbox", action="store_true", default=True,
                        help="Fall back to bounding box if segmentation fails")
    return parser.parse_args()


class BikeSegmenter:
    """Segments bikes from images using YOLO segmentation."""
    
    # COCO class IDs: bicycle=1, motorcycle=3
    BIKE_CLASSES = [1, 3]
    
    def __init__(self, model_name: str = "yolov8n-seg.pt", conf_threshold: float = 0.3):
        """Initialize the YOLO segmentation model."""
        print(f"Loading YOLO segmentation model: {model_name}")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        
        # Also load detection model as fallback
        self.detection_model = None
        
    def _load_detection_fallback(self):
        """Lazy load detection model for fallback."""
        if self.detection_model is None:
            print("Loading detection model as fallback...")
            self.detection_model = YOLO("yolov8n.pt")
    
    def get_bike_mask(self, image: np.ndarray, dilate_pixels: int = 5) -> tuple:
        """
        Get segmentation mask for bike only using YOLO-seg.
        
        Args:
            image: Input image (BGR)
            dilate_pixels: Number of pixels to dilate the mask
            
        Returns:
            (mask, success_flag) where mask is binary (0 or 255)
        """
        h, w = image.shape[:2]
        results = self.model(image, verbose=False, conf=self.conf_threshold)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        found_bike = False
        
        for result in results:
            if result.masks is not None:
                for i, cls in enumerate(result.boxes.cls):
                    cls_id = int(cls.item())
                    
                    # Only include bike classes (no people)
                    if cls_id in self.BIKE_CLASSES:
                        found_bike = True
                        seg_mask = result.masks.data[i].cpu().numpy()
                        # Resize mask to image size
                        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                        mask = np.maximum(mask, (seg_mask > 0.5).astype(np.uint8) * 255)
        
        if mask.max() > 0:
            # Dilate mask to ensure we get bike edges
            if dilate_pixels > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                    (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1))
                mask = cv2.dilate(mask, kernel, iterations=1)
            return mask, True
        
        return mask, False
    
    def get_bbox_mask(self, image: np.ndarray) -> tuple:
        """
        Fallback: Get bounding box mask using detection model.
        
        Returns:
            (mask, success_flag)
        """
        self._load_detection_fallback()
        
        h, w = image.shape[:2]
        results = self.detection_model(image, verbose=False, conf=self.conf_threshold)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        found_bike = False
        
        for result in results:
            boxes = result.boxes
            for i, cls in enumerate(boxes.cls):
                cls_id = int(cls.item())
                if cls_id in self.BIKE_CLASSES:
                    found_bike = True
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    mask[y1:y2, x1:x2] = 255
        
        return mask, found_bike
    
    def mask_bike(self, image: np.ndarray, target_size: int = 224, 
                  dilate_pixels: int = 5, fallback_to_bbox: bool = True) -> tuple:
        """
        Apply bike mask to image, setting non-bike pixels to black.
        
        Args:
            image: Input image (BGR)
            target_size: Output image size (square)
            dilate_pixels: Pixels to dilate mask by
            fallback_to_bbox: If True, use bounding box when segmentation fails
            
        Returns:
            (masked_image, success_flag)
        """
        # Try segmentation first
        mask, success = self.get_bike_mask(image, dilate_pixels)
        
        # Fallback to bounding box if segmentation failed
        if not success and fallback_to_bbox:
            mask, success = self.get_bbox_mask(image)
            if success:
                print("  Using bounding box fallback")
        
        if not success:
            # No bike detected at all - return black image
            print("  Warning: No bike detected")
            masked = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            return masked, False
        
        # Apply mask to image (set background to black)
        masked = cv2.bitwise_and(image, image, mask=mask)
        
        # Find bounding box of mask to crop tightly around bike
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, bw, bh = cv2.boundingRect(coords)
            
            # Add small padding
            pad = 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            bw = min(image.shape[1] - x, bw + 2 * pad)
            bh = min(image.shape[0] - y, bh + 2 * pad)
            
            # Make square (use larger dimension)
            if bw > bh:
                diff = bw - bh
                y = max(0, y - diff // 2)
                bh = bw
            else:
                diff = bh - bw
                x = max(0, x - diff // 2)
                bw = bh
            
            # Ensure we don't go out of bounds
            if y + bh > image.shape[0]:
                bh = image.shape[0] - y
            if x + bw > image.shape[1]:
                bw = image.shape[1] - x
            
            # Crop
            masked = masked[y:y+bh, x:x+bw]
        
        # Resize to target size
        if masked.shape[0] > 0 and masked.shape[1] > 0:
            masked = cv2.resize(masked, (target_size, target_size))
        else:
            masked = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            success = False
        
        return masked, success


def subsample_frames(df: pd.DataFrame, source_fps: float, target_fps: float) -> pd.DataFrame:
    """
    Subsample frames from source FPS to target FPS.
    Keeps every Nth frame where N = source_fps / target_fps.
    
    Processes each video separately to maintain proper subsampling.
    """
    if target_fps >= source_fps:
        print(f"Target FPS ({target_fps}) >= source FPS ({source_fps}), no subsampling needed")
        return df
    
    skip_factor = int(round(source_fps / target_fps))
    print(f"Subsampling: keeping every {skip_factor}th frame ({source_fps}Hz -> {target_fps}Hz)")
    
    # Group by source video and subsample each
    subsampled_dfs = []
    
    for video_name, video_df in df.groupby('source_video'):
        # Sort by frame number to ensure proper ordering
        video_df = video_df.sort_values('frame_number')
        
        # Keep every Nth frame
        subsampled = video_df.iloc[::skip_factor]
        subsampled_dfs.append(subsampled)
        
        print(f"  {video_name}: {len(video_df)} -> {len(subsampled)} frames")
    
    result = pd.concat(subsampled_dfs, ignore_index=True)
    print(f"Total: {len(df)} -> {len(result)} frames ({100*len(result)/len(df):.1f}%)")
    
    return result


def process_dataset(input_csv: str, output_dir: str, segmenter: BikeSegmenter,
                    target_size: int, dilate_pixels: int, fallback_to_bbox: bool,
                    source_fps: float, target_fps: float, base_dir: str = None):
    """Process entire dataset with masking and subsampling."""
    
    # Read input CSV
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} samples from {input_csv}")
    
    # Subsample frames first (reduces processing time significantly)
    df = subsample_frames(df, source_fps, target_fps)
    
    # Determine base directory for frame paths
    if base_dir is None:
        base_dir = Path(input_csv).parent
    else:
        base_dir = Path(base_dir)
    
    # Create output directories
    output_path = Path(output_dir)
    frames_dir = output_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    results = []
    success_count = 0
    fail_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Masking bikes"):
        frame_path = row["frame_path"]
        
        # Resolve full path
        full_path = base_dir / frame_path
        
        if not full_path.exists():
            print(f"  Warning: Image not found: {full_path}")
            continue
        
        # Read image
        image = cv2.imread(str(full_path))
        if image is None:
            print(f"  Warning: Could not read image: {full_path}")
            continue
        
        # Mask bike
        masked, success = segmenter.mask_bike(
            image, 
            target_size=target_size,
            dilate_pixels=dilate_pixels,
            fallback_to_bbox=fallback_to_bbox
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        # Save masked image
        output_filename = Path(frame_path).stem + "_masked.jpg"
        output_filepath = frames_dir / output_filename
        cv2.imwrite(str(output_filepath), masked)
        
        # Store result
        result_row = row.to_dict()
        result_row["original_frame_path"] = frame_path
        result_row["frame_path"] = f"frames/{output_filename}"
        result_row["segmentation_success"] = success
        results.append(result_row)
    
    # Save new CSV
    output_df = pd.DataFrame(results)
    output_csv_path = output_path / "dataset.csv"
    output_df.to_csv(output_csv_path, index=False)
    
    print(f"\n=== Processing Complete ===")
    print(f"Total processed: {len(results)}")
    print(f"Successful segmentations: {success_count} ({100*success_count/len(results):.1f}%)")
    print(f"Failed segmentations: {fail_count} ({100*fail_count/len(results):.1f}%)")
    print(f"Output directory: {output_path}")
    print(f"Output CSV: {output_csv_path}")
    
    return output_df


def main():
    args = parse_args()
    
    # Initialize segmenter
    segmenter = BikeSegmenter(
        model_name=args.yolo_model,
        conf_threshold=args.conf_threshold
    )
    
    # Process dataset
    process_dataset(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        segmenter=segmenter,
        target_size=args.target_size,
        dilate_pixels=args.dilate_mask,
        fallback_to_bbox=args.fallback_to_bbox,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
        base_dir=args.base_dir
    )


if __name__ == "__main__":
    main()
