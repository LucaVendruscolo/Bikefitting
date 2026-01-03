"""
Bike Segmenter - Masks bike pixels using YOLOv8 segmentation.

This is the EXACT same preprocessing as training to ensure model accuracy.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class BikeSegmenter:
    """
    Segments bikes from images using YOLO segmentation.
    
    IMPORTANT: This must match the preprocessing used during training exactly!
    - COCO classes: bicycle=1, motorcycle=3
    - Dilate mask by 5 pixels
    - Crop to square bounding box with 10px padding
    - Resize to 224x224
    - Background set to black (0, 0, 0)
    """
    
    # COCO class IDs for bikes
    BIKE_CLASSES = [1, 3]  # bicycle, motorcycle
    
    def __init__(self, model_path: str = "yolov8n-seg.pt", conf_threshold: float = 0.3):
        """
        Initialize the YOLO segmentation model.
        
        Args:
            model_path: Path to YOLOv8 segmentation model
            conf_threshold: Confidence threshold for detection
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
    
    def get_bike_mask(self, image: np.ndarray, dilate_pixels: int = 5) -> tuple:
        """
        Get segmentation mask for bike pixels only.
        
        Args:
            image: Input image (BGR format)
            dilate_pixels: Pixels to dilate the mask by (includes bike edges)
            
        Returns:
            (mask, success) - mask is binary (0 or 255), success indicates if bike found
        """
        h, w = image.shape[:2]
        results = self.model(image, verbose=False, conf=self.conf_threshold)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for result in results:
            if result.masks is not None:
                for i, cls in enumerate(result.boxes.cls):
                    cls_id = int(cls.item())
                    
                    # Only include bike classes (no people!)
                    if cls_id in self.BIKE_CLASSES:
                        seg_mask = result.masks.data[i].cpu().numpy()
                        # Resize mask to image size
                        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                        mask = np.maximum(mask, (seg_mask > 0.5).astype(np.uint8) * 255)
        
        if mask.max() > 0 and dilate_pixels > 0:
            # Dilate mask to ensure we capture bike edges
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)
            return mask, True
        
        return mask, False
    
    def mask_bike(self, image: np.ndarray, target_size: int = 224, 
                  dilate_pixels: int = 5) -> tuple:
        """
        Apply bike mask to image for model input.
        
        This is the EXACT preprocessing used during training:
        1. Segment bike pixels
        2. Dilate mask by 5 pixels
        3. Set background to black
        4. Crop to square bounding box with 10px padding
        5. Resize to 224x224
        
        Args:
            image: Input image (BGR format)
            target_size: Output size (must be 224 for the trained model)
            dilate_pixels: Pixels to dilate mask by
            
        Returns:
            (masked_image, raw_mask, success)
        """
        mask, success = self.get_bike_mask(image, dilate_pixels)
        
        if not success:
            # No bike detected - return black image
            return np.zeros((target_size, target_size, 3), dtype=np.uint8), mask, False
        
        # Apply mask (set background to black)
        masked = cv2.bitwise_and(image, image, mask=mask)
        
        # Find bounding box of mask for tight crop
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, bw, bh = cv2.boundingRect(coords)
            
            # Add padding
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
            
            # Ensure bounds
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
        
        return masked, mask, success

