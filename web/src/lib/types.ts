/**
 * Types for the BikeFitting AI application
 */

// Keypoint data from pose detection
export interface Keypoint {
  x: number;
  y: number;
  confidence: number;
}

// All 17 COCO keypoints
export interface Skeleton {
  keypoints: Keypoint[];
  side: 'left' | 'right' | null;
}

// Joint angles calculated from skeleton
export interface JointAngles {
  knee: number | null;
  hip: number | null;
  elbow: number | null;
}

// Processed data for a single frame
export interface FrameData {
  time: number;
  skeleton: Skeleton | null;
  jointAngles: JointAngles;
  bikeAngle: number | null;
  bikeMask: ImageData | null;
}

// Model loading state
export interface ModelState {
  poseModel: any;
  segModel: any;
  angleModel: any;
  isLoaded: boolean;
}

// Processing settings
export interface ProcessingSettings {
  startTime: number;
  endTime: number;
  fps: number;
}

// Current angles for display
export interface CurrentAngles {
  knee: number | null;
  hip: number | null;
  elbow: number | null;
  bike: number | null;
}

