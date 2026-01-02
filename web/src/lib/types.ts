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

// Processing timing metrics
export interface FrameMetrics {
  posePreprocess: number;
  poseInference: number;
  posePostprocess: number;
  segPreprocess: number;
  segInference: number;
  segPostprocess: number;
  anglePreprocess: number;
  angleInference: number;
  anglePostprocess: number;
  totalFrame: number;
}

// Aggregated metrics across all frames
export interface ProcessingMetrics {
  frameCount: number;
  avgTotalFrame: number;
  avgPosePreprocess: number;
  avgPoseInference: number;
  avgPosePostprocess: number;
  avgSegPreprocess: number;
  avgSegInference: number;
  avgSegPostprocess: number;
  avgAnglePreprocess: number;
  avgAngleInference: number;
  avgAnglePostprocess: number;
  modelLoadTime: number;
}

