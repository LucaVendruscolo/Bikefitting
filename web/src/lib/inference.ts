/**
 * Client-side inference using ONNX Runtime Web
 * 
 * This module handles loading and running the ML models:
 * - YOLOv8m-pose: Body pose detection
 * - YOLOv8n-seg: Bike segmentation (for masking)
 * - ConvNeXt: Bike angle prediction
 * 
 * ONNX Runtime is loaded from CDN to avoid webpack bundling issues.
 */

// Model paths (relative to public folder)
const MODEL_PATHS = {
  pose: 'models/yolov8m-pose.onnx',
  segmentation: 'models/yolov8n-seg.onnx',
  bikeAngle: 'models/bike_angle.onnx',
};

// Config for bike angle model
const BIKE_ANGLE_CONFIG = {
  numBins: 120,
  inputSize: 224,
};

export interface ModelState {
  poseSession: any | null;
  segSession: any | null;
  bikeAngleSession: any | null;
  ort: any;
}

export interface FrameResults {
  jointAngles: {
    knee: number | null;
    hip: number | null;
    elbow: number | null;
  };
  bikeAngle: number | null;
  skeleton: {
    keypoints: Array<{ x: number; y: number; confidence: number }>;
    side: 'left' | 'right' | null;
  } | null;
  bikeMask: ImageData | null;
}

// Keypoint indices (COCO format)
const KEYPOINTS = {
  nose: 0,
  left_eye: 1,
  right_eye: 2,
  left_ear: 3,
  right_ear: 4,
  left_shoulder: 5,
  right_shoulder: 6,
  left_elbow: 7,
  right_elbow: 8,
  left_wrist: 9,
  right_wrist: 10,
  left_hip: 11,
  right_hip: 12,
  left_knee: 13,
  right_knee: 14,
  left_ankle: 15,
  right_ankle: 16,
};

// Global ONNX Runtime reference
let ortInstance: any = null;

/**
 * Load ONNX Runtime from CDN
 */
async function loadOnnxRuntime(): Promise<any> {
  if (ortInstance) return ortInstance;
  
  // Check if already loaded via script tag
  if (typeof window !== 'undefined' && (window as any).ort) {
    ortInstance = (window as any).ort;
    return ortInstance;
  }
  
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort.min.js';
    script.async = true;
    script.onload = () => {
      ortInstance = (window as any).ort;
      resolve(ortInstance);
    };
    script.onerror = () => reject(new Error('Failed to load ONNX Runtime'));
    document.head.appendChild(script);
  });
}

/**
 * Load all models
 */
export async function loadModels(
  onProgress?: (progress: number) => void
): Promise<ModelState> {
  const state: ModelState = {
    poseSession: null,
    segSession: null,
    bikeAngleSession: null,
    ort: null,
  };

  try {
    // Load ONNX Runtime from CDN
    onProgress?.(0);
    console.log('Loading ONNX Runtime...');
    const ort = await loadOnnxRuntime();
    state.ort = ort;
    onProgress?.(10);
    
    // Configure WASM paths
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/';
    
    // Load pose model (largest, load first)
    console.log('Loading pose model...');
    state.poseSession = await ort.InferenceSession.create(MODEL_PATHS.pose, {
      executionProviders: ['wasm'],
    });
    onProgress?.(40);

    // Load segmentation model
    console.log('Loading segmentation model...');
    state.segSession = await ort.InferenceSession.create(MODEL_PATHS.segmentation, {
      executionProviders: ['wasm'],
    });
    onProgress?.(70);

    // Load bike angle model
    console.log('Loading bike angle model...');
    state.bikeAngleSession = await ort.InferenceSession.create(MODEL_PATHS.bikeAngle, {
      executionProviders: ['wasm'],
    });
    onProgress?.(100);

    console.log('All models loaded');
    return state;

  } catch (error) {
    console.error('Failed to load models:', error);
    throw error;
  }
}

/**
 * Process a single frame
 */
export async function processFrame(
  imageData: ImageData,
  models: ModelState
): Promise<FrameResults> {
  const results: FrameResults = {
    jointAngles: { knee: null, hip: null, elbow: null },
    bikeAngle: null,
    skeleton: null,
    bikeMask: null,
  };

  // Run pose detection
  if (models.poseSession) {
    try {
      const poseResult = await runPoseDetection(imageData, models.poseSession, models.ort);
      if (poseResult) {
        results.skeleton = poseResult.skeleton;
        results.jointAngles = poseResult.angles;
      }
    } catch (e) {
      console.warn('Pose detection failed:', e);
    }
  }

  // Run bike segmentation and angle prediction
  if (models.segSession && models.bikeAngleSession) {
    try {
      const bikeResult = await runBikeAnalysis(
        imageData,
        models.segSession,
        models.bikeAngleSession,
        models.ort
      );
      results.bikeAngle = bikeResult.angle;
      results.bikeMask = bikeResult.mask;
    } catch (e) {
      console.warn('Bike analysis failed:', e);
    }
  }

  return results;
}

/**
 * Run pose detection
 */
async function runPoseDetection(
  imageData: ImageData,
  session: any,
  ort: any
): Promise<{
  skeleton: { keypoints: Array<{ x: number; y: number; confidence: number }>; side: 'left' | 'right' | null };
  angles: { knee: number | null; hip: number | null; elbow: number | null };
} | null> {
  // Preprocess image for YOLO (resize to 640x640, normalize)
  const inputTensor = preprocessForYolo(imageData, 640, ort);
  
  // Run inference
  const feeds = { images: inputTensor };
  const results = await session.run(feeds);
  
  // Parse results (YOLO pose output format)
  const output = results.output0;
  if (!output) return null;

  const keypoints = parseYoloPoseOutput(output.data as Float32Array, imageData.width, imageData.height);
  if (!keypoints) return null;

  // Determine which side is more visible
  const side = determineSide(keypoints);
  
  // Calculate angles
  const angles = calculateJointAngles(keypoints, side);

  return {
    skeleton: { keypoints, side },
    angles,
  };
}

/**
 * Preprocess image for YOLO
 */
function preprocessForYolo(imageData: ImageData, size: number, ort: any): any {
  const { width, height } = imageData;
  
  // Create canvas for resizing
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  
  // Create temporary canvas with original image
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = width;
  tempCanvas.height = height;
  const tempCtx = tempCanvas.getContext('2d')!;
  tempCtx.putImageData(imageData, 0, 0);
  
  // Resize with letterboxing
  const scale = Math.min(size / width, size / height);
  const newWidth = Math.round(width * scale);
  const newHeight = Math.round(height * scale);
  const offsetX = (size - newWidth) / 2;
  const offsetY = (size - newHeight) / 2;
  
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(tempCanvas, offsetX, offsetY, newWidth, newHeight);
  
  // Get pixel data
  const resizedData = ctx.getImageData(0, 0, size, size);
  
  // Convert to tensor format [1, 3, H, W] with normalization
  const float32Data = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    const r = resizedData.data[i * 4] / 255;
    const g = resizedData.data[i * 4 + 1] / 255;
    const b = resizedData.data[i * 4 + 2] / 255;
    float32Data[i] = r;
    float32Data[size * size + i] = g;
    float32Data[2 * size * size + i] = b;
  }
  
  return new ort.Tensor('float32', float32Data, [1, 3, size, size]);
}

/**
 * Parse YOLO pose output
 */
function parseYoloPoseOutput(
  data: Float32Array,
  origWidth: number,
  origHeight: number
): Array<{ x: number; y: number; confidence: number }> | null {
  // YOLO pose output format: [1, 56, 8400] where 56 = 4 (box) + 1 (conf) + 51 (17 keypoints * 3)
  // This is a simplified parser - real implementation needs proper NMS
  
  const numDetections = 8400;
  
  let bestIdx = -1;
  let bestConf = 0;
  
  // Find best detection
  for (let i = 0; i < numDetections; i++) {
    const conf = data[4 * numDetections + i];
    if (conf > bestConf) {
      bestConf = conf;
      bestIdx = i;
    }
  }
  
  if (bestIdx === -1 || bestConf < 0.3) return null;
  
  // Extract keypoints
  const keypoints: Array<{ x: number; y: number; confidence: number }> = [];
  
  for (let k = 0; k < 17; k++) {
    const kpOffset = 5 + k * 3; // 5 = box (4) + conf (1), then 3 values per keypoint
    const x = data[kpOffset * numDetections + bestIdx];
    const y = data[(kpOffset + 1) * numDetections + bestIdx];
    const conf = data[(kpOffset + 2) * numDetections + bestIdx];
    
    // Scale back to original image size
    const scale = Math.min(640 / origWidth, 640 / origHeight);
    const offsetX = (640 - origWidth * scale) / 2;
    const offsetY = (640 - origHeight * scale) / 2;
    
    keypoints.push({
      x: (x - offsetX) / scale,
      y: (y - offsetY) / scale,
      confidence: conf,
    });
  }
  
  return keypoints;
}

/**
 * Determine which side is more visible
 */
function determineSide(keypoints: Array<{ x: number; y: number; confidence: number }>): 'left' | 'right' | null {
  const leftConfs = [
    keypoints[KEYPOINTS.left_shoulder]?.confidence || 0,
    keypoints[KEYPOINTS.left_elbow]?.confidence || 0,
    keypoints[KEYPOINTS.left_hip]?.confidence || 0,
    keypoints[KEYPOINTS.left_knee]?.confidence || 0,
    keypoints[KEYPOINTS.left_ankle]?.confidence || 0,
  ];
  
  const rightConfs = [
    keypoints[KEYPOINTS.right_shoulder]?.confidence || 0,
    keypoints[KEYPOINTS.right_elbow]?.confidence || 0,
    keypoints[KEYPOINTS.right_hip]?.confidence || 0,
    keypoints[KEYPOINTS.right_knee]?.confidence || 0,
    keypoints[KEYPOINTS.right_ankle]?.confidence || 0,
  ];
  
  const leftAvg = leftConfs.reduce((a, b) => a + b, 0) / leftConfs.length;
  const rightAvg = rightConfs.reduce((a, b) => a + b, 0) / rightConfs.length;
  
  if (leftAvg < 0.3 && rightAvg < 0.3) return null;
  
  return leftAvg > rightAvg ? 'left' : 'right';
}

/**
 * Calculate joint angles
 */
function calculateJointAngles(
  keypoints: Array<{ x: number; y: number; confidence: number }>,
  side: 'left' | 'right' | null
): { knee: number | null; hip: number | null; elbow: number | null } {
  if (!side) return { knee: null, hip: null, elbow: null };
  
  const prefix = side === 'left' ? 'left_' : 'right_';
  
  const shoulder = keypoints[KEYPOINTS[`${prefix}shoulder` as keyof typeof KEYPOINTS]];
  const elbow = keypoints[KEYPOINTS[`${prefix}elbow` as keyof typeof KEYPOINTS]];
  const wrist = keypoints[KEYPOINTS[`${prefix}wrist` as keyof typeof KEYPOINTS]];
  const hip = keypoints[KEYPOINTS[`${prefix}hip` as keyof typeof KEYPOINTS]];
  const knee = keypoints[KEYPOINTS[`${prefix}knee` as keyof typeof KEYPOINTS]];
  const ankle = keypoints[KEYPOINTS[`${prefix}ankle` as keyof typeof KEYPOINTS]];
  
  const minConf = 0.3;
  
  return {
    knee: calculateAngle(hip, knee, ankle, minConf),
    hip: calculateAngle(shoulder, hip, knee, minConf),
    elbow: calculateAngle(shoulder, elbow, wrist, minConf),
  };
}

/**
 * Calculate angle between three points
 */
function calculateAngle(
  a: { x: number; y: number; confidence: number } | undefined,
  b: { x: number; y: number; confidence: number } | undefined,
  c: { x: number; y: number; confidence: number } | undefined,
  minConf: number
): number | null {
  if (!a || !b || !c) return null;
  if (a.confidence < minConf || b.confidence < minConf || c.confidence < minConf) return null;
  
  const ba = { x: a.x - b.x, y: a.y - b.y };
  const bc = { x: c.x - b.x, y: c.y - b.y };
  
  const dot = ba.x * bc.x + ba.y * bc.y;
  const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y);
  const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y);
  
  if (magBA === 0 || magBC === 0) return null;
  
  const cosAngle = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
  return Math.acos(cosAngle) * (180 / Math.PI);
}

/**
 * Run bike segmentation and angle prediction
 */
async function runBikeAnalysis(
  imageData: ImageData,
  segSession: any,
  angleSession: any,
  ort: any
): Promise<{ angle: number | null; mask: ImageData | null }> {
  // For now, just run the angle model on the whole image
  // TODO: Implement proper bike segmentation and masking
  
  const inputTensor = preprocessForBikeAngle(imageData, ort);
  
  const feeds = { input: inputTensor };
  const results = await angleSession.run(feeds);
  
  const output = results.output?.data as Float32Array;
  if (!output) return { angle: null, mask: null };
  
  // Convert logits to angle using circular mean
  const angle = calculateAngleFromBins(output);
  
  return { angle, mask: null };
}

/**
 * Preprocess image for bike angle model (224x224, ImageNet normalization)
 */
function preprocessForBikeAngle(imageData: ImageData, ort: any): any {
  const size = BIKE_ANGLE_CONFIG.inputSize;
  
  // Create canvas for resizing
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  
  // Create temporary canvas with original image
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = imageData.width;
  tempCanvas.height = imageData.height;
  const tempCtx = tempCanvas.getContext('2d')!;
  tempCtx.putImageData(imageData, 0, 0);
  
  // Resize
  ctx.drawImage(tempCanvas, 0, 0, size, size);
  
  // Get pixel data
  const resizedData = ctx.getImageData(0, 0, size, size);
  
  // ImageNet normalization
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  
  const float32Data = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    const r = (resizedData.data[i * 4] / 255 - mean[0]) / std[0];
    const g = (resizedData.data[i * 4 + 1] / 255 - mean[1]) / std[1];
    const b = (resizedData.data[i * 4 + 2] / 255 - mean[2]) / std[2];
    float32Data[i] = r;
    float32Data[size * size + i] = g;
    float32Data[2 * size * size + i] = b;
  }
  
  return new ort.Tensor('float32', float32Data, [1, 3, size, size]);
}

/**
 * Calculate angle from bin probabilities using circular mean
 */
function calculateAngleFromBins(logits: Float32Array): number {
  const numBins = BIKE_ANGLE_CONFIG.numBins;
  
  // Softmax
  const logitsArray = Array.from(logits);
  const maxLogit = Math.max(...logitsArray);
  const expLogits = logitsArray.map(l => Math.exp(l - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  const probs = expLogits.map(e => e / sumExp);
  
  // Calculate bin centers
  const binSize = 360 / numBins;
  const binCenters = Array.from({ length: numBins }, (_, i) => -180 + binSize / 2 + i * binSize);
  
  // Circular mean using sin/cos
  let sinSum = 0;
  let cosSum = 0;
  for (let i = 0; i < numBins; i++) {
    const rad = binCenters[i] * Math.PI / 180;
    sinSum += probs[i] * Math.sin(rad);
    cosSum += probs[i] * Math.cos(rad);
  }
  
  return Math.atan2(sinSum, cosSum) * 180 / Math.PI;
}
