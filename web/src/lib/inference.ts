/**
 * Client-side ML inference for BikeFitting AI
 * 
 * Uses ONNX Runtime Web loaded from CDN.
 * Pipeline matches generate_demo_video.py exactly:
 * 1. Pose detection (YOLOv8m-pose) → skeleton + joint angles
 * 2. Bike segmentation (YOLOv8n-seg) → bike mask
 * 3. Angle classification (ConvNeXt) → bike angle
 */

import { FrameData, Keypoint, Skeleton, JointAngles, FrameMetrics } from './types';

// Model URLs - loaded from GitHub LFS
const GITHUB_LFS_BASE = 'https://media.githubusercontent.com/media/LucaVendruscolo/Bikefitting/main/web/public/models';
const MODEL_URLS = {
  pose: `${GITHUB_LFS_BASE}/yolov8m-pose.onnx`,
  seg: `${GITHUB_LFS_BASE}/yolov8n-seg.onnx`,
  angle: `${GITHUB_LFS_BASE}/bike_angle.onnx`,
};

// Model config (must match training)
const BIKE_ANGLE_CONFIG = {
  numBins: 120,
  inputSize: 224,
};

// COCO keypoint indices
const KEYPOINT_NAMES = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
];

const IDX: Record<string, number> = {};
KEYPOINT_NAMES.forEach((name, i) => { IDX[name] = i; });

// Bike segmentation classes (bicycle=1, motorcycle=3)
const BIKE_CLASSES = [1, 3];

export interface ModelState {
  poseSession: any;
  segSession: any;
  angleSession: any;
  ort: any;
  isLoaded: boolean;
}

// Global ONNX Runtime instance
let ortModule: any = null;

/**
 * Load ONNX Runtime from CDN
 */
async function loadOrt(): Promise<any> {
  if (ortModule) return ortModule;
  
  if (typeof window !== 'undefined' && (window as any).ort) {
    ortModule = (window as any).ort;
    return ortModule;
  }
  
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort.min.js';
    script.async = true;
    script.onload = () => {
      ortModule = (window as any).ort;
      resolve(ortModule);
    };
    script.onerror = () => reject(new Error('Failed to load ONNX Runtime'));
    document.head.appendChild(script);
  });
}

/**
 * Load all ML models
 */
export async function loadModels(onProgress?: (progress: number) => void): Promise<ModelState> {
  onProgress?.(0);
  
  // Load ONNX Runtime
  const ort = await loadOrt();
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/';
  onProgress?.(10);

  // Load pose model
  console.log('Loading pose model...');
  const poseSession = await ort.InferenceSession.create(MODEL_URLS.pose, {
    executionProviders: ['wasm'],
  });
  onProgress?.(40);

  // Load segmentation model
  console.log('Loading segmentation model...');
  const segSession = await ort.InferenceSession.create(MODEL_URLS.seg, {
    executionProviders: ['wasm'],
  });
  onProgress?.(70);

  // Load angle model
  console.log('Loading angle model...');
  const angleSession = await ort.InferenceSession.create(MODEL_URLS.angle, {
    executionProviders: ['wasm'],
  });
  onProgress?.(100);

  console.log('All models loaded');
  
  return {
    poseSession,
    segSession,
    angleSession,
    ort,
    isLoaded: true,
  };
}

/**
 * Process a single video frame with timing metrics
 */
export async function processVideoFrame(
  imageData: ImageData,
  time: number,
  models: ModelState
): Promise<{ frame: FrameData; metrics: FrameMetrics }> {
  const { poseSession, segSession, angleSession, ort } = models;
  const frameStart = performance.now();

  // 1. Pose detection with timing
  const poseResult = await runPoseDetectionTimed(imageData, poseSession, ort);
  
  // 2. Bike segmentation + angle prediction with timing
  const bikeResult = await runBikeAnglePredictionTimed(imageData, segSession, angleSession, ort);

  const totalFrame = performance.now() - frameStart;

  return {
    frame: {
      time,
      skeleton: poseResult.skeleton,
      jointAngles: poseResult.angles,
      bikeAngle: bikeResult.angle,
      bikeMask: null,
    },
    metrics: {
      posePreprocess: poseResult.timing.preprocess,
      poseInference: poseResult.timing.inference,
      posePostprocess: poseResult.timing.postprocess,
      segPreprocess: bikeResult.timing.segPreprocess,
      segInference: bikeResult.timing.segInference,
      segPostprocess: bikeResult.timing.segPostprocess,
      anglePreprocess: bikeResult.timing.anglePreprocess,
      angleInference: bikeResult.timing.angleInference,
      anglePostprocess: bikeResult.timing.anglePostprocess,
      totalFrame,
    },
  };
}

/**
 * Run pose detection with timing metrics
 */
async function runPoseDetectionTimed(
  imageData: ImageData,
  session: any,
  ort: any
): Promise<{ 
  skeleton: Skeleton | null; 
  angles: JointAngles;
  timing: { preprocess: number; inference: number; postprocess: number };
}> {
  const { width, height } = imageData;
  const inputSize = 640;
  
  // Preprocess
  const t0 = performance.now();
  const inputTensor = preprocessYolo(imageData, inputSize, ort);
  const preprocessTime = performance.now() - t0;
  
  // Inference
  const t1 = performance.now();
  const feeds = { images: inputTensor };
  const results = await session.run(feeds);
  const inferenceTime = performance.now() - t1;
  
  // Postprocess
  const t2 = performance.now();
  const output = results.output0;
  
  if (!output) {
    return { 
      skeleton: null, 
      angles: { knee: null, hip: null, elbow: null },
      timing: { preprocess: preprocessTime, inference: inferenceTime, postprocess: performance.now() - t2 }
    };
  }

  const keypoints = parseYoloPose(output.data as Float32Array, width, height, inputSize);
  if (!keypoints) {
    return { 
      skeleton: null, 
      angles: { knee: null, hip: null, elbow: null },
      timing: { preprocess: preprocessTime, inference: inferenceTime, postprocess: performance.now() - t2 }
    };
  }

  const side = determineSide(keypoints);
  const angles = calculateJointAngles(keypoints, side);
  const postprocessTime = performance.now() - t2;

  return {
    skeleton: { keypoints, side },
    angles,
    timing: { preprocess: preprocessTime, inference: inferenceTime, postprocess: postprocessTime }
  };
}

/**
 * Run bike segmentation and angle prediction with timing
 * Matches 1_preprocess.py and 2_train.py exactly
 */
async function runBikeAnglePredictionTimed(
  imageData: ImageData,
  segSession: any,
  angleSession: any,
  ort: any
): Promise<{ 
  angle: number | null;
  timing: {
    segPreprocess: number;
    segInference: number;
    segPostprocess: number;
    anglePreprocess: number;
    angleInference: number;
    anglePostprocess: number;
  };
}> {
  const { width, height } = imageData;
  
  // 1. Segmentation
  let bikeMask: Uint8Array | null = null;
  let segPreprocess = 0, segInference = 0, segPostprocess = 0;
  
  try {
    const t0 = performance.now();
    const segInput = preprocessYolo(imageData, 640, ort);
    segPreprocess = performance.now() - t0;
    
    const t1 = performance.now();
    const segFeeds = { images: segInput };
    const segResults = await segSession.run(segFeeds);
    segInference = performance.now() - t1;
    
    const t2 = performance.now();
    bikeMask = parseSegmentation(segResults, width, height);
    segPostprocess = performance.now() - t2;
  } catch (e) {
    console.warn('Segmentation failed, using center crop fallback');
  }

  // 2. Create masked image
  const t3 = performance.now();
  const maskedImage = createMaskedBikeImage(imageData, bikeMask, BIKE_ANGLE_CONFIG.inputSize);
  const maskCreateTime = performance.now() - t3;
  segPostprocess += maskCreateTime; // Include in seg postprocess
  
  if (!maskedImage) {
    return { 
      angle: null,
      timing: { segPreprocess, segInference, segPostprocess, anglePreprocess: 0, angleInference: 0, anglePostprocess: 0 }
    };
  }

  // 3. Angle prediction
  const t4 = performance.now();
  const angleInput = preprocessForAngleModel(maskedImage, ort);
  const anglePreprocess = performance.now() - t4;
  
  const t5 = performance.now();
  const angleFeeds = { input: angleInput };
  const angleResults = await angleSession.run(angleFeeds);
  const angleInference = performance.now() - t5;
  
  const t6 = performance.now();
  const logits = angleResults.output?.data as Float32Array;
  if (!logits) {
    return { 
      angle: null,
      timing: { segPreprocess, segInference, segPostprocess, anglePreprocess, angleInference, anglePostprocess: performance.now() - t6 }
    };
  }
  
  const angle = binsToAngle(logits);
  const anglePostprocess = performance.now() - t6;
  
  return { 
    angle,
    timing: { segPreprocess, segInference, segPostprocess, anglePreprocess, angleInference, anglePostprocess }
  };
}

/**
 * Preprocess image for YOLO models (letterbox to square)
 */
function preprocessYolo(imageData: ImageData, size: number, ort: any): any {
  const { width, height, data } = imageData;
  
  // Create canvas for resizing
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  
  // Letterbox resize
  const scale = Math.min(size / width, size / height);
  const newW = Math.round(width * scale);
  const newH = Math.round(height * scale);
  const offsetX = (size - newW) / 2;
  const offsetY = (size - newH) / 2;
  
  // Gray background
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, size, size);
  
  // Draw original image
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = width;
  tempCanvas.height = height;
  const tempCtx = tempCanvas.getContext('2d')!;
  tempCtx.putImageData(imageData, 0, 0);
  ctx.drawImage(tempCanvas, offsetX, offsetY, newW, newH);
  
  // Get pixel data and convert to tensor [1, 3, H, W]
  const resized = ctx.getImageData(0, 0, size, size);
  const float32 = new Float32Array(3 * size * size);
  
  for (let i = 0; i < size * size; i++) {
    float32[i] = resized.data[i * 4] / 255;                    // R
    float32[size * size + i] = resized.data[i * 4 + 1] / 255;  // G
    float32[2 * size * size + i] = resized.data[i * 4 + 2] / 255; // B
  }
  
  return new ort.Tensor('float32', float32, [1, 3, size, size]);
}

/**
 * Parse YOLO pose output to keypoints
 */
function parseYoloPose(
  data: Float32Array,
  origW: number,
  origH: number,
  inputSize: number
): Keypoint[] | null {
  // YOLO pose output: [1, 56, 8400] where 56 = 4 (box) + 1 (conf) + 51 (17 kpts * 3)
  const numDetections = 8400;
  
  // Find best detection
  let bestIdx = -1;
  let bestConf = 0;
  for (let i = 0; i < numDetections; i++) {
    const conf = data[4 * numDetections + i];
    if (conf > bestConf) {
      bestConf = conf;
      bestIdx = i;
    }
  }
  
  if (bestIdx === -1 || bestConf < 0.3) return null;
  
  // Calculate scale/offset for coordinate conversion
  const scale = Math.min(inputSize / origW, inputSize / origH);
  const offsetX = (inputSize - origW * scale) / 2;
  const offsetY = (inputSize - origH * scale) / 2;
  
  // Extract keypoints
  const keypoints: Keypoint[] = [];
  for (let k = 0; k < 17; k++) {
    const baseIdx = 5 + k * 3;
    const x = data[baseIdx * numDetections + bestIdx];
    const y = data[(baseIdx + 1) * numDetections + bestIdx];
    const conf = data[(baseIdx + 2) * numDetections + bestIdx];
    
    keypoints.push({
      x: (x - offsetX) / scale,
      y: (y - offsetY) / scale,
      confidence: conf,
    });
  }
  
  return keypoints;
}

/**
 * Parse YOLOv8-seg output to bike mask
 * output0: [1, 116, 8400] - detections (4 box + 80 classes + 32 mask coeffs)
 * output1: [1, 32, 160, 160] - prototype masks
 */
function parseSegmentation(results: any, origW: number, origH: number): Uint8Array | null {
  const output0 = results.output0?.data as Float32Array;
  const output1 = results.output1?.data as Float32Array;
  
  if (!output0 || !output1) return null;
  
  const numDetections = 8400;
  const numClasses = 80;
  const numMaskCoeffs = 32;
  const protoH = 160;
  const protoW = 160;
  
  // Find best bike detection (class 1=bicycle, 3=motorcycle)
  let bestIdx = -1;
  let bestConf = 0.3; // minimum threshold
  
  for (let i = 0; i < numDetections; i++) {
    // Class scores start at index 4 (after cx, cy, w, h)
    const bikeScore = output0[(4 + 1) * numDetections + i]; // class 1 (bicycle)
    const motoScore = output0[(4 + 3) * numDetections + i]; // class 3 (motorcycle)
    const score = Math.max(bikeScore, motoScore);
    
    if (score > bestConf) {
      bestConf = score;
      bestIdx = i;
    }
  }
  
  if (bestIdx === -1) return null;
  
  // Get bounding box for this detection
  const cx = output0[0 * numDetections + bestIdx];
  const cy = output0[1 * numDetections + bestIdx];
  const w = output0[2 * numDetections + bestIdx];
  const h = output0[3 * numDetections + bestIdx];
  
  // Get mask coefficients (last 32 values per detection)
  const maskCoeffs = new Float32Array(numMaskCoeffs);
  for (let j = 0; j < numMaskCoeffs; j++) {
    maskCoeffs[j] = output0[(4 + numClasses + j) * numDetections + bestIdx];
  }
  
  // Matrix multiply: coeffs [32] @ protos [32, 160*160] -> [160*160]
  const maskFlat = new Float32Array(protoH * protoW);
  for (let p = 0; p < protoH * protoW; p++) {
    let sum = 0;
    for (let c = 0; c < numMaskCoeffs; c++) {
      sum += maskCoeffs[c] * output1[c * protoH * protoW + p];
    }
    // Sigmoid activation
    maskFlat[p] = 1 / (1 + Math.exp(-sum));
  }
  
  // Scale factor from 640 input to prototype size (160)
  const inputSize = 640;
  const scale = Math.min(inputSize / origW, inputSize / origH);
  const scaledW = Math.round(origW * scale);
  const scaledH = Math.round(origH * scale);
  const offsetX = (inputSize - scaledW) / 2;
  const offsetY = (inputSize - scaledH) / 2;
  
  // Convert box coords from input space (640) to proto space (160)
  const protoScale = protoW / inputSize;
  const boxX1 = Math.max(0, Math.floor((cx - w/2) * protoScale));
  const boxY1 = Math.max(0, Math.floor((cy - h/2) * protoScale));
  const boxX2 = Math.min(protoW, Math.ceil((cx + w/2) * protoScale));
  const boxY2 = Math.min(protoH, Math.ceil((cy + h/2) * protoScale));
  
  // Create full-size mask
  const mask = new Uint8Array(origW * origH);
  
  // Map proto mask to original image coordinates
  for (let py = boxY1; py < boxY2; py++) {
    for (let px = boxX1; px < boxX2; px++) {
      const maskVal = maskFlat[py * protoW + px];
      if (maskVal > 0.5) {
        // Convert proto coord to input space
        const inputX = px / protoScale;
        const inputY = py / protoScale;
        
        // Convert input space to original image space
        const origX = Math.round((inputX - offsetX) / scale);
        const origY = Math.round((inputY - offsetY) / scale);
        
        if (origX >= 0 && origX < origW && origY >= 0 && origY < origH) {
          mask[origY * origW + origX] = 255;
        }
      }
    }
  }
  
  // Dilate mask slightly (5 pixel equivalent)
  const dilatedMask = dilateMask(mask, origW, origH, 5);
  
  return dilatedMask;
}

/**
 * Simple dilation of binary mask
 */
function dilateMask(mask: Uint8Array, w: number, h: number, radius: number): Uint8Array {
  const result = new Uint8Array(w * h);
  
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x] > 0) {
        // Set all pixels in radius
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
              if (dx*dx + dy*dy <= radius*radius) {
                result[ny * w + nx] = 255;
              }
            }
          }
        }
      }
    }
  }
  
  return result;
}

/**
 * Create masked bike image (matches 1_preprocess.py exactly)
 * 1. Apply mask (set non-bike pixels to black)
 * 2. Find bounding box of mask
 * 3. Make square with padding
 * 4. Crop and resize to targetSize
 */
function createMaskedBikeImage(
  imageData: ImageData,
  mask: Uint8Array | null,
  targetSize: number
): ImageData | null {
  const { width, height, data } = imageData;
  
  // If no mask, use center crop as fallback
  if (!mask) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCtx.putImageData(imageData, 0, 0);
    
    const canvas = document.createElement('canvas');
    canvas.width = targetSize;
    canvas.height = targetSize;
    const ctx = canvas.getContext('2d')!;
    
    const size = Math.min(width, height);
    const sx = (width - size) / 2;
    const sy = (height - size) / 2;
    ctx.drawImage(tempCanvas, sx, sy, size, size, 0, 0, targetSize, targetSize);
    return ctx.getImageData(0, 0, targetSize, targetSize);
  }
  
  // Apply mask - set non-bike pixels to black (like cv2.bitwise_and)
  const maskedData = new ImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    if (mask[i] > 0) {
      maskedData.data[i * 4] = data[i * 4];       // R
      maskedData.data[i * 4 + 1] = data[i * 4 + 1]; // G
      maskedData.data[i * 4 + 2] = data[i * 4 + 2]; // B
      maskedData.data[i * 4 + 3] = 255;            // A
    } else {
      // Black background (like Python cv2.bitwise_and)
      maskedData.data[i * 4] = 0;
      maskedData.data[i * 4 + 1] = 0;
      maskedData.data[i * 4 + 2] = 0;
      maskedData.data[i * 4 + 3] = 255;
    }
  }
  
  // Find bounding box of mask (like cv2.findNonZero + boundingRect)
  let minX = width, minY = height, maxX = 0, maxY = 0;
  let hasPixels = false;
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (mask[y * width + x] > 0) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
        hasPixels = true;
      }
    }
  }
  
  if (!hasPixels) {
    // No mask found, return black image
    const canvas = document.createElement('canvas');
    canvas.width = targetSize;
    canvas.height = targetSize;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, targetSize, targetSize);
    return ctx.getImageData(0, 0, targetSize, targetSize);
  }
  
  // Add padding (like Python pad=10)
  const pad = 10;
  let x = Math.max(0, minX - pad);
  let y = Math.max(0, minY - pad);
  let bw = Math.min(width - x, maxX - minX + 2 * pad);
  let bh = Math.min(height - y, maxY - minY + 2 * pad);
  
  // Make square (like Python code)
  if (bw > bh) {
    const diff = bw - bh;
    y = Math.max(0, y - Math.floor(diff / 2));
    bh = bw;
  } else {
    const diff = bh - bw;
    x = Math.max(0, x - Math.floor(diff / 2));
    bw = bh;
  }
  
  // Clamp to image bounds
  if (y + bh > height) bh = height - y;
  if (x + bw > width) bw = width - x;
  
  // Create canvas with masked image
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = width;
  tempCanvas.height = height;
  const tempCtx = tempCanvas.getContext('2d')!;
  tempCtx.putImageData(maskedData, 0, 0);
  
  // Crop and resize
  const canvas = document.createElement('canvas');
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d')!;
  
  // Fill with black first (in case crop is partially outside)
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, targetSize, targetSize);
  
  // Draw cropped region
  ctx.drawImage(tempCanvas, x, y, bw, bh, 0, 0, targetSize, targetSize);
  
  return ctx.getImageData(0, 0, targetSize, targetSize);
}

/**
 * Preprocess image for angle model (ImageNet normalization)
 */
function preprocessForAngleModel(imageData: ImageData, ort: any): any {
  const size = BIKE_ANGLE_CONFIG.inputSize;
  const { data } = imageData;
  
  // ImageNet normalization constants
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  
  const float32 = new Float32Array(3 * size * size);
  
  for (let i = 0; i < size * size; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    
    float32[i] = (r - mean[0]) / std[0];
    float32[size * size + i] = (g - mean[1]) / std[1];
    float32[2 * size * size + i] = (b - mean[2]) / std[2];
  }
  
  return new ort.Tensor('float32', float32, [1, 3, size, size]);
}

/**
 * Convert bin logits to angle using circular mean (same as training)
 */
function binsToAngle(logits: Float32Array): number {
  const numBins = BIKE_ANGLE_CONFIG.numBins;
  const binSize = 360 / numBins;
  
  // Softmax
  const logitsArray = Array.from(logits);
  const maxLogit = Math.max(...logitsArray);
  const expLogits = logitsArray.map(l => Math.exp(l - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  const probs = expLogits.map(e => e / sumExp);
  
  // Bin centers from -180 to 180
  const binCenters: number[] = [];
  for (let i = 0; i < numBins; i++) {
    binCenters.push(-180 + binSize / 2 + i * binSize);
  }
  
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

/**
 * Determine which side of the body is more visible
 */
function determineSide(keypoints: Keypoint[]): 'left' | 'right' | null {
  const leftJoints = [IDX.left_shoulder, IDX.left_elbow, IDX.left_hip, IDX.left_knee, IDX.left_ankle];
  const rightJoints = [IDX.right_shoulder, IDX.right_elbow, IDX.right_hip, IDX.right_knee, IDX.right_ankle];
  
  const leftConf = leftJoints.reduce((sum, idx) => sum + (keypoints[idx]?.confidence || 0), 0);
  const rightConf = rightJoints.reduce((sum, idx) => sum + (keypoints[idx]?.confidence || 0), 0);
  
  if (leftConf < 1.5 && rightConf < 1.5) return null;
  return leftConf > rightConf ? 'left' : 'right';
}

/**
 * Calculate joint angles from keypoints
 */
function calculateJointAngles(keypoints: Keypoint[], side: 'left' | 'right' | null): JointAngles {
  if (!side) return { knee: null, hip: null, elbow: null };
  
  const prefix = side === 'left' ? 'left_' : 'right_';
  
  const getPoint = (name: string) => {
    const idx = IDX[prefix + name];
    const kp = keypoints[idx];
    if (!kp || kp.confidence < 0.3) return null;
    return { x: kp.x, y: kp.y };
  };

  const calcAngle = (a: any, b: any, c: any) => {
    if (!a || !b || !c) return null;
    
    const ba = { x: a.x - b.x, y: a.y - b.y };
    const bc = { x: c.x - b.x, y: c.y - b.y };
    
    const dot = ba.x * bc.x + ba.y * bc.y;
    const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y);
    const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y);
    
    if (magBA === 0 || magBC === 0) return null;
    
    const cosAngle = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
    return Math.acos(cosAngle) * 180 / Math.PI;
  };

  return {
    knee: calcAngle(getPoint('hip'), getPoint('knee'), getPoint('ankle')),
    hip: calcAngle(getPoint('shoulder'), getPoint('hip'), getPoint('knee')),
    elbow: calcAngle(getPoint('shoulder'), getPoint('elbow'), getPoint('wrist')),
  };
}
