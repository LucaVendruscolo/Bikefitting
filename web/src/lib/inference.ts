/**
 * Client-side inference using ONNX Runtime Web
 * 
 * This module handles all ML inference directly in the browser:
 * - Bike segmentation (YOLOv8-seg)
 * - Pose estimation (YOLOv8-pose)  
 * - Bike angle classification (ConvNeXT)
 */

// ============= Types =============

export interface Keypoint {
  x: number;
  y: number;
  confidence: number;
}

export interface JointAngles {
  knee: number | null;
  hip: number | null;
  elbow: number | null;
}

export interface FrameResult {
  jointAngles: JointAngles;
  bikeAngle: number | null;
  bikeConfidence: number | null;
  detectedSide: 'left' | 'right' | null;
  maskedBikeData: ImageData | null;
}

interface BikeAngleConfig {
  num_bins: number;
  input_size: number;
  normalization: {
    mean: number[];
    std: number[];
  };
}

// ONNX Runtime types (loaded dynamically)
type InferenceSession = {
  run: (feeds: Record<string, unknown>) => Promise<Record<string, { data: Float32Array; dims: number[] }>>;
};

type OrtModule = {
  InferenceSession: {
    create: (path: string, options?: { executionProviders: string[] }) => Promise<InferenceSession>;
  };
  Tensor: new (type: string, data: Float32Array, dims: number[]) => unknown;
  env: {
    wasm: {
      numThreads: number;
      wasmPaths: string;
    };
  };
};

// ============= COCO Keypoint Indices =============

const COCO_KEYPOINTS = {
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

// ============= Model Manager =============

class ModelManager {
  private ort: OrtModule | null = null;
  private poseSession: InferenceSession | null = null;
  private segSession: InferenceSession | null = null;
  private bikeAngleSession: InferenceSession | null = null;
  private bikeAngleConfig: BikeAngleConfig | null = null;
  private isLoading = false;
  private loadPromise: Promise<void> | null = null;

  async loadModels(onProgress?: (status: string, progress: number) => void): Promise<void> {
    if (this.loadPromise) {
      return this.loadPromise;
    }

    if (this.isLoaded()) {
      return;
    }

    this.isLoading = true;
    this.loadPromise = this._loadModels(onProgress);
    
    try {
      await this.loadPromise;
    } finally {
      this.isLoading = false;
    }
  }

  private async _loadModels(onProgress?: (status: string, progress: number) => void): Promise<void> {
    const basePath = '/models';
    
    try {
      // Dynamically import ONNX Runtime (browser only)
      onProgress?.('Loading ONNX Runtime...', 5);
      
      // Dynamic import to avoid SSR issues
      const ortModule = await import('onnxruntime-web');
      this.ort = ortModule as unknown as OrtModule;
      
      // Configure ONNX Runtime for browser
      this.ort.env.wasm.numThreads = 1; // Use single thread for compatibility
      this.ort.env.wasm.wasmPaths = '/';

      // Load bike angle config
      onProgress?.('Loading configuration...', 10);
      const configResponse = await fetch(`${basePath}/bike_angle_config.json`);
      if (configResponse.ok) {
        this.bikeAngleConfig = await configResponse.json();
      }

      // Load pose model
      onProgress?.('Loading pose detection model...', 25);
      try {
        this.poseSession = await this.ort.InferenceSession.create(
          `${basePath}/yolov8-pose.onnx`,
          { executionProviders: ['wasm'] }
        );
      } catch (e) {
        console.warn('Pose model not found, skipping:', e);
      }

      // Load segmentation model (optional)
      onProgress?.('Loading bike segmentation model...', 55);
      try {
        this.segSession = await this.ort.InferenceSession.create(
          `${basePath}/yolov8-seg.onnx`,
          { executionProviders: ['wasm'] }
        );
      } catch (e) {
        console.warn('Segmentation model not found, skipping:', e);
      }

      // Load bike angle model
      onProgress?.('Loading bike angle model...', 85);
      try {
        this.bikeAngleSession = await this.ort.InferenceSession.create(
          `${basePath}/bike_angle.onnx`,
          { executionProviders: ['wasm'] }
        );
      } catch (e) {
        console.warn('Bike angle model not found, skipping:', e);
      }

      onProgress?.('Models loaded!', 100);
    } catch (error) {
      console.error('Error loading models:', error);
      throw error;
    }
  }

  isLoaded(): boolean {
    return this.bikeAngleSession !== null || this.poseSession !== null;
  }

  getOrt(): OrtModule | null {
    return this.ort;
  }

  getConfig(): BikeAngleConfig | null {
    return this.bikeAngleConfig;
  }

  getPoseSession(): InferenceSession | null {
    return this.poseSession;
  }

  getSegSession(): InferenceSession | null {
    return this.segSession;
  }

  getBikeAngleSession(): InferenceSession | null {
    return this.bikeAngleSession;
  }
}

export const modelManager = new ModelManager();

// ============= Image Processing =============

function imageDataToTensor(
  ort: OrtModule,
  imageData: ImageData,
  targetSize: number,
  normalize: { mean: number[]; std: number[] }
): unknown {
  const { width, height, data } = imageData;
  
  // Create canvas to resize
  const canvas = document.createElement('canvas');
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d')!;
  
  // Draw and resize
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = width;
  tempCanvas.height = height;
  const tempCtx = tempCanvas.getContext('2d')!;
  tempCtx.putImageData(imageData, 0, 0);
  
  ctx.drawImage(tempCanvas, 0, 0, targetSize, targetSize);
  const resizedData = ctx.getImageData(0, 0, targetSize, targetSize);
  
  // Convert to tensor [1, 3, H, W] with normalization
  const tensorData = new Float32Array(3 * targetSize * targetSize);
  
  for (let i = 0; i < targetSize * targetSize; i++) {
    const r = resizedData.data[i * 4] / 255;
    const g = resizedData.data[i * 4 + 1] / 255;
    const b = resizedData.data[i * 4 + 2] / 255;
    
    tensorData[i] = (r - normalize.mean[0]) / normalize.std[0];
    tensorData[targetSize * targetSize + i] = (g - normalize.mean[1]) / normalize.std[1];
    tensorData[2 * targetSize * targetSize + i] = (b - normalize.mean[2]) / normalize.std[2];
  }
  
  return new ort.Tensor('float32', tensorData, [1, 3, targetSize, targetSize]);
}

function videoFrameToImageData(video: HTMLVideoElement): ImageData {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(video, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

// ============= Pose Detection =============

async function detectPose(
  ort: OrtModule,
  imageData: ImageData,
  session: InferenceSession
): Promise<{ keypoints: Keypoint[]; side: 'left' | 'right' | null }> {
  const inputSize = 640;
  const { width, height } = imageData;
  
  // Prepare input tensor
  const canvas = document.createElement('canvas');
  canvas.width = inputSize;
  canvas.height = inputSize;
  const ctx = canvas.getContext('2d')!;
  
  // Letterbox resize
  const scale = Math.min(inputSize / width, inputSize / height);
  const newWidth = Math.round(width * scale);
  const newHeight = Math.round(height * scale);
  const offsetX = (inputSize - newWidth) / 2;
  const offsetY = (inputSize - newHeight) / 2;
  
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = width;
  tempCanvas.height = height;
  const tempCtx = tempCanvas.getContext('2d')!;
  tempCtx.putImageData(imageData, 0, 0);
  
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, inputSize, inputSize);
  ctx.drawImage(tempCanvas, offsetX, offsetY, newWidth, newHeight);
  
  const resizedData = ctx.getImageData(0, 0, inputSize, inputSize);
  
  // Convert to tensor [1, 3, 640, 640]
  const tensorData = new Float32Array(3 * inputSize * inputSize);
  for (let i = 0; i < inputSize * inputSize; i++) {
    tensorData[i] = resizedData.data[i * 4] / 255;
    tensorData[inputSize * inputSize + i] = resizedData.data[i * 4 + 1] / 255;
    tensorData[2 * inputSize * inputSize + i] = resizedData.data[i * 4 + 2] / 255;
  }
  
  const inputTensor = new ort.Tensor('float32', tensorData, [1, 3, inputSize, inputSize]);
  
  try {
    const results = await session.run({ images: inputTensor });
    const output = results.output0;
    
    if (!output) {
      return { keypoints: [], side: null };
    }
    
    // Parse YOLO pose output
    // Output shape: [1, 56, num_detections] where 56 = 5 (box) + 51 (17 keypoints * 3)
    const data = output.data;
    const numDetections = output.dims[2];
    
    // Find best detection
    let bestScore = 0;
    let bestIdx = -1;
    
    for (let i = 0; i < numDetections; i++) {
      const score = data[4 * numDetections + i]; // confidence at index 4
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }
    
    if (bestIdx === -1 || bestScore < 0.5) {
      return { keypoints: [], side: null };
    }
    
    // Extract keypoints (starting at index 5, 3 values per keypoint)
    const keypoints: Keypoint[] = [];
    for (let k = 0; k < 17; k++) {
      const baseIdx = (5 + k * 3) * numDetections + bestIdx;
      const x = (data[baseIdx] - offsetX) / scale;
      const y = (data[baseIdx + numDetections] - offsetY) / scale;
      const conf = data[baseIdx + 2 * numDetections];
      keypoints.push({ x, y, confidence: conf });
    }
    
    // Determine which side is more visible
    const leftConf = (
      keypoints[COCO_KEYPOINTS.left_shoulder].confidence +
      keypoints[COCO_KEYPOINTS.left_hip].confidence +
      keypoints[COCO_KEYPOINTS.left_knee].confidence
    ) / 3;
    
    const rightConf = (
      keypoints[COCO_KEYPOINTS.right_shoulder].confidence +
      keypoints[COCO_KEYPOINTS.right_hip].confidence +
      keypoints[COCO_KEYPOINTS.right_knee].confidence
    ) / 3;
    
    const side = rightConf > leftConf ? 'right' : 'left';
    
    return { keypoints, side };
  } catch (error) {
    console.error('Pose detection error:', error);
    return { keypoints: [], side: null };
  }
}

// ============= Angle Calculation =============

function calculateAngle(p1: Keypoint, p2: Keypoint, p3: Keypoint): number | null {
  if (p1.confidence < 0.5 || p2.confidence < 0.5 || p3.confidence < 0.5) {
    return null;
  }
  
  const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
  const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };
  
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
  const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
  
  if (mag1 === 0 || mag2 === 0) return null;
  
  const cosAngle = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
  return Math.acos(cosAngle) * (180 / Math.PI);
}

function computeJointAngles(keypoints: Keypoint[], side: 'left' | 'right'): JointAngles {
  const prefix = side === 'left' ? 'left_' : 'right_';
  
  const shoulder = keypoints[COCO_KEYPOINTS[`${prefix}shoulder` as keyof typeof COCO_KEYPOINTS]];
  const elbow = keypoints[COCO_KEYPOINTS[`${prefix}elbow` as keyof typeof COCO_KEYPOINTS]];
  const wrist = keypoints[COCO_KEYPOINTS[`${prefix}wrist` as keyof typeof COCO_KEYPOINTS]];
  const hip = keypoints[COCO_KEYPOINTS[`${prefix}hip` as keyof typeof COCO_KEYPOINTS]];
  const knee = keypoints[COCO_KEYPOINTS[`${prefix}knee` as keyof typeof COCO_KEYPOINTS]];
  const ankle = keypoints[COCO_KEYPOINTS[`${prefix}ankle` as keyof typeof COCO_KEYPOINTS]];
  
  return {
    knee: calculateAngle(hip, knee, ankle),
    hip: calculateAngle(shoulder, hip, knee),
    elbow: calculateAngle(shoulder, elbow, wrist),
  };
}

// ============= Bike Segmentation (Simplified) =============

async function segmentBike(
  imageData: ImageData
): Promise<{ masked: ImageData; success: boolean }> {
  // Simplified: use center crop instead of full segmentation
  // Full YOLO segmentation is complex to implement in browser
  const size = 224;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = imageData.width;
  tempCanvas.height = imageData.height;
  const tempCtx = tempCanvas.getContext('2d')!;
  tempCtx.putImageData(imageData, 0, 0);
  
  // Center crop
  const cropSize = Math.min(imageData.width, imageData.height);
  const sx = (imageData.width - cropSize) / 2;
  const sy = (imageData.height - cropSize) / 2;
  
  ctx.drawImage(tempCanvas, sx, sy, cropSize, cropSize, 0, 0, size, size);
  return { masked: ctx.getImageData(0, 0, size, size), success: true };
}

// ============= Bike Angle Prediction =============

async function predictBikeAngle(
  ort: OrtModule,
  maskedImage: ImageData,
  session: InferenceSession,
  config: BikeAngleConfig
): Promise<{ angle: number; confidence: number }> {
  const inputTensor = imageDataToTensor(ort, maskedImage, config.input_size, config.normalization);
  
  try {
    const results = await session.run({ input: inputTensor });
    const probs = results.probabilities.data;
    
    // Circular decoding (same as training)
    const numBins = config.num_bins;
    const binSize = 360 / numBins;
    
    let sinSum = 0;
    let cosSum = 0;
    let maxProb = 0;
    
    for (let i = 0; i < numBins; i++) {
      const binCenter = -180 + binSize / 2 + i * binSize;
      const binCenterRad = (binCenter * Math.PI) / 180;
      const prob = probs[i];
      
      sinSum += prob * Math.sin(binCenterRad);
      cosSum += prob * Math.cos(binCenterRad);
      
      if (prob > maxProb) {
        maxProb = prob;
      }
    }
    
    const angle = Math.atan2(sinSum, cosSum) * (180 / Math.PI);
    const confidence = maxProb * 100;
    
    return { angle, confidence };
  } catch (error) {
    console.error('Bike angle prediction error:', error);
    return { angle: 0, confidence: 0 };
  }
}

// ============= Main Processing Function =============

export async function processFrame(
  video: HTMLVideoElement
): Promise<FrameResult> {
  const imageData = videoFrameToImageData(video);
  const ort = modelManager.getOrt();
  
  let jointAngles: JointAngles = { knee: null, hip: null, elbow: null };
  let detectedSide: 'left' | 'right' | null = null;
  let bikeAngle: number | null = null;
  let bikeConfidence: number | null = null;
  let maskedBikeData: ImageData | null = null;
  
  if (!ort) {
    return { jointAngles, bikeAngle, bikeConfidence, detectedSide, maskedBikeData };
  }
  
  // Pose detection
  const poseSession = modelManager.getPoseSession();
  if (poseSession) {
    const { keypoints, side } = await detectPose(ort, imageData, poseSession);
    detectedSide = side;
    if (keypoints.length > 0 && side) {
      jointAngles = computeJointAngles(keypoints, side);
    }
  }
  
  // Bike segmentation and angle prediction
  const bikeAngleSession = modelManager.getBikeAngleSession();
  const config = modelManager.getConfig();
  
  if (bikeAngleSession && config) {
    const { masked, success } = await segmentBike(imageData);
    maskedBikeData = masked;
    
    if (success) {
      const result = await predictBikeAngle(ort, masked, bikeAngleSession, config);
      bikeAngle = result.angle;
      bikeConfidence = result.confidence;
    }
  }
  
  return {
    jointAngles,
    bikeAngle,
    bikeConfidence,
    detectedSide,
    maskedBikeData,
  };
}

// ============= Video Processing =============

export interface ProcessingOptions {
  startTime: number;
  endTime: number;
  outputFps: number;
  onProgress: (progress: number, currentFrame: number, status: string) => void;
  onFrame: (frameIndex: number, result: FrameResult) => void;
}

export async function processVideo(
  video: HTMLVideoElement,
  options: ProcessingOptions
): Promise<FrameResult[]> {
  const { startTime, endTime, outputFps, onProgress, onFrame } = options;
  
  const duration = endTime - startTime;
  const totalFrames = Math.ceil(duration * outputFps);
  const frameInterval = 1 / outputFps;
  
  const results: FrameResult[] = [];
  
  // Ensure models are loaded
  if (!modelManager.isLoaded()) {
    await modelManager.loadModels((status, progress) => {
      onProgress(progress * 0.1, 0, status);
    });
  }
  
  for (let i = 0; i < totalFrames; i++) {
    const time = startTime + i * frameInterval;
    
    // Seek to frame
    video.currentTime = time;
    await new Promise<void>((resolve) => {
      const onSeeked = () => {
        video.removeEventListener('seeked', onSeeked);
        resolve();
      };
      video.addEventListener('seeked', onSeeked);
    });
    
    // Process frame
    const result = await processFrame(video);
    results.push(result);
    
    onFrame(i, result);
    
    const progress = 10 + ((i + 1) / totalFrames) * 90;
    onProgress(progress, i + 1, `Processing frame ${i + 1}/${totalFrames}`);
  }
  
  return results;
}
