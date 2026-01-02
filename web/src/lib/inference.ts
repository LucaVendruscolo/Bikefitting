/**
 * Client-side inference using ONNX Runtime Web (loaded from CDN)
 * 
 * This module handles all ML inference directly in the browser:
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

// Global ONNX Runtime reference (loaded from CDN)
declare global {
  interface Window {
    ort: OrtModule;
  }
}

interface OrtTensor {
  data: Float32Array;
  dims: number[];
}

interface OrtSession {
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
}

interface OrtModule {
  InferenceSession: {
    create(path: string, options?: { executionProviders: string[] }): Promise<OrtSession>;
  };
  Tensor: new (type: string, data: Float32Array, dims: number[]) => OrtTensor;
  env: {
    wasm: {
      wasmPaths: string;
    };
  };
}

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

// ============= Load ONNX Runtime from CDN =============

let ortLoadPromise: Promise<OrtModule> | null = null;

async function loadOrt(): Promise<OrtModule> {
  if (typeof window !== 'undefined' && window.ort) {
    return window.ort;
  }
  
  if (ortLoadPromise) {
    return ortLoadPromise;
  }
  
  ortLoadPromise = new Promise((resolve, reject) => {
    console.log('[ONNX] Loading ONNX Runtime from CDN...');
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js';
    script.async = true;
    script.onload = () => {
      if (window.ort) {
        console.log('[ONNX] ONNX Runtime loaded successfully');
        // Set WASM paths to CDN
        window.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
        resolve(window.ort);
      } else {
        reject(new Error('ONNX Runtime failed to load'));
      }
    };
    script.onerror = () => reject(new Error('Failed to load ONNX Runtime script'));
    document.head.appendChild(script);
  });
  
  return ortLoadPromise;
}

// ============= Model Manager =============

class ModelManager {
  private ort: OrtModule | null = null;
  private poseSession: OrtSession | null = null;
  private bikeAngleSession: OrtSession | null = null;
  private bikeAngleConfig: BikeAngleConfig | null = null;
  private isLoading = false;
  private loadPromise: Promise<void> | null = null;
  private loadErrors: string[] = [];

  async loadModels(onProgress?: (status: string, progress: number) => void): Promise<void> {
    if (this.loadPromise) {
      return this.loadPromise;
    }

    if (this.isLoaded()) {
      return;
    }

    this.isLoading = true;
    this.loadErrors = [];
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
      // Load ONNX Runtime from CDN
      onProgress?.('Loading ONNX Runtime...', 5);
      this.ort = await loadOrt();

      // Load bike angle config
      onProgress?.('Loading configuration...', 15);
      try {
        const configResponse = await fetch(`${basePath}/bike_angle_config.json`);
        if (configResponse.ok) {
          this.bikeAngleConfig = await configResponse.json();
          console.log('[Config] Loaded bike angle config:', this.bikeAngleConfig);
        } else {
          const error = `Config fetch failed: ${configResponse.status}`;
          console.error('[Config]', error);
          this.loadErrors.push(error);
        }
      } catch (e) {
        const error = `Config error: ${e}`;
        console.error('[Config]', error);
        this.loadErrors.push(error);
      }

      // Load pose model
      onProgress?.('Loading pose detection model (~13MB)...', 30);
      try {
        console.log('[Pose] Loading from:', `${basePath}/yolov8-pose.onnx`);
        
        // First check if the file exists
        const checkResponse = await fetch(`${basePath}/yolov8-pose.onnx`, { method: 'HEAD' });
        console.log('[Pose] File check:', checkResponse.status, checkResponse.headers.get('content-type'));
        
        if (!checkResponse.ok) {
          throw new Error(`Pose model not found: ${checkResponse.status}`);
        }
        
        this.poseSession = await this.ort.InferenceSession.create(
          `${basePath}/yolov8-pose.onnx`,
          { executionProviders: ['wasm'] }
        );
        console.log('[Pose] Model loaded successfully!');
        onProgress?.('Pose model loaded!', 60);
      } catch (e) {
        const error = `Pose model error: ${e}`;
        console.error('[Pose]', error);
        this.loadErrors.push(error);
      }

      // Load bike angle model
      onProgress?.('Loading bike angle model (~97MB)...', 70);
      try {
        console.log('[BikeAngle] Loading from:', `${basePath}/bike_angle.onnx`);
        
        // First check if the file exists
        const checkResponse = await fetch(`${basePath}/bike_angle.onnx`, { method: 'HEAD' });
        console.log('[BikeAngle] File check:', checkResponse.status, checkResponse.headers.get('content-type'));
        
        if (!checkResponse.ok) {
          throw new Error(`Bike angle model not found: ${checkResponse.status}`);
        }
        
        this.bikeAngleSession = await this.ort.InferenceSession.create(
          `${basePath}/bike_angle.onnx`,
          { executionProviders: ['wasm'] }
        );
        console.log('[BikeAngle] Model loaded successfully!');
        onProgress?.('Bike angle model loaded!', 95);
      } catch (e) {
        const error = `Bike angle model error: ${e}`;
        console.error('[BikeAngle]', error);
        this.loadErrors.push(error);
      }

      if (this.loadErrors.length > 0) {
        console.warn('[Models] Some models failed to load:', this.loadErrors);
        onProgress?.(`Loaded with errors: ${this.loadErrors.length} model(s) failed`, 100);
      } else {
        onProgress?.('All models loaded!', 100);
      }
    } catch (error) {
      console.error('[Models] Fatal error loading models:', error);
      throw error;
    }
  }

  isLoaded(): boolean {
    return this.poseSession !== null || this.bikeAngleSession !== null;
  }
  
  getLoadErrors(): string[] {
    return this.loadErrors;
  }

  getOrt(): OrtModule | null {
    return this.ort;
  }

  getConfig(): BikeAngleConfig | null {
    return this.bikeAngleConfig;
  }

  getPoseSession(): OrtSession | null {
    return this.poseSession;
  }

  getBikeAngleSession(): OrtSession | null {
    return this.bikeAngleSession;
  }
}

export const modelManager = new ModelManager();

// ============= Image Processing =============

function createTensor(
  ort: OrtModule,
  data: Float32Array,
  dims: number[]
): OrtTensor {
  return new ort.Tensor('float32', data, dims);
}

function imageDataToTensor(
  ort: OrtModule,
  imageData: ImageData,
  targetSize: number,
  normalize: { mean: number[]; std: number[] }
): OrtTensor {
  // Create canvas to resize
  const canvas = document.createElement('canvas');
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d')!;
  
  // Draw and resize
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = imageData.width;
  tempCanvas.height = imageData.height;
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
  
  return createTensor(ort, tensorData, [1, 3, targetSize, targetSize]);
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
  session: OrtSession
): Promise<{ keypoints: Keypoint[]; side: 'left' | 'right' | null }> {
  const inputSize = 640;
  const { width, height } = imageData;
  
  // Prepare input tensor with letterbox
  const canvas = document.createElement('canvas');
  canvas.width = inputSize;
  canvas.height = inputSize;
  const ctx = canvas.getContext('2d')!;
  
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
  
  const inputTensor = createTensor(ort, tensorData, [1, 3, inputSize, inputSize]);
  
  try {
    const results = await session.run({ images: inputTensor });
    const output = results.output0;
    
    if (!output) {
      console.warn('[Pose] No output from model');
      return { keypoints: [], side: null };
    }
    
    const data = output.data;
    const numDetections = output.dims[2];
    
    // Find best detection
    let bestScore = 0;
    let bestIdx = -1;
    
    for (let i = 0; i < numDetections; i++) {
      const score = data[4 * numDetections + i];
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }
    
    if (bestIdx === -1 || bestScore < 0.3) {
      console.log('[Pose] No detection above threshold, best score:', bestScore);
      return { keypoints: [], side: null };
    }
    
    console.log('[Pose] Detection found with score:', bestScore);
    
    // Extract keypoints
    const keypoints: Keypoint[] = [];
    for (let k = 0; k < 17; k++) {
      const baseIdx = (5 + k * 3) * numDetections + bestIdx;
      const x = (data[baseIdx] - offsetX) / scale;
      const y = (data[baseIdx + numDetections] - offsetY) / scale;
      const conf = data[baseIdx + 2 * numDetections];
      keypoints.push({ x, y, confidence: conf });
    }
    
    // Determine visible side
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
    console.log('[Pose] Detected side:', side, 'leftConf:', leftConf.toFixed(2), 'rightConf:', rightConf.toFixed(2));
    
    return { keypoints, side };
  } catch (error) {
    console.error('[Pose] Inference error:', error);
    return { keypoints: [], side: null };
  }
}

// ============= Angle Calculation =============

function calculateAngle(p1: Keypoint, p2: Keypoint, p3: Keypoint): number | null {
  if (p1.confidence < 0.3 || p2.confidence < 0.3 || p3.confidence < 0.3) {
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
  
  const angles = {
    knee: calculateAngle(hip, knee, ankle),
    hip: calculateAngle(shoulder, hip, knee),
    elbow: calculateAngle(shoulder, elbow, wrist),
  };
  
  console.log('[Angles] Computed:', angles);
  return angles;
}

// ============= Bike Angle Prediction =============

async function predictBikeAngle(
  ort: OrtModule,
  imageData: ImageData,
  session: OrtSession,
  config: BikeAngleConfig
): Promise<{ angle: number; confidence: number }> {
  const inputTensor = imageDataToTensor(ort, imageData, config.input_size, config.normalization);
  
  try {
    const results = await session.run({ input: inputTensor });
    const probs = results.probabilities.data;
    
    // Circular decoding
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
    
    console.log('[BikeAngle] Predicted:', angle.toFixed(1), 'confidence:', confidence.toFixed(1));
    return { angle, confidence };
  } catch (error) {
    console.error('[BikeAngle] Inference error:', error);
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
  const maskedBikeData: ImageData | null = null;
  
  if (!ort) {
    console.warn('[ProcessFrame] ORT not loaded');
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
  } else {
    console.warn('[ProcessFrame] Pose session not available');
  }
  
  // Bike angle prediction (using center crop)
  const bikeAngleSession = modelManager.getBikeAngleSession();
  const config = modelManager.getConfig();
  
  if (bikeAngleSession && config) {
    // Center crop for bike angle
    const size = config.input_size;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d')!;
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCtx.putImageData(imageData, 0, 0);
    
    const cropSize = Math.min(imageData.width, imageData.height);
    const sx = (imageData.width - cropSize) / 2;
    const sy = (imageData.height - cropSize) / 2;
    ctx.drawImage(tempCanvas, sx, sy, cropSize, cropSize, 0, 0, size, size);
    
    const croppedData = ctx.getImageData(0, 0, size, size);
    const result = await predictBikeAngle(ort, croppedData, bikeAngleSession, config);
    bikeAngle = result.angle;
    bikeConfidence = result.confidence;
  } else {
    console.warn('[ProcessFrame] Bike angle session not available, session:', !!bikeAngleSession, 'config:', !!config);
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
    console.log('[ProcessVideo] Loading models...');
    await modelManager.loadModels((status, progress) => {
      onProgress(progress * 0.1, 0, status);
    });
  }
  
  const errors = modelManager.getLoadErrors();
  if (errors.length > 0) {
    console.warn('[ProcessVideo] Model load errors:', errors);
  }
  
  console.log('[ProcessVideo] Starting processing of', totalFrames, 'frames');
  console.log('[ProcessVideo] Pose session:', !!modelManager.getPoseSession());
  console.log('[ProcessVideo] BikeAngle session:', !!modelManager.getBikeAngleSession());
  
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
    const frameStart = performance.now();
    const result = await processFrame(video);
    const frameTime = performance.now() - frameStart;
    
    if (i === 0) {
      console.log('[ProcessVideo] First frame took', frameTime.toFixed(0), 'ms');
      console.log('[ProcessVideo] First frame result:', result);
    }
    
    results.push(result);
    onFrame(i, result);
    
    const progress = 10 + ((i + 1) / totalFrames) * 90;
    onProgress(progress, i + 1, `Processing frame ${i + 1}/${totalFrames}`);
  }
  
  return results;
}
