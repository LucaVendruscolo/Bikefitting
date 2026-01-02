'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import { ProcessingState, AnalysisResults } from '@/app/page';
import { loadModels, processFrame, ModelState } from '@/lib/inference';

interface VideoProcessorProps {
  videoUrl: string;
  processingState: ProcessingState;
  setProcessingState: (state: ProcessingState) => void;
  progress: number;
  setProgress: (progress: number) => void;
  onComplete: (results: AnalysisResults) => void;
  onError: (error: string) => void;
}

export default function VideoProcessor({
  videoUrl,
  processingState,
  setProcessingState,
  progress,
  setProgress,
  onComplete,
  onError,
}: VideoProcessorProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [modelState, setModelState] = useState<ModelState | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);

  // Load models on mount
  useEffect(() => {
    const init = async () => {
      setProcessingState('loading-models');
      try {
        const state = await loadModels((p) => setProgress(p * 0.3)); // 0-30% for model loading
        setModelState(state);
        setProcessingState('idle');
        setProgress(0);
      } catch (err) {
        onError(`Failed to load models: ${err}`);
      }
    };
    init();
  }, []);

  // Handle video metadata
  const handleLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    if (video) {
      const fps = 30; // Assume 30fps, will adjust based on duration
      const frames = Math.floor(video.duration * fps);
      setTotalFrames(frames);
    }
  }, []);

  // Start processing
  const startProcessing = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !modelState) return;

    setProcessingState('processing');
    setProgress(30); // Start at 30% (models already loaded)

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const fps = 10; // Process at 10fps
    const frameInterval = 1 / fps;
    const totalDuration = video.duration;
    const framesToProcess = Math.floor(totalDuration * fps);

    const results: AnalysisResults = {
      jointAngles: { knee: [], hip: [], elbow: [] },
      bikeAngles: [],
      frameCount: framesToProcess,
      fps: fps,
    };

    // Process frames
    for (let i = 0; i < framesToProcess; i++) {
      const time = i * frameInterval;
      video.currentTime = time;

      // Wait for frame to load
      await new Promise<void>((resolve) => {
        const onSeeked = () => {
          video.removeEventListener('seeked', onSeeked);
          resolve();
        };
        video.addEventListener('seeked', onSeeked);
      });

      // Draw frame to canvas
      ctx.drawImage(video, 0, 0);

      // Get image data
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      try {
        // Process frame with models
        const frameResults = await processFrame(imageData, modelState);

        // Store results
        if (frameResults.jointAngles.knee !== null) {
          results.jointAngles.knee.push(frameResults.jointAngles.knee);
        }
        if (frameResults.jointAngles.hip !== null) {
          results.jointAngles.hip.push(frameResults.jointAngles.hip);
        }
        if (frameResults.jointAngles.elbow !== null) {
          results.jointAngles.elbow.push(frameResults.jointAngles.elbow);
        }
        if (frameResults.bikeAngle !== null) {
          results.bikeAngles.push(frameResults.bikeAngle);
        }

        // Draw skeleton overlay
        if (frameResults.skeleton) {
          drawSkeleton(ctx, frameResults.skeleton);
        }

        // Draw bike mask outline
        if (frameResults.bikeMask) {
          drawBikeMask(ctx, frameResults.bikeMask);
        }

      } catch (err) {
        console.warn(`Frame ${i} processing error:`, err);
      }

      // Update progress (30-100%)
      const frameProgress = 30 + (i / framesToProcess) * 70;
      setProgress(frameProgress);
      setCurrentFrame(i);
    }

    onComplete(results);
  }, [modelState, onComplete, setProcessingState, setProgress]);

  // Draw skeleton on canvas
  const drawSkeleton = (ctx: CanvasRenderingContext2D, skeleton: any) => {
    if (!skeleton || !skeleton.keypoints) return;

    const connections = [
      [5, 7], [7, 9],   // Left arm
      [6, 8], [8, 10],  // Right arm
      [5, 11], [6, 12], // Torso
      [11, 13], [13, 15], // Left leg
      [12, 14], [14, 16], // Right leg
    ];

    // Draw connections
    ctx.strokeStyle = '#00FFFF';
    ctx.lineWidth = 3;
    for (const [start, end] of connections) {
      const p1 = skeleton.keypoints[start];
      const p2 = skeleton.keypoints[end];
      if (p1 && p2 && p1.confidence > 0.3 && p2.confidence > 0.3) {
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    }

    // Draw keypoints
    ctx.fillStyle = '#FF00FF';
    for (const kp of skeleton.keypoints) {
      if (kp && kp.confidence > 0.3) {
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  };

  // Draw bike mask outline
  const drawBikeMask = (ctx: CanvasRenderingContext2D, mask: any) => {
    // Placeholder - will implement actual mask drawing
    if (!mask) return;
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 2;
    // Draw mask contours when implemented
  };

  return (
    <div className="space-y-4">
      {/* Video Preview */}
      <div className="video-container aspect-video bg-black rounded-xl overflow-hidden relative">
        <video
          ref={videoRef}
          src={videoUrl}
          onLoadedMetadata={handleLoadedMetadata}
          className="w-full h-full object-contain"
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />

        {/* Processing Overlay */}
        {processingState === 'loading-models' && (
          <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center">
            <div className="spinner mb-4" />
            <p className="text-white">Loading AI models...</p>
            <p className="text-gray-400 text-sm mt-1">{Math.round(progress)}%</p>
          </div>
        )}

        {processingState === 'processing' && (
          <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center">
            <p className="text-white mb-2">Analyzing video...</p>
            <p className="text-gray-400 text-sm">
              Frame {currentFrame + 1} / {totalFrames}
            </p>
          </div>
        )}
      </div>

      {/* Progress Bar */}
      {(processingState === 'loading-models' || processingState === 'processing') && (
        <div className="progress-bar">
          <div
            className="progress-bar-fill"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {/* Start Button */}
      {processingState === 'idle' && modelState && (
        <button
          onClick={startProcessing}
          className="w-full py-3 bg-gradient-to-r from-blue-500 to-emerald-500 hover:from-blue-600 hover:to-emerald-600 rounded-xl font-semibold transition"
        >
          Start Analysis
        </button>
      )}
    </div>
  );
}

