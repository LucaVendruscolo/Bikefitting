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
  const [currentFrame, setCurrentFrame] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const stopRef = useRef(false);

  // Video settings
  const [videoDuration, setVideoDuration] = useState(0);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [fps, setFps] = useState(5);

  // Load models on mount
  useEffect(() => {
    const init = async () => {
      setProcessingState('loading-models');
      try {
        const state = await loadModels((p) => setProgress(p * 0.3));
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
      setVideoDuration(video.duration);
      setEndTime(video.duration);
    }
  }, []);

  // Format time as MM:SS
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Stop processing
  const stopProcessing = useCallback(() => {
    stopRef.current = true;
  }, []);

  // Start processing
  const startProcessing = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !modelState) return;

    stopRef.current = false;
    setProcessingState('processing');
    setProgress(30);

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const frameInterval = 1 / fps;
    const duration = endTime - startTime;
    const framesToProcess = Math.floor(duration * fps);
    setTotalFrames(framesToProcess);

    const results: AnalysisResults = {
      jointAngles: { knee: [], hip: [], elbow: [] },
      bikeAngles: [],
      frameCount: framesToProcess,
      fps: fps,
    };

    // Track which side is dominant
    let leftCount = 0;
    let rightCount = 0;

    for (let i = 0; i < framesToProcess; i++) {
      if (stopRef.current) {
        setProcessingState('idle');
        setProgress(0);
        return;
      }

      const time = startTime + i * frameInterval;
      video.currentTime = time;

      await new Promise<void>((resolve) => {
        const onSeeked = () => {
          video.removeEventListener('seeked', onSeeked);
          resolve();
        };
        video.addEventListener('seeked', onSeeked);
      });

      ctx.drawImage(video, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      try {
        const frameResults = await processFrame(imageData, modelState);

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

        // Draw skeleton (only visible side, no head)
        if (frameResults.skeleton) {
          if (frameResults.skeleton.side === 'left') leftCount++;
          else if (frameResults.skeleton.side === 'right') rightCount++;
          
          const dominantSide = leftCount >= rightCount ? 'left' : 'right';
          drawSkeleton(ctx, frameResults.skeleton, dominantSide);
        }

      } catch (err) {
        console.warn(`Frame ${i} processing error:`, err);
      }

      const frameProgress = 30 + (i / framesToProcess) * 70;
      setProgress(frameProgress);
      setCurrentFrame(i);
    }

    onComplete(results);
  }, [modelState, onComplete, setProcessingState, setProgress, startTime, endTime, fps]);

  // Draw skeleton - only visible side, no head features
  const drawSkeleton = (ctx: CanvasRenderingContext2D, skeleton: any, dominantSide: 'left' | 'right') => {
    if (!skeleton || !skeleton.keypoints) return;

    const side = skeleton.side || dominantSide;
    
    // Connections for each side (no head, no cross-body)
    const leftConnections = [
      [5, 7], [7, 9],     // Left arm: shoulder-elbow-wrist
      [5, 11],            // Left torso: shoulder-hip
      [11, 13], [13, 15], // Left leg: hip-knee-ankle
    ];
    
    const rightConnections = [
      [6, 8], [8, 10],    // Right arm: shoulder-elbow-wrist
      [6, 12],            // Right torso: shoulder-hip
      [12, 14], [14, 16], // Right leg: hip-knee-ankle
    ];

    const connections = side === 'left' ? leftConnections : rightConnections;
    
    // Keypoint indices for the visible side (no head: 0-4)
    const visibleKeypoints = side === 'left' 
      ? [5, 7, 9, 11, 13, 15]  // Left: shoulder, elbow, wrist, hip, knee, ankle
      : [6, 8, 10, 12, 14, 16]; // Right: shoulder, elbow, wrist, hip, knee, ankle

    // Draw connections
    ctx.strokeStyle = '#00FFFF';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    
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

    // Draw keypoints (only for visible side, no head)
    ctx.fillStyle = '#FF00FF';
    for (const idx of visibleKeypoints) {
      const kp = skeleton.keypoints[idx];
      if (kp && kp.confidence > 0.3) {
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
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
          controls={processingState === 'idle'}
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />

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

      {/* Settings Panel - only show when idle and models loaded */}
      {processingState === 'idle' && modelState && videoDuration > 0 && (
        <div className="bg-slate-800/50 rounded-xl p-4 space-y-4">
          <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Processing Settings</h3>
          
          {/* Time Range */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Start Time</label>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min={0}
                  max={videoDuration}
                  step={0.1}
                  value={startTime}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setStartTime(Math.min(val, endTime - 1));
                    if (videoRef.current) videoRef.current.currentTime = val;
                  }}
                  className="flex-1 accent-cyan-500"
                />
                <span className="text-sm text-gray-300 w-12">{formatTime(startTime)}</span>
              </div>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">End Time</label>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min={0}
                  max={videoDuration}
                  step={0.1}
                  value={endTime}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setEndTime(Math.max(val, startTime + 1));
                    if (videoRef.current) videoRef.current.currentTime = val;
                  }}
                  className="flex-1 accent-cyan-500"
                />
                <span className="text-sm text-gray-300 w-12">{formatTime(endTime)}</span>
              </div>
            </div>
          </div>

          {/* Frame Rate */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">
              Frame Rate: {fps} fps ({Math.floor((endTime - startTime) * fps)} frames)
            </label>
            <input
              type="range"
              min={1}
              max={30}
              step={1}
              value={fps}
              onChange={(e) => setFps(parseInt(e.target.value))}
              className="w-full accent-cyan-500"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>1 fps (fast)</span>
              <span>30 fps (detailed)</span>
            </div>
          </div>

          {/* Duration info */}
          <p className="text-xs text-gray-500">
            Processing {formatTime(endTime - startTime)} of video ({Math.floor((endTime - startTime) * fps)} frames)
          </p>
        </div>
      )}

      {/* Buttons */}
      {processingState === 'idle' && modelState && (
        <button
          onClick={startProcessing}
          className="w-full py-3 bg-gradient-to-r from-cyan-500 to-emerald-500 hover:from-cyan-600 hover:to-emerald-600 rounded-xl font-semibold transition"
        >
          Start Analysis
        </button>
      )}

      {processingState === 'processing' && (
        <button
          onClick={stopProcessing}
          className="w-full py-3 bg-red-500 hover:bg-red-600 rounded-xl font-semibold transition"
        >
          Stop Processing
        </button>
      )}
    </div>
  );
}
