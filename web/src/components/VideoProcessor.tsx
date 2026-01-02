'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import { loadModels, processFrame, ModelState, FrameResults } from '@/lib/inference';

interface VideoProcessorProps {
  videoUrl: string;
  onReset: () => void;
}

type Phase = 'settings' | 'loading-models' | 'processing' | 'playback';

interface FrameData {
  time: number;
  skeleton: any;
  jointAngles: { knee: number | null; hip: number | null; elbow: number | null };
  bikeAngle: number | null;
}

export default function VideoProcessor({ videoUrl, onReset }: VideoProcessorProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stopRef = useRef(false);

  const [phase, setPhase] = useState<Phase>('settings');
  const [modelState, setModelState] = useState<ModelState | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Settings
  const [videoDuration, setVideoDuration] = useState(0);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [fps, setFps] = useState(5);

  // Processing state
  const [currentFrame, setCurrentFrame] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);

  // Results storage
  const [frameResults, setFrameResults] = useState<FrameData[]>([]);
  const [detectedSide, setDetectedSide] = useState<'left' | 'right'>('right');

  // Playback state
  const [currentAngles, setCurrentAngles] = useState({
    knee: null as number | null,
    hip: null as number | null,
    elbow: null as number | null,
    bike: null as number | null,
  });

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

  // Start processing
  const startProcessing = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    stopRef.current = false;
    setPhase('loading-models');
    setProgress(0);
    setFrameResults([]);

    // Load models if not already loaded
    let models = modelState;
    if (!models) {
      try {
        models = await loadModels((p) => setProgress(p * 0.3));
        setModelState(models);
      } catch (err) {
        setError(`Failed to load models: ${err}`);
        setPhase('settings');
        return;
      }
    }

    setPhase('processing');
    setProgress(30);

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const frameInterval = 1 / fps;
    const duration = endTime - startTime;
    const framesToProcess = Math.floor(duration * fps);
    setTotalFrames(framesToProcess);

    const results: FrameData[] = [];
    let leftCount = 0;
    let rightCount = 0;

    for (let i = 0; i < framesToProcess; i++) {
      if (stopRef.current) {
        setPhase('settings');
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
        const frameResult = await processFrame(imageData, models!);

        if (frameResult.skeleton?.side === 'left') leftCount++;
        else if (frameResult.skeleton?.side === 'right') rightCount++;

        results.push({
          time,
          skeleton: frameResult.skeleton,
          jointAngles: frameResult.jointAngles,
          bikeAngle: frameResult.bikeAngle,
        });

      } catch (err) {
        console.warn(`Frame ${i} processing error:`, err);
        results.push({
          time,
          skeleton: null,
          jointAngles: { knee: null, hip: null, elbow: null },
          bikeAngle: null,
        });
      }

      const frameProgress = 30 + (i / framesToProcess) * 70;
      setProgress(frameProgress);
      setCurrentFrame(i + 1);
    }

    setDetectedSide(leftCount >= rightCount ? 'left' : 'right');
    setFrameResults(results);
    setPhase('playback');
    
    // Reset video to start
    video.currentTime = startTime;
  }, [modelState, startTime, endTime, fps]);

  // Stop processing
  const stopProcessing = useCallback(() => {
    stopRef.current = true;
  }, []);

  // Handle video time update during playback
  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || phase !== 'playback' || frameResults.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Find the closest frame result
    const currentTime = video.currentTime;
    let closestFrame = frameResults[0];
    let minDiff = Math.abs(currentTime - closestFrame.time);

    for (const frame of frameResults) {
      const diff = Math.abs(currentTime - frame.time);
      if (diff < minDiff) {
        minDiff = diff;
        closestFrame = frame;
      }
    }

    // Draw video frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Draw skeleton if available
    if (closestFrame.skeleton) {
      drawSkeleton(ctx, closestFrame.skeleton, detectedSide);
    }

    // Update angles display
    setCurrentAngles({
      knee: closestFrame.jointAngles.knee,
      hip: closestFrame.jointAngles.hip,
      elbow: closestFrame.jointAngles.elbow,
      bike: closestFrame.bikeAngle,
    });
  }, [phase, frameResults, detectedSide]);

  // Draw skeleton - only visible side, no head
  const drawSkeleton = (ctx: CanvasRenderingContext2D, skeleton: any, side: 'left' | 'right') => {
    if (!skeleton || !skeleton.keypoints) return;

    const leftConnections = [[5, 7], [7, 9], [5, 11], [11, 13], [13, 15]];
    const rightConnections = [[6, 8], [8, 10], [6, 12], [12, 14], [14, 16]];
    const connections = side === 'left' ? leftConnections : rightConnections;
    const visibleKeypoints = side === 'left' ? [5, 7, 9, 11, 13, 15] : [6, 8, 10, 12, 14, 16];

    ctx.strokeStyle = '#00FFFF';
    ctx.lineWidth = 4;
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

    ctx.fillStyle = '#FF00FF';
    for (const idx of visibleKeypoints) {
      const kp = skeleton.keypoints[idx];
      if (kp && kp.confidence > 0.3) {
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 8, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  };

  const formatAngle = (angle: number | null) => {
    if (angle === null) return '—';
    return `${Math.round(angle)}°`;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
      {/* Video Container */}
      <div className="lg:col-span-3">
        <div className="aspect-video bg-black rounded-xl overflow-hidden relative">
          <video
            ref={videoRef}
            src={videoUrl}
            onLoadedMetadata={handleLoadedMetadata}
            onTimeUpdate={handleTimeUpdate}
            className="w-full h-full object-contain"
            controls={phase === 'playback'}
            playsInline
            muted
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            style={{ display: phase === 'playback' ? 'block' : 'none' }}
          />

          {/* Loading Models Overlay */}
          {phase === 'loading-models' && (
            <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center">
              <div className="spinner mb-4" />
              <p className="text-white">Loading AI models...</p>
              <p className="text-gray-400 text-sm mt-1">{Math.round(progress)}%</p>
            </div>
          )}

          {/* Processing Overlay */}
          {phase === 'processing' && (
            <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center">
              <p className="text-white text-lg mb-2">Processing video...</p>
              <p className="text-gray-400">Frame {currentFrame} / {totalFrames}</p>
            </div>
          )}

          {/* Error Overlay */}
          {error && (
            <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center p-4">
              <p className="text-red-400 text-center">{error}</p>
            </div>
          )}
        </div>

        {/* Progress Bar */}
        {(phase === 'loading-models' || phase === 'processing') && (
          <div className="mt-2 h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 to-emerald-500 transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}

        {/* Settings Panel */}
        {phase === 'settings' && videoDuration > 0 && (
          <div className="mt-4 bg-slate-800/50 rounded-xl p-4 space-y-4">
            <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
              Processing Settings
            </h3>

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

            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Frame Rate: {fps} fps • {Math.floor((endTime - startTime) * fps)} frames
              </label>
              <input
                type="range"
                min={1}
                max={15}
                step={1}
                value={fps}
                onChange={(e) => setFps(parseInt(e.target.value))}
                className="w-full accent-cyan-500"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>1 fps (fast)</span>
                <span>15 fps (detailed)</span>
              </div>
            </div>
          </div>
        )}

        {/* Buttons */}
        <div className="mt-4 flex gap-4">
          {phase === 'settings' && (
            <>
              <button
                onClick={startProcessing}
                disabled={videoDuration === 0}
                className="flex-1 py-3 bg-gradient-to-r from-cyan-500 to-emerald-500 hover:from-cyan-600 hover:to-emerald-600 disabled:opacity-50 rounded-xl font-semibold transition"
              >
                Start Processing
              </button>
              <button
                onClick={onReset}
                className="px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-xl transition"
              >
                Upload New Video
              </button>
            </>
          )}

          {phase === 'processing' && (
            <button
              onClick={stopProcessing}
              className="flex-1 py-3 bg-red-500 hover:bg-red-600 rounded-xl font-semibold transition"
            >
              Stop Processing
            </button>
          )}

          {phase === 'playback' && (
            <button
              onClick={onReset}
              className="px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-xl transition"
            >
              Upload New Video
            </button>
          )}
        </div>
      </div>

      {/* Angles Panel */}
      <div className="lg:col-span-1 space-y-4">
        <div className="bg-slate-800/70 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">
            Joint Angles {phase === 'playback' && `(${detectedSide})`}
          </h3>

          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Knee</span>
              <span className="text-2xl font-bold text-cyan-400 font-mono">
                {formatAngle(currentAngles.knee)}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-gray-300">Hip</span>
              <span className="text-2xl font-bold text-cyan-400 font-mono">
                {formatAngle(currentAngles.hip)}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-gray-300">Elbow</span>
              <span className="text-2xl font-bold text-cyan-400 font-mono">
                {formatAngle(currentAngles.elbow)}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-slate-800/70 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">
            Bike Angle
          </h3>

          <div className="text-center">
            <span className="text-4xl font-bold text-emerald-400 font-mono">
              {formatAngle(currentAngles.bike)}
            </span>
            <p className="text-gray-500 text-xs mt-2">
              0° = facing camera
            </p>
          </div>
        </div>

        {/* Status */}
        <div className="bg-slate-800/50 rounded-xl p-4 text-center">
          {phase === 'settings' && (
            <p className="text-gray-400 text-sm">Configure settings and press Start</p>
          )}
          {phase === 'loading-models' && (
            <p className="text-yellow-400 text-sm">Loading AI models...</p>
          )}
          {phase === 'processing' && (
            <p className="text-yellow-400 text-sm">Processing frames...</p>
          )}
          {phase === 'playback' && (
            <p className="text-green-400 text-sm">✓ Ready • Play video to see results</p>
          )}
        </div>
      </div>
    </div>
  );
}

