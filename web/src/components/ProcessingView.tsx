'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { FrameData } from '@/lib/types';
import { loadModels, processVideoFrame, ModelState } from '@/lib/inference';

interface ProcessingViewProps {
  videoUrl: string;
  isProcessing: boolean;
  onStartProcessing: () => void;
  onProcessingComplete: (frames: FrameData[], start: number, end: number, fps: number) => void;
  onReset: () => void;
}

export default function ProcessingView({
  videoUrl,
  isProcessing,
  onStartProcessing,
  onProcessingComplete,
  onReset,
}: ProcessingViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stopRef = useRef(false);

  // Video info
  const [duration, setDuration] = useState(0);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [fps, setFps] = useState(5);

  // Processing state
  const [modelState, setModelState] = useState<ModelState | null>(null);
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelProgress, setModelProgress] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Handle video metadata
  const handleLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    if (video) {
      setDuration(video.duration);
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
    onStartProcessing();
    setError(null);

    // Load models if not loaded
    let models = modelState;
    if (!models || !models.isLoaded) {
      setLoadingModels(true);
      try {
        models = await loadModels((p) => setModelProgress(p));
        setModelState(models);
      } catch (err) {
        setError(`Failed to load AI models: ${err}`);
        return;
      } finally {
        setLoadingModels(false);
      }
    }

    // Setup canvas
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Calculate frames to process
    const frameInterval = 1 / fps;
    const processDuration = endTime - startTime;
    const framesToProcess = Math.floor(processDuration * fps);
    setTotalFrames(framesToProcess);

    const frames: FrameData[] = [];

    // Process each frame
    for (let i = 0; i < framesToProcess; i++) {
      if (stopRef.current) {
        return; // Stopped by user
      }

      const time = startTime + i * frameInterval;
      video.currentTime = time;

      // Wait for seek
      await new Promise<void>((resolve) => {
        const onSeeked = () => {
          video.removeEventListener('seeked', onSeeked);
          resolve();
        };
        video.addEventListener('seeked', onSeeked);
      });

      // Draw frame
      ctx.drawImage(video, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      try {
        const frameData = await processVideoFrame(imageData, time, models!);
        frames.push(frameData);
      } catch (err) {
        console.warn(`Frame ${i} error:`, err);
        frames.push({
          time,
          skeleton: null,
          jointAngles: { knee: null, hip: null, elbow: null },
          bikeAngle: null,
          bikeMask: null,
        });
      }

      setCurrentFrame(i + 1);
      setProgress(((i + 1) / framesToProcess) * 100);
    }

    onProcessingComplete(frames, startTime, endTime, fps);
  }, [modelState, startTime, endTime, fps, onStartProcessing, onProcessingComplete]);

  // Stop processing
  const stopProcessing = useCallback(() => {
    stopRef.current = true;
  }, []);

  // Seek video when adjusting time
  const seekVideo = useCallback((time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
    }
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Video Preview */}
      <div className="lg:col-span-2">
        <div className="glass rounded-3xl overflow-hidden">
          <div className="aspect-video bg-black relative">
            <video
              ref={videoRef}
              src={videoUrl}
              onLoadedMetadata={handleLoadedMetadata}
              className="w-full h-full object-contain"
              playsInline
              muted
            />
            <canvas ref={canvasRef} className="hidden" />

            {/* Loading overlay */}
            {loadingModels && (
              <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center">
                <div className="spinner mb-4" />
                <p className="text-white font-medium">Loading AI Models...</p>
                <p className="text-white/50 text-sm mt-1">{Math.round(modelProgress)}%</p>
              </div>
            )}

            {/* Processing overlay */}
            {isProcessing && !loadingModels && (
              <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center">
                <p className="text-white text-xl font-medium mb-2">Processing</p>
                <p className="text-white/60">
                  Frame {currentFrame} of {totalFrames}
                </p>
              </div>
            )}

            {/* Error overlay */}
            {error && (
              <div className="absolute inset-0 bg-red-900/80 flex items-center justify-center p-6">
                <p className="text-white text-center">{error}</p>
              </div>
            )}
          </div>

          {/* Progress bar */}
          {isProcessing && (
            <div className="px-6 py-4">
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Settings Panel */}
      <div className="lg:col-span-1">
        <div className="glass rounded-3xl p-6 space-y-6">
          <h3 className="text-lg font-medium text-white">Processing Settings</h3>

          {/* Start Time */}
          <div>
            <label className="block text-white/50 text-sm mb-2">
              Start: {formatTime(startTime)}
            </label>
            <input
              type="range"
              min={0}
              max={duration}
              step={0.1}
              value={startTime}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setStartTime(Math.min(val, endTime - 1));
                seekVideo(val);
              }}
              disabled={isProcessing}
              className="w-full"
            />
          </div>

          {/* End Time */}
          <div>
            <label className="block text-white/50 text-sm mb-2">
              End: {formatTime(endTime)}
            </label>
            <input
              type="range"
              min={0}
              max={duration}
              step={0.1}
              value={endTime}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setEndTime(Math.max(val, startTime + 1));
                seekVideo(val);
              }}
              disabled={isProcessing}
              className="w-full"
            />
          </div>

          {/* Frame Rate */}
          <div>
            <label className="block text-white/50 text-sm mb-2">
              Frame Rate: {fps} fps
            </label>
            <input
              type="range"
              min={1}
              max={15}
              step={1}
              value={fps}
              onChange={(e) => setFps(parseInt(e.target.value))}
              disabled={isProcessing}
              className="w-full"
            />
            <p className="text-white/30 text-xs mt-1">
              {Math.floor((endTime - startTime) * fps)} frames total
            </p>
          </div>

          {/* Duration info */}
          <div className="glass-light rounded-xl p-4">
            <div className="flex justify-between text-sm">
              <span className="text-white/50">Duration</span>
              <span className="text-white">{formatTime(endTime - startTime)}</span>
            </div>
            <div className="flex justify-between text-sm mt-2">
              <span className="text-white/50">Frames</span>
              <span className="text-white">{Math.floor((endTime - startTime) * fps)}</span>
            </div>
          </div>

          {/* Buttons */}
          <div className="space-y-3">
            {!isProcessing ? (
              <>
                <button
                  onClick={startProcessing}
                  disabled={duration === 0}
                  className="btn-primary w-full"
                >
                  Start Processing
                </button>
                <button
                  onClick={onReset}
                  className="w-full py-3 rounded-xl bg-white/5 hover:bg-white/10 text-white/70 transition"
                >
                  Choose Different Video
                </button>
              </>
            ) : (
              <button
                onClick={stopProcessing}
                className="w-full py-3 rounded-xl bg-red-500/80 hover:bg-red-500 text-white transition"
              >
                Stop Processing
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

