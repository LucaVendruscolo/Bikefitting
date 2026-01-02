'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { FrameData, FrameMetrics, ProcessingMetrics } from '@/lib/types';
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

  // Timing metrics
  const [modelLoadTime, setModelLoadTime] = useState(0);
  const [currentMetrics, setCurrentMetrics] = useState<FrameMetrics | null>(null);
  const [avgMetrics, setAvgMetrics] = useState<ProcessingMetrics | null>(null);
  const [showMetrics, setShowMetrics] = useState(true);

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
    setCurrentMetrics(null);
    setAvgMetrics(null);

    // Load models if not loaded
    let models = modelState;
    let loadTime = modelLoadTime;
    if (!models || !models.isLoaded) {
      setLoadingModels(true);
      const loadStart = performance.now();
      try {
        models = await loadModels((p) => setModelProgress(p));
        setModelState(models);
        loadTime = performance.now() - loadStart;
        setModelLoadTime(loadTime);
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
    
    // Metrics accumulator
    const allMetrics: FrameMetrics[] = [];

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
        const { frame: frameData, metrics } = await processVideoFrame(imageData, time, models!);
        frames.push(frameData);
        allMetrics.push(metrics);
        setCurrentMetrics(metrics);
        
        // Update running average
        if (allMetrics.length > 0) {
          const avg: ProcessingMetrics = {
            frameCount: allMetrics.length,
            avgTotalFrame: allMetrics.reduce((s, m) => s + m.totalFrame, 0) / allMetrics.length,
            avgPosePreprocess: allMetrics.reduce((s, m) => s + m.posePreprocess, 0) / allMetrics.length,
            avgPoseInference: allMetrics.reduce((s, m) => s + m.poseInference, 0) / allMetrics.length,
            avgPosePostprocess: allMetrics.reduce((s, m) => s + m.posePostprocess, 0) / allMetrics.length,
            avgSegPreprocess: allMetrics.reduce((s, m) => s + m.segPreprocess, 0) / allMetrics.length,
            avgSegInference: allMetrics.reduce((s, m) => s + m.segInference, 0) / allMetrics.length,
            avgSegPostprocess: allMetrics.reduce((s, m) => s + m.segPostprocess, 0) / allMetrics.length,
            avgAnglePreprocess: allMetrics.reduce((s, m) => s + m.anglePreprocess, 0) / allMetrics.length,
            avgAngleInference: allMetrics.reduce((s, m) => s + m.angleInference, 0) / allMetrics.length,
            avgAnglePostprocess: allMetrics.reduce((s, m) => s + m.anglePostprocess, 0) / allMetrics.length,
            modelLoadTime: loadTime,
          };
          setAvgMetrics(avg);
        }
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
  }, [modelState, modelLoadTime, startTime, endTime, fps, onStartProcessing, onProcessingComplete]);

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

  // Format milliseconds
  const formatMs = (ms: number) => ms.toFixed(1) + 'ms';

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
          {isProcessing && totalFrames > 0 && (
            <div className="px-6 py-4">
              <div className="progress-track">
                <div 
                  className="progress-fill" 
                  style={{ width: `${Math.round((currentFrame / totalFrames) * 100)}%` }} 
                />
              </div>
            </div>
          )}
        </div>

        {/* Performance Metrics Panel */}
        {(isProcessing || avgMetrics) && showMetrics && (
          <div className="glass rounded-3xl p-4 mt-4">
            <div className="flex justify-between items-center mb-3">
              <h4 className="text-white font-medium text-sm">Performance Metrics</h4>
              <button 
                onClick={() => setShowMetrics(false)}
                className="text-white/40 hover:text-white/60 text-xs"
              >
                Hide
              </button>
            </div>
            
            {avgMetrics && (
              <div className="space-y-3 text-xs">
                {/* Summary */}
                <div className="grid grid-cols-3 gap-2">
                  <div className="glass-light rounded-lg p-2 text-center">
                    <div className="text-white/40">Avg Frame</div>
                    <div className="text-emerald-400 font-mono text-sm">{formatMs(avgMetrics.avgTotalFrame)}</div>
                  </div>
                  <div className="glass-light rounded-lg p-2 text-center">
                    <div className="text-white/40">Est. FPS</div>
                    <div className="text-blue-400 font-mono text-sm">{(1000 / avgMetrics.avgTotalFrame).toFixed(1)}</div>
                  </div>
                  <div className="glass-light rounded-lg p-2 text-center">
                    <div className="text-white/40">Model Load</div>
                    <div className="text-violet-400 font-mono text-sm">{(avgMetrics.modelLoadTime / 1000).toFixed(1)}s</div>
                  </div>
                </div>

                {/* Breakdown by model */}
                <div className="space-y-2">
                  {/* Pose Model */}
                  <div className="glass-light rounded-lg p-2">
                    <div className="flex justify-between text-white/60 mb-1">
                      <span>Pose (YOLOv8m)</span>
                      <span className="text-cyan-400 font-mono">
                        {formatMs(avgMetrics.avgPosePreprocess + avgMetrics.avgPoseInference + avgMetrics.avgPosePostprocess)}
                      </span>
                    </div>
                    <div className="flex gap-2 text-white/40">
                      <span>Pre: {formatMs(avgMetrics.avgPosePreprocess)}</span>
                      <span>•</span>
                      <span className="text-cyan-400/80">Inf: {formatMs(avgMetrics.avgPoseInference)}</span>
                      <span>•</span>
                      <span>Post: {formatMs(avgMetrics.avgPosePostprocess)}</span>
                    </div>
                    <div className="mt-1 h-1 bg-white/10 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-cyan-500" 
                        style={{ width: `${((avgMetrics.avgPosePreprocess + avgMetrics.avgPoseInference + avgMetrics.avgPosePostprocess) / avgMetrics.avgTotalFrame) * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Segmentation Model */}
                  <div className="glass-light rounded-lg p-2">
                    <div className="flex justify-between text-white/60 mb-1">
                      <span>Seg (YOLOv8n)</span>
                      <span className="text-yellow-400 font-mono">
                        {formatMs(avgMetrics.avgSegPreprocess + avgMetrics.avgSegInference + avgMetrics.avgSegPostprocess)}
                      </span>
                    </div>
                    <div className="flex gap-2 text-white/40">
                      <span>Pre: {formatMs(avgMetrics.avgSegPreprocess)}</span>
                      <span>•</span>
                      <span className="text-yellow-400/80">Inf: {formatMs(avgMetrics.avgSegInference)}</span>
                      <span>•</span>
                      <span>Post: {formatMs(avgMetrics.avgSegPostprocess)}</span>
                    </div>
                    <div className="mt-1 h-1 bg-white/10 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-yellow-500" 
                        style={{ width: `${((avgMetrics.avgSegPreprocess + avgMetrics.avgSegInference + avgMetrics.avgSegPostprocess) / avgMetrics.avgTotalFrame) * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Angle Model */}
                  <div className="glass-light rounded-lg p-2">
                    <div className="flex justify-between text-white/60 mb-1">
                      <span>Angle (ConvNeXt)</span>
                      <span className="text-pink-400 font-mono">
                        {formatMs(avgMetrics.avgAnglePreprocess + avgMetrics.avgAngleInference + avgMetrics.avgAnglePostprocess)}
                      </span>
                    </div>
                    <div className="flex gap-2 text-white/40">
                      <span>Pre: {formatMs(avgMetrics.avgAnglePreprocess)}</span>
                      <span>•</span>
                      <span className="text-pink-400/80">Inf: {formatMs(avgMetrics.avgAngleInference)}</span>
                      <span>•</span>
                      <span>Post: {formatMs(avgMetrics.avgAnglePostprocess)}</span>
                    </div>
                    <div className="mt-1 h-1 bg-white/10 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-pink-500" 
                        style={{ width: `${((avgMetrics.avgAnglePreprocess + avgMetrics.avgAngleInference + avgMetrics.avgAnglePostprocess) / avgMetrics.avgTotalFrame) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Current frame time */}
                {currentMetrics && isProcessing && (
                  <div className="text-white/40 text-center pt-1 border-t border-white/10">
                    Current frame: <span className="text-white font-mono">{formatMs(currentMetrics.totalFrame)}</span>
                  </div>
                )}
              </div>
            )}

            {!avgMetrics && isProcessing && (
              <p className="text-white/40 text-sm">Collecting metrics...</p>
            )}
          </div>
        )}

        {/* Show metrics button when hidden */}
        {!showMetrics && avgMetrics && (
          <button 
            onClick={() => setShowMetrics(true)}
            className="mt-4 text-white/40 hover:text-white/60 text-xs"
          >
            Show Performance Metrics
          </button>
        )}
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

