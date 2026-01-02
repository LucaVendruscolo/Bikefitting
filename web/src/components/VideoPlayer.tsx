'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import { loadModels, processFrame, ModelState } from '@/lib/inference';

interface VideoPlayerProps {
  videoUrl: string;
}

interface CurrentAngles {
  knee: number | null;
  hip: number | null;
  elbow: number | null;
  bike: number | null;
}

export default function VideoPlayer({ videoUrl }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const lastProcessTimeRef = useRef<number>(0);
  
  const [modelState, setModelState] = useState<ModelState | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadProgress, setLoadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentAngles, setCurrentAngles] = useState<CurrentAngles>({
    knee: null,
    hip: null,
    elbow: null,
    bike: null,
  });
  const [detectedSide, setDetectedSide] = useState<'left' | 'right' | null>(null);

  // Load models on mount
  useEffect(() => {
    const init = async () => {
      try {
        const state = await loadModels((p) => setLoadProgress(p));
        setModelState(state);
        setIsLoading(false);
      } catch (err) {
        setError(`Failed to load models: ${err}`);
        setIsLoading(false);
      }
    };
    init();
  }, []);

  // Process current frame
  const processCurrentFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !modelState || video.paused) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Throttle processing to ~10fps
    const now = performance.now();
    if (now - lastProcessTimeRef.current < 100) {
      animationRef.current = requestAnimationFrame(processCurrentFrame);
      return;
    }
    lastProcessTimeRef.current = now;

    // Draw video frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Get image data for processing
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    try {
      const results = await processFrame(imageData, modelState);

      // Update angles
      setCurrentAngles({
        knee: results.jointAngles.knee,
        hip: results.jointAngles.hip,
        elbow: results.jointAngles.elbow,
        bike: results.bikeAngle,
      });

      // Draw skeleton overlay
      if (results.skeleton) {
        setDetectedSide(results.skeleton.side);
        drawSkeleton(ctx, results.skeleton);
      }

    } catch (err) {
      console.warn('Frame processing error:', err);
    }

    // Continue loop if playing
    if (!video.paused) {
      animationRef.current = requestAnimationFrame(processCurrentFrame);
    }
  }, [modelState]);

  // Draw skeleton - only visible side, no head
  const drawSkeleton = (ctx: CanvasRenderingContext2D, skeleton: any) => {
    if (!skeleton || !skeleton.keypoints) return;

    const side = skeleton.side || 'right';
    
    const leftConnections = [
      [5, 7], [7, 9],     // Left arm
      [5, 11],            // Left torso
      [11, 13], [13, 15], // Left leg
    ];
    
    const rightConnections = [
      [6, 8], [8, 10],    // Right arm
      [6, 12],            // Right torso
      [12, 14], [14, 16], // Right leg
    ];

    const connections = side === 'left' ? leftConnections : rightConnections;
    const visibleKeypoints = side === 'left' 
      ? [5, 7, 9, 11, 13, 15]
      : [6, 8, 10, 12, 14, 16];

    // Draw connections
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

    // Draw keypoints
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

  // Handle video play/pause
  const handlePlay = () => {
    setIsPlaying(true);
    animationRef.current = requestAnimationFrame(processCurrentFrame);
  };

  const handlePause = () => {
    setIsPlaying(false);
    cancelAnimationFrame(animationRef.current);
  };

  // Set canvas size when video loads
  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
  };

  // Cleanup
  useEffect(() => {
    return () => {
      cancelAnimationFrame(animationRef.current);
    };
  }, []);

  // Format angle for display
  const formatAngle = (angle: number | null) => {
    if (angle === null) return '—';
    return `${Math.round(angle)}°`;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
      {/* Video Container */}
      <div className="lg:col-span-3 relative">
        <div className="aspect-video bg-black rounded-xl overflow-hidden relative">
          <video
            ref={videoRef}
            src={videoUrl}
            onLoadedMetadata={handleLoadedMetadata}
            onPlay={handlePlay}
            onPause={handlePause}
            className="w-full h-full object-contain"
            controls
            playsInline
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            style={{ display: isPlaying ? 'block' : 'none' }}
          />

          {/* Loading Overlay */}
          {isLoading && (
            <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center">
              <div className="spinner mb-4" />
              <p className="text-white">Loading AI models...</p>
              <p className="text-gray-400 text-sm mt-1">{Math.round(loadProgress)}%</p>
            </div>
          )}

          {/* Error Overlay */}
          {error && (
            <div className="absolute inset-0 bg-black/80 flex flex-col items-center justify-center p-4">
              <p className="text-red-400 text-center">{error}</p>
            </div>
          )}
        </div>

        {!isLoading && !error && (
          <p className="text-gray-500 text-sm mt-2 text-center">
            Press play to analyze • Skeleton and angles update in real-time
          </p>
        )}
      </div>

      {/* Angles Panel */}
      <div className="lg:col-span-1 space-y-4">
        {/* Joint Angles */}
        <div className="bg-slate-800/70 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">
            Joint Angles {detectedSide && `(${detectedSide})`}
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

        {/* Bike Angle */}
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
          {isLoading ? (
            <p className="text-yellow-400 text-sm">Loading models...</p>
          ) : error ? (
            <p className="text-red-400 text-sm">Error loading models</p>
          ) : isPlaying ? (
            <p className="text-green-400 text-sm">● Analyzing</p>
          ) : (
            <p className="text-gray-500 text-sm">Ready • Press play</p>
          )}
        </div>
      </div>
    </div>
  );
}

