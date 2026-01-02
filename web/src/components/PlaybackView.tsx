'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { FrameData, CurrentAngles } from '@/lib/types';
import AngleDisplay from './AngleDisplay';

interface PlaybackViewProps {
  videoUrl: string;
  frames: FrameData[];
  startTime: number;
  endTime: number;
  fps: number;
  onReprocess: () => void;
  onReset: () => void;
}

export default function PlaybackView({
  videoUrl,
  frames,
  startTime,
  endTime,
  fps,
  onReprocess,
  onReset,
}: PlaybackViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const lastFrameTimeRef = useRef<number>(0);
  const frameIndexRef = useRef<number>(0);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(startTime);
  const frameInterval = 1000 / fps; // ms between frames
  const [detectedSide, setDetectedSide] = useState<'left' | 'right'>('right');
  const [currentAngles, setCurrentAngles] = useState<CurrentAngles>({
    knee: null,
    hip: null,
    elbow: null,
    bike: null,
  });

  // Determine dominant side from frames
  useEffect(() => {
    let leftCount = 0;
    let rightCount = 0;
    for (const frame of frames) {
      if (frame.skeleton?.side === 'left') leftCount++;
      else if (frame.skeleton?.side === 'right') rightCount++;
    }
    setDetectedSide(leftCount > rightCount ? 'left' : 'right');
  }, [frames]);

  // Initialize canvas and set video to start time
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas) {
      video.currentTime = startTime;
      
      const handleLoaded = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        renderFrame();
      };
      
      if (video.readyState >= 2) {
        handleLoaded();
      } else {
        video.addEventListener('loadeddata', handleLoaded, { once: true });
      }
    }
    
    return () => {
      cancelAnimationFrame(animationRef.current);
    };
  }, [startTime]);

  // Find closest frame data for current time
  const getFrameData = useCallback((time: number): FrameData | null => {
    if (frames.length === 0) return null;
    
    let closest = frames[0];
    let minDiff = Math.abs(time - closest.time);
    
    for (const frame of frames) {
      const diff = Math.abs(time - frame.time);
      if (diff < minDiff) {
        minDiff = diff;
        closest = frame;
      }
    }
    
    return closest;
  }, [frames]);

  // Render current frame with skeleton overlay
  const renderFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || frames.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw video frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get frame data for current frame index
    const frameData = frames[frameIndexRef.current] || frames[0];
    if (!frameData) return;

    // Draw skeleton
    if (frameData.skeleton && frameData.skeleton.keypoints) {
      drawSkeleton(ctx, frameData.skeleton.keypoints, detectedSide);
    }

    // Update angles display
    setCurrentAngles({
      knee: frameData.jointAngles.knee,
      hip: frameData.jointAngles.hip,
      elbow: frameData.jointAngles.elbow,
      bike: frameData.bikeAngle,
    });

    setCurrentTime(frameData.time);
  }, [frames, detectedSide]);

  // Animation loop for playback - step through frames at specified FPS
  const playbackLoop = useCallback((timestamp: number) => {
    if (!isPlaying) return;
    
    const elapsed = timestamp - lastFrameTimeRef.current;
    
    // Only advance frame if enough time has passed
    if (elapsed >= frameInterval) {
      lastFrameTimeRef.current = timestamp;
      
      // Move to next frame
      frameIndexRef.current++;
      if (frameIndexRef.current >= frames.length) {
        frameIndexRef.current = 0; // Loop back to start
      }
      
      // Seek video to frame time
      const video = videoRef.current;
      if (video && frames[frameIndexRef.current]) {
        video.currentTime = frames[frameIndexRef.current].time;
      }
      
      renderFrame();
    }
    
    animationRef.current = requestAnimationFrame(playbackLoop);
  }, [isPlaying, frameInterval, frames, renderFrame]);

  // Draw skeleton on canvas
  const drawSkeleton = (ctx: CanvasRenderingContext2D, keypoints: any[], side: 'left' | 'right') => {
    const connections = side === 'left'
      ? [[5, 7], [7, 9], [5, 11], [11, 13], [13, 15]]
      : [[6, 8], [8, 10], [6, 12], [12, 14], [14, 16]];
    
    const jointIndices = side === 'left'
      ? [5, 7, 9, 11, 13, 15]
      : [6, 8, 10, 12, 14, 16];

    // Draw connections (cyan with black outline)
    ctx.lineCap = 'round';
    for (const [start, end] of connections) {
      const p1 = keypoints[start];
      const p2 = keypoints[end];
      if (p1 && p2 && p1.confidence > 0.3 && p2.confidence > 0.3) {
        // Black outline
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 7;
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
        
        // Cyan line
        ctx.strokeStyle = '#00FFFF';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    }

    // Draw joints (magenta with white border)
    for (const idx of jointIndices) {
      const kp = keypoints[idx];
      if (kp && kp.confidence > 0.3) {
        // Black outline
        ctx.fillStyle = '#000';
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 12, 0, Math.PI * 2);
        ctx.fill();
        
        // Magenta fill
        ctx.fillStyle = '#FF00FF';
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 10, 0, Math.PI * 2);
        ctx.fill();
        
        // White border
        ctx.strokeStyle = '#FFF';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 10, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  };

  // Play/Pause toggle
  const togglePlayback = useCallback(() => {
    if (isPlaying) {
      // Pause
      setIsPlaying(false);
      cancelAnimationFrame(animationRef.current);
    } else {
      // Play
      setIsPlaying(true);
      lastFrameTimeRef.current = performance.now();
      animationRef.current = requestAnimationFrame(playbackLoop);
    }
  }, [isPlaying, playbackLoop]);

  // Seek to time
  const seekTo = useCallback((time: number) => {
    const video = videoRef.current;
    if (!video || frames.length === 0) return;
    
    // Find closest frame index
    let closestIdx = 0;
    let minDiff = Math.abs(time - frames[0].time);
    for (let i = 1; i < frames.length; i++) {
      const diff = Math.abs(time - frames[i].time);
      if (diff < minDiff) {
        minDiff = diff;
        closestIdx = i;
      }
    }
    
    frameIndexRef.current = closestIdx;
    video.currentTime = frames[closestIdx].time;
    setCurrentTime(frames[closestIdx].time);
    
    // Render the seeked frame
    setTimeout(renderFrame, 50);
  }, [frames, renderFrame]);

  // Format time
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
      {/* Video Player */}
      <div className="lg:col-span-3">
        <div className="glass rounded-3xl overflow-hidden">
          {/* Canvas (visible) */}
          <div className="aspect-video bg-black relative">
            <canvas
              ref={canvasRef}
              className="w-full h-full object-contain"
            />
            
            {/* Hidden video element */}
            <video
              ref={videoRef}
              src={videoUrl}
              className="hidden"
              playsInline
              muted
            />
          </div>

          {/* Custom Controls */}
          <div className="video-controls">
            {/* Play/Pause */}
            <button
              onClick={togglePlayback}
              className="w-12 h-12 flex items-center justify-center rounded-full bg-white/10 hover:bg-white/20 transition"
            >
              {isPlaying ? (
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="4" width="4" height="16" />
                  <rect x="14" y="4" width="4" height="16" />
                </svg>
              ) : (
                <svg className="w-5 h-5 ml-1" fill="currentColor" viewBox="0 0 24 24">
                  <polygon points="5,3 19,12 5,21" />
                </svg>
              )}
            </button>

            {/* Timeline */}
            <div className="flex-1 flex items-center gap-3">
              <span className="text-white/60 text-sm font-mono w-12">
                {formatTime(currentTime - startTime)}
              </span>
              <input
                type="range"
                min={startTime}
                max={endTime}
                step={0.1}
                value={currentTime}
                onChange={(e) => seekTo(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="text-white/60 text-sm font-mono w-12">
                {formatTime(endTime - startTime)}
              </span>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="mt-4 flex gap-4">
          <button
            onClick={onReprocess}
            className="px-6 py-3 rounded-xl bg-white/5 hover:bg-white/10 text-white/70 transition"
          >
            Reprocess Video
          </button>
          <button
            onClick={onReset}
            className="px-6 py-3 rounded-xl bg-white/5 hover:bg-white/10 text-white/70 transition"
          >
            Upload New Video
          </button>
        </div>
      </div>

      {/* Angles Panel */}
      <div className="lg:col-span-1">
        <AngleDisplay
          angles={currentAngles}
          detectedSide={detectedSide}
        />
      </div>
    </div>
  );
}

