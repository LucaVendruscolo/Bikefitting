'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { modelManager, processVideo, type FrameResult } from '@/lib/inference';

// Types
interface ProcessingMetrics {
  totalFrames: number;
  processedFrames: number;
  avgTimePerFrame: number;
  totalTime: number;
}

interface FrameData extends FrameResult {
  frameIndex: number;
  timestamp: number;
}

export default function HomePage() {
  // State
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState(0);
  
  // Selection state
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [outputFps, setOutputFps] = useState(4);
  
  // Model loading state
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [modelLoadingStatus, setModelLoadingStatus] = useState('');
  const [modelLoadingProgress, setModelLoadingProgress] = useState(0);
  const [modelErrors, setModelErrors] = useState<string[]>([]);
  
  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [currentProcessingFrame, setCurrentProcessingFrame] = useState(0);
  const [processingStatus, setProcessingStatus] = useState('');
  
  // Result state
  const [processedFrames, setProcessedFrames] = useState<FrameData[]>([]);
  const [currentFrameData, setCurrentFrameData] = useState<FrameData | null>(null);
  const [metrics, setMetrics] = useState<ProcessingMetrics | null>(null);
  
  // Player state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [playbackFrame, setPlaybackFrame] = useState(0);
  
  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const playbackIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Load models on mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        console.log('[App] Starting model load...');
        await modelManager.loadModels((status, progress) => {
          setModelLoadingStatus(status);
          setModelLoadingProgress(progress);
          console.log('[App] Model load progress:', progress, status);
        });
        
        const errors = modelManager.getLoadErrors();
        setModelErrors(errors);
        
        if (errors.length > 0) {
          console.warn('[App] Model load errors:', errors);
        }
        
        setModelsLoaded(true);
        console.log('[App] Models loaded! Pose:', !!modelManager.getPoseSession(), 'BikeAngle:', !!modelManager.getBikeAngleSession());
      } catch (error) {
        console.error('[App] Failed to load models:', error);
        setModelLoadingStatus(`Failed: ${error}`);
        setModelErrors([`${error}`]);
      }
    };
    loadModels();
  }, []);

  // Handle file upload
  const handleFileSelect = useCallback((file: File) => {
    if (!file.type.startsWith('video/')) {
      alert('Please select a video file');
      return;
    }
    
    setUploadedFile(file);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setProcessedFrames([]);
    setCurrentFrameData(null);
    setMetrics(null);
  }, []);

  // Handle video metadata load
  const handleVideoLoad = useCallback(() => {
    if (videoRef.current) {
      const duration = videoRef.current.duration;
      setVideoDuration(duration);
      setEndTime(Math.min(duration, 30));
    }
  }, []);

  // Handle drag and drop
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  }, [handleFileSelect]);

  // Process video
  const handleProcess = async () => {
    if (!uploadedFile || !videoRef.current) return;
    
    setIsProcessing(true);
    setProcessingProgress(0);
    setCurrentProcessingFrame(0);
    setProcessingStatus('Initializing...');
    setProcessedFrames([]);
    
    const startTimestamp = performance.now();
    const frames: FrameData[] = [];
    
    try {
      await processVideo(videoRef.current, {
        startTime,
        endTime,
        outputFps,
        onProgress: (progress, currentFrame, status) => {
          setProcessingProgress(progress);
          setCurrentProcessingFrame(currentFrame);
          setProcessingStatus(status);
        },
        onFrame: (frameIndex, result) => {
          const frameData: FrameData = {
            ...result,
            frameIndex,
            timestamp: startTime + frameIndex / outputFps,
          };
          frames.push(frameData);
          setProcessedFrames([...frames]);
        },
      });
      
      const totalTime = performance.now() - startTimestamp;
      setMetrics({
        totalFrames: frames.length,
        processedFrames: frames.length,
        avgTimePerFrame: totalTime / frames.length,
        totalTime,
      });
      
      setProcessingStatus('Processing complete!');
      setPlaybackFrame(0);
      setCurrentFrameData(frames[0] || null);
      
    } catch (error) {
      console.error('Processing error:', error);
      setProcessingStatus(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Update frame data helper
  const updateFrameDisplay = useCallback((frameIndex: number) => {
    if (processedFrames.length > 0 && frameIndex < processedFrames.length) {
      const frameData = processedFrames[frameIndex];
      setCurrentFrameData(frameData);
      setCurrentTime(frameData.timestamp);
      
      // Sync video position
      if (videoRef.current) {
        videoRef.current.currentTime = frameData.timestamp;
      }
    }
  }, [processedFrames]);

  // Playback controls using requestAnimationFrame for smoother updates
  const togglePlay = useCallback(() => {
    if (processedFrames.length === 0) return;
    
    if (isPlaying) {
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
    }
  }, [isPlaying, processedFrames.length]);

  // Handle playback animation loop
  useEffect(() => {
    if (!isPlaying || processedFrames.length === 0) return;
    
    const frameDelay = 1000 / outputFps;
    let lastFrameTime = performance.now();
    let currentFrame = playbackFrame;
    let animationId: number;
    
    const animate = (now: number) => {
      const elapsed = now - lastFrameTime;
      
      if (elapsed >= frameDelay) {
        currentFrame++;
        
        if (currentFrame >= processedFrames.length) {
          setIsPlaying(false);
          setPlaybackFrame(0);
          updateFrameDisplay(0);
          return;
        }
        
        setPlaybackFrame(currentFrame);
        updateFrameDisplay(currentFrame);
        lastFrameTime = now;
      }
      
      animationId = requestAnimationFrame(animate);
    };
    
    animationId = requestAnimationFrame(animate);
    
    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [isPlaying, processedFrames.length, outputFps, playbackFrame, updateFrameDisplay]);

  // Update display when manually changing frame (seeking)
  useEffect(() => {
    if (!isPlaying && processedFrames.length > 0) {
      updateFrameDisplay(playbackFrame);
    }
  }, [playbackFrame, processedFrames, isPlaying, updateFrameDisplay]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current);
      }
    };
  }, []);

  // Seek to position
  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (processedFrames.length === 0) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = x / rect.width;
    const frameIndex = Math.floor(percentage * processedFrames.length);
    setPlaybackFrame(Math.min(frameIndex, processedFrames.length - 1));
  };

  // Format time display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const hasResults = processedFrames.length > 0;
  const hasModelErrors = modelErrors.length > 0;

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <div className="logo-icon">🚴</div>
          <span className="logo-text">BikeFit Pro</span>
        </div>
        <div className="flex items-center gap-4">
          {!modelsLoaded ? (
            <div className="text-muted text-sm flex items-center gap-2">
              <span className="loading-spinner" />
              {modelLoadingStatus || 'Loading models...'}
            </div>
          ) : hasModelErrors ? (
            <div className="text-warning text-sm">⚠️ Some models failed</div>
          ) : (
            <div className="text-accent text-sm">✓ Models ready</div>
          )}
        </div>
      </header>

      {/* Model Loading Overlay */}
      <AnimatePresence>
        {!modelsLoaded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="loading-overlay"
          >
            <div className="loading-card">
              <div className="loading-icon">🧠</div>
              <h2>Loading AI Models</h2>
              <p className="text-muted">{modelLoadingStatus}</p>
              <div className="progress-container mt-4">
                <div 
                  className="progress-bar"
                  style={{ width: `${modelLoadingProgress}%` }}
                />
              </div>
              <p className="text-muted text-sm mt-2">
                Models run locally in your browser for privacy
              </p>
              <p className="text-muted text-xs mt-4">
                First load may take a minute (~110MB download)
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <div className="main-grid">
        {/* Left Column - Video Area */}
        <div className="main-content">
          {/* Model Error Warning */}
          {hasModelErrors && (
            <div className="card" style={{ background: 'rgba(245, 158, 11, 0.1)', borderColor: 'rgba(245, 158, 11, 0.3)' }}>
              <div className="card-header" style={{ borderBottom: 'none', paddingBottom: 0 }}>
                <span className="card-title" style={{ color: '#f59e0b' }}>⚠️ Model Loading Issues</span>
              </div>
              <div className="text-sm text-muted">
                {modelErrors.map((err, i) => (
                  <div key={i} className="mb-2">{err}</div>
                ))}
                <div className="mt-2">
                  Check browser console (F12) for details. Models may not be deployed correctly.
                </div>
              </div>
            </div>
          )}

          {/* Upload / Preview / Result */}
          <AnimatePresence mode="wait">
            {!videoUrl ? (
              // Upload Zone
              <motion.div
                key="upload"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <div
                  className={`upload-zone ${isDragging ? 'dragging' : ''}`}
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="video/*"
                    className="hidden"
                    onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                  />
                  <div className="upload-icon">📹</div>
                  <div className="upload-text">
                    <div className="upload-title">Drop your video here</div>
                    <div className="upload-subtitle">or click to browse • MP4, MOV, AVI supported</div>
                  </div>
                </div>
              </motion.div>
            ) : (
              // Video Player with Controls
              <motion.div
                key="preview"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="card"
              >
                <div className="card-header">
                  <div className="card-icon">{hasResults ? '🎬' : '📹'}</div>
                  <span className="card-title">{hasResults ? 'Analysis Result' : 'Video Preview'}</span>
                  <div className="flex-1" />
                  <button 
                    className="btn btn-secondary"
                    onClick={() => {
                      setUploadedFile(null);
                      setVideoUrl(null);
                      setProcessedFrames([]);
                      setCurrentFrameData(null);
                      setMetrics(null);
                    }}
                  >
                    Change Video
                  </button>
                </div>
                
                <div className="video-container">
                  <video
                    ref={videoRef}
                    className="video-player"
                    src={videoUrl}
                    onLoadedMetadata={handleVideoLoad}
                    controls={!hasResults}
                    muted
                  />
                  
                  {/* Custom Controls for Results */}
                  {hasResults && (
                    <div className="video-controls">
                      {/* Timeline */}
                      <div className="timeline" onClick={handleSeek}>
                        <div 
                          className="timeline-progress"
                          style={{ 
                            width: `${(playbackFrame / Math.max(1, processedFrames.length - 1)) * 100}%` 
                          }}
                        >
                          <div className="timeline-handle" />
                        </div>
                      </div>
                      
                      {/* Controls Row */}
                      <div className="controls-row">
                        <button className="btn btn-icon btn-secondary" onClick={togglePlay}>
                          {isPlaying ? '⏸️' : '▶️'}
                        </button>
                        
                        <span className="time-display">
                          {formatTime(currentTime)} / {formatTime(endTime - startTime)}
                        </span>
                        
                        <span className="text-muted text-sm">
                          Frame {playbackFrame + 1} / {processedFrames.length}
                        </span>
                        
                        <div className="flex-1" />
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Selection Controls (when not processed) */}
                {!hasResults && (
                  <div className="mt-4">
                    <div className="flex gap-4 mb-4">
                      <div className="input-group flex-1">
                        <label className="input-label">Start Time (seconds)</label>
                        <input
                          type="number"
                          className="input"
                          value={startTime}
                          min={0}
                          max={endTime}
                          step={0.1}
                          onChange={(e) => setStartTime(parseFloat(e.target.value) || 0)}
                        />
                      </div>
                      <div className="input-group flex-1">
                        <label className="input-label">End Time (seconds)</label>
                        <input
                          type="number"
                          className="input"
                          value={endTime}
                          min={startTime}
                          max={videoDuration}
                          step={0.1}
                          onChange={(e) => setEndTime(parseFloat(e.target.value) || 0)}
                        />
                      </div>
                      <div className="input-group flex-1">
                        <label className="input-label">Output FPS</label>
                        <input
                          type="number"
                          className="input"
                          value={outputFps}
                          min={1}
                          max={15}
                          onChange={(e) => setOutputFps(parseInt(e.target.value) || 4)}
                        />
                      </div>
                    </div>
                    
                    <div className="text-muted text-sm mb-4">
                      Duration: {formatTime(endTime - startTime)} • 
                      ~{Math.ceil((endTime - startTime) * outputFps)} frames to process
                    </div>
                    
                    <button
                      className="btn btn-primary w-full"
                      onClick={handleProcess}
                      disabled={isProcessing || !modelsLoaded}
                    >
                      {!modelsLoaded 
                        ? 'Loading models...' 
                        : isProcessing 
                          ? 'Processing...' 
                          : 'Start Analysis'}
                    </button>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Processing Progress */}
          <AnimatePresence>
            {isProcessing && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="card"
              >
                <div className="card-header">
                  <div className="card-icon">⚙️</div>
                  <span className="card-title">Processing</span>
                </div>
                
                <div className="progress-container mb-4">
                  <div 
                    className="progress-bar"
                    style={{ width: `${processingProgress}%` }}
                  />
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-muted">{processingStatus}</span>
                  <span className="text-mono text-accent">
                    Frame {currentProcessingFrame} • {processingProgress.toFixed(1)}%
                  </span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Processing Metrics */}
          {metrics && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="card"
            >
              <div className="card-header">
                <div className="card-icon">📊</div>
                <span className="card-title">Processing Metrics</span>
              </div>
              
              <div className="metrics-grid">
                <div className="metric-item">
                  <div className="metric-value">{metrics.totalFrames}</div>
                  <div className="metric-label">Total Frames</div>
                </div>
                <div className="metric-item">
                  <div className="metric-value">{metrics.avgTimePerFrame.toFixed(0)}ms</div>
                  <div className="metric-label">Avg per Frame</div>
                </div>
                <div className="metric-item">
                  <div className="metric-value">{(metrics.totalTime / 1000).toFixed(1)}s</div>
                  <div className="metric-label">Total Time</div>
                </div>
                <div className="metric-item">
                  <div className="metric-value">{(1000 / metrics.avgTimePerFrame).toFixed(1)}</div>
                  <div className="metric-label">FPS</div>
                </div>
              </div>
            </motion.div>
          )}
        </div>

        {/* Right Column - Angle Data */}
        <div className="sidebar">
          {/* Current Angles */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="card"
          >
            <div className="card-header">
              <div className="card-icon">📐</div>
              <span className="card-title">Joint Angles</span>
            </div>
            
            <div className="angle-panel">
              <div className="angle-item">
                <div className="angle-label">
                  <span className="angle-dot knee" />
                  Knee Angle
                </div>
                <div className="angle-value">
                  {currentFrameData?.jointAngles.knee?.toFixed(1) ?? '--'}°
                </div>
              </div>
              
              <div className="angle-item">
                <div className="angle-label">
                  <span className="angle-dot hip" />
                  Hip Angle
                </div>
                <div className="angle-value">
                  {currentFrameData?.jointAngles.hip?.toFixed(1) ?? '--'}°
                </div>
              </div>
              
              <div className="angle-item">
                <div className="angle-label">
                  <span className="angle-dot elbow" />
                  Elbow Angle
                </div>
                <div className="angle-value">
                  {currentFrameData?.jointAngles.elbow?.toFixed(1) ?? '--'}°
                </div>
              </div>
            </div>
            
            {currentFrameData?.detectedSide && (
              <div className="mt-4 text-center text-muted text-sm">
                Detected Side: <span className="text-accent">{currentFrameData.detectedSide.toUpperCase()}</span>
              </div>
            )}
          </motion.div>

          {/* Bike Angle */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="card"
          >
            <div className="card-header">
              <div className="card-icon">🚲</div>
              <span className="card-title">Bike Angle</span>
            </div>
            
            <div className="angle-panel">
              <div className="angle-item">
                <div className="angle-label">
                  <span className="angle-dot bike" />
                  Predicted Angle
                </div>
                <div className="angle-value">
                  {currentFrameData?.bikeAngle?.toFixed(1) ?? '--'}°
                </div>
              </div>
              
              <div className="angle-item">
                <div className="angle-label">
                  Confidence
                </div>
                <div className="angle-value text-accent">
                  {currentFrameData?.bikeConfidence?.toFixed(0) ?? '--'}%
                </div>
              </div>
            </div>
          </motion.div>

          {/* Placeholder for future recommendations */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="card"
            style={{ opacity: 0.6 }}
          >
            <div className="card-header">
              <div className="card-icon">💡</div>
              <span className="card-title">Recommendations</span>
            </div>
            
            <div className="text-muted text-sm text-center py-4">
              Seat height and handlebar recommendations coming soon...
            </div>
          </motion.div>

          {/* Client-side notice */}
          <div className="text-muted text-xs text-center mt-4">
            🔒 All processing happens locally in your browser.<br/>
            Your video never leaves your device.
          </div>
        </div>
      </div>
    </div>
  );
}
