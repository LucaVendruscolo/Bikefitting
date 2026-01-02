'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Types
interface ProcessingMetrics {
  totalFrames: number;
  processedFrames: number;
  avgTimePerFrame: number;
  segmentationTime: number;
  poseDetectionTime: number;
  bikeAngleTime: number;
  totalTime: number;
}

interface FrameData {
  frameIndex: number;
  timestamp: number;
  jointAngles: {
    knee: number | null;
    hip: number | null;
    elbow: number | null;
  };
  bikeAngle: number | null;
  bikeConfidence: number | null;
  detectedSide: 'left' | 'right' | null;
}

interface ProcessedVideo {
  videoUrl: string;
  frames: FrameData[];
  startFrame: number;
  endFrame: number;
  fps: number;
  metrics: ProcessingMetrics;
}

export default function HomePage() {
  // State
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState(0);
  const [videoFps, setVideoFps] = useState(30);
  
  // Selection state
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [outputFps, setOutputFps] = useState(10);
  
  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [currentProcessingFrame, setCurrentProcessingFrame] = useState(0);
  const [processingStatus, setProcessingStatus] = useState('');
  
  // Result state
  const [processedVideo, setProcessedVideo] = useState<ProcessedVideo | null>(null);
  const [currentFrameData, setCurrentFrameData] = useState<FrameData | null>(null);
  
  // Player state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  
  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const previewVideoRef = useRef<HTMLVideoElement>(null);
  const resultVideoRef = useRef<HTMLVideoElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Handle file upload
  const handleFileSelect = useCallback((file: File) => {
    if (!file.type.startsWith('video/')) {
      alert('Please select a video file');
      return;
    }
    
    setUploadedFile(file);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setProcessedVideo(null);
  }, []);

  // Handle video metadata load
  const handleVideoLoad = useCallback(() => {
    if (previewVideoRef.current) {
      const duration = previewVideoRef.current.duration;
      setVideoDuration(duration);
      setEndTime(Math.min(duration, 30)); // Default to 30 seconds or video length
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
    if (!uploadedFile) return;
    
    setIsProcessing(true);
    setProcessingProgress(0);
    setCurrentProcessingFrame(0);
    setProcessingStatus('Uploading video...');
    
    try {
      // Create form data
      const formData = new FormData();
      formData.append('video', uploadedFile);
      formData.append('start_time', startTime.toString());
      formData.append('end_time', endTime.toString());
      formData.append('output_fps', outputFps.toString());
      
      // Start processing
      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Processing failed');
      }
      
      // Read streaming response for progress
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      
      let result: ProcessedVideo | null = null;
      let buffer = '';
      
      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer
        
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            
            if (data.type === 'progress') {
              setProcessingProgress(data.progress);
              setCurrentProcessingFrame(data.currentFrame);
              setProcessingStatus(data.status);
            } else if (data.type === 'complete') {
              result = data.result;
            } else if (data.type === 'error') {
              throw new Error(data.message);
            }
          } catch {
            // Ignore parse errors for partial data
          }
        }
      }
      
      if (result) {
        setProcessedVideo(result);
        setProcessingStatus('Processing complete!');
      }
      
    } catch (error) {
      console.error('Processing error:', error);
      setProcessingStatus(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Update current frame data based on video time
  useEffect(() => {
    if (processedVideo && resultVideoRef.current) {
      const video = resultVideoRef.current;
      const frameIndex = Math.floor(currentTime * processedVideo.fps);
      const frameData = processedVideo.frames.find(f => f.frameIndex === frameIndex);
      setCurrentFrameData(frameData || null);
    }
  }, [currentTime, processedVideo]);

  // Video player time update
  const handleTimeUpdate = useCallback(() => {
    if (resultVideoRef.current) {
      setCurrentTime(resultVideoRef.current.currentTime);
    }
  }, []);

  // Format time display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Play/Pause toggle
  const togglePlay = () => {
    if (resultVideoRef.current) {
      if (isPlaying) {
        resultVideoRef.current.pause();
      } else {
        resultVideoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  // Seek to position
  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (resultVideoRef.current && processedVideo) {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percentage = x / rect.width;
      const duration = processedVideo.frames.length / processedVideo.fps;
      resultVideoRef.current.currentTime = percentage * duration;
    }
  };

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <div className="logo-icon">🚴</div>
          <span className="logo-text">BikeFit Pro</span>
        </div>
        <div className="text-muted text-sm">AI-Powered Bike Fitting Analysis</div>
      </header>

      {/* Main Content */}
      <div className="main-grid">
        {/* Left Column - Video Area */}
        <div className="main-content">
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
            ) : processedVideo ? (
              // Result Video Player
              <motion.div
                key="result"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="card"
              >
                <div className="card-header">
                  <div className="card-icon">🎬</div>
                  <span className="card-title">Analysis Result</span>
                </div>
                
                <div className="video-container">
                  <video
                    ref={resultVideoRef}
                    className="video-player"
                    src={processedVideo.videoUrl}
                    onTimeUpdate={handleTimeUpdate}
                    onEnded={() => setIsPlaying(false)}
                  />
                  
                  {/* Video Controls */}
                  <div className="video-controls">
                    {/* Timeline */}
                    <div className="timeline" onClick={handleSeek}>
                      <div 
                        className="timeline-progress"
                        style={{ 
                          width: `${(currentTime / (processedVideo.frames.length / processedVideo.fps)) * 100}%` 
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
                        {formatTime(currentTime)} / {formatTime(processedVideo.frames.length / processedVideo.fps)}
                      </span>
                      
                      <div className="flex-1" />
                      
                      <button 
                        className="btn btn-secondary"
                        onClick={() => {
                          setProcessedVideo(null);
                          setCurrentFrameData(null);
                        }}
                      >
                        New Video
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            ) : (
              // Preview Video
              <motion.div
                key="preview"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="card"
              >
                <div className="card-header">
                  <div className="card-icon">📹</div>
                  <span className="card-title">Video Preview</span>
                  <div className="flex-1" />
                  <button 
                    className="btn btn-secondary btn-sm"
                    onClick={() => {
                      setUploadedFile(null);
                      setVideoUrl(null);
                    }}
                  >
                    Change Video
                  </button>
                </div>
                
                <div className="video-container">
                  <video
                    ref={previewVideoRef}
                    className="video-player"
                    src={videoUrl}
                    controls
                    onLoadedMetadata={handleVideoLoad}
                  />
                </div>
                
                {/* Selection Controls */}
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
                        max={30}
                        onChange={(e) => setOutputFps(parseInt(e.target.value) || 10)}
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
                    disabled={isProcessing}
                  >
                    {isProcessing ? 'Processing...' : 'Start Analysis'}
                  </button>
                </div>
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
          {processedVideo?.metrics && (
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
                  <div className="metric-value">{processedVideo.metrics.totalFrames}</div>
                  <div className="metric-label">Total Frames</div>
                </div>
                <div className="metric-item">
                  <div className="metric-value">{processedVideo.metrics.avgTimePerFrame.toFixed(0)}ms</div>
                  <div className="metric-label">Avg per Frame</div>
                </div>
                <div className="metric-item">
                  <div className="metric-value">{processedVideo.metrics.segmentationTime.toFixed(0)}ms</div>
                  <div className="metric-label">Segmentation</div>
                </div>
                <div className="metric-item">
                  <div className="metric-value">{processedVideo.metrics.poseDetectionTime.toFixed(0)}ms</div>
                  <div className="metric-label">Pose Detection</div>
                </div>
                <div className="metric-item">
                  <div className="metric-value">{processedVideo.metrics.bikeAngleTime.toFixed(0)}ms</div>
                  <div className="metric-label">Bike Angle</div>
                </div>
                <div className="metric-item">
                  <div className="metric-value">{(processedVideo.metrics.totalTime / 1000).toFixed(1)}s</div>
                  <div className="metric-label">Total Time</div>
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
        </div>
      </div>
    </div>
  );
}

