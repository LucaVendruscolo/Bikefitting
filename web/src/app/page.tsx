'use client';

import { useState, useCallback } from 'react';
import VideoUploader from '@/components/VideoUploader';
import ProcessingView from '@/components/ProcessingView';
import PlaybackView from '@/components/PlaybackView';
import { FrameData } from '@/lib/types';

type AppState = 'upload' | 'configure' | 'processing' | 'playback';

export default function Home() {
  const [appState, setAppState] = useState<AppState>('upload');
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [processedFrames, setProcessedFrames] = useState<FrameData[]>([]);
  const [processedRange, setProcessedRange] = useState({ start: 0, end: 0 });

  const handleVideoSelect = useCallback((file: File) => {
    // Clean up previous URL
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    
    setVideoFile(file);
    setVideoUrl(URL.createObjectURL(file));
    setAppState('configure');
    setProcessedFrames([]);
  }, [videoUrl]);

  const handleProcessingComplete = useCallback((frames: FrameData[], start: number, end: number) => {
    setProcessedFrames(frames);
    setProcessedRange({ start, end });
    setAppState('playback');
  }, []);

  const handleReset = useCallback(() => {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoFile(null);
    setVideoUrl(null);
    setProcessedFrames([]);
    setAppState('upload');
  }, [videoUrl]);

  const handleReprocess = useCallback(() => {
    setProcessedFrames([]);
    setAppState('configure');
  }, []);

  return (
    <main className="min-h-screen p-6 md:p-10">
      {/* Header */}
      <header className="text-center mb-10">
        <h1 className="text-4xl md:text-5xl font-semibold tracking-tight">
          <span className="bg-gradient-to-r from-sky-400 via-cyan-300 to-emerald-400 bg-clip-text text-transparent">
            BikeFitting
          </span>
          <span className="text-white/90 ml-2">AI</span>
        </h1>
        <p className="text-white/50 mt-3 text-lg">
          Analyze your cycling posture with AI
        </p>
      </header>

      <div className="max-w-6xl mx-auto">
        {/* Upload State */}
        {appState === 'upload' && (
          <VideoUploader onVideoSelect={handleVideoSelect} />
        )}

        {/* Configure & Processing States */}
        {(appState === 'configure' || appState === 'processing') && videoUrl && (
          <ProcessingView
            videoUrl={videoUrl}
            isProcessing={appState === 'processing'}
            onStartProcessing={() => setAppState('processing')}
            onProcessingComplete={handleProcessingComplete}
            onReset={handleReset}
          />
        )}

        {/* Playback State */}
        {appState === 'playback' && videoUrl && (
          <PlaybackView
            videoUrl={videoUrl}
            frames={processedFrames}
            startTime={processedRange.start}
            endTime={processedRange.end}
            onReprocess={handleReprocess}
            onReset={handleReset}
          />
        )}
      </div>

      {/* Footer */}
      <footer className="mt-16 text-center text-white/30 text-sm">
        <p>All processing happens locally in your browser</p>
        <p className="mt-1">Your video never leaves your device</p>
      </footer>
    </main>
  );
}
