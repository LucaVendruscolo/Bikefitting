'use client';

import { useState, useRef, useCallback } from 'react';
import VideoUploader from '@/components/VideoUploader';
import VideoProcessor from '@/components/VideoProcessor';
import ResultsPanel from '@/components/ResultsPanel';

export type ProcessingState = 'idle' | 'loading-models' | 'processing' | 'complete' | 'error';

export interface AnalysisResults {
  jointAngles: {
    knee: number[];
    hip: number[];
    elbow: number[];
  };
  bikeAngles: number[];
  frameCount: number;
  fps: number;
}

export default function Home() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [processingState, setProcessingState] = useState<ProcessingState>('idle');
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleVideoSelect = useCallback((file: File) => {
    setVideoFile(file);
    setVideoUrl(URL.createObjectURL(file));
    setResults(null);
    setError(null);
    setProcessingState('idle');
    setProgress(0);
  }, []);

  const handleProcessingComplete = useCallback((analysisResults: AnalysisResults) => {
    setResults(analysisResults);
    setProcessingState('complete');
  }, []);

  const handleError = useCallback((errorMessage: string) => {
    setError(errorMessage);
    setProcessingState('error');
  }, []);

  const handleReset = useCallback(() => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setVideoFile(null);
    setVideoUrl(null);
    setResults(null);
    setError(null);
    setProcessingState('idle');
    setProgress(0);
  }, [videoUrl]);

  return (
    <main className="min-h-screen p-4 md:p-8">
      {/* Header */}
      <header className="text-center mb-8">
        <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
          BikeFitting AI
        </h1>
        <p className="text-gray-400 mt-2 text-lg">
          Upload a video to analyze your bike fit using AI
        </p>
      </header>

      <div className="max-w-7xl mx-auto">
        {/* Video Upload Section */}
        {!videoFile && (
          <VideoUploader onVideoSelect={handleVideoSelect} />
        )}

        {/* Video Processing Section */}
        {videoFile && videoUrl && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Video & Canvas */}
            <div className="lg:col-span-2">
              <VideoProcessor
                videoUrl={videoUrl}
                processingState={processingState}
                setProcessingState={setProcessingState}
                progress={progress}
                setProgress={setProgress}
                onComplete={handleProcessingComplete}
                onError={handleError}
              />
              
              {/* Controls */}
              <div className="mt-4 flex gap-4">
                <button
                  onClick={handleReset}
                  className="px-6 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition"
                >
                  Upload New Video
                </button>
              </div>
            </div>

            {/* Results Panel */}
            <div className="lg:col-span-1">
              <ResultsPanel
                results={results}
                processingState={processingState}
                progress={progress}
                error={error}
              />
            </div>
          </div>
        )}

        {/* Info Section */}
        <section className="mt-12 text-center text-gray-500 text-sm">
          <p>All processing happens locally in your browser. Your video never leaves your device.</p>
          <p className="mt-1">Works best with a side-view video of you cycling.</p>
        </section>
      </div>
    </main>
  );
}

