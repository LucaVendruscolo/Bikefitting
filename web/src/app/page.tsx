'use client';

import { useState, useCallback } from 'react';
import VideoUploader from '@/components/VideoUploader';
import VideoPlayer from '@/components/VideoPlayer';

export default function Home() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  const handleVideoSelect = useCallback((file: File) => {
    setVideoFile(file);
    setVideoUrl(URL.createObjectURL(file));
  }, []);

  const handleReset = useCallback(() => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setVideoFile(null);
    setVideoUrl(null);
  }, [videoUrl]);

  return (
    <main className="min-h-screen p-4 md:p-8">
      {/* Header */}
      <header className="text-center mb-8">
        <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-cyan-400 to-emerald-400 bg-clip-text text-transparent">
          BikeFitting AI
        </h1>
        <p className="text-gray-400 mt-2 text-lg">
          Upload a video to analyze your bike fit using AI
        </p>
      </header>

      <div className="max-w-6xl mx-auto">
        {/* Video Upload Section */}
        {!videoFile && (
          <VideoUploader onVideoSelect={handleVideoSelect} />
        )}

        {/* Video Player Section */}
        {videoFile && videoUrl && (
          <div className="space-y-4">
            <VideoPlayer videoUrl={videoUrl} />
            
            <button
              onClick={handleReset}
              className="px-6 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition"
            >
              Upload New Video
            </button>
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
