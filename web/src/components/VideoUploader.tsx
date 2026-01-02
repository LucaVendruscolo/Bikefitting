'use client';

import { useCallback, useState } from 'react';

interface VideoUploaderProps {
  onVideoSelect: (file: File) => void;
}

export default function VideoUploader({ onVideoSelect }: VideoUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragOut = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('video/')) {
        onVideoSelect(file);
      }
    }
  }, [onVideoSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onVideoSelect(files[0]);
    }
  }, [onVideoSelect]);

  return (
    <div className="max-w-2xl mx-auto">
      <div
        className={`
          glass rounded-3xl p-12 text-center transition-all duration-300 cursor-pointer
          ${isDragging ? 'border-sky-500/50 bg-sky-500/5 scale-[1.02]' : 'hover:border-white/20'}
        `}
        onDragEnter={handleDragIn}
        onDragLeave={handleDragOut}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-input')?.click()}
      >
        {/* Upload icon */}
        <div className={`
          w-20 h-20 mx-auto mb-6 rounded-2xl flex items-center justify-center transition-all duration-300
          ${isDragging ? 'bg-sky-500/20' : 'bg-white/5'}
        `}>
          <svg 
            className={`w-10 h-10 transition-colors ${isDragging ? 'text-sky-400' : 'text-white/40'}`}
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={1.5}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
        </div>

        <h2 className="text-xl font-medium text-white mb-2">
          Upload Your Cycling Video
        </h2>
        <p className="text-white/40 mb-6">
          Drag and drop or click to browse
        </p>

        <input
          id="file-input"
          type="file"
          accept="video/*"
          onChange={handleFileInput}
          className="hidden"
        />

        <div className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-white/5 text-white/60 text-sm">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          MP4, MOV, WebM supported
        </div>
      </div>

      {/* Tips */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { icon: '📷', title: 'Side View', desc: 'Best results with side profile' },
          { icon: '🚴', title: 'Full Body', desc: 'Ensure rider is fully visible' },
          { icon: '💡', title: 'Good Lighting', desc: 'Clear visibility improves accuracy' },
        ].map((tip, i) => (
          <div key={i} className="glass-light rounded-2xl p-4 text-center">
            <div className="text-2xl mb-2">{tip.icon}</div>
            <div className="text-white/80 font-medium text-sm">{tip.title}</div>
            <div className="text-white/40 text-xs mt-1">{tip.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

