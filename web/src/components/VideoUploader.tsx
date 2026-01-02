'use client';

import { useCallback, useRef, useState } from 'react';

interface VideoUploaderProps {
  onVideoSelect: (file: File) => void;
}

export default function VideoUploader({ onVideoSelect }: VideoUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('video/')) {
        onVideoSelect(file);
      }
    },
    [onVideoSelect]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        onVideoSelect(file);
      }
    },
    [onVideoSelect]
  );

  const handleClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  return (
    <div
      onClick={handleClick}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={`
        relative cursor-pointer
        border-2 border-dashed rounded-2xl
        p-12 md:p-20
        text-center
        transition-all duration-300
        ${isDragging 
          ? 'border-blue-400 bg-blue-500/10 scale-[1.02]' 
          : 'border-gray-600 hover:border-gray-500 bg-dark/50'
        }
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        onChange={handleFileSelect}
        className="hidden"
      />

      {/* Upload Icon */}
      <div className="mb-6">
        <svg
          className={`w-16 h-16 mx-auto transition-colors ${
            isDragging ? 'text-blue-400' : 'text-gray-500'
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
          />
        </svg>
      </div>

      <h2 className="text-xl md:text-2xl font-semibold mb-2">
        {isDragging ? 'Drop your video here' : 'Upload Your Cycling Video'}
      </h2>
      
      <p className="text-gray-400 mb-4">
        Drag and drop or click to select
      </p>

      <div className="flex flex-wrap justify-center gap-2 text-sm text-gray-500">
        <span className="px-3 py-1 bg-gray-800 rounded-full">MP4</span>
        <span className="px-3 py-1 bg-gray-800 rounded-full">MOV</span>
        <span className="px-3 py-1 bg-gray-800 rounded-full">WebM</span>
      </div>

      {/* Tips */}
      <div className="mt-8 pt-8 border-t border-gray-700">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Tips for best results:</h3>
        <ul className="text-sm text-gray-500 space-y-1">
          <li>• Film from the side (perpendicular to bike)</li>
          <li>• Keep the full body and bike in frame</li>
          <li>• Good lighting helps accuracy</li>
          <li>• 10-30 seconds is ideal</li>
        </ul>
      </div>
    </div>
  );
}

