'use client'

import { useState } from 'react'
import { Bike } from 'lucide-react'
import VideoUploader from '@/components/VideoUploader'
import ResultsViewer from '@/components/ResultsViewer'

type ProcessingStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'error'

interface FrameData {
  frame: number
  time: number
  bike_angle: number | null
  knee_angle: number | null
  hip_angle: number | null
  elbow_angle: number | null
  detected_side: string | null
}

interface ProcessingResult {
  resultUrl: string
  stats: { frames_processed: number; output_fps?: number }
  frameData?: FrameData[]
}

export default function Home() {
  const [status, setStatus] = useState<ProcessingStatus>('idle')
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ProcessingResult | null>(null)

  const handleReset = () => {
    setStatus('idle')
    setProgress(0)
    setError(null)
    setResult(null)
  }

  return (
    <main className="min-h-screen">
      {/* Header */}
      <header className="border-b border-surface-800/50 glass sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-brand-600 flex items-center justify-center glow-green">
                <Bike className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-lg font-semibold tracking-tight">BikeFit AI</h1>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <section className="py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          {status === 'completed' && result ? (
            <ResultsViewer result={result} onReset={handleReset} />
          ) : (
            <VideoUploader
              status={status}
              progress={progress}
              error={error}
              onUploadStart={() => { setStatus('uploading'); setProgress(0); setError(null); setResult(null) }}
              onUploadProgress={(p) => setProgress(p * 0.3)}
              onProcessingStart={() => { setStatus('processing'); setProgress(30) }}
              onProcessingProgress={(p) => setProgress(30 + p * 0.7)}
              onComplete={(data) => { setStatus('completed'); setProgress(100); setResult(data) }}
              onError={(msg) => { setStatus('error'); setError(msg) }}
            />
          )}
        </div>
      </section>
    </main>
  )
}
