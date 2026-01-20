'use client'

import { useState } from 'react'
import VideoUploader from '@/components/VideoUploader'
import ResultsViewer, { ProcessingResult } from '@/components/ResultsViewer'

type ProcessingStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'error'

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
    <main className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="glass sticky top-0 z-50 border-b border-white/5">
        <div className="max-w-3xl mx-auto px-6 h-14 flex items-center">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#30d158] flex items-center justify-center">
              <svg className="w-4 h-4 text-black" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <circle cx="5" cy="18" r="3" />
                <circle cx="19" cy="18" r="3" />
                <path d="M5 18l4-10h6l4 10M9 8l3-5 3 5" />
              </svg>
            </div>
            <span className="font-semibold text-[15px]">BikeFit</span>
          </div>
        </div>
      </header>

      {/* Main */}
      <section className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-2xl">
          {status === 'completed' && result ? (
            <ResultsViewer result={result} onReset={handleReset} />
          ) : (
            <div className="space-y-8">
              {/* Hero text */}
              {status === 'idle' && (
                <div className="text-center animate-fade-in">
                  <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight">
                    Bike Fit Analysis
                  </h1>
                  <p className="text-secondary text-lg mt-3 max-w-md mx-auto">
                    Upload a side-view cycling video and get instant fit recommendations
                  </p>
                </div>
              )}
              
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
            </div>
          )}
        </div>
      </section>
    </main>
  )
}
