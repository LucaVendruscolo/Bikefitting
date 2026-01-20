'use client'

import { useCallback, useState, useRef } from 'react'
import { Upload, AlertCircle } from 'lucide-react'
import { Recommendations } from './ResultsViewer'

type ProcessingStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'error'

interface ProcessingResult {
  stats: {
    frames_processed: number
    samples_taken?: number
    recommendations?: Recommendations
  }
}

interface VideoUploaderProps {
  status: ProcessingStatus
  progress: number
  error: string | null
  onUploadStart: () => void
  onUploadProgress: (percent: number) => void
  onProcessingStart: () => void
  onProcessingProgress: (percent: number) => void
  onComplete: (result: ProcessingResult) => void
  onError: (message: string) => void
}

export default function VideoUploader({
  status,
  error,
  onUploadStart,
  onUploadProgress,
  onProcessingStart,
  onProcessingProgress,
  onComplete,
  onError,
}: VideoUploaderProps) {
  const [dragOver, setDragOver] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [progressMessage, setProgressMessage] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file?.type.startsWith('video/')) {
      handleFileSelect(file)
    } else {
      onError('Please drop a valid video file')
    }
  }, [onError])

  const handleFileSelect = (file: File) => {
    setSelectedFile(file)
    // Auto-start analysis
    setTimeout(() => handleUpload(file), 100)
  }

  const handleUpload = async (file: File) => {
    try {
      onUploadStart()
      setProgressMessage('Preparing video...')

      if (file.size > 200 * 1024 * 1024) {
        onError('File too large. Maximum is 200MB.')
        return
      }

      // Convert to base64
      const base64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve((reader.result as string).split(',')[1])
        reader.onerror = reject
        reader.onprogress = (e) => {
          if (e.lengthComputable) {
            onUploadProgress((e.loaded / e.total) * 100)
            setProgressMessage('Uploading...')
          }
        }
        reader.readAsDataURL(file)
      })

      onProcessingStart()
      setProgressMessage('Connecting to AI...')

      const modalUrl = process.env.NEXT_PUBLIC_MODAL_STREAM_URL
      if (!modalUrl) throw new Error('Server URL not configured')

      const response = await fetch(modalUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_base64: base64 }),
      })

      if (!response.ok || !response.body) {
        throw new Error('Failed to connect to server')
      }

      // Parse SSE stream
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let finalResult: { stats?: Record<string, unknown> } | null = null

      const processLine = (line: string) => {
        if (!line.startsWith('data: ')) return
        try {
          const data = JSON.parse(line.slice(6))
          if (data.type === 'error') throw new Error(data.error)
          if (data.type === 'progress') {
            setProgressMessage(data.message || 'Processing...')
            onProcessingProgress(data.percent)
          }
          if (data.type === 'complete') {
            finalResult = data
          }
        } catch (e) {
          if (e instanceof Error && e.message !== 'No result received') throw e
        }
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        for (const line of lines) processLine(line)
      }
      if (buffer.trim()) processLine(buffer)

      if (!finalResult) throw new Error('No result received')

      onComplete({
        stats: (finalResult.stats as ProcessingResult['stats']) || { frames_processed: 0 },
      })
    } catch (err) {
      onError(err instanceof Error ? err.message : 'Analysis failed')
    }
  }

  const isProcessing = status === 'uploading' || status === 'processing'

  return (
    <div className="space-y-8">
      {/* Upload Zone */}
      <div
        className={`upload-zone p-12 sm:p-16 text-center cursor-pointer transition-all ${
          dragOver ? 'dragover' : ''
        } ${isProcessing ? 'opacity-60 pointer-events-none' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isProcessing && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
          className="hidden"
        />

        {isProcessing ? (
          <div className="space-y-6 animate-fade-in">
            {/* Spinner */}
            <div className="w-16 h-16 mx-auto relative">
              <svg className="w-16 h-16 progress-ring" viewBox="0 0 64 64">
                <circle
                  cx="32" cy="32" r="28"
                  fill="none"
                  stroke="rgba(255,255,255,0.1)"
                  strokeWidth="4"
                />
                <circle
                  cx="32" cy="32" r="28"
                  fill="none"
                  stroke="#30d158"
                  strokeWidth="4"
                  strokeLinecap="round"
                  strokeDasharray="176"
                  strokeDashoffset="44"
                  className="animate-spin"
                  style={{ transformOrigin: 'center' }}
                />
              </svg>
            </div>
            <div>
              <p className="text-lg font-medium">{progressMessage}</p>
              <p className="text-sm text-secondary mt-1">This typically takes 15-30 seconds</p>
            </div>
          </div>
        ) : (
          <div className="space-y-5">
            <div className="w-14 h-14 rounded-2xl bg-[var(--accent-dim)] flex items-center justify-center mx-auto">
              <Upload className="w-6 h-6 text-[#30d158]" />
            </div>
            <div>
              <p className="text-xl font-medium">Drop your video here</p>
              <p className="text-secondary mt-1">or click to browse</p>
            </div>
            <p className="text-xs text-secondary">
              MP4, MOV, AVI • Max 200MB • Side-view cycling footage
            </p>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="card p-4 status-error animate-fade-in">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">Error</p>
              <p className="text-sm opacity-80">{error}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
