'use client'

import { useCallback, useState, useRef } from 'react'
import { Upload, Video, Loader2, AlertCircle, CheckCircle2, Settings2, Clock, Play, Square } from 'lucide-react'

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
  stats: {
    frames_processed: number
    output_fps?: number
  }
  frameData?: FrameData[]
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

// Format seconds to MM:SS
function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

// Time input with separate minute/second fields
function TimeInput({ value, onChange, maxSeconds }: { 
  value: number
  onChange: (seconds: number) => void
  maxSeconds: number 
}) {
  const mins = Math.floor(value / 60)
  const secs = Math.floor(value % 60)

  const handleMinsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newMins = Math.max(0, parseInt(e.target.value) || 0)
    const newTotal = newMins * 60 + secs
    if (newTotal <= maxSeconds) onChange(newTotal)
  }

  const handleSecsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    let newSecs = parseInt(e.target.value) || 0
    newSecs = Math.max(0, Math.min(59, newSecs))
    const newTotal = mins * 60 + newSecs
    if (newTotal <= maxSeconds) onChange(newTotal)
  }

  return (
    <div className="flex items-center gap-1">
      <input
        type="number"
        min="0"
        max="99"
        value={mins}
        onChange={handleMinsChange}
        className="w-14 px-2 py-2 rounded-lg bg-surface-800 border border-surface-700 text-white font-mono text-sm text-center focus:border-brand-500 focus:outline-none"
      />
      <span className="text-surface-500 font-mono">:</span>
      <input
        type="number"
        min="0"
        max="59"
        value={secs.toString().padStart(2, '0')}
        onChange={handleSecsChange}
        className="w-14 px-2 py-2 rounded-lg bg-surface-800 border border-surface-700 text-white font-mono text-sm text-center focus:border-brand-500 focus:outline-none"
      />
    </div>
  )
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
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [videoDuration, setVideoDuration] = useState(0)
  const [settings, setSettings] = useState({ outputFps: 10, startTime: 0, endTime: 0 })
  const [showSettings, setShowSettings] = useState(true)
  const [progressInfo, setProgressInfo] = useState<ProgressInfo | null>(null)
  
  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)

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
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(URL.createObjectURL(file))
    setSettings({ outputFps: 10, startTime: 0, endTime: 0 })
    setVideoDuration(0)
  }

  const handleVideoLoaded = () => {
    if (videoRef.current) {
      const duration = videoRef.current.duration
      setVideoDuration(duration)
      setSettings(prev => ({ ...prev, endTime: Math.min(duration, 120) }))
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    try {
      onUploadStart()

      const effectiveEndTime = settings.endTime > 0 ? settings.endTime : videoDuration
      const clipDuration = effectiveEndTime - settings.startTime
      const durationRatio = videoDuration > 0 ? clipDuration / videoDuration : 1
      const estimatedSize = selectedFile.size * durationRatio

      if (selectedFile.size > 200 * 1024 * 1024) {
        onError(`File too large. Maximum is 200MB.`)
        return
      }
      if (estimatedSize > 50 * 1024 * 1024) {
        onError(`Selected clip too large (~${(estimatedSize / (1024 * 1024)).toFixed(1)}MB). Try a shorter range.`)
        return
      }

      // Convert to base64
      const base64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve((reader.result as string).split(',')[1])
        reader.onerror = reject
        reader.onprogress = (e) => e.lengthComputable && onUploadProgress((e.loaded / e.total) * 100)
        reader.readAsDataURL(selectedFile)
      })

      onUploadProgress(100)
      onProcessingStart()
      setProgressInfo({ stage: 'setup', message: 'Connecting...', percent: 10 })

      // Call streaming API
      const response = await fetch('/api/process-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_base64: base64,
          output_fps: settings.outputFps,
          start_time: settings.startTime,
          end_time: effectiveEndTime,
          max_duration_sec: Math.min(clipDuration, 120),
        }),
      })

      if (!response.ok || !response.body) {
        throw new Error('Failed to connect to server')
      }

      // Parse SSE stream
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let finalResult: { job_id?: string; stats?: Record<string, unknown>; frame_data?: FrameData[] } | null = null

      const processLine = (line: string) => {
        if (!line.startsWith('data: ')) return
        try {
          const data = JSON.parse(line.slice(6))
          if (data.type === 'error') throw new Error(data.error)
          if (data.type === 'progress') {
            setProgressInfo({
              stage: data.stage,
              message: data.message,
              percent: data.percent,
              current: data.current,
              total: data.total || data.total_frames
            })
            onProcessingProgress(data.percent)
          }
          if (data.type === 'complete') {
            finalResult = data
          }
        } catch (e) {
          // Only throw if it's an actual error from the server
          if (e instanceof Error && e.message !== 'No result received') throw e
        }
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          processLine(line)
        }
      }

      // Process any remaining data in buffer
      if (buffer.trim()) {
        processLine(buffer)
      }

      if (!finalResult) throw new Error('No result received')

      const downloadBaseUrl = process.env.NEXT_PUBLIC_MODAL_DOWNLOAD_URL
      if (!downloadBaseUrl) throw new Error('Download URL not configured')
      
      const result = finalResult as { job_id?: string; stats?: Record<string, unknown>; frame_data?: FrameData[] }
      const downloadUrl = result.job_id
        ? `${downloadBaseUrl}?job_id=${result.job_id}`
        : ''

      onComplete({
        resultUrl: downloadUrl,
        stats: (result.stats as ProcessingResult['stats']) || { frames_processed: 0 },
        frameData: result.frame_data || [],
      })
    } catch (err) {
      onError(err instanceof Error ? err.message : 'Upload failed')
    }
  }

  const isProcessing = status === 'uploading' || status === 'processing'

  return (
    <div className="space-y-6">
      {/* Upload Zone */}
      <div
        className={`upload-zone rounded-2xl p-8 sm:p-12 text-center transition-all ${
          dragOver ? 'dragover' : ''
        } ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
          className="hidden"
        />

        {!selectedFile ? (
          <div className="space-y-4">
            <div className="w-16 h-16 rounded-2xl bg-brand-500/10 flex items-center justify-center mx-auto">
              <Upload className="w-8 h-8 text-brand-400" />
            </div>
            <div>
              <p className="text-lg font-medium mb-1">Drop your video here</p>
              <p className="text-sm text-surface-400">or click to browse</p>
            </div>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-6 py-3 rounded-xl bg-brand-500 hover:bg-brand-600 text-white font-medium transition-colors"
            >
              Select Video
            </button>
            <p className="text-xs text-surface-500">
              MP4, MOV, AVI • Max 200MB • Side-view cycling footage works best
            </p>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Video Preview */}
            <div className="relative max-w-md mx-auto rounded-xl overflow-hidden bg-black">
              <video
                ref={videoRef}
                src={previewUrl || undefined}
                className="w-full aspect-video object-contain"
                controls
                muted
                onLoadedMetadata={handleVideoLoaded}
              />
              {videoDuration > 0 && (
                <div className="absolute bottom-2 right-2 px-2 py-1 bg-black/70 rounded text-xs text-white font-mono">
                  {formatTime(videoDuration)}
                </div>
              )}
            </div>

            {/* File Info */}
            <div className="flex items-center justify-center gap-3 text-sm">
              <Video className="w-4 h-4 text-brand-400" />
              <span className="text-surface-300">{selectedFile.name}</span>
              <span className="text-surface-500">({(selectedFile.size / (1024 * 1024)).toFixed(1)} MB)</span>
            </div>

            {/* Actions */}
            {!isProcessing && status !== 'completed' && (
              <div className="flex items-center justify-center gap-4">
                <button
                  onClick={() => {
                    setSelectedFile(null)
                    if (previewUrl) URL.revokeObjectURL(previewUrl)
                    setPreviewUrl(null)
                  }}
                  className="px-4 py-2 rounded-lg text-surface-400 hover:text-white transition-colors"
                >
                  Choose Different
                </button>
                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className="px-4 py-2 rounded-lg bg-surface-800 hover:bg-surface-700 transition-colors flex items-center gap-2"
                >
                  <Settings2 className="w-4 h-4" />
                  Settings
                </button>
                <button
                  onClick={handleUpload}
                  className="px-6 py-3 rounded-xl bg-brand-500 hover:bg-brand-600 text-white font-medium transition-colors glow-green"
                >
                  Analyze Video
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Settings Panel */}
      {showSettings && !isProcessing && selectedFile && (
        <div className="glass rounded-xl p-6 space-y-6">
          <h3 className="font-medium flex items-center gap-2">
            <Settings2 className="w-4 h-4 text-brand-400" />
            Processing Settings
          </h3>

          {/* FPS */}
          <div>
            <label className="block text-sm text-surface-400 mb-2">Output FPS</label>
            <div className="grid grid-cols-4 gap-2">
              {[5, 10, 15, 30].map(fps => (
                <button
                  key={fps}
                  onClick={() => setSettings({ ...settings, outputFps: fps })}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    settings.outputFps === fps
                      ? 'bg-brand-500 text-white'
                      : 'bg-surface-800 text-surface-300 hover:bg-surface-700'
                  }`}
                >
                  {fps} FPS
                </button>
              ))}
            </div>
            <p className="text-xs text-surface-500 mt-2">Lower = faster processing</p>
          </div>

          {/* Time Range */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-sm text-surface-400 flex items-center gap-2">
                <Clock className="w-4 h-4" />
                Time Range
              </label>
              {videoDuration > 0 && (
                <span className="text-xs text-surface-500">Length: {formatTime(videoDuration)}</span>
              )}
            </div>

            <div className="grid sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-surface-500 mb-2">Start (mm:ss)</label>
                <div className="flex items-center gap-2">
                  <TimeInput
                    value={settings.startTime}
                    onChange={(v) => v < (settings.endTime || videoDuration) && setSettings({ ...settings, startTime: v })}
                    maxSeconds={videoDuration}
                  />
                  <button
                    onClick={() => videoRef.current && setSettings({ 
                      ...settings, 
                      startTime: videoRef.current.currentTime,
                      endTime: Math.max(settings.endTime, videoRef.current.currentTime + 1)
                    })}
                    className="px-3 py-2 rounded-lg bg-surface-700 hover:bg-surface-600 text-surface-300"
                    title="Use current position"
                  >
                    <Play className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-xs text-surface-500 mb-2">End (mm:ss)</label>
                <div className="flex items-center gap-2">
                  <TimeInput
                    value={settings.endTime || videoDuration}
                    onChange={(v) => v > settings.startTime && setSettings({ ...settings, endTime: v })}
                    maxSeconds={videoDuration}
                  />
                  <button
                    onClick={() => videoRef.current && setSettings({
                      ...settings,
                      endTime: videoRef.current.currentTime,
                      startTime: Math.min(settings.startTime, videoRef.current.currentTime - 1)
                    })}
                    className="px-3 py-2 rounded-lg bg-surface-700 hover:bg-surface-600 text-surface-300"
                    title="Use current position"
                  >
                    <Square className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Timeline */}
            {videoDuration > 0 && (
              <div className="space-y-2">
                <div className="relative h-2 bg-surface-800 rounded-full">
                  <div
                    className="absolute h-full bg-brand-500/50 rounded-full"
                    style={{
                      left: `${(settings.startTime / videoDuration) * 100}%`,
                      width: `${(((settings.endTime || videoDuration) - settings.startTime) / videoDuration) * 100}%`,
                    }}
                  />
                </div>
                <div className="flex justify-between text-xs text-surface-500">
                  <span>Duration: {formatTime((settings.endTime || videoDuration) - settings.startTime)}</span>
                  <span className={((settings.endTime || videoDuration) - settings.startTime) > 120 ? 'text-orange-400' : ''}>
                    {((settings.endTime || videoDuration) - settings.startTime) > 120 ? '⚠️ Capped at 2:00' : 'Max 2 min'}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Processing Progress */}
      {isProcessing && <ProcessingProgress status={status} progressInfo={progressInfo} />}

      {/* Error */}
      {error && (
        <div className="glass rounded-xl p-4 border border-red-500/30 bg-red-500/10">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-red-400">Error</p>
              <p className="text-sm text-surface-400">{error}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

interface ProgressInfo {
  stage: string
  message: string
  percent: number
  current?: number
  total?: number
}

function ProcessingProgress({ status, progressInfo }: { status: ProcessingStatus; progressInfo: ProgressInfo | null }) {
  const steps = [
    { id: 'upload', label: 'Uploading video' },
    { id: 'setup', label: 'Loading AI models' },
    { id: 'models', label: 'Models ready' },
    { id: 'processing', label: 'Analyzing frames' },
    { id: 'encoding', label: 'Encoding video' },
    { id: 'finalizing', label: 'Saving results' },
  ]

  const getStepStatus = (stepId: string, index: number): 'pending' | 'active' | 'done' => {
    if (!progressInfo) {
      return status === 'uploading' && index === 0 ? 'active' : 'pending'
    }
    const currentIndex = steps.findIndex(s => s.id === progressInfo.stage)
    if (index < currentIndex) return 'done'
    if (index === currentIndex) return 'active'
    return 'pending'
  }

  return (
    <div className="glass rounded-xl p-6">
      <div className="space-y-3">
        {steps.map((step, index) => {
          const stepStatus = getStepStatus(step.id, index)
          return (
            <div key={step.id} className="flex items-center gap-3">
              <div className="w-5 h-5 flex items-center justify-center">
                {stepStatus === 'active' && <Loader2 className="w-5 h-5 animate-spin text-brand-400" />}
                {stepStatus === 'done' && <CheckCircle2 className="w-5 h-5 text-brand-400" />}
                {stepStatus === 'pending' && <div className="w-3 h-3 rounded-full bg-surface-600" />}
              </div>
              <span className={
                stepStatus === 'active' ? 'text-white' :
                stepStatus === 'done' ? 'text-surface-400' : 'text-surface-600'
              }>
                {step.label}
              </span>
            </div>
          )
        })}
      </div>

      {progressInfo && (
        <div className="mt-4 pt-4 border-t border-surface-800">
          <p className="text-sm text-brand-400">{progressInfo.message}</p>
          {progressInfo.current != null && progressInfo.total != null && (
            <p className="text-xs text-surface-500 mt-1">
              Frame {progressInfo.current} / {progressInfo.total}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
