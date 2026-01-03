'use client'

import { useRef, useState, useEffect } from 'react'
import { Download, RotateCcw, CheckCircle2, Bike, Activity } from 'lucide-react'

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

interface ResultsViewerProps {
  result: ProcessingResult
  onReset: () => void
}

export default function ResultsViewer({ result, onReset }: ResultsViewerProps) {
  const { stats, frameData } = result
  const videoRef = useRef<HTMLVideoElement>(null)
  const [currentFrame, setCurrentFrame] = useState<FrameData | null>(null)

  // Sync video playback with frame data
  useEffect(() => {
    const video = videoRef.current
    if (!video || !frameData || frameData.length === 0) return

    const handleTimeUpdate = () => {
      const currentTime = video.currentTime
      
      // Find the closest frame by time
      let closestFrame = frameData[0]
      let minDiff = Math.abs(currentTime - closestFrame.time)

      for (const frame of frameData) {
        const diff = Math.abs(currentTime - frame.time)
        if (diff < minDiff) {
          minDiff = diff
          closestFrame = frame
        }
      }

      setCurrentFrame(closestFrame)
    }

    video.addEventListener('timeupdate', handleTimeUpdate)
    handleTimeUpdate() // Set initial frame

    return () => video.removeEventListener('timeupdate', handleTimeUpdate)
  }, [frameData])

  return (
    <div className="space-y-6">
      {/* Success Header */}
      <div className="glass rounded-2xl p-4 border border-brand-500/30 bg-brand-500/5">
        <div className="flex items-center gap-3">
          <CheckCircle2 className="w-5 h-5 text-brand-400" />
          <p className="text-brand-400">
            Analysis Complete — {stats.frames_processed} frames
          </p>
        </div>
      </div>

      {/* Video + Live Data */}
      <div className="grid lg:grid-cols-[1fr,280px] gap-4">
        {/* Video Player */}
        <div className="video-container">
          <video
            ref={videoRef}
            src={result.resultUrl}
            className="w-full aspect-video rounded-xl"
            controls
            autoPlay
            loop
            muted
          />
        </div>

        {/* Live Data Panel */}
        <div className="glass rounded-xl p-4 space-y-4">
          <div className="text-xs text-surface-500 uppercase tracking-wide">
            Live Analysis
          </div>

          {/* Bike Angle */}
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-sm text-surface-400">
              <Bike className="w-4 h-4" />
              <span>Bike Angle</span>
            </div>
            <div className="text-3xl font-bold font-mono text-brand-400">
              {currentFrame?.bike_angle != null
                ? `${currentFrame.bike_angle.toFixed(1)}°`
                : '—'}
            </div>
          </div>

          {/* Joint Angles */}
          <div className="border-t border-surface-800 pt-4">
            <div className="flex items-center gap-2 text-sm text-surface-400 mb-3">
              <Activity className="w-4 h-4" />
              <span>Joint Angles</span>
              {currentFrame?.detected_side && (
                <span className="text-xs px-2 py-0.5 bg-surface-800 rounded">
                  {currentFrame.detected_side}
                </span>
              )}
            </div>

            <div className="space-y-3">
              <AngleRow label="Knee" value={currentFrame?.knee_angle} color="text-cyan-400" />
              <AngleRow label="Hip" value={currentFrame?.hip_angle} color="text-orange-400" />
              <AngleRow label="Elbow" value={currentFrame?.elbow_angle} color="text-purple-400" />
            </div>
          </div>

          {/* Frame Counter */}
          <div className="border-t border-surface-800 pt-4 text-xs text-surface-500">
            Frame {currentFrame?.frame ?? 0} / {stats.frames_processed}
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
        <button
          onClick={() => {
            const a = document.createElement('a')
            a.href = result.resultUrl
            a.download = 'bikefitting_analysis.mp4'
            a.click()
          }}
          className="w-full sm:w-auto px-6 py-3 rounded-xl bg-brand-500 hover:bg-brand-600 text-white font-medium transition-colors flex items-center justify-center gap-2 glow-green"
        >
          <Download className="w-4 h-4" />
          Download Video
        </button>
        <button
          onClick={onReset}
          className="w-full sm:w-auto px-6 py-3 rounded-xl bg-surface-800 hover:bg-surface-700 text-white font-medium transition-colors flex items-center justify-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Analyze Another Video
        </button>
      </div>
    </div>
  )
}

function AngleRow({ label, value, color }: { label: string; value?: number | null; color: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-surface-400">{label}</span>
      <span className={`text-lg font-mono font-semibold ${value != null ? color : 'text-surface-600'}`}>
        {value != null ? `${value.toFixed(1)}°` : '—'}
      </span>
    </div>
  )
}
