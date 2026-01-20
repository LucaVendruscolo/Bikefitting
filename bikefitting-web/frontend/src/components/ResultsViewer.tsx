'use client'

import { useRef, useState, useEffect } from 'react'
import { Download, RotateCcw, CheckCircle2, Bike, Activity, Wrench, ArrowUp, ArrowDown, ArrowLeft, ArrowRight, AlertTriangle, Check } from 'lucide-react'

export interface FrameData {
  frame: number
  time: number
  bike_angle: number | null
  knee_angle: number | null
  hip_angle: number | null
  elbow_angle: number | null
  detected_side: string | null
  is_valid?: boolean
}

export interface Recommendation {
  status: string
  action?: string | null
  adjustment_mm?: number
  details?: string
  stack_action?: string | null
  reach_action?: string | null
}

export interface Recommendations {
  saddle_height: Recommendation
  saddle_fore_aft: Recommendation
  crank_length: Recommendation
  cockpit: Recommendation
  summary: string[]
  metrics: {
    knee_max_extension: number | null
    knee_min_flexion: number | null
    min_hip_angle: number | null
    avg_elbow_angle: number | null
  }
}

export interface ProcessingResult {
  resultUrl: string
  stats: {
    frames_processed: number
    output_fps?: number
    valid_frames?: number
    recommendations?: Recommendations
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

      {/* Recommendations Panel */}
      {stats.recommendations && (
        <RecommendationsPanel recommendations={stats.recommendations} />
      )}

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

function RecommendationsPanel({ recommendations }: { recommendations: Recommendations }) {
  const { metrics, summary, saddle_height, saddle_fore_aft, crank_length, cockpit } = recommendations
  
  return (
    <div className="glass rounded-2xl p-6 border border-surface-700">
      <div className="flex items-center gap-3 mb-6">
        <Wrench className="w-5 h-5 text-brand-400" />
        <h3 className="text-lg font-semibold text-white">Bike Fit Recommendations</h3>
      </div>

      {/* Quick Summary */}
      {summary && summary.length > 0 && (
        <div className="mb-6 p-4 rounded-xl bg-surface-800/50 border border-surface-700">
          <div className="text-sm text-surface-400 mb-2">Summary</div>
          <div className="space-y-2">
            {summary.map((item, idx) => (
              <div key={idx} className="flex items-center gap-2 text-sm">
                {item.includes('optimal') || item.includes('good') || item.includes('✅') ? (
                  <Check className="w-4 h-4 text-green-400 flex-shrink-0" />
                ) : item.includes('Consider') || item.includes('⚠️') ? (
                  <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0" />
                ) : (
                  <Wrench className="w-4 h-4 text-brand-400 flex-shrink-0" />
                )}
                <span className="text-surface-300">{item.replace(/[🔧✅⚠️]/g, '').trim()}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Detailed Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard 
          label="Max Knee Extension" 
          value={metrics.knee_max_extension} 
          unit="°"
          target="140-150°"
          status={getKneeStatus(metrics.knee_max_extension)}
        />
        <MetricCard 
          label="Min Knee Flexion" 
          value={metrics.knee_min_flexion} 
          unit="°"
          target=">70°"
          status={metrics.knee_min_flexion && metrics.knee_min_flexion >= 70 ? 'ok' : 'warning'}
        />
        <MetricCard 
          label="Min Hip Angle" 
          value={metrics.min_hip_angle} 
          unit="°"
          target=">48°"
          status={metrics.min_hip_angle && metrics.min_hip_angle >= 48 ? 'ok' : 'warning'}
        />
        <MetricCard 
          label="Avg Elbow Angle" 
          value={metrics.avg_elbow_angle} 
          unit="°"
          target="150-165°"
          status={getElbowStatus(metrics.avg_elbow_angle)}
        />
      </div>

      {/* Adjustment Details */}
      <div className="grid md:grid-cols-2 gap-4">
        <AdjustmentCard
          title="Saddle Height"
          recommendation={saddle_height}
          icon={saddle_height.action === 'raise' ? <ArrowUp className="w-4 h-4" /> : 
                saddle_height.action === 'lower' ? <ArrowDown className="w-4 h-4" /> : 
                <Check className="w-4 h-4" />}
        />
        <AdjustmentCard
          title="Saddle Position"
          recommendation={saddle_fore_aft}
          icon={saddle_fore_aft.action === 'move_back' ? <ArrowLeft className="w-4 h-4" /> : 
                <Check className="w-4 h-4" />}
        />
        <AdjustmentCard
          title="Stem / Reach"
          recommendation={cockpit}
          icon={cockpit.reach_action === 'shorten' ? <ArrowLeft className="w-4 h-4" /> : 
                cockpit.reach_action === 'lengthen' ? <ArrowRight className="w-4 h-4" /> : 
                <Check className="w-4 h-4" />}
        />
        <AdjustmentCard
          title="Crank Length"
          recommendation={crank_length}
          icon={crank_length.action === 'consider_shorter' ? <AlertTriangle className="w-4 h-4" /> : 
                <Check className="w-4 h-4" />}
        />
      </div>
    </div>
  )
}

function MetricCard({ label, value, unit, target, status }: { 
  label: string
  value: number | null
  unit: string
  target: string
  status: 'ok' | 'warning' | 'error'
}) {
  const statusColors = {
    ok: 'text-green-400 border-green-500/30 bg-green-500/5',
    warning: 'text-yellow-400 border-yellow-500/30 bg-yellow-500/5',
    error: 'text-red-400 border-red-500/30 bg-red-500/5'
  }
  
  return (
    <div className={`p-3 rounded-lg border ${statusColors[status]}`}>
      <div className="text-xs text-surface-500 mb-1">{label}</div>
      <div className="text-xl font-bold font-mono">
        {value != null ? `${value.toFixed(0)}${unit}` : '—'}
      </div>
      <div className="text-xs text-surface-600 mt-1">Target: {target}</div>
    </div>
  )
}

function AdjustmentCard({ title, recommendation, icon }: {
  title: string
  recommendation: Recommendation
  icon: React.ReactNode
}) {
  const isOk = recommendation.status === 'ok'
  
  return (
    <div className={`p-4 rounded-lg border ${isOk ? 'border-surface-700 bg-surface-800/30' : 'border-brand-500/30 bg-brand-500/5'}`}>
      <div className="flex items-center gap-2 mb-2">
        <span className={isOk ? 'text-green-400' : 'text-brand-400'}>{icon}</span>
        <span className="font-medium text-white">{title}</span>
        {recommendation.adjustment_mm && recommendation.adjustment_mm > 0 && (
          <span className="ml-auto text-sm font-mono text-brand-400">
            {recommendation.action === 'raise' || recommendation.reach_action === 'lengthen' ? '+' : '-'}
            {recommendation.adjustment_mm}mm
          </span>
        )}
      </div>
      <p className="text-sm text-surface-400">{recommendation.details || 'No adjustments needed'}</p>
    </div>
  )
}

function getKneeStatus(value: number | null): 'ok' | 'warning' | 'error' {
  if (!value) return 'warning'
  if (value >= 140 && value <= 150) return 'ok'
  if (value >= 135 && value <= 155) return 'warning'
  return 'error'
}

function getElbowStatus(value: number | null): 'ok' | 'warning' | 'error' {
  if (!value) return 'warning'
  if (value >= 150 && value <= 165) return 'ok'
  if (value >= 145 && value <= 170) return 'warning'
  return 'error'
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
