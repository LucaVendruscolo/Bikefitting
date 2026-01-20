'use client'

import { RotateCcw, CheckCircle, AlertTriangle, ArrowUp, ArrowDown, Minus } from 'lucide-react'

export interface Recommendation {
  status: string
  action?: string | null
  adjustment_mm?: number
  details?: string
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
  stats: {
    frames_processed: number
    samples_taken?: number
    recommendations?: Recommendations
  }
}

interface ResultsViewerProps {
  result: ProcessingResult
  onReset: () => void
}

export default function ResultsViewer({ result, onReset }: ResultsViewerProps) {
  const { stats } = result
  const recommendations = stats.recommendations

  if (!recommendations) {
    return (
      <div className="text-center py-12">
        <p className="text-secondary">No recommendations available</p>
        <button onClick={onReset} className="btn-primary mt-6">
          Try Again
        </button>
      </div>
    )
  }

  const { metrics, summary, saddle_height, saddle_fore_aft, crank_length, cockpit } = recommendations

  // Count issues
  const issueCount = summary.filter(s => 
    !s.includes('optimal') && !s.includes('good')
  ).length

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center animate-fade-in">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-[var(--accent-dim)] mb-4">
          <CheckCircle className="w-8 h-8 text-[#30d158]" />
        </div>
        <h2 className="text-3xl font-semibold tracking-tight">Analysis Complete</h2>
        <p className="text-secondary mt-2">
          {stats.samples_taken || stats.frames_processed} frames analyzed
          {issueCount > 0 ? ` • ${issueCount} adjustment${issueCount > 1 ? 's' : ''} suggested` : ' • All optimal'}
        </p>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 animate-fade-in stagger-1">
        <MetricCard
          label="Knee Extension"
          value={metrics.knee_max_extension}
          target="140-150°"
          status={getKneeStatus(metrics.knee_max_extension)}
        />
        <MetricCard
          label="Knee Flexion"
          value={metrics.knee_min_flexion}
          target=">70°"
          status={metrics.knee_min_flexion && metrics.knee_min_flexion >= 70 ? 'ok' : 'warning'}
        />
        <MetricCard
          label="Hip Angle"
          value={metrics.min_hip_angle}
          target=">48°"
          status={metrics.min_hip_angle && metrics.min_hip_angle >= 48 ? 'ok' : 'warning'}
        />
        <MetricCard
          label="Elbow Angle"
          value={metrics.avg_elbow_angle}
          target="150-160°"
          status={getElbowStatus(metrics.avg_elbow_angle)}
        />
      </div>

      {/* Recommendations */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium text-secondary uppercase tracking-wider animate-fade-in stagger-2">
          Adjustments
        </h3>
        <div className="grid md:grid-cols-2 gap-3">
          <RecommendationCard
            title="Saddle Height"
            recommendation={saddle_height}
            className="animate-fade-in stagger-2"
          />
          <RecommendationCard
            title="Saddle Position"
            recommendation={saddle_fore_aft}
            className="animate-fade-in stagger-3"
          />
          <RecommendationCard
            title="Crank Length"
            recommendation={crank_length}
            className="animate-fade-in stagger-4"
          />
          <RecommendationCard
            title="Stem / Reach"
            recommendation={cockpit}
            className="animate-fade-in stagger-5"
          />
        </div>
      </div>

      {/* Action */}
      <div className="flex justify-center pt-4 animate-fade-in stagger-6">
        <button onClick={onReset} className="btn-secondary flex items-center gap-2">
          <RotateCcw className="w-4 h-4" />
          Analyze Another Video
        </button>
      </div>
    </div>
  )
}

function MetricCard({ label, value, target, status }: {
  label: string
  value: number | null
  target: string
  status: 'ok' | 'warning' | 'error'
}) {
  const statusClass = status === 'ok' ? 'status-ok' : status === 'warning' ? 'status-warning' : 'status-error'
  
  return (
    <div className={`card p-4 ${statusClass}`}>
      <p className="text-xs opacity-60 mb-1">{label}</p>
      <p className="text-2xl font-semibold tabular-nums">
        {value != null ? `${Math.round(value)}°` : '—'}
      </p>
      <p className="text-xs opacity-50 mt-1">{target}</p>
    </div>
  )
}

function RecommendationCard({ title, recommendation, className }: {
  title: string
  recommendation: Recommendation
  className?: string
}) {
  const isOk = recommendation.status === 'ok'
  const hasAction = recommendation.action || recommendation.reach_action
  
  const getIcon = () => {
    if (isOk) return <CheckCircle className="w-5 h-5 text-[#30d158]" />
    
    const action = recommendation.action || recommendation.reach_action
    if (action === 'raise' || action === 'lengthen') return <ArrowUp className="w-5 h-5 text-[#ffd60a]" />
    if (action === 'lower' || action === 'shorten' || action === 'move_back') return <ArrowDown className="w-5 h-5 text-[#ffd60a]" />
    if (action === 'consider_shorter') return <AlertTriangle className="w-5 h-5 text-[#ffd60a]" />
    return <Minus className="w-5 h-5 text-[var(--surface-300)]" />
  }

  return (
    <div className={`card p-4 flex items-start gap-4 ${className}`}>
      <div className="flex-shrink-0 mt-0.5">
        {getIcon()}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <h4 className="font-medium">{title}</h4>
          {hasAction && recommendation.adjustment_mm != null && recommendation.adjustment_mm > 0 && (
            <span className="text-sm font-medium text-[#30d158] tabular-nums">
              {recommendation.action === 'raise' || recommendation.reach_action === 'lengthen' ? '+' : '−'}
              {recommendation.adjustment_mm}mm
            </span>
          )}
        </div>
        <p className="text-sm text-secondary mt-1">
          {recommendation.details || 'No adjustment needed'}
        </p>
      </div>
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
  if (value >= 150 && value <= 160) return 'ok'
  if (value >= 145 && value <= 165) return 'warning'
  return 'error'
}
