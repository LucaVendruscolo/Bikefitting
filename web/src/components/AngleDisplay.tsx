'use client';

import { CurrentAngles } from '@/lib/types';

interface AngleDisplayProps {
  angles: CurrentAngles;
  detectedSide: 'left' | 'right';
}

export default function AngleDisplay({ angles, detectedSide }: AngleDisplayProps) {
  const formatAngle = (angle: number | null) => {
    if (angle === null) return '—';
    return `${Math.round(angle)}°`;
  };

  return (
    <div className="space-y-4">
      {/* Joint Angles Card */}
      <div className="glass rounded-3xl p-6">
        <div className="flex items-center gap-2 mb-6">
          <div className="w-2 h-2 rounded-full bg-cyan-400" />
          <h3 className="text-sm font-medium text-white/60 uppercase tracking-wider">
            Joint Angles
          </h3>
          <span className="ml-auto text-xs text-white/40 capitalize">
            {detectedSide} side
          </span>
        </div>

        <div className="space-y-5">
          {/* Knee */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-cyan-500/10 flex items-center justify-center">
                <svg className="w-5 h-5 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 4v4m0 4v8M8 8l4 4 4-4" />
                </svg>
              </div>
              <span className="text-white/80">Knee</span>
            </div>
            <span className="text-2xl font-semibold text-cyan-400 angle-value">
              {formatAngle(angles.knee)}
            </span>
          </div>

          {/* Hip */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-orange-500/10 flex items-center justify-center">
                <svg className="w-5 h-5 text-orange-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="8" r="4" />
                  <path d="M12 12v4m-4 4l4-4 4 4" />
                </svg>
              </div>
              <span className="text-white/80">Hip</span>
            </div>
            <span className="text-2xl font-semibold text-orange-400 angle-value">
              {formatAngle(angles.hip)}
            </span>
          </div>

          {/* Elbow */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-green-500/10 flex items-center justify-center">
                <svg className="w-5 h-5 text-green-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M8 8l8 8M16 8l-8 8" />
                </svg>
              </div>
              <span className="text-white/80">Elbow</span>
            </div>
            <span className="text-2xl font-semibold text-green-400 angle-value">
              {formatAngle(angles.elbow)}
            </span>
          </div>
        </div>
      </div>

      {/* Bike Angle Card */}
      <div className="glass rounded-3xl p-6">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-2 h-2 rounded-full bg-emerald-400" />
          <h3 className="text-sm font-medium text-white/60 uppercase tracking-wider">
            Bike Angle
          </h3>
        </div>

        <div className="text-center py-4">
          <div className="text-5xl font-semibold text-emerald-400 angle-value">
            {formatAngle(angles.bike)}
          </div>
          <p className="text-white/40 text-sm mt-3">
            0° = perpendicular to camera
          </p>
        </div>

        {/* Visual angle indicator */}
        <div className="mt-4 flex justify-center">
          <div className="relative w-24 h-24">
            {/* Circle */}
            <div className="absolute inset-0 rounded-full border-2 border-white/10" />
            
            {/* Reference line (0°) */}
            <div className="absolute top-1/2 left-1/2 w-1/2 h-0.5 bg-white/20 origin-left" />
            
            {/* Angle line */}
            {angles.bike !== null && (
              <div
                className="absolute top-1/2 left-1/2 w-1/2 h-1 bg-emerald-400 origin-left rounded-full transition-transform duration-200"
                style={{
                  transform: `rotate(${-angles.bike}deg)`,
                }}
              />
            )}
            
            {/* Center dot */}
            <div className="absolute top-1/2 left-1/2 w-3 h-3 -ml-1.5 -mt-1.5 rounded-full bg-emerald-400" />
          </div>
        </div>
      </div>

      {/* Future: Recommendations placeholder */}
      <div className="glass-light rounded-3xl p-6 opacity-50">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-2 h-2 rounded-full bg-violet-400" />
          <h3 className="text-sm font-medium text-white/40 uppercase tracking-wider">
            Recommendations
          </h3>
        </div>
        <p className="text-white/30 text-sm">
          Seat and handlebar recommendations coming soon...
        </p>
      </div>
    </div>
  );
}

