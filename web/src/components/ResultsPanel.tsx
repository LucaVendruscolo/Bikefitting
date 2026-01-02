'use client';

import { ProcessingState, AnalysisResults } from '@/app/page';

interface ResultsPanelProps {
  results: AnalysisResults | null;
  processingState: ProcessingState;
  progress: number;
  error: string | null;
}

function average(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function stdDev(arr: number[]): number {
  if (arr.length === 0) return 0;
  const avg = average(arr);
  const squareDiffs = arr.map(value => Math.pow(value - avg, 2));
  return Math.sqrt(average(squareDiffs));
}

export default function ResultsPanel({
  results,
  processingState,
  progress,
  error,
}: ResultsPanelProps) {
  
  // Loading state
  if (processingState === 'loading-models') {
    return (
      <div className="bg-dark rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-4">Loading Models</h2>
        <div className="space-y-3">
          <ModelLoadingItem name="Pose Detection" progress={progress * 3} />
          <ModelLoadingItem name="Bike Segmentation" progress={Math.max(0, (progress - 10) * 3)} />
          <ModelLoadingItem name="Angle Estimation" progress={Math.max(0, (progress - 20) * 3)} />
        </div>
        <p className="text-gray-500 text-sm mt-4">
          Models are cached after first load
        </p>
      </div>
    );
  }

  // Processing state
  if (processingState === 'processing') {
    return (
      <div className="bg-dark rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-4">Analyzing Video</h2>
        <div className="text-center py-8">
          <div className="spinner mx-auto mb-4" />
          <p className="text-gray-400">{Math.round(progress)}% complete</p>
        </div>
        <div className="text-gray-500 text-sm">
          <p>• Detecting body pose</p>
          <p>• Tracking bike position</p>
          <p>• Calculating joint angles</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="bg-dark rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-4 text-red-400">Error</h2>
        <p className="text-gray-400">{error}</p>
        <p className="text-gray-500 text-sm mt-4">
          Try refreshing the page or using a different browser.
        </p>
      </div>
    );
  }

  // Results state
  if (results) {
    const kneeAvg = average(results.jointAngles.knee);
    const hipAvg = average(results.jointAngles.hip);
    const elbowAvg = average(results.jointAngles.elbow);
    const bikeAngleAvg = average(results.bikeAngles);

    return (
      <div className="bg-dark rounded-xl p-6 space-y-6">
        <h2 className="text-xl font-semibold">Analysis Results</h2>

        {/* Joint Angles */}
        <div>
          <h3 className="text-sm font-medium text-gray-400 mb-3">Joint Angles (Average)</h3>
          <div className="space-y-3">
            <AngleResult
              label="Knee"
              value={kneeAvg}
              stdDev={stdDev(results.jointAngles.knee)}
              color="cyan"
              idealRange={[70, 145]}
            />
            <AngleResult
              label="Hip"
              value={hipAvg}
              stdDev={stdDev(results.jointAngles.hip)}
              color="orange"
              idealRange={[85, 95]}
            />
            <AngleResult
              label="Elbow"
              value={elbowAvg}
              stdDev={stdDev(results.jointAngles.elbow)}
              color="green"
              idealRange={[150, 170]}
            />
          </div>
        </div>

        {/* Bike Angle */}
        <div>
          <h3 className="text-sm font-medium text-gray-400 mb-3">Bike Orientation</h3>
          <div className="bg-gray-800/50 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-300">Average Angle</span>
              <span className="text-xl font-mono">{bikeAngleAvg.toFixed(1)}°</span>
            </div>
            <p className="text-gray-500 text-xs mt-2">
              0° = facing camera, ±180° = facing away
            </p>
          </div>
        </div>

        {/* Recommendations Placeholder */}
        <div className="border border-dashed border-gray-700 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-400 mb-2">
            Recommendations
          </h3>
          <p className="text-gray-500 text-sm">
            Personalized bike fit recommendations coming soon...
          </p>
          <div className="mt-3 flex gap-2">
            <span className="text-xs px-2 py-1 bg-gray-800 rounded">Seat Height</span>
            <span className="text-xs px-2 py-1 bg-gray-800 rounded">Handlebar</span>
            <span className="text-xs px-2 py-1 bg-gray-800 rounded">Reach</span>
          </div>
        </div>

        {/* Stats */}
        <div className="text-gray-500 text-xs">
          <p>Analyzed {results.frameCount} frames at {results.fps} FPS</p>
        </div>
      </div>
    );
  }

  // Idle state
  return (
    <div className="bg-dark rounded-xl p-6">
      <h2 className="text-xl font-semibold mb-4">How It Works</h2>
      <div className="space-y-4 text-gray-400 text-sm">
        <Step number={1} title="Upload Video">
          Record yourself cycling from the side and upload the video.
        </Step>
        <Step number={2} title="AI Analysis">
          Our AI detects your body pose and bike position in each frame.
        </Step>
        <Step number={3} title="Get Results">
          See your joint angles and bike fit measurements.
        </Step>
        <Step number={4} title="Recommendations" badge="Coming Soon">
          Get personalized suggestions to optimize your bike fit.
        </Step>
      </div>
    </div>
  );
}

function ModelLoadingItem({ name, progress }: { name: string; progress: number }) {
  const clampedProgress = Math.min(100, Math.max(0, progress));
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-400">{name}</span>
        <span className="text-gray-500">{Math.round(clampedProgress)}%</span>
      </div>
      <div className="progress-bar">
        <div
          className="progress-bar-fill"
          style={{ width: `${clampedProgress}%` }}
        />
      </div>
    </div>
  );
}

function AngleResult({
  label,
  value,
  stdDev,
  color,
  idealRange,
}: {
  label: string;
  value: number;
  stdDev: number;
  color: 'cyan' | 'orange' | 'green';
  idealRange: [number, number];
}) {
  const colorClasses = {
    cyan: 'text-cyan-400 bg-cyan-500/20',
    orange: 'text-orange-400 bg-orange-500/20',
    green: 'text-green-400 bg-green-500/20',
  };

  const isInRange = value >= idealRange[0] && value <= idealRange[1];

  return (
    <div className={`rounded-lg p-3 ${colorClasses[color]}`}>
      <div className="flex items-center justify-between">
        <span className="font-medium">{label}</span>
        <div className="text-right">
          <span className="text-xl font-mono">{value.toFixed(1)}°</span>
          <span className="text-xs text-gray-400 ml-1">±{stdDev.toFixed(1)}</span>
        </div>
      </div>
      <div className="text-xs mt-1 opacity-70">
        Ideal: {idealRange[0]}° - {idealRange[1]}°
        {!isInRange && value > 0 && (
          <span className="ml-2 text-yellow-400">⚠ Outside range</span>
        )}
      </div>
    </div>
  );
}

function Step({
  number,
  title,
  children,
  badge,
}: {
  number: number;
  title: string;
  children: React.ReactNode;
  badge?: string;
}) {
  return (
    <div className="flex gap-3">
      <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center text-sm">
        {number}
      </div>
      <div>
        <div className="flex items-center gap-2">
          <h3 className="font-medium text-white">{title}</h3>
          {badge && (
            <span className="text-xs px-2 py-0.5 bg-emerald-500/20 text-emerald-400 rounded">
              {badge}
            </span>
          )}
        </div>
        <p className="mt-1">{children}</p>
      </div>
    </div>
  );
}

