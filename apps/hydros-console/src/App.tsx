import { useState, useEffect } from 'react';
import HeaderBar from './layout/HeaderBar';
import TopTelemetryBand from './components/TopTelemetryBand';
import ConicalCascadeView from './components/ConicalCascadeView';
import WebGLCascadeView from './components/WebGLCascadeView';
import BottomNarrativeBand from './components/BottomNarrativeBand';
import RightAdvisoryPanel from './components/RightAdvisoryPanel';
import PlaybackBar from './components/PlaybackBar';
import { listScenarios, runScenario } from './api';
import type { ScenarioInfo, ScenarioExecutionHistory } from './api';
import { useScenarioPlayback } from './scenarios/useScenarioPlayback';

type ViewMode = '2d' | '3d';

export default function App() {
  const [scenarios, setScenarios]   = useState<ScenarioInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string>('baseline_nominal');
  const [isRunning, setIsRunning]   = useState(false);
  const [loadError, setLoadError]   = useState<string | null>(null);
  const [history, setHistory]       = useState<ScenarioExecutionHistory | null>(null);
  const [viewMode, setViewMode]     = useState<ViewMode>('3d');

  const playback = useScenarioPlayback(history);

  useEffect(() => {
    listScenarios().then(result => {
      if (result.ok) setScenarios(result.data);
    });
  }, []);

  const handleRun = async () => {
    setIsRunning(true);
    setLoadError(null);
    setHistory(null);
    const result = await runScenario(selectedId);
    setIsRunning(false);
    if (result.ok) setHistory(result.data);
    else setLoadError(result.error.message);
  };

  const {
    displayState, eventMarkers,
    stepIndex, totalSteps, currentTime, isPlaying, speedMultiplier,
    play, pause, nextStep, prevStep, jumpToStep, jumpToMarker, setSpeed,
  } = playback;

  const scenarioDuration = history?.steps.length
    ? history.steps[history.steps.length - 1].t
    : 0;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      background: '#080D18',
      overflow: 'hidden',
    }}>

      {/* Region 1 — Header */}
      <HeaderBar />

      {/* Region 2 — Top Telemetry Band */}
      <TopTelemetryBand state={displayState} />

      {/* Region 3 — Core Machine View + Right Advisory */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        {/* Machine View (dominant) — 2D SVG or 3D WebGL */}
        <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
          {viewMode === '2d'
            ? <ConicalCascadeView state={displayState} />
            : <WebGLCascadeView
                state={displayState}
                playbackTime={currentTime}
                isPlaying={isPlaying}
                speedMultiplier={speedMultiplier}
              />}

          {/* 2D / 3D toggle */}
          <div style={{
            position: 'absolute',
            top: 12,
            right: 12,
            display: 'flex',
            gap: 4,
            padding: 3,
            background: 'rgba(8, 13, 24, 0.85)',
            border: '1px solid #1E293B',
            borderRadius: 4,
            zIndex: 10,
          }}>
            {(['2d', '3d'] as const).map(m => (
              <button
                key={m}
                onClick={() => setViewMode(m)}
                style={{
                  padding: '4px 10px',
                  background: viewMode === m ? '#38BDF8' : 'transparent',
                  color: viewMode === m ? '#080D18' : '#7DD3FC',
                  border: 'none',
                  borderRadius: 3,
                  cursor: 'pointer',
                  font: '11px "JetBrains Mono", monospace',
                  fontWeight: 600,
                  letterSpacing: 0.5,
                }}
              >
                {m.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* Right Advisory Panel */}
        <RightAdvisoryPanel state={displayState} />
      </div>

      {/* Region 4 — Bottom Narrative Band */}
      <BottomNarrativeBand state={displayState} />

      {/* Region 5 — Scenario selector + playback controls */}
      <div style={{ borderTop: '1px solid #1E293B', flexShrink: 0 }}>
        <PlaybackBar
          scenarios={scenarios}
          selectedId={selectedId}
          onSelectScenario={setSelectedId}
          onRun={handleRun}
          isRunning={isRunning}
          loadError={loadError}
          hasHistory={!!history}
          stepIndex={stepIndex}
          totalSteps={totalSteps}
          currentTime={currentTime}
          scenarioDuration={scenarioDuration}
          isPlaying={isPlaying}
          speedMultiplier={speedMultiplier}
          eventMarkers={eventMarkers}
          onPlay={play}
          onPause={pause}
          onNextStep={nextStep}
          onPrevStep={prevStep}
          onJumpToStep={jumpToStep}
          onJumpToMarker={jumpToMarker}
          onSetSpeed={setSpeed}
        />
      </div>
    </div>
  );
}
