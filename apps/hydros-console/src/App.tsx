import { useState, useEffect } from 'react';
import HeaderBar from './layout/HeaderBar';
import TopTelemetryBand from './components/TopTelemetryBand';
import MachineCore from './components/MachineCore';
import BottomNarrativeBand from './components/BottomNarrativeBand';
import RightAdvisoryPanel from './components/RightAdvisoryPanel';
import PlaybackBar from './components/PlaybackBar';
import { listScenarios, runScenario } from './api';
import type { ScenarioInfo, ScenarioExecutionHistory } from './api';
import { useScenarioPlayback } from './scenarios/useScenarioPlayback';

export default function App() {
  const [scenarios, setScenarios]   = useState<ScenarioInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string>('baseline_nominal');
  const [isRunning, setIsRunning]   = useState(false);
  const [loadError, setLoadError]   = useState<string | null>(null);
  const [history, setHistory]       = useState<ScenarioExecutionHistory | null>(null);

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
        {/* Machine SVG (dominant) */}
        <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
          <MachineCore state={displayState} />
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
