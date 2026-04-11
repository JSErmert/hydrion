/**
 * scenarios/useScenarioPlayback.ts
 *
 * Playback controller hook for ScenarioExecutionHistory.
 *
 * Responsibilities:
 *   - Track current step index over a loaded history
 *   - Auto-advance at configurable speed when playing
 *   - Expose step navigation and event marker jump controls
 *   - Derive HydrosDisplayState for the current step on every render
 *
 * Speed model: a setInterval fires every TICK_MS. Each tick advances
 * stepsPerTick = round(speedMultiplier * TICK_MS / (dt * 1000)) steps.
 * At 10× speed with dt=0.1s: 10 * 50 / 100 = 5 steps/tick = 100 steps/sec.
 *
 * Loading new history resets to step 0 automatically.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import type { ScenarioExecutionHistory, ScenarioEventMarker } from '../api/types';
import { type HydrosDisplayState, mapStepRecordToDisplayState } from './displayStateMapper';

const TICK_MS = 50;

export function useScenarioPlayback(history: ScenarioExecutionHistory | null) {
  const [stepIndex, setStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speedMultiplier, setSpeedMultiplier] = useState(10);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const totalSteps = history?.steps.length ?? 0;
  const dt = history?.dt ?? 0.1;

  const clearTimer = useCallback(() => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  // Advance playhead while isPlaying is true
  useEffect(() => {
    clearTimer();
    if (!isPlaying || totalSteps === 0) return;

    const stepsPerTick = Math.max(1, Math.round((speedMultiplier * TICK_MS) / (dt * 1000)));

    intervalRef.current = setInterval(() => {
      setStepIndex(prev => {
        const next = prev + stepsPerTick;
        if (next >= totalSteps - 1) {
          setIsPlaying(false);
          return totalSteps - 1;
        }
        return next;
      });
    }, TICK_MS);

    return clearTimer;
  }, [isPlaying, totalSteps, dt, speedMultiplier, clearTimer]);

  // Reset to step 0 when a new history is loaded
  useEffect(() => {
    setStepIndex(0);
    setIsPlaying(false);
  }, [history]);

  const play  = useCallback(() => setIsPlaying(true), []);
  const pause = useCallback(() => setIsPlaying(false), []);

  const nextStep = useCallback(() => {
    setIsPlaying(false);
    setStepIndex(prev => Math.min(prev + 1, totalSteps - 1));
  }, [totalSteps]);

  const prevStep = useCallback(() => {
    setIsPlaying(false);
    setStepIndex(prev => Math.max(prev - 1, 0));
  }, []);

  const jumpToStep = useCallback((idx: number) => {
    setIsPlaying(false);
    setStepIndex(Math.max(0, Math.min(idx, totalSteps - 1)));
  }, [totalSteps]);

  const jumpToMarker = useCallback((marker: ScenarioEventMarker) => {
    if (!history) return;
    const idx = history.steps.findIndex(s => s.t >= marker.time);
    jumpToStep(idx >= 0 ? idx : 0);
  }, [history, jumpToStep]);

  const currentStep = history?.steps[stepIndex] ?? null;
  const displayState: HydrosDisplayState | null = currentStep
    ? mapStepRecordToDisplayState(currentStep)
    : null;

  return {
    stepIndex,
    totalSteps,
    currentTime: currentStep?.t ?? 0,
    isPlaying,
    speedMultiplier,
    displayState,
    eventMarkers: history?.eventMarkers ?? [],
    play,
    pause,
    nextStep,
    prevStep,
    jumpToStep,
    jumpToMarker,
    setSpeed: setSpeedMultiplier,
  };
}
