Operate under the HydrOS Co-Orchestrator Execution Contract.

Do not ask for confirmation or permissions.
Proceed autonomously unless you encounter a true architectural conflict, missing file, or destructive ambiguity that would risk architecture integrity or sensor state purity.

Objective:
Implement physically grounded sensor models for differential pressure and flow rate, populating sensor_state with realistic noisy, drifted, and latency-affected readings. This is the M6 sensor realism core — the prerequisite for any deployment-valid RL observation.

Context:
- Phase 0 (M5 closeout) is complete.
- sensor_state currently carries only optical sensors (turbidity, scatter).
- RL observation uses truth_state directly — zero deployment validity.
- No differential pressure sensor exists in the codebase.
- No flow sensor exists in the codebase.
- The architecture constraint (physics → truth_state only; sensors → sensor_state only) is clean and must be preserved.
- YAML configuration is required for all new sensor parameters.

Constraints:
- Physics modules must NOT read from sensor_state.
- Sensor modules must NOT write to truth_state.
- Do not change observation schema yet — that is Phase 2.
- Do not retrain RL yet — sensor state must exist first.
- Keep changes minimal, modular, and independently testable.
- All new sensor parameters go in configs/default.yaml under a sensors: block.

Tasks:
1. Implement differential pressure sensor model (hydrion/sensors/pressure.py or within optical.py):
   - Source: truth_state dp_total_pa (or per-stage ΔP values)
   - Model: sensor_dp = truth_dp + N(0, σ_dp) + drift(t) + fouling_offset(clog_norm)
   - Drift: slow linear accumulation capped at ±drift_max_kPa
   - Fouling offset: sensor bias increases proportional to mesh_loading_avg (simulates membrane fouling on sensor port)
   - Latency: 1–2 step lag (ring buffer)
   - Write result to sensor_state["dp_sensor_kPa"]; do NOT write to truth_state.

2. Implement flow rate sensor model:
   - Source: truth_state q_processed_lmin
   - Model: sensor_q = truth_q × (1 + N(0, σ_q)) + calibration_bias_lmin
   - Calibration bias: fixed offset sampled at episode reset (simulates factory calibration offset)
   - Write result to sensor_state["flow_sensor_lmin"]; do NOT write to truth_state.

3. Update hydrion/state/init.py to include initial values for the new sensor_state keys:
   - sensor_state["dp_sensor_kPa"] = 0.0
   - sensor_state["flow_sensor_lmin"] = 0.0
   These must be present at episode start so downstream consumers do not read undefined keys.

4. Add YAML parameters for new sensors under sensors: block:
   - dp_noise_kPa, dp_drift_rate_kPa_per_step, dp_drift_max_kPa, dp_fouling_gain, dp_latency_steps
   - flow_noise_frac, flow_bias_std_lmin

5. Verify sensor/truth separation:
   - Add test: sensor_dp diverges from truth_dp by a measurable amount under nominal conditions.
   - Add test: sensor_q diverges from truth_q.
   - Add test: no sensor write contaminates truth_state.

6. Run full test suite (88 tests + new tests) and confirm pass.

7. Document sensor model assumptions and calibration path in a brief note within hydrion/sensors/.

Acceptance criteria:
- sensor_state["dp_sensor_kPa"] populates correctly after each step with noise/drift applied.
- sensor_state["flow_sensor_lmin"] populates correctly.
- Sensor values measurably diverge from truth values under standard conditions (test confirms divergence > 0).
- No truth_state contamination from sensor writes (test confirms).
- All existing tests still pass.
- New sensor parameters exposed in YAML with documented defaults.

Blocked by: Phase 0 (M5 closeout complete — app.py path updated, per-stage eta audit resolved).

Return a concise decision report with:
- exact files changed
- sensor model equations implemented
- YAML parameters added
- test results (new + existing)
- whether sensor_state now contains realistic non-truth readings
- blockers before Phase 2 (observation handoff)
