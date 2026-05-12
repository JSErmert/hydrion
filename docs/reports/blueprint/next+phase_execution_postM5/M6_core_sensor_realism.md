Operate under the HydrOS Co-Orchestrator Execution Contract.

Do not ask for confirmation or permissions.
Proceed autonomously unless you encounter a true architectural conflict, missing file, or destructive ambiguity that would risk architecture or state separation.

Objective:
Implement the core of M6 sensor realism so RL observation can begin transitioning from privileged truth-state to realistic sensor-mediated state.

Context:
- Post-M5 benchmark work is complete or sufficiently stabilized.
- M6 is the current gating stage for deployment-relevant RL.
- Critical missing sensors: differential pressure and flow.
- truth_state must remain authoritative; sensor_state must remain observational.

Constraints:
- Do not collapse truth_state and sensor_state
- Do not silently alter observation schema
- Do not skip validation
- Do not broaden scope into reward redesign or UI
- Preserve locked architecture and pipeline order

Tasks:
1. Audit the current sensor pipeline and identify the minimum implementation path for:
   - differential pressure sensor
   - flow sensor
2. Implement a differential pressure sensor model:
   - derive from hydraulic truth outputs
   - add realistic bounded noise
   - add bias/offset if appropriate
   - add latency/smoothing if appropriate
3. Implement a flow sensor model:
   - derive from truth flow
   - add realistic bounded noise
   - add bias/lag if appropriate
4. Write outputs only to sensor_state.
5. Preserve all state separation guarantees.
6. Add validation/tests that confirm:
   - sensor outputs diverge from truth in realistic bounded ways
   - fixed seed remains deterministic
   - no truth/sensor contamination occurs

Return a concise decision report with:
- affected modules
- exact sensor models added
- validation performed
- failure conditions considered
- whether M6 core sensors now exist
- what remains before RL can consume sensor_state