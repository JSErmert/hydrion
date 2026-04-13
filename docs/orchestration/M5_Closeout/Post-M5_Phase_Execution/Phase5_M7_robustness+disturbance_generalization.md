Operate under the HydrOS Co-Orchestrator Execution Contract.

Do not ask for confirmation or permissions.
Proceed autonomously unless you encounter a true architectural conflict, missing file, or destructive ambiguity that would risk result interpretation.

Blocked by: Phase 4 (reward shaping complete — revised reward benchmarked and documented).

Objective:
Extend M7 beyond narrow benchmark optimization into robustness and disturbance generalization.

Context:
- Baseline PPO under M6 exists
- Reward shaping has been expanded and benchmarked
- Current open RL gap includes lack of curriculum / disturbance generalization

Constraints:
- Do not change locked interfaces unnecessarily
- Keep scenarios explicit and traceable
- Maintain simulation-only claim discipline
- Do not overclaim robustness beyond tested regimes

Tasks:
1. Define a minimal robustness evaluation suite including:
   - nominal baseline
   - threshold-edge case
   - high-flow stress case: pump_cmd ≥ 0.85 (Q ≈ 12-15 L/min, matching residential laundry drain flow range);
     tests whether PPO maintains capture performance at real-world inlet flow rates or collapses to the
     DEP-inactive regime
   - rapid fouling / clogging buildup case
   - recovery / backflush case
   - at least one unseen disturbance condition
2. Implement the minimum scenario or evaluation support needed.
3. Evaluate PPO vs Heuristic vs Random across the robustness suite.
4. Identify where PPO generalizes and where it fails.
5. Recommend whether curriculum-style training is now warranted.

Return a concise decision report with:
- scenarios tested
- benchmark outcomes by scenario
- robustness strengths
- failure modes
- whether curriculum / broader generalization training should be the next RL step; any curriculum
  recommendation must explicitly note that (a) all results remain simulation-only, (b) DEP flow-rate
  hardware architecture is unresolved, and (c) PSD breadth has not been expanded beyond the current
  benchmark regime