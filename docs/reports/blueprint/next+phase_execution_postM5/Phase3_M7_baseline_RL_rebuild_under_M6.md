Operate under the HydrOS Co-Orchestrator Execution Contract.

Do not ask for confirmation or permissions.
Proceed autonomously unless you encounter a true architectural conflict, missing file, or destructive ambiguity that would risk training validity.

Objective:
Rebuild the baseline PPO benchmark under sensor-mediated observation so HydrOS can quantify how much current PPO performance survives realistic sensing.

Context:
- M6 core sensor realism exists (Phase 1 complete).
- RL observation is now sensor-mediated under obs12_v3 (Phase 2 complete).
- The truth-state baseline is ppo_cce_v2: models/ppo_cce_v2.zip, 500k steps, seed=42, d_p=1.0µm,
  obs12_v2, trained entirely under truth_state observation. All Phase 3 comparisons MUST use ppo_cce_v2
  as the prior-state anchor. Do not compare against ppo_cce_v1 (archived, benchmark-invalid).
- The submicron benchmark regime (d_p=1.0µm) is a physics-validation regime chosen to activate DEP
  sensitivity. It is not representative of the full laundry outflow particle distribution (fibers
  100–5000µm, fragments 10–500µm). Label all Phase 3 results accordingly.

Blocked by: Phase 2 (M6 observation handoff complete — obs12_v3 defined, sensor_state reads active).

Constraints:
- Do not broaden scope into reward redesign yet
- Keep benchmark structure as stable as possible
- Preserve reproducibility
- Keep artifact naming and metadata explicit
- Do not overclaim deployment readiness

Tasks:
1. Retrain baseline PPO under the new M6 observation path using the canonical benchmark regime.
2. Save clean artifacts and metadata.
3. Re-run PPO vs Heuristic vs Random under the new observation regime.
4. Compare:
   - truth-state PPO benchmark vs sensor-state PPO benchmark
   - PPO vs Heuristic
   - PPO vs Random
5. Quantify performance degradation or robustness under realistic sensing.

Return a concise decision report with:
- training configuration
- artifact paths
- benchmark result table
- comparison vs prior truth-state benchmark
- whether PPO remains meaningfully better than baselines under sensor realism
- whether PPO's learned flow-rate operating point under M6 observation remains consistent with or conflicts with
  the hardware drain flow range (12-15 L/min); note explicitly if the policy continues to prefer Q ≤ 7.7 L/min
- whether RL is now closer to deployment-validity