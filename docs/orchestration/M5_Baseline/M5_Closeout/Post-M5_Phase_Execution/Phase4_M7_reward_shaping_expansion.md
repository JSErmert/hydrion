Operate under the HydrOS Co-Orchestrator Execution Contract.

Do not ask for confirmation or permissions.
Proceed autonomously unless you encounter a true architectural conflict, missing file, or destructive ambiguity that would risk benchmark comparability or realism.

Blocked by: Phase 3 (M7 baseline RL rebuild complete — sensor-mediated PPO benchmark established).

Objective:
Expand M7 reward design so the controller is optimized against a more mission-aligned objective after sensor realism is in place.

Context:
- Sensor-mediated RL baseline has already been established.
- Current reward logic is still Phase 1 and not fully production-aligned.
- Reward changes should now be evaluated in the more realistic M6 regime, not under privileged truth-state.

Constraints:
- Do not redesign unrelated architecture
- Preserve reproducibility
- Keep reward changes explicit and documented
- Do not silently change benchmark meaning
- Maintain claim discipline: simulation-only unless hardware calibration exists

Tasks:
1. Audit the existing reward formulation and identify the minimum reward changes needed to better align with mission outcomes.
2. Promote capture-centric reward design:
   - capture outcome / capture mass primary
   - energy and pressure secondary costs
   - maintenance/backflush timing costs retained where appropriate
3. Improve energy realism only under the following guard condition: only add energy terms if (a) the energy
   parameter can be derived from existing YAML-defined operating parameters (V_max, Q, pump efficiency) rather
   than newly assumed constants, and (b) the reward change is documented with its physical basis. Do not
   introduce energy penalty weights that are not derivable from existing parameters.
4. Retrain PPO under the revised reward.
5. Re-run benchmark comparisons against Heuristic and Random.
6. Document whether behavior becomes more mission-aligned than the prior baseline.

Return a concise decision report with:
- exact reward changes
- rationale
- benchmark impact
- tradeoffs introduced
- whether the revised reward improves mission alignment without breaking realism