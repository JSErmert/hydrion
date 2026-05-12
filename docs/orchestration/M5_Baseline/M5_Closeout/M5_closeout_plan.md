Operate under the HydrOS Co-Orchestrator Execution Contract.

Do not ask for confirmation or permissions.
Proceed autonomously unless you encounter a true architectural conflict, missing file, or destructive ambiguity that would risk repo integrity or benchmark validity.

Objective:
Close the remaining post-M5 proof and runtime hygiene items so the current benchmark stack becomes fully trustworthy and operationally aligned.

Context:
- The blueprint is now internally clean and current.
- ppo_cce_v2 is the canonical benchmark artifact.
- ppo_cce_v1 is archived / benchmark-invalid.
- Remaining known open items:
  1. app.py still points _PPO_CCE_MODEL_PATH to ppo_cce_v1.zip
  2. per-stage eta audit is still open to confirm PPO’s eta_cascade = 1.000 is genuine and not residual saturation

Constraints:
- Preserve architecture and locked interfaces
- Do not broaden scope into M6 or reward redesign
- Do not change observation schema
- Keep changes minimal and traceable

Tasks:
1. Update hydrion/service/app.py so _PPO_CCE_MODEL_PATH points to ppo_cce_v2.zip.
2. Run a short smoke test through the relevant service/API path to confirm the canonical model is now being served.
   - Pass criteria: POST to /api/run with policy_type=ppo_cce; response completes without error; logged telemetry step
     values are non-zero (confirms live CCE state is being read, not stale initial-state defaults).
3. Perform a per-stage eta audit at the PPO operating point:
   - inspect eta_s1, eta_s2, eta_s3
   - confirm whether eta_cascade = 1.000 reflects genuine nDEP-optimal behavior or residual saturation
   - Note: per-stage η values are accessible from cce._state or by running eval_ppo_cce.py with per-stage logging
     enabled; a short diagnostic script may be required if these keys are not currently exposed in the eval output.
4. Record the audit result in the appropriate internal report or note if needed.
5. Confirm docs/runtime alignment:
   - ppo_cce_v2 canonical
   - ppo_cce_v1 archived / invalid for benchmark claims

Return a concise decision report with:
- exact files changed
- runtime verification result
- per-stage eta audit result
- whether M5 closeout is now complete
- any remaining blockers before M6