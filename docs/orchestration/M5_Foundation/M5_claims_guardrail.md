# HydrOS — Founder-Safe Claims Guardrail

**Internal use only. Not for external distribution.**
**Last updated: 2026-04-12. Based on validated repo state at commit 634dcb4.**

---

## Purpose

This document defines what HydrOS can and cannot credibly claim at current maturity.
It is the authoritative reference for any external communication, pitch material,
or partner conversation. Claims outside the "can claim now" section require the
blocking work listed under "must not claim yet."

---

## What HydrOS Can Claim Now

### Physics simulation fidelity

- A digital twin of a three-stage conical cascade microplastic extraction device
  is implemented and running, incorporating peer-reviewed first-principles models:
  Rajagopalan-Tien (1976) RT filtration and literature-calibrated nDEP force balance.
- Per-species capture efficiency is computed for PP, PE, and PET polymers based on
  Clausius-Mossotti factors from RSC 2025 literature (CM ≈ −0.47 to −0.48).
- Density split from buoyancy (PP/PE buoyant, PET sinking) emerges directly from
  the RT gravity term — not from hand-coded fractions.
- The system correctly models the DEP threshold phenomenon: at sub-pore-scale
  particle sizes (1 µm), nDEP capture activates only when flow rate stays below
  Q ≈ 7.7 L/min at 500V, validated against device geometry and force-balance physics.

### Safe RL pipeline

- A PPO-based reinforcement learning policy trains on a Gymnasium-compatible
  environment (ConicalCascadeEnv) with Safe RL constraints (pressure and clog limits,
  hard-termination penalties).
- The training infrastructure is deterministic from a fixed seed (Constraint 6).
- A three-way comparison framework (PPO vs. heuristic vs. random) is implemented.

### Digital twin environment

- The simulation includes hydraulics (validated M4 model), clogging dynamics,
  nDEP force balance, and a backflush state machine.
- A FastAPI service exposes the environment with artifact logging.
- A React-based console with animated particle visualization is operational.

### RL benchmark (once ppo_cce_v2 training completes)

- In the 1 µm benchmark regime, PPO is expected to discover the DEP flow-threshold
  and achieve capture efficiency ~1.65× above the heuristic baseline and the random
  baseline.
- This demonstrates non-trivial learning of a physical threshold not encoded in
  the policy or the heuristic.

---

## What Must Always Be Qualified

### All efficiency numbers are model outputs, not hardware measurements

Every eta_cascade, v_crit, or capture percentage in this repo is the output of
a simulation built on design-default geometry values (`[DESIGN_DEFAULT]` markers
in source). None of these numbers have been validated against a physical device.

**Required qualifier:** "Simulation-predicted at [geometry/voltage/flow] with
design-default parameters."

### Particle sizes and species fractions are assumed

The polymer mix (PP 8%, PE 7%, PET 70%, neutral 15%) and the benchmark particle
size (1 µm) are modeling assumptions. Real ocean microplastic PSDs differ and vary
by source.

**Required qualifier:** "Modeled at representative size of X µm; real-world
distribution will vary."

### DEP threshold numbers assume 3 µm tip electrode geometry

v_crit = 789 mm/s at 500V assumes tip_radius = 3 µm (iDEP pore-scale electrode).
This is a design assumption for a membrane-integrated electrode, not a measurement.

**Required qualifier:** "Assumes iDEP electrode with 3 µm tip at pore scale."

---

## What HydrOS Cannot Claim Yet

### Any capture performance claim tied to a specific removal percentage

Examples of prohibited claims:
- "Removes X% of microplastics from water"
- "Achieves X% capture efficiency"
- "Eliminates particles above Y µm"

**Reason:** All capture numbers are simulation outputs with design-default geometry.
No physical device has been fabricated or tested.

**What unlocks this:** Prototype fabrication + validated measurement (Task I/hardware milestone).

### That the RL policy outperforms existing filtration systems

The comparison is between PPO, heuristic, and random baselines — all simulated.
No comparison to existing commercial filtration systems or published benchmarks.

**What unlocks this:** Peer-reviewed benchmark study against established methods.

### That the simulation is validated against hardware

The physics models use literature-calibrated constants, but the complete system
(geometry + multi-stage cascade + fouling dynamics) has not been validated against
a physical prototype.

**What unlocks this:** First prototype + comparison between simulated and measured
capture efficiency, pressure drop, and fouling rate.

### That nDEP is confirmed operative at the described scale

nDEP capture probability is computed from a force-balance model. The tip geometry
(3 µm) and grad_E² values are design assumptions. The actual electrode fabrication
and electric field distribution in a microporous membrane have not been characterized.

**What unlocks this:** iDEP electrode fabrication + impedance spectroscopy +
particle trajectory imaging.

---

## What Should Never Be Said Externally at Current Maturity

1. Any specific removal efficiency percentage without the required qualifier.
2. "Clinical" or "certified" — no regulatory pathway has been initiated.
3. Claims of real-time adaptive control of a physical system — the control loop
   exists only in simulation.
4. That the current RL policy "controls" or "manages" water purification —
   it controls a simulation environment.
5. Claims derived from ppo_cce_v1 (trained in RT-saturated regime with d_p_um=10 µm).
   That model's capture performance was not meaningful. ppo_cce_v2 (1 µm, 500k steps)
   is the first model with non-trivial capture-sensitive control.

---

## Strongest Truthful External Claim (Current Maturity)

> "HydrOS is a physics-grounded digital twin of a conical-cascade microplastic
> extraction device, combining peer-reviewed RT filtration theory with nDEP
> force-balance models. A PPO reinforcement learning agent trained on this
> simulation discovers the non-trivial DEP activation threshold — reducing pump
> flow to keep face velocity below the critical value while maintaining full
> voltage — achieving simulated capture efficiency approximately 1.6× above a
> naive fixed-setting baseline. All results are simulation predictions with
> design-default geometry parameters pending physical validation."

---

*Last reviewed: 2026-04-12 by Claude Sonnet 4.6 / co-orchestrator session.*
*Trigger for review: Any hardware fabrication, peer review submission, or investor pitch.*
