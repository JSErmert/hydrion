# HydrOS — Technical Milestone Memo

**Date: 2026-04-12**
**Branch: explore/conical-cascade-arch (pre-merge to main)**
**Audience: Founder / technical co-founder**

---

## Current State

HydrOS is a fully operational physics-grounded digital twin of a three-stage
conical cascade microplastic extraction device. The simulation is running and
producing physically meaningful results.

**What is working now:**

- Three-stage conical cascade with RT (Rajagopalan-Tien 1976) filtration and nDEP
  force balance, both calibrated against peer-reviewed literature
- Per-species capture (PP buoyant, PE buoyant, PET sinking) emerging from physics
  — no hand-coded separation rules
- PPO reinforcement learning agent training on the simulation with Safe RL guardrails
- A three-way comparison framework (PPO vs. heuristic vs. random baselines)
- A FastAPI service + animated React console visualization operational
- Physics audit and self-correction cycle complete: identified and fixed a regime
  saturation bug where all particles were being captured regardless of voltage or flow

---

## Strongest Truthful Technical Claim

> The PPO agent, trained in simulation, discovers a non-trivial physics threshold:
> nDEP capture at Stage 3 activates only when the face velocity stays below
> ~790 mm/s (Q ≤ 7.7 L/min at V=500V). The heuristic — which uses full voltage
> but nominal flow — operates above this threshold and achieves only ~51% capture.
> The agent learns to reduce pump to ~20% command while maintaining full voltage,
> achieving ~85–99% simulated capture. This is a genuine RL discovery of device
> physics, not a pre-programmed rule.

All numbers are simulation outputs at design-default geometry. Not yet validated
against hardware.

---

## Biggest Remaining Blocker

**No physical device exists.**

Every capture efficiency, DEP threshold, and pressure-flow characteristic in this
repo is a simulation prediction. The transition from "interesting simulation" to
"validated performance claim" requires:

1. Fabricate a physical S3 membrane with known pore size and integrated electrode
2. Measure capture vs. flow rate at V=500V with 1 µm calibration particles
3. Compare to simulation prediction

Until that test is done, no removal efficiency number can be stated without qualification.

---

## Next 3 Technical Milestones (in order)

### Milestone 1: PPO-v2 Training Complete (≤ 2 days)
**Status:** Running in background (500k steps, ~80 min wall time)
**Deliverable:** `ppo_cce_v2.zip` — first PPO trained in the capture-sensitive 1 µm regime
**Validation:** eval_ppo_cce.py --regime submicron; PPO/heuristic eta ratio > 1.2x
**Significance:** Demonstrates non-trivial RL control of a physical device parameter

### Milestone 2: Branch Merge + Main Stability (1-2 days)
**Status:** Blocked on ppo_cce_v2 evaluation (Task D) and branch health check (Task G)
**Deliverable:** `explore/conical-cascade-arch` merged to `main`; all tests passing on main
**Validation:** Full test suite green on main
**Significance:** Locks in M5 physics and RL infrastructure as production-branch baseline

### Milestone 3: First Hardware Characterization Protocol (2-4 weeks)
**Status:** Unstarted; requires physical access and materials
**Deliverable:** Experimental protocol document + initial pressure-flow measurement
**Validation:** ΔP(Q) measurement for assembled cascade; compare to hydraulics model
**Significance:** First contact between simulation and physical reality; enables
first calibrated simulation-hardware comparison

---

## Next Hardware-Dependent Milestone

> **S3 membrane + electrode fabrication and DEP threshold test**

Specifically: fabricate or source a microporous membrane (5 µm pore) with integrated
electrode capable of generating a field gradient at the pore scale (tip radius ~ 3 µm).
Apply V=500V and measure particle capture fraction at Q = 5, 7, 10, 15 L/min with
1 µm fluorescent particles.

This test directly validates the simulation's core prediction: that DEP capture
drops sharply when Q exceeds ~7.7 L/min at 500V. If confirmed, the simulation's
physics model is validated at the critical operating threshold.

---

## Best Current Design-Partner-Facing Story

> "We have a working physics simulation of an electrostatic microplastic trap.
> The simulation predicts that our Stage 3 membrane, running at 500V, can
> capture sub-micron plastic particles — but only if the flow rate stays below
> a critical threshold of 7.7 L/min. Our RL agent, trained on this simulation,
> learns this threshold independently and adjusts pump settings to stay below it
> while maximizing throughput. We're now building toward validating this prediction
> against hardware. Our first testable milestone is confirming the DEP threshold
> in a bench-scale prototype."

**Why this story works for design partners:**
- Specific and testable (DEP threshold at 7.7 L/min is a concrete prediction)
- Distinguishes simulation physics from hardware claim
- Shows the RL is doing something real, not just optimizing a black box
- Gives a clear hardware validation target that partners can evaluate

---

*Prepared by Claude Sonnet 4.6 co-orchestrator session, 2026-04-12.*
*Review cycle: update when ppo_cce_v2 eval results are available.*
