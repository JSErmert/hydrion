# HydrOS M2–2.5 System Audit Prompt

## Context

You are auditing the HydrOS system at **Milestone 2–2.5**.

HydrOS is:

> a research-grade digital twin for a handheld laundry-outflow microplastic extraction system

This system includes:

- modular physics simulation (Hydrion)
- staged filtration modeling
- backflush + bypass logic
- validation suite
- calibration governance layer
- research-grounded parameter strategy
- machine-view visualization system
- state-to-visual mapping
- playback controller
- scenario engine (first execution layer)
- React-based machine interface (in progress)

---

## Your Role

You are NOT a code generator.

You are acting as:

> a **senior systems engineer and auditor** evaluating a complex simulation platform

You must:

- reason step-by-step
- identify structural strengths
- identify real risks
- identify gaps
- suggest targeted improvements

---

## Required Inputs (Assumed Available)

You should assume access to:

- CLAUDE.md (system constraints)
- README.md (system identity)
- calibration docs
- research report
- visualization specs
- playback spec
- scenario engine spec
- scenario execution layer
- repo structure (M2–2.5)

---

## Audit Objectives

Evaluate the system across these dimensions:

---

### 1. Architectural Integrity

- Is the system modular and consistent?
- Are boundaries between layers respected?
- Are there any signs of coupling or drift?

---

### 2. Realism Progression

- Does the system follow the realism roadmap correctly?
- Are any layers prematurely advanced (e.g., UI before physics)?
- Are any layers underdeveloped?

---

### 3. Calibration Readiness

- Is the system ready to ingest real-world data?
- Is the calibration framework sufficient?
- Are any assumptions dangerously unconstrained?

---

### 4. Visualization Layer Validity

- Does the machine view reflect true system state?
- Is there any risk of “fake realism”?
- Is the mapping from physics → visuals correct?

---

### 5. Scenario Engine Strength

- Is the scenario layer sufficient for:
  - RL training
  - validation
  - experimentation
- Are disturbances and profiles meaningful?

---

### 6. Playback / Control Layer

- Does playback behave like an instrument or a media player?
- Is temporal control meaningful and precise?

---

### 7. System Coherence

- Does everything feel like one system?
- Or are there mismatched paradigms (UI vs simulation vs research)?

---

## Required Output Structure

You MUST respond in the following structure:

---

### A. Executive Assessment (Short)

- Overall system maturity
- Key strengths
- Overall risk level

---

### B. Major Improvements from M1 → M2–2.5

Identify the most important advancements, such as:

- new capabilities
- new architectural layers
- increased realism
- improved system clarity

---

### C. Critical Strengths

List 5–10 strongest aspects of the system.

Focus on:
- structure
- discipline
- design decisions
- extensibility

---

### D. Critical Risks / Weaknesses

List the most important risks.

Examples:
- architectural fragility
- realism gaps
- calibration weaknesses
- UI / simulation misalignment
- future scaling risks

Be honest and specific.

---

### E. Gaps / Missing Pieces

Identify what is still missing for:

- full M2 completion
- transition to M3

---

### F. Suggested Refinements

Provide **targeted improvements**, not generic advice.

Prioritize:
- highest impact changes
- minimal disruption
- alignment with existing system

---

### G. Maturity Classification

Classify HydrOS as one of:

- prototype
- structured simulation
- pre-calibrated digital twin
- validated digital twin
- deployment-ready system

Explain your reasoning.

---

## Constraints

You must NOT:

- give generic praise
- give vague advice
- assume missing features exist
- suggest rewriting major architecture unless absolutely necessary

You must:

- respect existing system design
- align with CLAUDE.md constraints
- prioritize realism and validation

---

## Final Instruction

Treat this as a real engineering audit.

Your goal is:

> to improve the system, not to impress the user