# HydrOS Console — Mockup L Generation Prompt
**Date:** 2026-04-10
**Status:** LOCKED execution prompt
**Framework:** docs/visualization/M4_L_M_N_O_MOCKUP_PRIORITY_FRAMEWORK.md

---

You are generating **HydrOS Console Mockup L**.

This is NOT a general UI design task.

You are operating under a **locked visual framework**:

- M4_L_M_N_O_MOCKUP_PRIORITY_FRAMEWORK.md
- M2-2.5_HYDROS_MACHINE_VIEW_SPEC.md
- M2-2.5_UI_SCHEMA_SPEC.md
- M2-2.5_HYDROS_VISUAL_STATE_MAPPING_SPEC.md

You must follow these strictly.

---

# Objective

Generate **Mockup L**:

> The η_nominal research framing layer

This mockup establishes how efficiency is understood in HydrOS.

It is NOT about layout exploration.
It is NOT about aesthetics.

It is about:

> **making efficiency physically interpretable and research-valid**

---

# Base Requirement

You MUST build on:

> **Mockup K (cinematic + correct topology)**

DO NOT:
- redesign the machine layout
- change stage geometry
- alter flow direction
- introduce new layout paradigms

Mockup L is:

> **K + η instrumentation layer**

---

# Primary Task

Implement the η display as a **research instrument**.

---

# Required η Block (LOCKED)

You MUST render:

```
η_ref   85.4%  @ 10µm / 13.5 L/min
η_live  71.2%
Δη     -14.2 pts   cause: FLOW
```

## Rules for η Block

### 1. Placement (LOCKED)

Dedicated panel section.

NOT top telemetry band.
NOT overlaid on machine.
NOT hidden.

Recommended: right-side advisory panel (upper-middle region)

### 2. Visual Hierarchy (MANDATORY)

**η_live**
- largest
- brightest
- primary reading

**η_ref**
- smaller
- muted tone
- clearly labeled as reference

**Δη**
- color-coded:
  - neutral → small delta (0–5 pts)
  - amber → moderate degradation (5–15 pts)
  - red → significant degradation (>15 pts)

### 3. Qualification (MANDATORY)

This must ALWAYS be visible:

```
@ 10µm / 13.5 L/min
```

Do NOT:
- hide it
- abbreviate it
- move it to tooltip

### 4. Attribution (LOCKED)

Allowed values ONLY:

- FOULING
- FLOW
- VOLTAGE

No variation. Attribution is the dominant single cause.
When multiple causes active: priority order is FOULING > FLOW > VOLTAGE.

---

# Visual Integration Rules

## Machine (unchanged)
- preserve K exactly
- do NOT redesign geometry

## Outside machine
- clean
- minimal
- no particles

## Inside machine
- unchanged from K
- no new particle systems unless already grounded

---

# What L Must Solve

Mockup L must answer:

> "How do I understand system efficiency instantly?"

User should be able to see:

- what the system should achieve (η_ref)
- what it is achieving now (η_live)
- how far it has degraded (Δη)
- why it degraded (cause)

---

# What L Must NOT Do

Do NOT:

- equalize metrics visually
- treat η as a generic card
- introduce dashboard grid layout
- add new telemetry
- redesign playback controls
- introduce M/N features (stage gradients, density split, E-field redesign)

---

# Style Direction

- cinematic (as in K)
- high contrast
- controlled glow
- precision over decoration
- premium instrument feel

---

# Deliverable

Provide:

**1. Visual mockup**

Must clearly show:
- machine (unchanged K)
- η block (new)
- integration with panel system

**2. Short explanation**

Explain:
- where η block is placed
- how hierarchy is enforced
- how it supports research usage

---

# Success Condition

Mockup L is successful if:

- η_ref / η_live / Δη are immediately readable
- qualification is always visible
- hierarchy is clear without explanation
- machine remains dominant
- UI does not feel like a dashboard

---

# Final Instruction

This is a precision design task.

Do not improvise.

Do not expand scope.

Generate Mockup L now.
