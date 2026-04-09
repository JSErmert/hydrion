# Product Surfaces Strategy

This document defines which product surfaces HydrOS should build, in what order, and why.

Its purpose is to prevent:
- surface sprawl
- duplicated UI effort
- unstable API assumptions
- premature mobile development

---

# 1. Product Surface Philosophy

HydrOS should not build every surface at once.

The system must first stabilize:

- backend truth
- telemetry contract
- validation outputs
- mission-control workflows

Only then should additional product surfaces be expanded.

---

# 2. Canonical Surface

The **canonical first surface** for HydrOS is:

> the web-based research console

Location:
- `apps/hydros-console/`

This surface should become the primary interface for:
- observability
- run inspection
- comparison
- validation status
- telemetry review

---

# 3. Why Web First

Web console is the right first surface because it:

- matches Phase 1.5 goals
- supports high-density visualization
- is easiest to iterate with backend telemetry
- is best for engineering workflows
- avoids early mobile feature duplication

---

# 4. Mobile Strategy

A mobile app (e.g. Flutter) is NOT the first implementation priority.

Mobile should be deferred until all of the following are stable:

- telemetry contract
- run history model
- console state model
- validation indicators
- API design
- security expectations

---

# 5. Backend Strategy

The canonical backend direction should remain:

- Python simulation engine
- Python API layer (likely FastAPI)
- structured telemetry interface

Vite is frontend tooling, not backend architecture.

---

# 6. Surface Order

HydrOS should be built in this order:

## Surface 1
Web research console

## Surface 2
Expanded engineering UI / telemetry tooling

## Surface 3
Optional mobile companion or operator-facing app

Only after Surface 1 is mature.

---

# 7. Firebase Rule

Firebase is not default architecture.

It may be considered later only if:
- a clear product need exists
- authentication or sync requirements are explicit
- data model is defined
- security implications are documented

---

# 8. Parallel Surface Rule

Do NOT build:
- full web console
- full Flutter app

in parallel during current realism phases.

Reason:
- duplicated effort
- premature API lock-in
- split debugging surface
- unstable feature parity

---

# 9. Current Recommendation

Current recommendation is:

1. build backend truth + telemetry
2. build canonical web research console
3. stabilize validation and comparison workflows
4. then consider mobile only if justified

---

# 10. Final Directive

HydrOS must prioritize:

> one canonical, high-value engineering surface first

That surface is the web research console.

Everything else follows after contracts and truth stabilize.