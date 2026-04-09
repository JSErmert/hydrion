# Debugging and Superpowers

This document defines the default debugging protocol and “superpowers” Claude must use when working inside HydrOS.

The purpose is to prevent:
- random patching
- guess-based debugging
- architecture-breaking fixes
- wasted iteration cycles

---

# 1. Debugging Philosophy

HydrOS debugging must be:

- structured
- state-aware
- validation-first
- minimally invasive
- architecture-preserving

The goal is not to “make the error disappear.”

The goal is to:
- identify root cause
- preserve modular integrity
- produce a stable correction
- validate the fix properly

---

# 2. Default Debugging Order

Claude must debug in this order:

1. read relevant context docs
2. identify affected modules
3. trace state ownership
4. inspect current implementation
5. inspect logs / outputs / validation
6. propose smallest correct fix
7. test the fix
8. re-check architecture boundaries

---

# 3. Default Superpowers

Claude should use the following by default:

## 3.1 Structural Repo Scan
Before editing:
- identify canonical files
- identify duplicate logic
- confirm module ownership

## 3.2 State Ownership Tracing
Always determine:
- which module owns the behavior
- whether truth_state or sensor_state is involved
- whether reward, obs, or safety depends on the broken field

## 3.3 Pipeline Position Check
Claude must identify:
- where in the pipeline the bug originates
- whether it occurs before or after observation generation
- whether safety is involved

## 3.4 Validation Awareness
Claude must check:
- existing tests
- validation modules
- whether a current validator already covers the issue

## 3.5 Minimal Diff Discipline
Fix the smallest correct layer first.

Do not:
- rewrite unrelated modules
- refactor broadly during debugging
- expand scope while fixing a bug

---

# 4. Required Debugging Questions

Before patching, Claude must answer:

1. what is broken?
2. where does the broken behavior originate?
3. which module owns the fix?
4. what state fields are involved?
5. what downstream systems depend on them?
6. how do we verify the fix?

---

# 5. Preferred Debugging Tools

## Preferred first
- repo search
- code trace
- validation scripts
- logs
- deterministic reproduction
- targeted print / instrumentation
- minimal smoke tests

## Use later
- UI automation
- screenshot-based inspection
- heavy integration testing

---

# 6. Playwright Activation Rule

Playwright should NOT be the default first tool.

## Activate Playwright only when:
- frontend state is wired to real telemetry
- UI interaction is broken
- visual regression matters
- DOM/event behavior is unclear from code inspection
- end-to-end console behavior must be verified

## Do NOT use Playwright first for:
- physics bugs
- reward bugs
- YAML config bugs
- state ownership bugs
- validation failures

Playwright is a **targeted UI verification tool**, not a primary debugging tool.

---

# 7. Logging Expectations

When debugging, Claude should prefer:
- explicit state variable inspection
- narrow logging
- readable intermediate values
- no noisy permanent debug spam

If extra instrumentation is added, Claude should:
- keep it temporary unless genuinely useful
- remove or gate it after validation

---

# 8. Debugging Safety Rules

Do not:
- patch around broken validation
- bypass safety logic to “make it run”
- mutate truth_state from UI-side assumptions
- silently change observation schema
- silently alter reward meaning

---

# 9. Required Output Style During Debugging

Claude should report debugging in this format:

## Issue
What is broken.

## Root Cause
Where it originates.

## Affected Modules
What files / layers are involved.

## Proposed Fix
Smallest correct patch.

## Validation
How the fix will be verified.

---

# 10. Final Directive

Debugging in HydrOS must behave like engineering diagnosis, not trial-and-error improvisation.

Claude’s superpower is not speed alone.

It is:
- structured reasoning
- system awareness
- minimal correct intervention
- validation-backed repair