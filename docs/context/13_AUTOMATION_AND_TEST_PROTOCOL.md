# Automation and Test Protocol

This document defines when and how testing, automation, and verification should be used in HydrOS.

Its purpose is to ensure that:
- changes are testable
- realism upgrades are verified
- regressions are caught early
- UI automation is used appropriately

---

# 1. Testing Philosophy

In HydrOS, “working” means more than “runs without crashing.”

A valid change should be:
- functionally correct
- physically plausible
- numerically stable
- architecturally consistent

---

# 2. Test Categories

HydrOS should use the following test categories.

## 2.1 Smoke Tests
Purpose:
- verify basic execution
- catch immediate breakage

Examples:
- env reset
- env step
- logging init
- wrapper startup

---

## 2.2 Validation Tests
Purpose:
- verify realism and invariants

Examples:
- mass balance
- stress matrix
- envelope sweep
- recovery latency

---

## 2.3 Regression Tests
Purpose:
- ensure previous correct behavior was not broken

Examples:
- observation shape stability
- reward output remains finite
- safety wrapper still enforces limits

---

## 2.4 UI Verification
Purpose:
- verify console behavior once telemetry is live

Examples:
- page loads
- telemetry panels update
- system cutaway reflects live state
- comparison flows behave correctly

---

# 3. When Tests Are Required

## Required after physics changes
Run:
- smoke tests
- relevant validation tests
- regression checks for state and observation

## Required after reward / RL changes
Run:
- smoke tests
- reward consistency checks
- baseline vs PPO comparison when applicable

## Required after UI telemetry changes
Run:
- frontend build / run verification
- targeted interaction testing
- Playwright only if UI behavior needs end-to-end verification

---

# 4. Milestone Testing Rule

Each milestone must define:

- expected behavior
- failure conditions
- required tests
- acceptance criteria

No milestone is complete without this.

---

# 5. Playwright Policy

Playwright is optional until:
- telemetry endpoints exist
- live console binding exists
- UI behavior becomes stateful

Then it becomes useful for:
- frontend interaction regression
- visual state verification
- mission-control workflow checks

---

# 6. Default Automation Order

For any new implementation:

1. code change
2. smoke test
3. targeted validation
4. regression check
5. UI verification if relevant
6. document results

---

# 7. Acceptance Criteria Rule

Every meaningful change must define:

## What should happen
Expected correct behavior

## What must not happen
Failure conditions / regressions

## How it will be verified
Tests, validation, or inspection steps

---

# 8. Output Expectations for Claude

Whenever Claude makes a meaningful change, it should report:

- what changed
- what tests should be run
- what validations matter
- what remains unverified

If a test cannot be run, Claude must say so clearly.

---

# 9. Final Directive

HydrOS does not accept “probably works.”

All important changes must be:
- tested
- validated
- explainable
- reviewable