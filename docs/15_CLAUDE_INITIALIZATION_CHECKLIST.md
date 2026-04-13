# Claude Initialization Checklist

This checklist must be completed before handing HydrOS to Claude Code for deep analysis or implementation work.

Its purpose is to ensure that:
- the repository is structurally clean
- core doctrine is documented
- context is sufficient
- branch state is correct
- Claude will not waste time inferring missing structure

---

# 1. Repository State

## Branch
- [ ] Working branch is correct
- [ ] Branch is intended for Claude-guided development
- [ ] Branch is up to date with remote

## Clean Working Tree
- [ ] `git status` is clean
- [ ] no accidental untracked runtime artifacts
- [ ] no local backup folders inside active repo
- [ ] no secrets or env files staged

## Frozen Baseline
- [ ] frozen snapshot exists outside active repo
- [ ] frozen snapshot is not tracked by Git
- [ ] active repo is clearly separated from backup

---

# 2. Core Entry Files

- [ ] `README.md` exists and is updated
- [ ] `CLAUDE.md` exists and is updated
- [ ] `docs/MASTER_ARCHITECTURE.md` exists and is updated

These three files must work together as:
- README = system identity
- MASTER_ARCHITECTURE = system structure
- CLAUDE.md = execution contract

---

# 3. Context Package

Confirm all context docs exist under:

```text
docs/context/

Required files:

 01_SYSTEM_IDENTITY.md
 02_ARCHITECTURE_CONSTRAINTS.md
 03_REPO_MAP.md
 04_CURRENT_ENGINE_STATUS.md
 05_REALISM_TARGET.md
 06_LOCKED_SYSTEM_CONSTRAINTS.md
 07_TELEMETRY_AND_CONSOLE.md
 08_VALIDATION_AND_SAFETY.md
 09_REALISM_ROADMAP.md
 10_COUNCIL_OF_HYDROS.md
 11_SECURITY_PROTOCOL.md
 12_DEBUGGING_AND_SUPERPOWERS.md
 13_AUTOMATION_AND_TEST_PROTOCOL.md
 14_PRODUCT_SURFACES_STRATEGY.md
4. Architectural Readiness
 truth_state vs sensor_state separation is explicitly documented
 immutable pipeline ordering is documented
 current observation schema is documented
 realism roadmap is documented
 current engine strengths and limitations are documented
 validation doctrine is documented
 telemetry doctrine is documented
5. Milestone Readiness

Before Claude starts implementation work:

 current milestone is explicitly defined
 milestone scope is constrained
 milestone order is documented
 milestone acceptance criteria exist
 out-of-scope features are explicitly excluded

For current repo state, Claude should begin with:

Milestone 1 — Hydraulic + Fouling + Backflush Realism Backbone

6. Security & Hygiene
 .gitignore is correct
 no tracked runtime outputs remain
 no tracked checkpoints remain unless intentional
 no secrets are present in markdown, configs, logs, or commits
 dependency additions are controlled
7. Debugging & Testing Readiness
 debugging protocol is documented
 superpowers / default debugging order is documented
 automation / test protocol is documented
 validation expectations are clear
 Playwright is defined as conditional, not default
8. Product Surface Discipline
 canonical first surface is defined
 web console is designated primary surface
 mobile app is explicitly deferred
 backend / telemetry is prioritized over multi-surface expansion
9. Asset Readiness

If an image is provided to Claude:

 image is the correct HydrOS system image
 image is described as a system-intent anchor
 image is not treated as rigid implementation law

Do NOT provide unrelated images.

10. Claude Launch Prompt Readiness

Before launch:

 Claude audit prompt is prepared
 prompt instructs Claude to read README, MASTER_ARCHITECTURE, CLAUDE.md, and all context docs first
 prompt defines milestone scope clearly
 prompt tells Claude to audit before coding
11. Final Readiness Test

HydrOS is ready for Claude Code only if the answer to all of the following is YES:

 Can Claude understand what the system is immediately?
 Can Claude understand what the system is NOT?
 Can Claude identify current strengths without guessing?
 Can Claude identify current gaps without hallucinating features?
 Can Claude see the exact next milestone?
 Can Claude act without breaking architecture?
 Can Claude validate changes instead of improvising them?

If any answer is NO, fix documentation first.

12. Final Directive

Do not initialize Claude into ambiguity.

Initialize Claude only when:

structure is clear
doctrine is written
milestone is constrained
repo is clean
context is sufficient

HydrOS must enter Claude Code as a governed engineering system, not as an unfinished thought.