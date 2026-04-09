# Security Protocol

This document defines the **baseline security doctrine** for HydrOS development.

HydrOS is not yet a public production platform, but security discipline must still be present from the beginning.

The goal of this document is to prevent:
- accidental credential exposure
- unsafe local development patterns
- insecure API assumptions
- uncontrolled dependency risk
- premature cloud deployment mistakes

---

# 1. Security Philosophy

HydrOS security must be:

- minimal
- deliberate
- layered
- proportional to maturity
- compatible with research development

This means:

- do not over-engineer enterprise security too early
- do not ignore basic security hygiene
- do not build public-facing assumptions by default

---

# 2. Current Security Posture

HydrOS is currently a:

- local-first
- research-oriented
- branch-driven
- non-public-facing engineering system

Therefore the current security focus is:

1. repository hygiene
2. credential safety
3. dependency discipline
4. safe API design
5. controlled environment handling

---

# 3. Secrets Handling

## Rules

- never hardcode secrets
- never commit API keys
- never commit tokens
- never commit credentials
- never place secrets in markdown documentation

## Required Practice

Use environment variables or local secret files excluded from Git.

Examples:
- `.env`
- `.env.local`
- local config override files ignored by Git

## Git Rule

Secrets must never enter:
- commits
- PRs
- screenshots
- logs
- example configs

---

# 4. Environment Configuration Safety

## Allowed
- default example configs
- non-sensitive placeholders
- documented variable names

## Forbidden
- real private tokens in config files
- Firebase credentials committed by default
- production connection strings
- hidden auth assumptions

---

# 5. Repository Security Hygiene

## Required
- `.gitignore` must exclude:
  - env files
  - runtime artifacts
  - checkpoints
  - outputs
  - local caches
  - build artifacts
  - temporary logs

## Required practice
- review `git status` before commit
- verify no sensitive files are staged
- use clean branch discipline

---

# 6. Dependency Security

HydrOS must treat dependencies cautiously.

## Rules
- do not add a dependency without clear justification
- prefer mature, widely used libraries
- avoid unnecessary packages
- document why a dependency exists

## For Claude
Before adding a dependency, Claude must explain:
- why it is needed
- what it replaces or avoids
- what security or maintenance risk it introduces

---

# 7. Backend / API Safety

Future telemetry and control APIs must be designed safely even in local-first mode.

## Rules
- API must expose read-only telemetry by default
- UI must not be able to mutate internal truth_state directly
- run-control endpoints must be explicit and limited
- no arbitrary code execution endpoints
- no insecure file-system manipulation endpoints

## Allowed endpoint categories
- run creation
- run reset
- run step / execute
- read telemetry
- read validation status
- read history

---

# 8. Frontend Safety

The frontend must assume:
- backend is authoritative
- no direct simulation mutation
- no secret storage in browser code
- no embedded private credentials

If cloud services are introduced later, their security model must be documented separately.

---

# 9. Firebase / Cloud Rule

Firebase or other cloud services may be considered later, but are NOT default architecture.

## Rule
Do not add Firebase by default.

Only add it if:
- explicit use case is defined
- security assumptions are written
- data model is justified
- product-surface strategy says it is necessary

---

# 10. Logging Safety

Logs must not expose:
- credentials
- tokens
- secret URLs
- private system identifiers not needed for engineering

Logs should focus on:
- simulation state
- validation outcomes
- run metadata
- telemetry diagnostics

---

# 11. Deployment Safety

HydrOS must not assume public deployment by default.

## Current doctrine
- local development first
- no implicit exposure to public internet
- no production deployment assumptions unless explicitly documented

---

# 12. Security Escalation Rule

If HydrOS evolves toward:
- cloud API access
- shared multi-user access
- persistent user data
- external integrations
- mobile deployment

then a more advanced security architecture document must be added.

---

# 13. Final Directive

Security in HydrOS must be:

- practical
- disciplined
- proportional
- enforced through habit

HydrOS is not yet a public platform,
but it must never be built with careless security assumptions.