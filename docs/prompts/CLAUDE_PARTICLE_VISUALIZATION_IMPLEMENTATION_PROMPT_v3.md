# CLAUDE_PARTICLE_VISUALIZATION_IMPLEMENTATION_PROMPT_v3

**Status: LOCKED — execution spec**

---

## Physics ↔ Visualization Contract

1. **No orphan physics** — every backend variable is visible or drives something visible.
2. **No fake visualization** — every visible behavior comes from backend state or an explicit physical transition.
3. **Controlled transitions only** — stage entry, mesh sliding, capture, channel drop, storage transfer, backflush movement.
4. **No premature physics** — extend backend first, then map, then render.
5. **Strict pipeline** — `ParticleDynamicsEngine → API → Mapper → Renderer`.

---

## Cascade Sequencing

- 30-step cycle: S1 window steps 0–9, S2 window steps 10–19, S3 window steps 20–29.
- Stage stream is EMPTY outside its window.
- Deposited channel dots shown independently at all times.

---

## Particle Shapes

- PP, PE → circles (pellets/fragments).
- PET → thin curved line segment (fiber), oriented by trail direction.
- Size from `d_p_um`: 10µm small, 25µm medium, 50µm large.

---

## Channel Accumulation

- `_channel_deposits[i]` tracks captures per stage across cycles.
- Deposits added at cycle boundary (step % 30 == 0).
- Max 30 deposits per stage.
- Rendered as small dots in the collection channel band.

---

## Backflush Channel Flush

- During backflush: deposits drain from channel → `_storage_particles`.
- `ch-forward-flush` L→R animation fires per stage.
- Storage dots accumulate (max 50, rendered in chamber).

---

## Rendering Rules

- No CSS keyframe animation on particles.
- No browser physics.
- All positions from `(x_norm, r_norm)` backend output.
- Particle identity maintained across frames via `phase_frac` cycling.

---

## Success Condition

- Single particle traceable: entry → trajectory → capture → channel deposit → storage.
- Stage transitions are temporally sequential.
- Behavior changes with flow, field, and fouling are visible.
