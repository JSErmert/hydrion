HYDRION DEVELOPMENT RULES

Hydrion is a research-grade digital twin.

Architecture constraints:

1. Physics modules are authoritative (hydrion/physics/*)
2. Sensors may not modify truth state.
3. Safety shield (hydrion/safety/*) must never be bypassed.
4. Observation vector ordering must remain stable.
5. Config-driven parameters must not be hard-coded.
6. Mass balance integrity must be preserved.
7. Electrostatic stage must remain independently controllable.
8. Backflush must remain RL-actuated.
9. Do not flatten module boundaries.
10. Any refactor must preserve multi-physics separation.

Cursor should treat changes like pull requests.
