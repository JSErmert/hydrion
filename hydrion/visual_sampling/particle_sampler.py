from dataclasses import dataclass
import numpy as np
from typing import List, Tuple


@dataclass
class VisualParticle:
    x: float
    y: float
    vx: float
    vy: float
    captured: bool = False


class ParticleSampler:
    """
    Deterministic visual witness sampler.

    Particles are semantic tokens sampled from aggregate state.
    They DO NOT affect mechanics or reward.
    """

    def __init__(
        self,
        max_particles: int = 500,
        inlet_x_range: Tuple[float, float] = (-0.5, 0.5),
        inlet_y: float = 1.0,
        flow_speed: float = 0.5,
        sink_x: float = 0.8,
    ):
        self.max_particles = max_particles
        self.inlet_x_range = inlet_x_range
        self.inlet_y = inlet_y
        self.flow_speed = flow_speed
        self.sink_x = sink_x

        self._rng = None
        self._particles: List[VisualParticle] = []

    def reset(self, state: dict, seed: int):
        """
        Initialize particles deterministically from aggregate state.
        Capture is sampled ONCE here to preserve aggregate semantics.
        """
        self._rng = np.random.default_rng(seed)
        self._particles.clear()

        C_in = float(state.get("C_in", 0.0))
        capture_prob = float(state.get("particle_capture_eff", 0.0))

        n_particles = int(round(C_in * self.max_particles))
        n_particles = max(0, min(self.max_particles, n_particles))

        for _ in range(n_particles):
            x = self._rng.uniform(*self.inlet_x_range)
            y = self.inlet_y

            captured = self._rng.uniform(0.0, 1.0) < capture_prob

            if captured:
                # Visual routing to sink (no physics implied)
                vx = (self.sink_x - x) * 0.5
                vy = -self.flow_speed * 0.5
            else:
                vx = 0.0
                vy = -self.flow_speed

            self._particles.append(
                VisualParticle(
                    x=x,
                    y=y,
                    vx=vx,
                    vy=vy,
                    captured=captured,
                )
            )

    def step(self, dt: float) -> List[VisualParticle]:
        """
        Deterministically advance particles in time.
        """
        for p in self._particles:
            p.x += p.vx * dt
            p.y += p.vy * dt

        return self._particles
