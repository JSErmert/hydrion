# hydrion/runtime/run_context.py
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Optional


@dataclass(frozen=True)
class RunContext:
    """
    RunContext is the identity + reproducibility anchor for Hydrion runs.

    Philosophy:
    - Everything downstream (logging, noise, evaluation) references run_id.
    - Determinism is controlled by `seed` and `noise_enabled`.
    """
    run_id: str
    version: str
    seed: int
    noise_enabled: bool
    config_hash: str

    @staticmethod
    def derive_run_id(version: str, seed: int, config_hash: str, salt: Optional[str] = None) -> str:
        """
        Deterministic-by-default run_id (stable across reruns with same inputs),
        unless a salt is provided.

        - For research experiments: keep salt=None for stable IDs.
        - For interactive runs: pass salt=str(time.time()) for unique IDs.
        """
        base = f"{version}|{seed}|{config_hash}"
        if salt is not None:
            base += f"|{salt}"
        h = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
        return f"{version.replace('.', '-')}_{h}"

    @staticmethod
    def create(
        *,
        version: str,
        seed: int,
        noise_enabled: bool,
        config_hash: str,
        deterministic_id: bool = True,
    ) -> "RunContext":
        salt = None if deterministic_id else str(time.time())
        run_id = RunContext.derive_run_id(version, seed, config_hash, salt=salt)
        return RunContext(
            run_id=run_id,
            version=version,
            seed=int(seed),
            noise_enabled=bool(noise_enabled),
            config_hash=config_hash,
        )
