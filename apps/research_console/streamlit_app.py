"""apps/research_console/streamlit_app.py

Hydrion Research Console (v0)

A thin, research-grade instrument panel over HydrionEnv.

Run:
  python -m streamlit run apps/research_console/streamlit_app.py

Setup (from repo root):
  python -m pip install -e .
  python -m pip install streamlit
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st
import yaml

# Hydrion imports (requires: pip install -e .)
from hydrion.env import HydrionEnv
from hydrion.rendering import Observatory


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
OUTPUT_DIR = REPO_ROOT / "outputs" / "research_console"


# -------------------------
# YAML + formatting helpers
# -------------------------

def list_yaml_configs() -> List[Path]:
    if not CONFIG_DIR.exists():
        return []
    return sorted(CONFIG_DIR.rglob("*.yaml"))


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def yaml_text(obj: Dict[str, Any]) -> str:
    return yaml.safe_dump(obj, sort_keys=False, default_flow_style=False)


def unified_diff(a_text: str, b_text: str, a_name: str, b_name: str) -> str:
    diff = difflib.unified_diff(
        a_text.splitlines(True),
        b_text.splitlines(True),
        fromfile=a_name,
        tofile=b_name,
    )
    return "".join(diff)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def format_kv(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return {k: d.get(k) for k in keys if k in d}


# -------------------------
# Export helpers
# -------------------------

def summarize_truth(truth: Dict[str, Any]) -> Dict[str, Any]:
    core_keys = [
        "flow",
        "pressure",
        "clog",
        "Q_out_Lmin",
        "P_in",
        "P_m1",
        "P_m2",
        "P_m3",
        "P_out",
        "E_norm",
        "C_in",
        "C_out",
        "particle_capture_eff",
        "capture_eff",
        "C_fibers",
        "fiber_fraction",
    ]
    core = format_kv(truth, core_keys)

    bin_keys = sorted([k for k in truth.keys() if k.startswith("C_in_bin_") or k.startswith("C_out_bin_")])
    if bin_keys:
        core["psd_bins"] = {k: safe_float(truth.get(k)) for k in bin_keys}

    return {"core": core, "raw_keys": sorted(truth.keys())}


def random_action(seed: int, step: int) -> np.ndarray:
    """Deterministic pseudo-random action in [0,1]^4 for reproducible baselines."""
    rng = np.random.default_rng(seed + step * 9973)
    return rng.random(4, dtype=np.float32)


def export_history(sess: "RunSession") -> Dict[str, Any]:
    h = sess.observatory.history
    assert h is not None
    return {
        "run_id": sess.run_id,
        "config_path": sess.config_path,
        "seed": sess.seed,
        "steps": h.steps,
        "timesteps": h.timesteps,
        "truth_states": h.truth_states,
        "sensor_states": h.sensor_states,
        "actions": [a.tolist() for a in h.actions],
        "rewards": h.rewards,
        "infos": h.infos,
        "terminated": h.terminated,
        "truncated": h.truncated,
    }


def history_csv(sess: "RunSession") -> str:
    """Flat CSV for quick plotting in pandas/Excel (no pandas dependency)."""
    h = sess.observatory.history
    assert h is not None

    rows: List[Dict[str, Any]] = []
    for i in range(len(h.steps)):
        t = h.timesteps[i] if i < len(h.timesteps) else i
        truth = h.truth_states[i]
        sensor = h.sensor_states[i]
        action = h.actions[i] if i < len(h.actions) else np.zeros(4)
        rows.append(
            {
                "step": h.steps[i],
                "time": t,
                "reward": h.rewards[i] if i < len(h.rewards) else 0.0,
                "valve": float(action[0]),
                "pump": float(action[1]),
                "backflush": float(action[2]),
                "node_voltage": float(action[3]),
                "flow": safe_float(truth.get("flow")),
                "pressure": safe_float(truth.get("pressure")),
                "clog": safe_float(truth.get("clog")),
                "E_norm": safe_float(truth.get("E_norm")),
                "C_out": safe_float(truth.get("C_out")),
                "particle_capture_eff": safe_float(truth.get("particle_capture_eff")),
                "sensor_turbidity": safe_float(sensor.get("sensor_turbidity")),
                "sensor_scatter": safe_float(sensor.get("sensor_scatter")),
            }
        )

    if not rows:
        return ""

    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(str(r.get(k, "")) for k in keys))
    return "\n".join(lines)


# -------------------------
# Run session
# -------------------------

@dataclass
class RunSession:
    run_id: str
    config_path: str
    seed: int
    env: HydrionEnv
    observatory: Observatory
    step: int = 0
    terminated: bool = False
    truncated: bool = False
    last_obs: Optional[np.ndarray] = None
    last_reward: float = 0.0
    last_info: Dict[str, Any] = None


def new_run_id() -> str:
    return f"run_{np.random.randint(0, 10**9):09d}"


def create_session(config_path: str, seed: int, time_axis: str = "time") -> RunSession:
    env = HydrionEnv(config_path=config_path)
    observatory = Observatory(save_dir=OUTPUT_DIR, time_axis=time_axis)
    observatory.reset()
    obs, info = env.reset(seed=seed)

    return RunSession(
        run_id=new_run_id(),
        config_path=config_path,
        seed=seed,
        env=env,
        observatory=observatory,
        step=0,
        terminated=False,
        truncated=False,
        last_obs=obs,
        last_reward=0.0,
        last_info=info or {},
    )


def get_session() -> Optional[RunSession]:
    return st.session_state.get("run_session")


def set_session(sess: Optional[RunSession]) -> None:
    st.session_state["run_session"] = sess


def step_env(sess: RunSession, action: np.ndarray) -> None:
    if sess.terminated or sess.truncated:
        return

    obs, reward, term, trunc, info = sess.env.step(action)
    sess.last_obs = obs
    sess.last_reward = float(reward)
    sess.last_info = info or {}
    sess.terminated = bool(term)
    sess.truncated = bool(trunc)

    # Record AFTER the step (recorded truth/sensor state is the resulting state)
    sess.observatory.record_step(
        step=sess.step,
        truth_state=sess.env.truth_state,
        sensor_state=sess.env.sensor_state,
        action=action,
        reward=float(reward),
        info=info or {},
        observation=obs,
        dt=getattr(sess.env, "dt", None),
    )

    sess.step += 1

    if sess.terminated or sess.truncated:
        sess.observatory.finalize_episode(terminated=sess.terminated, truncated=sess.truncated)


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Hydrion Research Console", layout="wide")

st.title("Hydrion Research Console")
st.caption("Config → Run → Observe → Actuate → Log → Compare")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Overview", "System", "Scenarios", "Run Console", "Benchmarks (stub)", "Docs"],
        index=0,
    )

    st.divider()
    st.header("Runtime")

    configs = list_yaml_configs()
    if not configs:
        st.warning(f"No YAML configs found under {CONFIG_DIR}")
        config_path = str(CONFIG_DIR / "default.yaml")
    else:
        rels = [str(p.relative_to(REPO_ROOT)) for p in configs]
        default_idx = rels.index("configs/default.yaml") if "configs/default.yaml" in rels else 0
        chosen_rel = st.selectbox("Config", options=rels, index=default_idx)
        config_path = str(REPO_ROOT / chosen_rel)

    seed = int(st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=0, step=1))
    time_axis = st.selectbox("Time axis", options=["time", "step"], index=0)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start / Reset Run", use_container_width=True):
            sess = create_session(config_path=config_path, seed=seed, time_axis=time_axis)
            set_session(sess)
            st.success(f"Started {sess.run_id}")
    with col_b:
        if st.button("Clear Session", use_container_width=True):
            set_session(None)
            st.info("Cleared")

    sess = get_session()
    if sess is not None:
        st.write(f"**Active:** {sess.run_id}")
        st.write(f"Step: {sess.step}")
        if sess.terminated or sess.truncated:
            st.warning(f"Done: terminated={sess.terminated}, truncated={sess.truncated}")


if page == "Overview":
    st.subheader("What you’re looking at")
    st.markdown(
        """
This is a **research console**—a thin instrument panel over the simulation.

- It reads **truth_state**, **sensor_state**, and **info**.
- It does **not** change physics, safety behavior, observation dimensions, or pipeline ordering.

The plots are generated by the **Observatory** module (`hydrion/rendering/*`).
    """
    )

    st.subheader("Where the visuals come from")
    st.markdown(
        """
Every time you click **Step**, the console:

1) calls `env.step(action)`
2) then records the resulting state into `EpisodeHistory`:
   - `truth_state` (ground truth)
   - `sensor_state` (measurements)
   - action, reward, info

Dashboard plots are *computed from that recorded history*, not from an old cache.
    """
    )


elif page == "System":
    st.subheader("Pipeline")
    st.code("Hydraulics → Clogging → Electrostatics → Particles → Optical/Sensors → Observation (12D)")

    st.subheader("Module cards")
    cols = st.columns(3)
    cards = [
        ("Hydraulics", "hydrion/physics/hydraulics.py"),
        ("Clogging", "hydrion/physics/clogging.py"),
        ("Electrostatics", "hydrion/physics/electrostatics.py"),
        ("Particles", "hydrion/physics/particles.py"),
        ("Sensors", "hydrion/sensors"),
        ("Safety Shield", "hydrion/safety"),
        ("Anomalies", "hydrion/anomalies"),
        ("Rendering/Observatory", "hydrion/rendering"),
        ("Validation", "hydrion/validation"),
    ]
    for i, (name, path) in enumerate(cards):
        with cols[i % 3]:
            with st.expander(name, expanded=False):
                st.write(f"Backend: `{path}`")
                st.write("UI goal: expose the same concepts in panels and timelines.")


elif page == "Scenarios":
    st.subheader("Scenario library")
    configs = list_yaml_configs()
    if not configs:
        st.error(f"No configs found under {CONFIG_DIR}")
    else:
        rels = [str(p.relative_to(REPO_ROOT)) for p in configs]
        c1, c2 = st.columns([1, 1])

        with c1:
            left = st.selectbox("Config A", options=rels, index=0)
            left_path = REPO_ROOT / left
            left_obj = load_yaml(left_path)
            st.caption(str(left_path))
            st.code(yaml_text(left_obj), language="yaml")

        with c2:
            right = st.selectbox("Config B", options=rels, index=min(1, len(rels) - 1))
            right_path = REPO_ROOT / right
            right_obj = load_yaml(right_path)
            st.caption(str(right_path))
            st.code(yaml_text(right_obj), language="yaml")

        st.subheader("Diff")
        diff = unified_diff(yaml_text(left_obj), yaml_text(right_obj), left, right)
        if diff.strip():
            st.code(diff, language="diff")
        else:
            st.success("No differences")


elif page == "Run Console":
    sess = get_session()
    if sess is None:
        st.warning("Start a run from the sidebar first.")
    else:
        st.subheader("Instrument Panel")

        top = st.columns([1.2, 1, 1, 1])
        with top[0]:
            st.metric("Run", sess.run_id)
        with top[1]:
            st.metric("Step", sess.step)
        with top[2]:
            st.metric("Reward", f"{sess.last_reward:.4f}")
        with top[3]:
            st.metric("Done", str(sess.terminated or sess.truncated))

        st.divider()

        st.markdown("### Controls")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

        with c1:
            mode = st.selectbox("Action source", ["Manual", "Random"], index=0)
        with c2:
            n_steps = int(st.number_input("Step N", min_value=1, max_value=10000, value=10, step=1))
        with c3:
            auto_refresh = st.checkbox("Auto-refresh plots", value=False)
        with c4:
            st.caption("Manual actions are 4D: [valve, pump, backflush, node_voltage] in [0,1].")

        manual_action = np.zeros(4, dtype=np.float32)
        if mode == "Manual":
            a_cols = st.columns(4)
            manual_action[0] = a_cols[0].slider("Valve", 0.0, 1.0, 0.5, 0.01)
            manual_action[1] = a_cols[1].slider("Pump", 0.0, 1.0, 0.5, 0.01)
            manual_action[2] = a_cols[2].slider("Backflush", 0.0, 1.0, 0.0, 0.01)
            manual_action[3] = a_cols[3].slider("Node Voltage", 0.0, 1.0, 0.5, 0.01)

        btn_cols = st.columns([1, 1, 1, 2])
        with btn_cols[0]:
            if st.button("Step 1", use_container_width=True, disabled=(sess.terminated or sess.truncated)):
                action = manual_action if mode == "Manual" else random_action(seed=sess.seed, step=sess.step)
                step_env(sess, action)
                set_session(sess)
        with btn_cols[1]:
            if st.button(f"Step {n_steps}", use_container_width=True, disabled=(sess.terminated or sess.truncated)):
                for _ in range(n_steps):
                    if sess.terminated or sess.truncated:
                        break
                    action = manual_action if mode == "Manual" else random_action(seed=sess.seed, step=sess.step)
                    step_env(sess, action)
                set_session(sess)
        with btn_cols[2]:
            if st.button("Finalize", use_container_width=True):
                sess.observatory.finalize_episode(terminated=sess.terminated, truncated=sess.truncated)
                set_session(sess)
                st.success("Finalized (anomaly detection computed)")
        with btn_cols[3]:
            st.caption("Tip: fixed seed + Random mode gives reproducible baselines.")

        st.divider()

        tabs = st.tabs(["Live State", "Observations", "Safety", "Dashboard Plots", "Export"])

        with tabs[0]:
            truth = dict(sess.env.truth_state)
            sensor = dict(sess.env.sensor_state)
            left, right = st.columns(2)
            with left:
                st.markdown("#### truth_state")
                st.json(summarize_truth(truth))
            with right:
                st.markdown("#### sensor_state")
                st.json(sensor)

        with tabs[1]:
            st.markdown("#### 12D observation vector (what RL sees)")
            if sess.last_obs is None:
                st.info("No observation yet")
            else:
                st.write(sess.last_obs)

        with tabs[2]:
            st.markdown("#### Safety / constraints")
            safety = (sess.last_info or {}).get("safety", {})
            st.json(safety)

        with tabs[3]:
            st.markdown("#### Observatory")
            st.caption("These plots are generated from the EpisodeHistory recorded during this session.")

            gen = st.button("Generate / Refresh Plots", type="primary")
            if gen or auto_refresh:
                sess.observatory.finalize_episode(terminated=sess.terminated, truncated=sess.truncated)
                figs = sess.observatory.plot_dashboard(save=False, show=False)
                set_session(sess)

                names = list(figs.keys())
                for i in range(0, len(names), 2):
                    row = st.columns(2)
                    for j in range(2):
                        if i + j >= len(names):
                            continue
                        name = names[i + j]
                        with row[j]:
                            st.caption(name)
                            st.pyplot(figs[name], clear_figure=True)

            st.markdown("#### Anomaly summary")
            st.json(sess.observatory.get_anomaly_summary())

        with tabs[4]:
            st.markdown("#### Export run history")
            if sess.observatory.history is None or len(sess.observatory) == 0:
                st.info("No history recorded yet. Step the env first.")
            else:
                payload = export_history(sess)
                st.download_button(
                    "Download history.json",
                    data=json.dumps(payload, indent=2),
                    file_name=f"{sess.run_id}_history.json",
                    mime="application/json",
                )

                csv = history_csv(sess)
                st.download_button(
                    "Download history.csv",
                    data=csv,
                    file_name=f"{sess.run_id}_history.csv",
                    mime="text/csv",
                )


elif page == "Benchmarks (stub)":
    st.subheader("Benchmarks")
    st.markdown(
        """
This page is intentionally a stub for the next phase.

Next capabilities:
- Run **N episodes** per policy (Random baseline, PPO)
- Aggregate metrics and produce comparison tables
- Export results + reproduce run by (config, seed, policy)

For UI work, a **1,000-step** run is plenty.
For PPO training and comparison, use longer runs (e.g. 50k+ training steps), then evaluate on a fixed episode budget.
        """
    )


elif page == "Docs":
    st.subheader("Data contracts")
    st.markdown(
        """
Conceptually, the frontend consumes these payloads:

- **SimState**: `truth_state` (ground truth, physics-owned)
- **Observations**: `sensor_state` + 12D observation vector
- **SafetyStatus**: constraint + shield decisions (from `info["safety"]`)

This v0 console reads them **in-process** from `HydrionEnv`.
The next step is a minimal local API that serves those same payloads.
        """
    )

    st.subheader("Pointers")
    st.code(
        """
- hydrion/env.py               (gym Env)
- hydrion/state/types.py       (TruthState / SensorState)
- hydrion/rendering/*          (Observatory, EpisodeHistory)
- hydrion/safety/*             (shield + constraints)
- configs/*.yaml               (scenario definitions)
        """.strip()
    )
