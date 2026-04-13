"""
Hydrion — PPO Evaluation Script for ConicalCascadeEnv
------------------------------------------------------
Evaluates a trained PPO policy vs heuristic and random baselines on CCE.

Convergence criteria (Constraint 5, revised 2026-04-12 after physics audit):
    PPO must beat BOTH baselines on capture to be considered non-trivial.
    Updated thresholds account for the RT floor (eta ~ 0.51 even at zero voltage):

        PPO vs Random eta ratio:    > 1.5x  (was 3.0x — floor was 0.57, 3x impossible)
        PPO vs Heuristic eta ratio: > 1.2x  (heuristic doesn't know about DEP threshold)

    Why these thresholds:
        RT single-screen floor: eta ~ 0.51-0.59 regardless of voltage.
        Max possible eta: 1.00.  Max ratio from floor: 1.0 / 0.57 = 1.75x.
        1.5x is ~85% of achievable headroom — meaningful differentiation.
        1.2x PPO/heuristic: heuristic uses pump=0.7 (Q=14 L/min, above DEP threshold).
        PPO must discover: pump <= 0.20 (Q <= 7.7 L/min) + volt=1.0 to activate DEP.

Usage (from repo root):
    python -m hydrion.eval_ppo_cce
    python -m hydrion.eval_ppo_cce --model models/ppo_cce_v2.zip --episodes 5

    # Canonical capture-sensitive benchmark (retrained model required):
    python -m hydrion.eval_ppo_cce --regime submicron
    python -m hydrion.eval_ppo_cce --regime submicron --model models/ppo_cce_v2.zip

    Regime 'default':   d_p = 10 um.  N_R = 6.67 at S3.  RT-saturated.
                        eta = 1.0 for all policies.  Comparison is energy/hydraulic only.
                        (ppo_cce_v1 trained in this regime — capture comparison invalid.)

    Regime 'submicron': d_p = 1 um.   N_R = 0.67 at S3 (below collector diameter).
                        CANONICAL benchmark for capture-sensitive evaluation.

                        Physics (validated 2026-04-12):
                          v_crit(V=500V) = 789 mm/s at S3 tip_radius=3um.
                          DEP threshold: Q <= 7.7 L/min (pump_cmd <= 0.22).
                          At pump=0.70 (heuristic): Q=14 L/min >> threshold -> DEP off.
                          At pump=0.10: Q=3.9 L/min < threshold -> eta=0.997.
                          At pump=0.20: Q=7.1 L/min ~ threshold -> eta=0.845.

                        Policy separability:
                          Heuristic (pump=0.7, V=1.0):   eta ~ 0.509 (above threshold)
                          Random    (pump~0.5, V~0.5):   eta ~ 0.517 (above threshold)
                          PPO target (pump<=0.20, V=1.0): eta ~ 0.845-0.997
                          PPO/heuristic on capture: up to 1.65-1.96x (BEATS heuristic)

                        Key RL insight: PPO must discover the DEP threshold is at Q~7.7
                        L/min and reduce pump to stay below it. This is not obvious from
                        the observation space and is genuinely non-trivial to learn.
                        Retraining at d_p_um=1.0 required (ppo_cce_v1 is d_p_um=10.0).
"""

from __future__ import annotations

import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hydrion.environments.conical_cascade_env import ConicalCascadeEnv
from hydrion.wrappers.shielded_env import ShieldedEnv
from hydrion.safety.shield import SafetyConfig


_CCE_SAFETY_CFG = SafetyConfig(
    max_pressure_soft=0.75,
    max_pressure_hard=1.00,
    max_clog_soft=0.70,
    max_clog_hard=0.95,
    terminate_on_hard_violation=True,
    hard_violation_penalty=10.0,
)

_MAX_STEPS_EVAL = 400

# ---------------------------------------------------------------------------
# Particle size regimes for evaluation
#
# default   -- 10 um (legacy training regime -- NOT capture-sensitive)
#             N_R = d_p / d_c = 10 / 1.5 = 6.67 at S3.
#             eta_R term in RT formula: N_R^(15/8) = 29.6.
#             Even with single-screen bed_depth_m = d_c, exponent = -4*alpha*eta_0/pi.
#             At eta_0 = 80+, exponent = -41 -> eta_bed = 1.000 always.
#             All three policies score identical capture. ppo_cce_v1 was trained here.
#
# submicron -- 1 um (CANONICAL benchmark -- capture-sensitive, DEP-sensitive)
#             N_R = 1 / 1.5 = 0.67.  eta_0 ~ 1.9 -> eta_bed(single-screen) ~ 0.63.
#             S3 face velocity: U_face = Q / S3_area_mean (area_mean ~ 1.63e-4 m2).
#             v_crit(V=500V, tip=3um, r=0.5um) = 789 mm/s.
#             DEP active when U_face < 789 mm/s -> Q < 7.7 L/min -> pump_cmd < 0.22.
#
#             Eta landscape (actual env, validated 2026-04-12):
#               pump=0.10, V=1.0 -> Q=3.9  L/min, eta=0.997  (DEP active, ratio=1.97x)
#               pump=0.15, V=1.0 -> Q=5.6  L/min, eta=0.970  (DEP active, ratio=1.37x)
#               pump=0.20, V=1.0 -> Q=7.1  L/min, eta=0.845  (DEP marginal, ratio=1.08x)
#               pump=0.25, V=1.0 -> Q=8.4  L/min, eta=0.661  (DEP inactive, threshold passed)
#               pump=0.70, V=1.0 -> Q=14.2 L/min, eta=0.509  (heuristic operating point)
#               any pump,  V=0.0 -> eta~0.51-0.59              (RT floor, no DEP)
#
#             The heuristic (pump=0.7, V=1.0) operates above the DEP threshold.
#             PPO can beat heuristic by 1.65x on capture if it discovers the threshold.
# ---------------------------------------------------------------------------
_REGIME_DEFAULT   = "default"
_REGIME_SUBMICRON = "submicron"

_D_P_UM_BY_REGIME: dict[str, float] = {
    _REGIME_DEFAULT:    10.0,  # canonical — above S3 5 µm opening
    _REGIME_SUBMICRON:   1.0,  # below S3 d_c (1.5 µm); nDEP capture non-trivial
}


def make_env(seed: int, d_p_um: float = 10.0):
    def _init():
        env = ConicalCascadeEnv(
            config_path="configs/default.yaml",
            seed=seed,
            randomize_on_reset=False,   # deterministic eval — always clean start
            d_p_um=d_p_um,
        )
        env._max_steps = _MAX_STEPS_EVAL
        env = ShieldedEnv(env, cfg=_CCE_SAFETY_CFG)
        env.reset(seed=seed)
        return env
    return _init


class HeuristicPolicy:
    """
    Rule-based baseline (Constraint 5).

    Strategy: full voltage, nominal flow, trigger backflush when fouling_mean > 0.6.
    This is the non-trivial baseline — PPO must beat it to demonstrate learning.

    obs12_v2 layout:
        [0] q_in          [1] delta_p       [2] fouling_mean  [3] eta_cascade
        [4] C_in          [5] C_out         [6] E_field_norm  [7] v_crit_norm
        [8] step_norm     [9] bf_active     [10] eta_PP       [11] eta_PET
    """
    def predict(self, obs: np.ndarray, deterministic: bool = True):
        fouling_mean = float(obs[0][2])   # obs is (1, 12) from VecEnv
        bf_cmd = 1.0 if fouling_mean > 0.6 else 0.0
        action = np.array([[0.7, 0.7, bf_cmd, 1.0]], dtype=np.float32)
        return action, None


def _run_episodes(
    vec_env,
    model,
    n_episodes: int,
    policy_label: str,
    use_vecnorm: bool = False,
) -> dict:
    """Run n_episodes and collect per-episode statistics."""
    returns, eta_list, shield_violations, bf_steps_list, ep_lengths = [], [], [], [], []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_ret, ep_eta_sum, ep_steps = 0.0, 0.0, 0
        ep_shield_v, ep_bf = 0, 0

        while not done:
            if use_vecnorm and hasattr(model, 'predict'):
                action, _ = model.predict(obs, deterministic=True)
            elif hasattr(model, 'predict'):
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = vec_env.action_space.sample()[np.newaxis, :]

            obs, reward, done_arr, info_arr = vec_env.step(action)
            done = bool(done_arr[0])
            ep_ret += float(reward[0])
            ep_steps += 1

            info = info_arr[0]
            ep_eta_sum += float(info.get("eta_cascade", 0.0))

            safety = info.get("safety", {})
            if any([
                safety.get("soft_pressure_violation"),
                safety.get("hard_pressure_violation"),
                safety.get("soft_clog_violation"),
                safety.get("hard_clog_violation"),
                safety.get("blockage_violation"),
            ]):
                ep_shield_v += 1

            # Count steps where backflush is active
            action_flat = action[0] if hasattr(action, '__len__') else action
            if float(action_flat[2]) > 0.5:
                ep_bf += 1

        returns.append(ep_ret)
        eta_list.append(ep_eta_sum / max(ep_steps, 1))
        shield_violations.append(ep_shield_v)
        bf_steps_list.append(ep_bf)
        ep_lengths.append(ep_steps)

        print(
            f"  [{policy_label:4s}] ep {ep:02d}: "
            f"return={ep_ret:7.3f}  "
            f"eta={ep_eta_sum / max(ep_steps, 1):.3f}  "
            f"steps={ep_steps}  "
            f"violations={ep_shield_v}  "
            f"bf_steps={ep_bf}"
        )

    return {
        "mean_return":     float(np.mean(returns)),
        "std_return":      float(np.std(returns)),
        "mean_eta":        float(np.mean(eta_list)),
        "mean_violations": float(np.mean(shield_violations)),
        "mean_bf_steps":   float(np.mean(bf_steps_list)),
        "mean_ep_length":  float(np.mean(ep_lengths)),
    }


def evaluate(
    model_path: str = "models/ppo_cce_v2.zip",
    vecnorm_path: str = "models/ppo_cce_v2_vecnorm.pkl",
    n_episodes: int = 5,
    regime: str = _REGIME_SUBMICRON,  # submicron is the canonical benchmark
) -> None:

    d_p_um = _D_P_UM_BY_REGIME.get(regime, 10.0)
    print(f"\nParticle regime : {regime}  (d_p = {d_p_um:.1f} um)")
    if regime == _REGIME_SUBMICRON:
        print(
            "  N_R at S3 = {:.2f}  (< 1 -- interception sub-dominant)".format(
                d_p_um / 1.5  # d_c_um for S3_MEMBRANE
            )
        )
        print("  DEP threshold: pump_cmd <= 0.22 (Q <= 7.7 L/min) at V=500V")
        print("  Expected: PPO ~ 0.85-0.997, heuristic ~ 0.509, random ~ 0.52 on eta")
        print("  (ppo_cce_v1 trained at d_p_um=10 -- retrain ppo_cce_v2 first)")

    # ── PPO policy ────────────────────────────────────────────────────────
    if not os.path.exists(model_path) or not os.path.exists(vecnorm_path):
        print(f"Model not found: {model_path} / {vecnorm_path}")
        print("Run `python -m hydrion.train_ppo_cce` first.")
        ppo_stats = None
    else:
        ppo_vec = DummyVecEnv([make_env(seed=100, d_p_um=d_p_um)])
        ppo_vec = VecNormalize.load(vecnorm_path, ppo_vec)
        ppo_vec.training    = False
        ppo_vec.norm_reward = False
        model = PPO.load(model_path, env=ppo_vec)
        print(f"\nEvaluating PPO policy: {model_path}")
        ppo_stats = _run_episodes(ppo_vec, model, n_episodes, "PPO", use_vecnorm=True)
        ppo_vec.close()

    # ── Heuristic baseline ────────────────────────────────────────────────
    heur_vec = DummyVecEnv([make_env(seed=200, d_p_um=d_p_um)])
    print(f"\nEvaluating heuristic baseline ({n_episodes} episodes)...")
    heur_stats = _run_episodes(heur_vec, HeuristicPolicy(), n_episodes, "HEUR")
    heur_vec.close()

    # ── Random baseline ───────────────────────────────────────────────────
    rand_vec = DummyVecEnv([make_env(seed=300, d_p_um=d_p_um)])
    print(f"\nEvaluating random baseline ({n_episodes} episodes)...")
    rand_stats = _run_episodes(rand_vec, None, n_episodes, "RAND")
    rand_vec.close()

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    if ppo_stats:
        print(f"{'Metric':<28} {'PPO':>10} {'Heuristic':>10} {'Random':>10}")
    else:
        print(f"{'Metric':<28} {'Heuristic':>10} {'Random':>10}")
    print("-" * 68)

    metrics = [
        ("Mean return",     "mean_return"),
        ("Mean eta_cascade", "mean_eta"),
        ("Mean violations",  "mean_violations"),
        ("Mean BF steps",   "mean_bf_steps"),
    ]
    for label, key in metrics:
        h = heur_stats[key]
        r = rand_stats[key]
        if ppo_stats:
            p = ppo_stats[key]
            print(f"  {label:<26} {p:>10.3f} {h:>10.3f} {r:>10.3f}")
        else:
            print(f"  {label:<26} {h:>10.3f} {r:>10.3f}")
    print("=" * 68)

    if ppo_stats:
        ppo_eta  = ppo_stats["mean_eta"]
        heur_eta = heur_stats["mean_eta"]
        rand_eta = rand_stats["mean_eta"]
        print(f"\nConvergence criteria (Constraint 5, revised thresholds):")
        print(f"  PPO vs Random eta ratio:    {ppo_eta / max(rand_eta, 1e-6):.2f}x  (need > 1.5x)")
        print(f"  PPO vs Heuristic eta ratio: {ppo_eta / max(heur_eta, 1e-6):.2f}x  (need > 1.2x)")
        all_eta_clustered = (
            abs(ppo_eta - heur_eta) < 0.02
            and abs(ppo_eta - rand_eta) < 0.02
        )
        all_eta_saturated = (
            abs(ppo_eta - 1.0) < 0.01
            and abs(heur_eta - 1.0) < 0.01
            and abs(rand_eta - 1.0) < 0.01
        )
        if all_eta_saturated:
            print(
                "\n  [PHYSICS AUDIT] eta = 1.000 for all policies.\n"
                "\n"
                "  Root cause: particle size >= S3 pore scale.\n"
                "    For d_p >= d_c (1.5um): N_R >= 1 --> eta_0 grows as N_R^(15/8).\n"
                "    Even with single-screen bed_depth_m = d_c_m, the exponent\n"
                "    -4*alpha*eta_0/pi is large -> eta_bed = 1.000 at S3.\n"
                "\n"
                "  Fix: use --regime submicron (d_p = 1 um, N_R = 0.67).\n"
                "    CCE must be retrained with d_p_um=1.0 (ppo_cce_v1 is d_p_um=10.0).\n"
                "    See ppo_cce_v2 artifacts once Task C retraining is complete."
            )
        elif all_eta_clustered:
            print(
                "\n  [AUDIT] All policies cluster near eta={:.3f}. Likely operating above\n"
                "  DEP threshold (pump too high) or model was trained in different regime.\n"
                "  Expected: PPO ~ 0.85-0.997, heuristic ~ 0.51, random ~ 0.52.\n"
                "  If using ppo_cce_v1 (trained at d_p_um=10): retrain with d_p_um=1.0.".format(ppo_eta)
            )
    else:
        print(f"\nHeuristic vs Random eta ratio: {heur_stats['mean_eta'] / max(rand_stats['mean_eta'], 1e-6):.2f}x")
        print("(No trained model found — heuristic/random comparison only)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO-CCE policy")
    parser.add_argument("--model",    default="models/ppo_cce_v2.zip")
    parser.add_argument("--vecnorm",  default="models/ppo_cce_v2_vecnorm.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--regime",
        default=_REGIME_SUBMICRON,
        choices=[_REGIME_DEFAULT, _REGIME_SUBMICRON],
        help=(
            f"'{_REGIME_DEFAULT}': 10 um particles (canonical training regime, capture saturated). "
            f"'{_REGIME_SUBMICRON}': 1 um particles (sub-S3-d_c; exposes slant_length saturation)."
        ),
    )
    args = parser.parse_args()
    evaluate(args.model, args.vecnorm, args.episodes, regime=args.regime)


if __name__ == "__main__":
    main()
