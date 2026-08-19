"""Is the learning signal for walking toward food really smaller than its noise?

The nav-spread postmortem left one hypothesis standing, and it is arithmetic
rather than a measurement: V falls ~1.2 across the 10-20 band, a move covers 0.8
units, so one step toward food should be worth ~0.06 of advantage against ~1.0 of
per-tick reward noise from a gather landing or not. This module measures that
instead of trusting it.

For every move an alive agent makes, compute the two quantities PPO actually
trains on -- the one-step TD residual delta = r + gamma*V(s') - V(s), and the GAE
advantage with the checkpoint's own gamma and lambda -- then split moves into
toward / away from the nearest berry-bearing bush, bucketed by current distance.
The signal is E[A | toward] - E[A | away] within a band; the noise is the standard
deviation of A that PPO's per-minibatch normalisation divides by. If the gap is a
few hundredths against a std near 1, the arithmetic is confirmed and a k-tick
commitment (which multiplies the per-decision slope without touching the reward)
is the lever it says it is.

The tick window is held (150-450 by default) for the same reason it is in
`sim.navigation --value`: far-from-food ticks bunch at the start of an episode,
and near the horizon V is reading remaining lifetime rather than distance.
Mean hunger is printed per bucket because toward-movers being systematically
hungrier would confound the gap; the two columns should be close.

    python -m sim.advantage --checkpoint checkpoints/nav-spread2/latest.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .agents import MOVE_VECTORS, N_MOVE_ACTIONS
from .config import Config
from .evaluate import load_checkpoint, policy_act_fn
from .navigation import BANDS, _band
from .world import World


def measure_advantage(cfg: Config, policy, episodes: int = 20, seed: int = 10000,
                      tick_band: tuple[int, int] = (150, 450)) -> dict:
    """Per band: delta and GAE advantage for toward- vs away-food moves.

    The GAE pass mirrors ppo.py exactly -- truncation bootstraps V(final obs)
    into the last reward, death is terminal, rows after a death are cut by the
    done flag so nothing leaks backwards across it.
    """
    gamma, lam = cfg.ppo.gamma, cfg.ppo.gae_lambda
    n_agents = cfg.world.num_agents
    ids = torch.arange(n_agents)
    act_fn = policy_act_fn(policy)
    lo_t, hi_t = tick_band
    # The policy SAMPLES, so without this the per-band gaps wobble run to run by
    # more than their printed +- (which is conditional on the sampled trajectories).
    # Same convention as sim.steering: seed torch, so the tool reproduces itself.
    torch.manual_seed(seed)

    # buckets[band][toward] -> list of (delta, gae_advantage, hunger)
    buckets: list[list[list[tuple[float, float, float]]]] = [
        [[], []] for _ in BANDS
    ]
    all_adv: list[float] = []   # every active transition, for the normaliser's std
    all_delta: list[float] = []

    for e in range(episodes):
        w = World(cfg, seed=seed + e)
        obs = w.observations()
        k_int = max(cfg.world.decision_interval, 1)
        values, rewards, dones, actives = [], [], [], []
        annos: list[dict[int, tuple[int, bool, float]]] = []  # agent -> (band, toward, hunger)
        final_obs = None
        truncated_alive = np.zeros(n_agents, dtype=bool)
        episode_over = False

        # One iteration = one DECISION, exactly as the trainer sees it: under
        # decision_interval > 1 the chosen action persists k ticks and the rewards
        # those ticks earn are summed into the one transition PPO stores.
        while not episode_over and w.tick < cfg.world.max_ticks:
            mask = w.action_mask()
            with torch.no_grad():
                v = policy.value(torch.as_tensor(obs), ids).numpy()
            actions = act_fn(obs, mask)

            loaded = w.bush_berries > 0
            ann: dict[int, tuple[int, bool, float]] = {}
            if loaded.any() and lo_t <= w.tick <= hi_t:
                bx, bz = w.bush_x[loaded], w.bush_z[loaded]
                for i in np.flatnonzero(w.pool.alive):
                    a = int(actions[i])
                    if a >= N_MOVE_ACTIONS:
                        continue
                    x, z = w.pool.x[i], w.pool.z[i]
                    d0 = float(np.hypot(bx - x, bz - z).min())
                    step_v = MOVE_VECTORS[a] * cfg.world.move_step
                    d1 = float(np.hypot(bx - (x + step_v[0]), bz - (z + step_v[1])).min())
                    k = _band(d0)
                    if k is not None:
                        ann[i] = (k, d1 < d0, float(w.pool.hunger[i]))

            rew_acc = np.zeros(n_agents)
            term_acc = np.zeros(n_agents, dtype=bool)
            acted0 = None
            for _ in range(k_int):
                res = w.step(actions)
                rew_acc += res.rewards
                term_acc |= res.terminated
                if acted0 is None:
                    acted0 = res.acted.copy()
                obs = res.obs
                if res.episode_done:
                    episode_over = True
                    final_obs = res.obs
                    truncated_alive = res.truncated & w.pool.alive
                    break

            values.append(v)
            rewards.append(rew_acc)
            actives.append(acted0)
            # done cuts the GAE recursion: death, or any episode end (the
            # truncation bootstrap is folded into the reward below, as in ppo.py)
            dones.append(term_acc | ~acted0 | episode_over)
            annos.append(ann)

        T = len(values)
        val = np.stack(values)          # (T, A)
        rew = np.stack(rewards)
        done = np.stack(dones).astype(np.float64)
        active = np.stack(actives)
        if truncated_alive.any():
            with torch.no_grad():
                v_final = policy.value(torch.as_tensor(final_obs), ids).numpy()
            rew[-1] = rew[-1] + gamma * v_final * truncated_alive

        delta = np.zeros_like(rew)
        adv = np.zeros_like(rew)
        gae = np.zeros(n_agents)
        next_value = np.zeros(n_agents)  # only read where done cuts anyway
        for t in reversed(range(T)):
            non_terminal = 1.0 - done[t]
            delta[t] = rew[t] + gamma * next_value * non_terminal - val[t]
            gae = delta[t] + gamma * lam * non_terminal * gae
            adv[t] = gae
            next_value = val[t]

        all_adv.extend(adv[active].tolist())
        all_delta.extend(delta[active].tolist())
        for t, ann in enumerate(annos):
            for i, (k, toward, hunger) in ann.items():
                buckets[k][int(toward)].append((float(delta[t, i]), float(adv[t, i]), hunger))

    def _stats(rows: list[tuple[float, float, float]], col: int) -> tuple[float, float]:
        if not rows:
            return float("nan"), float("nan")
        arr = np.array([r[col] for r in rows])
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    out = {"batch_adv_std": float(np.std(all_adv)), "batch_delta_std": float(np.std(all_delta)),
           "n_transitions": len(all_adv), "bands": []}
    for k, (lo, hi) in enumerate(BANDS):
        away, toward = buckets[k][0], buckets[k][1]
        n_a, n_t = len(away), len(toward)
        d_t, d_t_sd = _stats(toward, 0)
        d_a, d_a_sd = _stats(away, 0)
        a_t, a_t_sd = _stats(toward, 1)
        a_a, a_a_sd = _stats(away, 1)
        h_t, _ = _stats(toward, 2)
        h_a, _ = _stats(away, 2)
        # SE of the gap between two independent-ish sample means
        def _se(sd1: float, n1: int, sd2: float, n2: int) -> float:
            if n1 < 2 or n2 < 2:
                return float("nan")
            return float(np.sqrt(sd1 ** 2 / n1 + sd2 ** 2 / n2))
        out["bands"].append({
            "lo": lo, "hi": hi, "n_toward": n_t, "n_away": n_a,
            "delta_toward": d_t, "delta_away": d_a,
            "delta_gap": d_t - d_a, "delta_gap_se": _se(d_t_sd, n_t, d_a_sd, n_a),
            "delta_sd": float(np.sqrt(((n_t - 1) * d_t_sd ** 2 + (n_a - 1) * d_a_sd ** 2)
                                      / max(n_t + n_a - 2, 1))) if n_t + n_a > 2 else float("nan"),
            "adv_toward": a_t, "adv_away": a_a,
            "adv_gap": a_t - a_a, "adv_gap_se": _se(a_t_sd, n_t, a_a_sd, n_a),
            "adv_sd": float(np.sqrt(((n_t - 1) * a_t_sd ** 2 + (n_a - 1) * a_a_sd ** 2)
                                    / max(n_t + n_a - 2, 1))) if n_t + n_a > 2 else float("nan"),
            "hunger_toward": h_t, "hunger_away": h_a,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--report", default="viewer/reports/advantage.json")
    args = ap.parse_args()

    policy, cfg, blob = load_checkpoint(args.checkpoint)
    print(f"loaded {args.checkpoint} (update {blob.get('update', '?')})")
    print(f"  gamma {cfg.ppo.gamma}  lambda {cfg.ppo.gae_lambda}  "
          f"move_step {cfg.world.move_step}  tick window 150-450\n")

    r = measure_advantage(cfg, policy, args.episodes, args.seed)
    print(f"  std over ALL active transitions (what normalisation divides by): "
          f"GAE adv {r['batch_adv_std']:.3f}   one-step delta {r['batch_delta_std']:.3f}   "
          f"(n={r['n_transitions']:,})\n")
    header = (f"  {'band':>6}  {'n tow/away':>13}  "
              f"{'delta gap':>16}  {'gae adv gap':>16}  {'adv sd':>7}  "
              f"{'gap/std':>8}  {'hunger t/a':>12}")
    print(header)
    for b in r["bands"]:
        span = f"{b['lo']:.0f}+" if b["hi"] > 1e8 else f"{b['lo']:.0f}-{b['hi']:.0f}"
        if b["n_toward"] + b["n_away"] == 0:
            continue
        norm = b["adv_gap"] / r["batch_adv_std"] if r["batch_adv_std"] else float("nan")
        print(f"  {span:>6}  {b['n_toward']:>6}/{b['n_away']:>6}  "
              f"{b['delta_gap']:+7.3f} ± {b['delta_gap_se']:5.3f}  "
              f"{b['adv_gap']:+7.3f} ± {b['adv_gap_se']:5.3f}  {b['adv_sd']:7.3f}  "
              f"{norm:+8.3f}  {b['hunger_toward']:5.1f}/{b['hunger_away']:5.1f}")
    print("\n  the gap is the per-step learning signal for direction; gap/std is that")
    print("  signal in the units PPO's advantage normalisation actually hands the")
    print("  gradient. The arithmetic in CLAUDE.md predicts ~+0.06 against ~1.0.")

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, **r}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
