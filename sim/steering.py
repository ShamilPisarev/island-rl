"""What is NAVIGATION worth to a policy that already knows what to do?

`sim.opportunity --force` answers "is this declined *action* worth taking". This
answers the movement version of the same question, and it needs a different
override: keep every decision the policy makes and replace only the DIRECTION it
walks in. When the policy chooses a move, steer that move at the nearest
berry-bearing bush -- or, at night, the nearest finished shelter. Gathering,
stealing, chopping, mining, building and idling are untouched, and the number of
ticks spent moving is identical, so the difference is navigation and nothing else.

WHY THIS MATTERS HERE. Every "spend ticks on X" intervention in this project has
netted zero: forcing builds (+3.6 +- 7.7), forcing steals (-8.3 +- 4.1), going home
at dusk (-0.3 +- 7.6, with berries falling 34.7 -> 29.2). Meanwhile the scripted
builder does all of those things at once and is +72.6 ahead, which is what having
spare ticks looks like. Steering says how much of that slack is walking: on m4h it
is **+89.1 +- 11.4** for food alone and **+115.7 +- 10.7** for food by day and home
at night -- the latter beating the scripted builder by 43 ticks.

READ IT AS AN UPPER BOUND, NOT A PLAN. The steering rule reads world state: the
nearest *berry-bearing* bush among all of them, where the observation carries the
k_bushes nearest bushes loaded or not (on m4h the target is absent from the
observation on 54% of ticks beyond 20 units). So this measures what perfect
navigation would be worth, not what is learnable from the current observation. What
makes it more than a fantasy is that `nav_probe` shows PPO learning 86%
toward-food from long range, and the scripted forager reaches 100% off the
observation alone.

    python -m sim.steering --checkpoint checkpoints/m4h/latest.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .agents import MOVE_VECTORS, N_MOVE_ACTIONS, night_phase
from .config import Config
from .evaluate import (ActFn, baseline_kinds, load_checkpoint, make_act_fn, paired_lines,
                       policy_act_fn)
from .world import EpisodeStats, World

MODES = ("off", "food", "food+home")


def steered_act(cfg: Config, base: ActFn, mode: str):
    """Wrap ``base`` so its move actions point at the nearest useful target.

    Returns a callable taking the live ``World`` as well as the observation, because
    the target is world state rather than something the observation fully carries.
    """
    def act(w: World, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        actions = base(obs, mask)
        if mode == "off":
            return actions
        _, is_night = night_phase(w.tick, cfg) if cfg.construction.enabled else (0.0, False)
        loaded = w.bush_berries > 0
        done = ((w.site_wood_needed + w.site_stone_needed) == 0
                if cfg.construction.enabled else np.zeros(0, dtype=bool))
        home = mode == "food+home" and is_night and done.any()
        tx, tz = (w.site_x[done], w.site_z[done]) if home else (w.bush_x[loaded],
                                                                w.bush_z[loaded])
        if tx.size == 0:
            return actions
        for i in np.flatnonzero(w.pool.alive):
            if int(actions[i]) >= N_MOVE_ACTIONS:
                continue                  # redirect moves; never replace a decision
            dx, dz = tx - w.pool.x[i], tz - w.pool.z[i]
            d = np.hypot(dx, dz)
            j = int(np.argmin(d))
            v = np.array([dx[j], dz[j]]) / max(float(d[j]), 1e-9)
            actions[i] = int(np.argmax(MOVE_VECTORS[:N_MOVE_ACTIONS] @ v))
        return actions
    return act


def run_steered(cfg: Config, act, episodes: int, seed: int,
                world_aware: bool = True) -> list[EpisodeStats]:
    """Rollouts driven by a world-aware act function, torch re-seeded per call.

    Common random numbers: sampling noise alone moves a 40-island mean by ~10 ticks
    here, which is bigger than several of the effects being compared.
    """
    stats = []
    torch.manual_seed(0)
    for e in range(episodes):
        w = World(cfg, seed=seed + e)
        obs = w.observations()
        for _ in range(cfg.world.max_ticks):
            mask = w.action_mask()
            result = w.step(act(w, obs, mask) if world_aware else act(obs, mask))
            obs = result.obs
            if result.episode_done:
                break
        stats.append(w.stats())
    return stats


def compare(cfg: Config, policy, episodes: int = 40, seed: int = 10000
            ) -> list[tuple[str, list[EpisodeStats]]]:
    base = policy_act_fn(policy)
    runs: list[tuple[str, list[EpisodeStats]]] = []
    for mode in MODES:
        label = "learned policy" if mode == "off" else f"steered: {mode}"
        if mode == "food+home" and not cfg.construction.enabled:
            continue
        runs.append((label, run_steered(cfg, steered_act(cfg, base, mode), episodes, seed)))
    for kind, label in baseline_kinds(cfg):
        if kind == "random":
            continue
        runs.append((label, run_steered(cfg, make_act_fn(kind, cfg, None, seed),
                                        episodes, seed, world_aware=False)))
    return runs


def _row(label: str, stats: list[EpisodeStats], cfg: Config) -> str:
    life = np.array([s.mean_lifespan for s in stats])
    nights = np.mean([s.night_ticks_sheltered
                      / max(s.night_ticks_sheltered + s.night_ticks_exposed, 1)
                      for s in stats])
    out = (f"  {label:22} lifespan {life.mean():6.1f} +- {life.std(ddof=1):5.1f}"
           f"   berries {np.mean([s.berries_gathered for s in stats]):5.1f}"
           f"   deaths {np.mean([s.deaths for s in stats]):4.2f}")
    if cfg.construction.enabled:
        out += (f"   shelters {np.mean([s.shelters_completed for s in stats]):4.2f}"
                f"   nights in {nights:5.1%}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=40,
                    help="paired, so 40 islands is cheap and much tighter than 20")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--report", default="viewer/reports/steering.json")
    args = ap.parse_args()

    policy, cfg, blob = load_checkpoint(args.checkpoint)
    print(f"loaded {args.checkpoint} (update {blob.get('update', '?')})")
    print("  every decision is the policy's own; only the DIRECTION of its moves "
          "is replaced.\n")
    runs = compare(cfg, policy, args.episodes, args.seed)
    for label, stats in runs:
        print(_row(label, stats, cfg))
    print()
    for line in paired_lines(runs, "learned policy"):
        print(line)
    print("\n  steering reads world state, so treat it as an upper bound on what "
          "perfect\n  navigation is worth rather than as what the current "
          "observation supports.")

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"checkpoint": args.checkpoint, "episodes": args.episodes,
         "runs": {label: {"lifespan": float(np.mean([s.mean_lifespan for s in st])),
                          "berries": float(np.mean([s.berries_gathered for s in st])),
                          "deaths": float(np.mean([s.deaths for s in st])),
                          "shelters": float(np.mean([s.shelters_completed for s in st]))}
                  for label, st in runs}}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
