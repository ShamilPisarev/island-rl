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


def steered_act(cfg: Config, base: ActFn, mode: str, agents: set[int] | None = None):
    """Wrap ``base`` so its move actions point at the nearest useful target.

    Returns a callable taking the live ``World`` as well as the observation, because
    the target is world state rather than something the observation fully carries.

    ``agents`` limits the override to those agent indices (None = everyone). That
    is the subset experiment: steering all six is a GROUP counterfactual, while
    PPO's gradient sees the individual one -- whether ONE navigator among five
    unsteered agents gains anything is a different question, and the one that
    decides whether the +89..+141 headroom is a skill gap or a coordination prize.
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
            if agents is not None and i not in agents:
                continue                  # not part of the steered subset
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


def run_per_agent(cfg: Config, act, episodes: int, seed: int) -> np.ndarray:
    """(episodes, num_agents) lifespans under a world-aware act fn.

    Same common-random-numbers discipline as ``run_steered``; per-agent rather
    than pooled, because the subset experiment's whole question is what happens
    to the steered individuals as opposed to the population mean.
    """
    out = np.zeros((episodes, cfg.world.num_agents))
    torch.manual_seed(0)
    for e in range(episodes):
        w = World(cfg, seed=seed + e)
        obs = w.observations()
        for _ in range(cfg.world.max_ticks):
            result = w.step(act(w, obs, w.action_mask()))
            obs = result.obs
            if result.episode_done:
                break
        out[e] = w.alive_ticks
    return out


def subset_compare(cfg: Config, policy, counts: list[int], episodes: int = 40,
                   seed: int = 10000) -> list[dict[str, float]]:
    """Steer the first n agents' moves at food and pair everything per island.

    For each n: the steered agents' lifespan against THE SAME agents unsteered
    (the individual counterfactual), the unsteered agents' lifespan against the
    same agents in the control (the spillover), and the population mean.
    """
    base = policy_act_fn(policy)
    a_total = cfg.world.num_agents
    control = run_per_agent(cfg, steered_act(cfg, base, "off"), episodes, seed)
    rows = []
    for n in counts:
        subset = set(range(n))
        steered = run_per_agent(cfg, steered_act(cfg, base, "food", agents=subset),
                                episodes, seed)
        def _paired(cols: slice) -> tuple[float, float, int]:
            d = (steered[:, cols] - control[:, cols]).mean(axis=1)
            return (float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))),
                    int((d > 0).sum()))
        st = _paired(slice(0, n))
        un = _paired(slice(n, a_total)) if n < a_total else (float("nan"),) * 2 + (0,)
        pop = _paired(slice(0, a_total))
        rows.append({"n": n, "steered": st[0], "steered_se": st[1], "steered_wins": st[2],
                     "unsteered": un[0], "unsteered_se": un[1],
                     "population": pop[0], "population_se": pop[1],
                     "episodes": episodes})
    return rows


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
    ap.add_argument("--subset", default=None, metavar="N1,N2,...",
                    help="steer only the first n agents' moves at food, for each n "
                         "given, and report the steered individuals' paired gain, "
                         "the spillover onto the unsteered, and the population "
                         "mean. This is the individual-vs-collective question.")
    args = ap.parse_args()

    policy, cfg, blob = load_checkpoint(args.checkpoint)
    print(f"loaded {args.checkpoint} (update {blob.get('update', '?')})")
    print("  every decision is the policy's own; only the DIRECTION of its moves "
          "is replaced.\n")

    if args.subset:
        counts = [int(x) for x in args.subset.split(",")]
        rows = subset_compare(cfg, policy, counts, args.episodes, args.seed)
        print(f"  steer the first n of {cfg.world.num_agents} agents at food, "
              f"paired per island against the unsteered control ({args.episodes} islands):\n")
        print(f"  {'n':>3}  {'steered agents':>22}  {'unsteered agents':>22}  "
              f"{'population':>20}")
        for r in rows:
            un = ("        --        " if np.isnan(r["unsteered"])
                  else f"{r['unsteered']:+7.1f} ± {r['unsteered_se']:4.1f}")
            print(f"  {r['n']:>3}  {r['steered']:+7.1f} ± {r['steered_se']:4.1f} "
                  f"({r['steered_wins']:>2}/{r['episodes']})  {un:>22}  "
                  f"{r['population']:+7.1f} ± {r['population_se']:4.1f}")
        print("\n  'steered agents' pairs each steered agent against ITSELF unsteered on")
        print("  the same island. A one-step advantage may still be negative while a")
        print("  sustained deviation pays -- this measures the sustained one.")
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"checkpoint": args.checkpoint, "subset": rows}, indent=2))
        print(f"\nwrote {out}")
        return
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
