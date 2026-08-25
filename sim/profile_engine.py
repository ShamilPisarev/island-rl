"""Profile the raw env step at a given agent count (Island 2.0, stage 1).

No policy, no torch: actions are random legal-ish integers, which is exactly the
stage-1 exit condition ("no AI changes -- random policies"). Prints agent-steps/s
and, with --profile, the cProfile hotspots so the lever is measured before it is
chosen.

Usage:
    python -m sim.profile_engine --config config/island2/engine100.yaml
    python -m sim.profile_engine --config config/island2/engine100.yaml --profile
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time

import numpy as np

from .agents import num_actions
from .config import load_config
from .world import World


def run(cfg, seed: int, ticks: int) -> tuple[int, float]:
    """Step one world for `ticks` ticks (resetting at episode end).

    Returns (agent-steps taken, seconds). Agent-steps counts every agent slot
    every tick, matching how the 1.0 throughput figures were quoted.
    """
    world = World(cfg, seed=seed)
    rng = np.random.default_rng(seed + 1)
    n_act = num_actions(cfg)
    n = cfg.world.num_agents
    start = time.perf_counter()
    for _ in range(ticks):
        actions = rng.integers(0, n_act, size=n)
        res = world.step(actions)
        # VecWorld computes a mask after every decision, so it is part of the
        # real per-tick cost and belongs in the measurement.
        world.action_mask()
        if res.episode_done:
            world.reset()
    elapsed = time.perf_counter() - start
    return ticks * n, elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", action="store_true", help="print cProfile hotspots")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    cfg = load_config(args.config)
    # Warm-up run so numpy/allocator startup does not pollute the measurement.
    run(cfg, args.seed, min(200, args.ticks))

    if args.profile:
        prof = cProfile.Profile()
        prof.enable()
        steps, elapsed = run(cfg, args.seed, args.ticks)
        prof.disable()
        print(f"{cfg.world.num_agents} agents: {steps / elapsed:,.0f} agent-steps/s "
              f"({steps:,} steps in {elapsed:.2f}s)")
        stats = pstats.Stats(prof)
        stats.sort_stats("cumulative").print_stats(args.top)
    else:
        steps, elapsed = run(cfg, args.seed, args.ticks)
        print(f"{cfg.world.num_agents} agents: {steps / elapsed:,.0f} agent-steps/s "
              f"({steps:,} steps in {elapsed:.2f}s)")


if __name__ == "__main__":
    main()
