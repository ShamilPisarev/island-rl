"""Generate a replay without any training, so the viewer can be built and
verified before a policy exists.

It is "fake" only in that no learning produced it: the episode is a real run of
the real world, driven by the hand-written greedy forager from ``policy.py``. A
hand-fabricated JSON blob would have verified the renderer against numbers that
no simulation can actually produce; this verifies it against numbers that one
does -- bushes really deplete and regrow, agents really starve.

    python -m sim.make_fake_replay --out viewer/replays/fake_demo.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import load_config
from .policy import greedy_forager_actions, random_actions
from .replay import record_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="viewer/replays/fake_demo.json")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--policy", choices=["mixed", "greedy", "random"], default="mixed")
    parser.add_argument("--wanderers", type=int, default=2,
                        help="in mixed mode, how many agents act randomly")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)

    if args.policy == "random":
        def act(obs: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
            return random_actions(obs, rng)
        label, source = "random walk", "random"
    elif args.policy == "greedy":
        def act(obs: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
            return greedy_forager_actions(obs, cfg)
        label, source = "scripted forager", "scripted"
    else:
        # A uniformly greedy island is visually dull and, more to the point, never
        # exercises the viewer's death rendering: the greedy forager is a
        # homeostat that tops its inventory back up and simply never starves.
        # Handing the last few agents a random walk guarantees the replay contains
        # thriving agents, starving agents, and corpses.
        wanderers = np.zeros(cfg.world.num_agents, dtype=bool)
        wanderers[cfg.world.num_agents - args.wanderers:] = True

        def act(obs: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
            return np.where(wanderers, random_actions(obs, rng),
                            greedy_forager_actions(obs, cfg))
        label, source = f"scripted demo ({args.wanderers} wanderers)", "scripted"

    recorder = record_episode(cfg, seed=args.seed, act_fn=act, label=label, source=source)
    path = recorder.save(Path(args.out))
    summary = recorder.to_dict()["summary"]
    print(f"wrote {path} ({len(recorder.ticks)} ticks, {path.stat().st_size / 1024:.0f} KB)")
    print(f"  {summary}")


if __name__ == "__main__":
    main()
