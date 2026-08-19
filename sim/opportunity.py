"""Opportunity versus uptake, per action: is a rare behaviour the world or the policy?

Every "the policy does not do X often enough" finding in this project has turned
out to be one of two completely different problems, and a raw count of X cannot
tell them apart:

  * **no opportunity** -- X is almost never legal, so the policy is not declining
    anything. `m4f` was this: `chop` was reachable on 0.2% of ticks and taken on
    82-93% of those, because trees were nowhere near where agents lived. The fix
    was geography and it was worth 20x the shelters.
  * **declined opportunity** -- X is legal and the policy passes. `m4h` was this:
    `build` legal on 1.5% of ticks and taken on 28.6% of them.

The action mask is what makes this measurable: it is the world's own statement of
what is reachable this tick, so "legal" here is not a heuristic. Worlds without
`competition.mask_invalid_actions` have an all-ones mask and every action reads as
100% legal -- the report says so rather than pretending otherwise.

Uptake is a share of *legal* ticks, so it is directly comparable across actions
with wildly different opportunity rates. Note that a taken action can still fail
(a thief can empty the victim first, two givers can race for one slot), so uptake
is what the policy chose, not what it achieved.

    python -m sim.opportunity --checkpoint checkpoints/m4h/latest.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .agents import N_MOVE_ACTIONS, action_names
from .config import Config
from .evaluate import ActFn, load_checkpoint, policy_act_fn
from .world import World


def measure(cfg: Config, act_fn: ActFn, episodes: int = 10, seed: int = 10000) -> dict:
    """Legal-tick share and uptake for every action, over living agents."""
    names = action_names(cfg)
    legal = np.zeros(len(names))
    taken = np.zeros(len(names))
    taken_legal = np.zeros(len(names))
    ticks = 0

    for e in range(episodes):
        w = World(cfg, seed=seed + e)
        obs = w.observations()
        for _ in range(cfg.world.max_ticks):
            mask = w.action_mask()
            actions = act_fn(obs, mask)
            live = w.pool.alive
            rows = np.flatnonzero(live)
            ticks += rows.size
            legal += mask[live].sum(axis=0)
            for i in rows:
                a = int(actions[i])
                taken[a] += 1
                # A masked action should never be sampled; counted separately so a
                # regression in the mask plumbing shows up as a number rather than
                # hiding inside the uptake share.
                taken_legal[a] += bool(mask[i, a])
            result = w.step(actions)
            obs = result.obs
            if result.episode_done:
                break

    return {
        "ticks": ticks,
        "masked": bool(cfg.competition.mask_invalid_actions),
        "actions": [
            {"name": names[j],
             "legal_frac": float(legal[j] / ticks) if ticks else float("nan"),
             "uptake": float(taken[j] / legal[j]) if legal[j] else float("nan"),
             "taken": int(taken[j]), "legal": int(legal[j]),
             "taken_while_masked": int(taken[j] - taken_legal[j])}
            for j in range(len(names))
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--all-actions", action="store_true",
                    help="include the eight moves and idle, which are always legal")
    ap.add_argument("--report", default="viewer/reports/opportunity.json")
    args = ap.parse_args()

    policy, cfg, blob = load_checkpoint(args.checkpoint)
    report = measure(cfg, policy_act_fn(policy), args.episodes, args.seed)

    print(f"loaded {args.checkpoint} (update {blob.get('update', '?')})")
    print(f"{report['ticks']:,} living agent-ticks over {args.episodes} episodes\n")
    if not report["masked"]:
        print("  competition.mask_invalid_actions is OFF in this world: the mask is")
        print("  all ones, so every action reads as legal and `uptake` is just the")
        print("  action mix. Only the ordering means anything here.\n")
    print(f"  {'action':14} {'legal on':>10}  {'taken on':>10}   counts")
    for a in report["actions"]:
        if not args.all_actions and (a["name"] == "idle"
                                    or a["name"] in action_names(cfg)[:N_MOVE_ACTIONS]):
            continue
        print(f"  {a['name']:14} {a['legal_frac']:9.1%}  {a['uptake']:9.1%}   "
              f"({a['taken']:,} of {a['legal']:,} legal)")
        if a["taken_while_masked"]:
            print(f"    !! {a['taken_while_masked']} of those were MASKED when taken "
                  f"-- the mask plumbing is broken, not the policy")

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, **report}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
