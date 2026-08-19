"""Does a policy actually travel toward what it wants?

For every move an agent makes, ask whether it ends up closer to the nearest
*berry-bearing* bush (or, at night, the nearest completed shelter). 50% is a coin
flip -- a random walk scores it exactly.

THE WHOLE POINT OF THIS MODULE IS THE BUCKETING, and it is worth being exact
about why, because the obvious explanation is wrong.

An aggregate toward-target share is dominated by whichever distance band the
policy spends its ticks in -- and for a policy that eats, that is the near field.
M1 spends 17,347 of its ~22,700 move-ticks within 3 units of food and only 199
beyond 10. Its aggregate is therefore essentially its near-field number (47.5%)
and tells you almost nothing about whether it can cross the island, which it can:
88.4% at 10-20 units.

The near-field ~48% is NOT itself an artefact of "you can only move away from a
target you are standing on" -- the scripted forager scores 100% in that same
band, because it only moves when moving is the right thing to do and otherwise
gathers. A learned policy scoring 48% there is genuinely wandering. The point is
that averaging that real near-field deficiency together with real far-field
competence, weighted 87:1 by sample count, produces one number that describes
neither.

The first version of this measurement reported M1's aggregate 51.3% and concluded
the project had never learned to navigate. That was wrong. See rule 6 in
CLAUDE.md.

SECOND CAVEAT: 50% IS NOT ALWAYS THE FLOOR. Movement is projected back onto the
island disc, so a random walker near the shoreline drifts inward -- and if the
food is inward, that reads as navigation. In the nav_probe world a random policy
scores ~64% in the 10-20 band on short rollouts for exactly this reason. Always
run --baselines and compare against the random floor measured in the SAME world,
never against a nominal 50%.

    python -m sim.navigation --checkpoint checkpoints/m1/latest.pt
    python -m sim.navigation --checkpoint checkpoints/m4h/latest.pt --baselines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .agents import MOVE_VECTORS, N_MOVE_ACTIONS, night_phase
from .config import Config
from .evaluate import ActFn, load_checkpoint, make_act_fn, policy_act_fn
from .world import World

# Distance bands, in world units. The first is "on top of it" and is expected to
# read ~50% for every policy including a perfect one; it is printed so that fact
# stays visible rather than being quietly dropped.
BANDS: tuple[tuple[float, float], ...] = ((0, 3), (3, 6), (6, 10), (10, 20), (20, 1e9))


def _band(d: float) -> int | None:
    for k, (lo, hi) in enumerate(BANDS):
        if lo <= d < hi:
            return k
    return None


def measure(cfg: Config, act_fn: ActFn, episodes: int = 10, seed: int = 10000
            ) -> dict[str, list[dict[str, float]]]:
    """Toward-target share per distance band, for food and (if built) shelter.

    Also, per band, whether the food target was *perceivable* at all -- see
    ``vision`` in the returned dict and the block main() prints from it.
    """
    step = cfg.world.move_step
    k_bushes = cfg.observation.k_bushes
    scale = cfg.observation.distance_scale
    hit = {"food": np.zeros(len(BANDS)), "shelter": np.zeros(len(BANDS))}
    tot = {"food": np.zeros(len(BANDS)), "shelter": np.zeros(len(BANDS))}
    # Vision bookkeeping, food only: was the nearest berry-bearing bush among the
    # k_bushes NEAREST bushes (the ones the observation actually carries), and did
    # its offset saturate the +-1 clip. Split the toward-share by visibility so a
    # flat band can be attributed to perception or acquitted of it.
    seen = np.zeros(len(BANDS))
    sat1 = np.zeros(len(BANDS))
    sat2 = np.zeros(len(BANDS))
    vis_hit = np.zeros((2, len(BANDS)))
    vis_tot = np.zeros((2, len(BANDS)))

    for e in range(episodes):
        w = World(cfg, seed=seed + e)
        obs = w.observations()
        for _ in range(cfg.world.max_ticks):
            actions = act_fn(obs, w.action_mask())
            _, is_night = night_phase(w.tick, cfg) if cfg.construction.enabled else (0.0, False)
            done = ((w.site_wood_needed + w.site_stone_needed) == 0
                    if cfg.construction.enabled else np.zeros(0, dtype=bool))
            loaded = w.bush_berries > 0
            for i in np.flatnonzero(w.pool.alive):
                a = int(actions[i])
                if a >= N_MOVE_ACTIONS:
                    continue          # only moves can be toward or away
                v = MOVE_VECTORS[a] * step
                x, z = w.pool.x[i], w.pool.z[i]

                def closer(ex: np.ndarray, ez: np.ndarray) -> tuple[float, bool] | None:
                    if ex.size == 0:
                        return None
                    d0 = float(np.hypot(ex - x, ez - z).min())
                    d1 = float(np.hypot(ex - (x + v[0]), ez - (z + v[1])).min())
                    return d0, d1 < d0

                for key, res in (("food", closer(w.bush_x[loaded], w.bush_z[loaded])),
                                 ("shelter", closer(w.site_x[done], w.site_z[done])
                                  if (cfg.construction.enabled and is_night and done.any())
                                  else None)):
                    if res is None:
                        continue
                    d0, got_closer = res
                    k = _band(d0)
                    if k is not None:
                        tot[key][k] += 1
                        hit[key][k] += got_closer
                        if key == "food":
                            d_all = np.hypot(w.bush_x - x, w.bush_z - z)
                            tgt = int(np.flatnonzero(loaded)[np.argmin(d_all[loaded])])
                            # Rank among ALL bushes, empty ones included: the
                            # observation carries the k nearest bushes whether or
                            # not they hold anything, so an empty bush standing
                            # nearby costs a slot the target might have used.
                            rank = int((d_all < d_all[tgt]).sum())
                            visible = rank < k_bushes
                            dx = abs(float(w.bush_x[tgt] - x))
                            dz = abs(float(w.bush_z[tgt] - z))
                            seen[k] += visible
                            sat1[k] += (dx >= scale) or (dz >= scale)
                            sat2[k] += (dx >= scale) and (dz >= scale)
                            vis_tot[int(visible)][k] += 1
                            vis_hit[int(visible)][k] += got_closer

            result = w.step(actions)
            obs = result.obs
            if result.episode_done:
                break

    out: dict[str, list[dict[str, float]]] = {}
    for key in ("food", "shelter"):
        out[key] = [
            {"lo": BANDS[k][0], "hi": BANDS[k][1], "n": int(tot[key][k]),
             "toward": float(hit[key][k] / tot[key][k]) if tot[key][k] else float("nan")}
            for k in range(len(BANDS))
        ]

    def _share(num: np.ndarray, den: np.ndarray, k: int) -> float:
        return float(num[k] / den[k]) if den[k] else float("nan")

    out["vision"] = [
        {"lo": BANDS[k][0], "hi": BANDS[k][1], "n": int(tot["food"][k]),
         "visible": _share(seen, tot["food"], k),
         "sat_one_axis": _share(sat1, tot["food"], k),
         "sat_both_axes": _share(sat2, tot["food"], k),
         "n_visible": int(vis_tot[1][k]), "n_unseen": int(vis_tot[0][k]),
         "toward_visible": _share(vis_hit[1], vis_tot[1], k),
         "toward_unseen": _share(vis_hit[0], vis_tot[0], k)}
        for k in range(len(BANDS))
    ]
    return out


def _row(label: str, bands: list[dict[str, float]]) -> str:
    """One printed line, with the sample count in each cell.

    n is printed rather than used to suppress the cell: a competent policy has
    *few* far-band samples precisely because it does not stay far from things,
    so hiding thin bands would hide the ceiling. Judge the share against its n.
    """
    cells = []
    for b in bands:
        span = f"{b['lo']:.0f}+" if b["hi"] > 1e8 else f"{b['lo']:.0f}-{b['hi']:.0f}"
        value = "   -- " if b["n"] == 0 else f"{b['toward']:5.1%}"
        cells.append(f"{span}: {value} (n={b['n']:>5})")
    return f"  {label:26} " + "  ".join(cells)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--baselines", action="store_true",
                    help="also measure random actions and the scripted forager")
    ap.add_argument("--report", default="viewer/reports/navigation.json")
    args = ap.parse_args()

    policy, cfg, blob = load_checkpoint(args.checkpoint)
    print(f"loaded {args.checkpoint} (update {blob.get('update', '?')})\n")
    print("  toward-target share by CURRENT distance. 50% = a coin flip.")
    print("  the 0-3 band reads ~50% for EVERY policy -- an agent on its target")
    print("  can only move away from it. Read the far bands.\n")

    rows: dict[str, dict] = {}
    rows["learned"] = measure(cfg, policy_act_fn(policy), args.episodes, args.seed)
    if args.baselines:
        for kind in ("random", "greedy"):
            rows[kind] = measure(cfg, make_act_fn(kind, cfg, None, args.seed),
                                 args.episodes, args.seed)

    label = {"learned": "learned policy", "random": "random actions (floor)",
             "greedy": "scripted forager (ceiling)"}
    print("  --- toward FOOD ---")
    for k, v in rows.items():
        print(_row(label[k], v["food"]))
    if any(any(b["n"] >= 200 for b in v["shelter"]) for v in rows.values()):
        print("\n  --- toward SHELTER, at night ---")
        for k, v in rows.items():
            print(_row(label[k], v["shelter"]))

    # Could the policy SEE the bush it is being scored against? The observation
    # carries the k nearest bushes whether or not they hold berries, so in a
    # scarce world the nearest berry-bearing one can rank outside that set and be
    # absent altogether -- and the +-1 offset clip costs resolution beyond
    # distance_scale. A flat band with a visible, unclipped target is a flat band
    # that perception does not explain.
    print(f"\n  --- can the learned policy SEE the food? (k_bushes="
          f"{cfg.observation.k_bushes}, distance_scale={cfg.observation.distance_scale:g}) ---")
    for b in rows["learned"]["vision"]:
        if b["n"] == 0:
            continue
        span = f"{b['lo']:.0f}+" if b["hi"] > 1e8 else f"{b['lo']:.0f}-{b['hi']:.0f}"
        vis = "  --  " if b["n_visible"] == 0 else f"{b['toward_visible']:5.1%}"
        uns = "  --  " if b["n_unseen"] == 0 else f"{b['toward_unseen']:5.1%}"
        print(f"  {span:>6}: target in the observation {b['visible']:5.1%}   "
              f"offset clipped {b['sat_one_axis']:5.1%} one axis / "
              f"{b['sat_both_axes']:5.1%} both   "
              f"toward when seen {vis} (n={b['n_visible']:>5}) / "
              f"unseen {uns} (n={b['n_unseen']:>5})")

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "bands": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
