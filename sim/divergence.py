"""Behavioural divergence between agents (Milestone 2, extended for 3).

Splitting the shared brain into six is only interesting if the six then do
different things. This module measures whether they do, over a block of
evaluation episodes, and writes a JSON report the viewer renders.

    python -m sim.divergence --checkpoint checkpoints/m2/latest.pt

What it measures, per agent:

* **action mix** -- share of ticks spent gathering, travelling, or idle
* **gather success rate** -- how often a `gather` action actually took a berry,
  which separates "forages well" from "mashes the gather button"
* **foraging proximity** -- mean distance to the nearest bush, and share of ticks
  spent within gathering range
* **territory** -- a 2D histogram of where the agent spent its time
* **theft** (Milestone 3, zero when competition is off) -- steal attempts and
  successes, times robbed by others, and gather attempts lost to a closer agent

and pairwise Jensen-Shannon divergence between agents on the action and
territory distributions.

READ THE ACTION MATRIX, NOT THE TERRITORY MATRIX, as the specialisation signal.
Six agents *sharing one brain* already score ~0.67 bits of territory divergence,
because they spawn apart and each walks to whichever cluster is nearest --
different ground is the default, not a finding. Action divergence is near zero
under sharing, so it is the number that has to clear its control. Always run the
shared checkpoint through this tool too.

A NOTE ON TERRITORY. It is only meaningful on a fixed map. With
``bushes.resample_each_episode`` on (the training default) every episode puts the
clusters somewhere new, so averaging positions in absolute coordinates smears
every agent towards the same centred blob and the heatmaps say nothing. So this
tool pins the layout by default (``--fixed-map``); the report records which mode
produced it, and the viewer says so. The map-independent statistics above are
unaffected either way.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .agents import GATHER, IDLE, N_MOVE_ACTIONS, STEAL, action_names, num_actions
from .config import Config, load_config
from .replay import agent_color
from .world import World

TERRITORY_BINS = 24


@dataclass
class AgentBehaviour:
    """Per-agent summary over the whole evaluation block."""

    agent: int
    ticks_alive: int
    lifespan: float
    deaths: int
    action_counts: list[int] = field(default_factory=list)
    action_mix: dict[str, float] = field(default_factory=dict)
    gather_share: float = 0.0
    travel_share: float = 0.0
    idle_share: float = 0.0
    gather_attempts: int = 0
    gather_successes: int = 0
    gather_success_rate: float = 0.0
    mean_dist_to_nearest_bush: float = 0.0
    time_in_gather_range: float = 0.0
    mean_radius: float = 0.0
    # Milestone 3. Zero throughout when competition is disabled.
    steal_share: float = 0.0
    steal_attempts: int = 0
    steal_successes: int = 0
    steal_success_rate: float = 0.0
    times_robbed: int = 0
    contests_lost: int = 0
    territory: list[list[float]] = field(default_factory=list)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits: 0 = identical, 1 = disjoint support.

    Symmetric and always finite, unlike KL, which matters here because agents
    routinely put zero mass where another puts some (an agent that never visits
    the north shore would make KL infinite).
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    p = p / p.sum() if p.sum() > 0 else np.full_like(p, 1.0 / p.size)
    q = q / q.sum() if q.sum() > 0 else np.full_like(q, 1.0 / q.size)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def pairwise_divergence(distributions: list[np.ndarray]) -> list[list[float]]:
    n = len(distributions)
    return [[_js_divergence(distributions[i], distributions[j]) for j in range(n)]
            for i in range(n)]


def analyse(cfg: Config, act_fn: Callable[[np.ndarray], np.ndarray],
            episodes: int, seed: int, fixed_map: bool = True) -> dict[str, Any]:
    """Run ``episodes`` episodes and accumulate per-agent behaviour."""
    if fixed_map:
        cfg = cfg.replace(**{"bushes.resample_each_episode": False})

    n = cfg.world.num_agents
    radius = cfg.world.island_radius
    edges = np.linspace(-radius, radius, TERRITORY_BINS + 1)

    names = action_names(cfg)
    action_counts = np.zeros((n, num_actions(cfg)), dtype=np.int64)
    gather_success = np.zeros(n, dtype=np.int64)
    steal_success = np.zeros(n, dtype=np.int64)
    times_robbed = np.zeros(n, dtype=np.int64)
    contests_lost = np.zeros(n, dtype=np.int64)
    territory = np.zeros((n, TERRITORY_BINS, TERRITORY_BINS), dtype=np.float64)
    dist_sum = np.zeros(n)
    in_range = np.zeros(n, dtype=np.int64)
    radius_sum = np.zeros(n)
    ticks_alive = np.zeros(n, dtype=np.int64)
    lifespans = np.zeros(n)
    deaths = np.zeros(n, dtype=np.int64)

    bush_layout: list[dict[str, float]] = []
    for e in range(episodes):
        # A fixed map still needs one shared seed across episodes, or "fixed"
        # only means "fixed within an episode".
        world = World(cfg, seed=seed if fixed_map else seed + e)
        if not bush_layout and fixed_map:
            bush_layout = [{"x": float(x), "z": float(z)}
                           for x, z in zip(world.bush_x, world.bush_z)]
        obs = world.observations()
        for _ in range(cfg.world.max_ticks):
            alive = world.pool.alive.copy()
            actions = np.asarray(act_fn(obs)).reshape(n)

            x, z = world.pool.x.copy(), world.pool.z.copy()
            d = np.hypot(world.bush_x[None, :] - x[:, None],
                         world.bush_z[None, :] - z[:, None]).min(axis=1)

            for i in np.flatnonzero(alive):
                action_counts[i, actions[i]] += 1
                dist_sum[i] += d[i]
                in_range[i] += int(d[i] <= cfg.bushes.gather_radius)
                radius_sum[i] += float(np.hypot(x[i], z[i]))
                bx = np.clip(np.searchsorted(edges, x[i], side="right") - 1, 0, TERRITORY_BINS - 1)
                bz = np.clip(np.searchsorted(edges, z[i], side="right") - 1, 0, TERRITORY_BINS - 1)
                territory[i, bz, bx] += 1.0
                ticks_alive[i] += 1

            result = world.step(actions)
            gather_success += result.gathered
            if result.stole.size:
                steal_success += result.stole
                times_robbed += result.robbed
            if result.contested.size:
                contests_lost += result.contested
            obs = result.obs
            if result.episode_done:
                break

        lifespans += world.alive_ticks
        deaths += (~world.pool.alive).astype(np.int64)

    behaviours = []
    for i in range(n):
        total = max(int(action_counts[i].sum()), 1)
        moves = int(action_counts[i, :N_MOVE_ACTIONS].sum())
        attempts = int(action_counts[i, GATHER])
        steal_attempts = int(action_counts[i, STEAL]) if action_counts.shape[1] > STEAL else 0
        heat = territory[i] / max(territory[i].sum(), 1.0)
        behaviours.append(AgentBehaviour(
            agent=i,
            ticks_alive=int(ticks_alive[i]),
            lifespan=float(lifespans[i] / episodes),
            deaths=int(deaths[i]),
            action_counts=[int(v) for v in action_counts[i]],
            action_mix={name: float(action_counts[i, k]) / total
                        for k, name in enumerate(names)},
            gather_share=attempts / total,
            travel_share=moves / total,
            idle_share=float(action_counts[i, IDLE]) / total,
            gather_attempts=attempts,
            gather_successes=int(gather_success[i]),
            gather_success_rate=(gather_success[i] / attempts) if attempts else 0.0,
            mean_dist_to_nearest_bush=float(dist_sum[i] / max(ticks_alive[i], 1)),
            time_in_gather_range=float(in_range[i] / max(ticks_alive[i], 1)),
            mean_radius=float(radius_sum[i] / max(ticks_alive[i], 1)),
            steal_share=steal_attempts / total,
            steal_attempts=steal_attempts,
            steal_successes=int(steal_success[i]),
            steal_success_rate=(steal_success[i] / steal_attempts) if steal_attempts else 0.0,
            times_robbed=int(times_robbed[i]),
            contests_lost=int(contests_lost[i]),
            territory=[[float(v) for v in row] for row in heat],
        ))

    action_dists = [action_counts[i].astype(np.float64) for i in range(n)]
    territory_dists = [territory[i].ravel() for i in range(n)]

    return {
        "schema_version": 1,
        "episodes": episodes,
        "seed": seed,
        "fixed_map": fixed_map,
        "competition": {
            "contest_bushes": cfg.competition.contest_bushes,
            "enable_steal": cfg.competition.enable_steal,
        },
        "island_radius": cfg.world.island_radius,
        "territory_bins": TERRITORY_BINS,
        "action_names": list(names),
        "num_agents": n,
        # Same colours the replay viewer assigns, so an agent looks like itself
        # in both views.
        "agent_colors": [agent_color(i, n) for i in range(n)],
        "agents": [asdict(b) for b in behaviours],
        "action_divergence": pairwise_divergence(action_dists),
        "territory_divergence": pairwise_divergence(territory_dists),
        "bushes": bush_layout,
    }


def summarise(report: dict[str, Any]) -> str:
    """The console table. Reads top to bottom as "are these agents different?"."""
    lines = []
    lines.append(f"{report['episodes']} episodes, "
                 f"{'fixed' if report['fixed_map'] else 'resampled'} map, "
                 f"seed {report['seed']}")
    lines.append("")
    header = (f"{'agent':>5} {'lifespan':>9} {'gather%':>8} {'travel%':>8} {'idle%':>7} "
              f"{'hit rate':>9} {'bush dist':>10} {'in range%':>10} {'radius':>7}")
    lines.append(header)
    lines.append("-" * len(header))
    for a in report["agents"]:
        lines.append(
            f"{a['agent']:>5} {a['lifespan']:>9.1f} {a['gather_share'] * 100:>8.1f} "
            f"{a['travel_share'] * 100:>8.1f} {a['idle_share'] * 100:>7.1f} "
            f"{a['gather_success_rate'] * 100:>9.1f} {a['mean_dist_to_nearest_bush']:>10.2f} "
            f"{a['time_in_gather_range'] * 100:>10.1f} {a['mean_radius']:>7.1f}"
        )
    lines.append("-" * len(header))

    action_div = np.array(report["action_divergence"])
    terr_div = np.array(report["territory_divergence"])
    off = ~np.eye(action_div.shape[0], dtype=bool)
    lines.append(
        f"mean pairwise JS divergence — actions {action_div[off].mean():.4f} bits, "
        f"territory {terr_div[off].mean():.4f} bits  (0 = identical)"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--policy", choices=["learned", "random", "greedy", "thief"], default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="viewer/reports/divergence.json")
    parser.add_argument("--label", default=None)
    parser.add_argument("--fixed-map", dest="fixed_map", action="store_true", default=True)
    parser.add_argument("--no-fixed-map", dest="fixed_map", action="store_false",
                        help="resample bushes each episode; territory heatmaps become meaningless")
    args = parser.parse_args()

    # Imported here rather than at module scope: evaluate imports nothing from
    # this module, and keeping it that way avoids a cycle.
    from .evaluate import load_checkpoint, make_act_fn

    policy = None
    if args.checkpoint:
        policy, cfg, blob = load_checkpoint(args.checkpoint, device=args.device)
        print(f"loaded {args.checkpoint} "
              f"({blob['policy_config'].get('mode', 'shared')}, update {blob.get('update', '?')})")
    else:
        cfg = load_config(args.config)

    kind = args.policy or ("learned" if policy is not None else "greedy")
    act_fn = make_act_fn(kind, cfg, policy, args.seed, device=args.device)

    report = analyse(cfg, act_fn, args.episodes, args.seed, fixed_map=args.fixed_map)
    report["label"] = args.label or (Path(args.checkpoint).parent.name if args.checkpoint else kind)
    report["policy"] = kind
    report["policy_mode"] = (policy.config_dict()["mode"] if policy is not None else kind)

    print()
    print(summarise(report))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, separators=(",", ":"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
