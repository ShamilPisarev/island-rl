"""The transfer ledger and what the economy did with it (Milestone 5).

The brief asks for every transfer to be logged so the economics can be analysed
after the fact. So this module writes two artefacts from a block of evaluation
episodes:

* ``viewer/reports/transfers.jsonl`` -- one JSON object per transfer,
  ``{episode, tick, giver, receiver, item}``. The raw ledger, deliberately not a
  metrics column: a per-episode average of "gifts" cannot answer who gave to
  whom, in what direction, or whether it came back, and those are the questions
  that separate trade from noise.
* ``viewer/reports/exchange.json`` -- the aggregate report ``viewer/exchange.html``
  renders (flow matrix, per-agent roles, reciprocity, utilisation).

    python -m sim.exchange --checkpoint checkpoints/m5/latest.pt

What it measures, beyond counting:

* **flow matrix** -- who gave to whom. A trade economy has structure here; noise
  is uniform.
* **reciprocity** -- ``sum_ij min(F_ij, F_ji) / sum_ij F_ij``. The share of gifts
  that were matched by a gift back along the same edge: 0 is pure one-way flow
  (charity, or a relay), 1 is perfectly balanced pairs. Note that a *high* score
  is not automatically good news -- two agents passing one berry back and forth
  score 1.0, and that is a reward farm rather than a trade (see the M5 shaping
  ablation, which predicted exactly that).
* **utilisation** -- the share of gifts the receiver appeared to *use*: for food,
  ate within ``--use-window`` ticks; for material, delivered to a site within the
  same window. This is an upper bound, not an attribution -- the receiver may
  have eaten a berry it gathered itself -- so read a low number as damning and a
  high one as merely consistent with the gift mattering.
* **role split** -- what each agent produced against what it handed over.
  Specialisation-plus-exchange looks like one agent harvesting material it never
  delivers and another delivering material it never harvested.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .agents import (GIVE_FOOD, GIVE_MATERIAL, ITEM_FOOD, ITEM_NAMES,
                     action_names, num_actions)
from .config import Config, load_config
from .replay import agent_color
from .world import World

REPORT_SCHEMA = 1


@dataclass
class AgentExchange:
    """Per-agent summary over the whole evaluation block."""

    agent: int
    lifespan: float
    ticks_alive: int
    given: int = 0
    received: int = 0
    given_food: int = 0
    given_material: int = 0
    received_food: int = 0
    received_material: int = 0
    net: int = 0                      # given - received; +ve is a net donor
    give_attempts: int = 0
    give_success_rate: float = 0.0
    give_share: float = 0.0           # share of living ticks spent giving
    # production, for the specialisation question
    berries_gathered: int = 0
    materials_harvested: int = 0
    deliveries: int = 0
    harvest_share: float = 0.0        # of all material harvested by anyone
    delivery_share: float = 0.0       # of all deliveries made by anyone
    action_mix: dict[str, float] = field(default_factory=dict)


def _reciprocity(flow: np.ndarray) -> float:
    total = float(flow.sum())
    if total <= 0:
        return 0.0
    return float(np.minimum(flow, flow.T).sum() / total)


def analyse(cfg: Config, act_fn: Callable[..., np.ndarray], episodes: int, seed: int,
            use_window: int = 50) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run ``episodes`` episodes; return (report, raw transfer records)."""
    if not cfg.exchange.enabled:
        raise ValueError("this world has exchange disabled; there is nothing to analyse")
    cfg = cfg.replace(**{"exchange.log_transfers": True})

    n = cfg.world.num_agents
    names = action_names(cfg)
    counts = np.zeros((n, num_actions(cfg)), dtype=np.int64)
    flow = np.zeros((n, n), dtype=np.int64)
    flow_food = np.zeros((n, n), dtype=np.int64)
    given_item = np.zeros((n, 3), dtype=np.int64)
    received_item = np.zeros((n, 3), dtype=np.int64)
    gathered = np.zeros(n, dtype=np.int64)
    harvested = np.zeros(n, dtype=np.int64)
    deliveries = np.zeros(n, dtype=np.int64)
    ticks_alive = np.zeros(n, dtype=np.int64)
    lifespans = np.zeros(n)
    gifts_used = np.zeros(2, dtype=np.int64)      # [food, material]
    gifts_total = np.zeros(2, dtype=np.int64)
    records: list[dict[str, Any]] = []

    for e in range(episodes):
        world = World(cfg, seed=seed + e)
        obs = world.observations()
        # Gifts waiting to be "used": the tick each arrived, per receiver. A
        # queue rather than a count so a gift that sat unused past the window
        # can be retired instead of being redeemed by a much later meal.
        pending: list[list[deque]] = [[deque(), deque()] for _ in range(n)]

        for _ in range(cfg.world.max_ticks):
            alive = world.pool.alive.copy()
            actions = np.asarray(act_fn(obs, world.action_mask())).reshape(n)
            for i in np.flatnonzero(alive):
                counts[i, actions[i]] += 1
                ticks_alive[i] += 1

            result = world.step(actions)
            tick = world.tick
            gathered += result.gathered
            if result.built.size:
                deliveries += result.built
                harvested += result.harvested
            for g, r, item in result.transfers:
                kind = 0 if item == ITEM_FOOD else 1
                flow[g, r] += 1
                if kind == 0:
                    flow_food[g, r] += 1
                given_item[g, item] += 1
                received_item[r, item] += 1
                gifts_total[kind] += 1
                pending[r][kind].append(tick)
                records.append({"episode": e, "tick": int(tick), "giver": int(g),
                                "receiver": int(r), "item": ITEM_NAMES[item]})

            # Did the receiver do the thing the gift enables? Eating for food,
            # delivering for material. Upper bound: the agent may well have been
            # using its own stock, so this cannot attribute, only bound.
            for i in range(n):
                for kind, did in ((0, bool(result.ate[i])),
                                  (1, bool(result.built.size and result.built[i]))):
                    queue = pending[i][kind]
                    while queue and tick - queue[0] > use_window:
                        queue.popleft()
                    if did and queue:
                        queue.popleft()
                        gifts_used[kind] += 1

            obs = result.obs
            if result.episode_done:
                break

        lifespans += world.alive_ticks

    total_harvest = max(int(harvested.sum()), 1)
    total_delivery = max(int(deliveries.sum()), 1)
    agents = []
    for i in range(n):
        total = max(int(counts[i].sum()), 1)
        attempts = int(counts[i, GIVE_FOOD] + counts[i, GIVE_MATERIAL])
        given = int(given_item[i].sum())
        received = int(received_item[i].sum())
        agents.append(AgentExchange(
            agent=i,
            lifespan=float(lifespans[i] / episodes),
            ticks_alive=int(ticks_alive[i]),
            given=given,
            received=received,
            given_food=int(given_item[i, ITEM_FOOD]),
            given_material=int(given_item[i, 1:].sum()),
            received_food=int(received_item[i, ITEM_FOOD]),
            received_material=int(received_item[i, 1:].sum()),
            net=given - received,
            give_attempts=attempts,
            give_success_rate=(given / attempts) if attempts else 0.0,
            give_share=attempts / total,
            berries_gathered=int(gathered[i]),
            materials_harvested=int(harvested[i]),
            deliveries=int(deliveries[i]),
            harvest_share=float(harvested[i] / total_harvest),
            delivery_share=float(deliveries[i] / total_delivery),
            action_mix={name: float(counts[i, k]) / total for k, name in enumerate(names)},
        ))

    report = {
        "schema_version": REPORT_SCHEMA,
        "episodes": episodes,
        "seed": seed,
        "num_agents": n,
        "use_window": use_window,
        "action_names": list(names),
        "item_names": list(ITEM_NAMES),
        "agent_colors": [agent_color(i, n) for i in range(n)],
        "reward_give": cfg.reward.give,
        "totals": {
            "transfers": int(flow.sum()),
            "transfers_per_episode": float(flow.sum() / max(episodes, 1)),
            "food": int(gifts_total[0]),
            "material": int(gifts_total[1]),
            "food_used": int(gifts_used[0]),
            "material_used": int(gifts_used[1]),
            "food_utilisation": float(gifts_used[0] / max(gifts_total[0], 1)),
            "material_utilisation": float(gifts_used[1] / max(gifts_total[1], 1)),
        },
        "reciprocity": _reciprocity(flow),
        "flow": [[int(v) for v in row] for row in flow],
        "flow_food": [[int(v) for v in row] for row in flow_food],
        "agents": [asdict(a) for a in agents],
    }
    return report, records


def summarise(report: dict[str, Any]) -> str:
    lines = [f"{report['episodes']} episodes, seed {report['seed']}, "
             f"reward.give = {report['reward_give']}", ""]
    header = (f"{'agent':>5} {'lifespan':>9} {'gave':>6} {'got':>6} {'net':>6} "
              f"{'give%':>7} {'hit%':>6} {'berries':>8} {'harvest':>8} {'deliver':>8}")
    lines.append(header)
    lines.append("-" * len(header))
    for a in report["agents"]:
        lines.append(
            f"{a['agent']:>5} {a['lifespan']:>9.1f} {a['given']:>6d} {a['received']:>6d} "
            f"{a['net']:>6d} {a['give_share'] * 100:>7.2f} "
            f"{a['give_success_rate'] * 100:>6.1f} {a['berries_gathered']:>8d} "
            f"{a['materials_harvested']:>8d} {a['deliveries']:>8d}"
        )
    lines.append("-" * len(header))

    t = report["totals"]
    lines.append(f"transfers: {t['transfers']} total, {t['transfers_per_episode']:.1f} per episode "
                 f"({t['food']} food, {t['material']} material)")
    lines.append(f"utilisation (upper bound, {report['use_window']}-tick window): "
                 f"food {t['food_utilisation'] * 100:.0f}%, "
                 f"material {t['material_utilisation'] * 100:.0f}%")
    lines.append(f"reciprocity: {report['reciprocity']:.3f}  "
                 f"(0 = one-way flow, 1 = every gift matched back along the same edge)")

    lines.append("")
    lines.append("flow (row gave to column):")
    n = report["num_agents"]
    lines.append("      " + " ".join(f"{j:>5d}" for j in range(n)))
    for i, row in enumerate(report["flow"]):
        lines.append(f"{i:>5d} " + " ".join(f"{v:>5d}" for v in row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--policy",
                        choices=["learned", "random", "greedy", "thief", "builder", "trader"],
                        default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-window", type=int, default=50,
                        help="ticks a gift has to be used in to count as used")
    parser.add_argument("--out", default="viewer/reports/exchange.json")
    parser.add_argument("--jsonl", default="viewer/reports/transfers.jsonl",
                        help="raw transfer ledger, one JSON object per line")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    # Imported here rather than at module scope, for the same reason divergence.py
    # does it: evaluate imports nothing from here and a cycle is easy to create.
    from .evaluate import load_checkpoint, make_act_fn

    policy = None
    if args.checkpoint:
        policy, cfg, blob = load_checkpoint(args.checkpoint, device=args.device)
        print(f"loaded {args.checkpoint} "
              f"({blob['policy_config'].get('mode', 'shared')}, update {blob.get('update', '?')})")
    else:
        cfg = load_config(args.config)

    kind = args.policy or ("learned" if policy is not None else "trader")
    act_fn = make_act_fn(kind, cfg, policy, args.seed, device=args.device)

    report, records = analyse(cfg, act_fn, args.episodes, args.seed, args.use_window)
    report["label"] = args.label or (Path(args.checkpoint).parent.name if args.checkpoint else kind)
    report["policy"] = kind
    report["policy_mode"] = policy.config_dict()["mode"] if policy is not None else kind

    print()
    print(summarise(report))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, separators=(",", ":"))
    ledger = Path(args.jsonl)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "w") as fh:
        for record in records:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"\nwrote {out}\nwrote {ledger} ({len(records)} transfers)")


if __name__ == "__main__":
    main()
