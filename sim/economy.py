"""Subsistence arithmetic for a config, computed rather than estimated.

This exists because of the M3 sizing mistake. `scarce.yaml` was sized by hand
from `eat_restore / drain_per_tick`, giving 6 meals per agent, and the true
figure is 8 -- an agent eats at the *threshold*, not at empty, so a meal only
buys `(threshold + restore - threshold) / drain` ticks rather than a full tank.
The M3 world landed at exactly 100% of subsistence by accident, which made it a
knife-edge testbed nobody intended. CLAUDE.md's standing rule from that
postmortem is "recompute demand with `sim` rather than by hand"; this is the
tool that makes obeying it a one-liner.

    python -m sim.economy --config config/island2/society100.yaml

Nights are the part that is easy to get wrong twice: an exposed agent drains at
`night_drain_multiplier` for `night_fraction` of every cycle, so its demand is
strictly higher than a sheltered one's. Both figures are printed, because a
world should sit between them -- shelter is then load-bearing by arithmetic, as
the M4 economy note puts it, rather than by decoration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .config import Config, load_config


@dataclass(frozen=True)
class Economy:
    """Berries demanded and supplied over one full episode."""

    ticks: int
    num_agents: int
    ticks_per_meal: float
    meals_sheltered: float
    meals_exposed: float
    demand_sheltered: float
    demand_exposed: float
    supply: float
    bushes: int

    @property
    def ratio_sheltered(self) -> float:
        return self.supply / max(self.demand_sheltered, 1e-9)

    @property
    def ratio_exposed(self) -> float:
        return self.supply / max(self.demand_exposed, 1e-9)

    def report(self) -> str:
        lines = [
            f"episode: {self.ticks} ticks, {self.num_agents} agents",
            f"one meal buys {self.ticks_per_meal:.1f} ticks of life",
            "",
            f"demand, everyone sheltered  {self.demand_sheltered:8.1f} berries"
            f"   ({self.meals_sheltered:.1f} meals/agent)",
            f"demand, everyone exposed    {self.demand_exposed:8.1f} berries"
            f"   ({self.meals_exposed:.1f} meals/agent)",
            f"supply ({self.bushes} bushes)        {self.supply:8.1f} berries",
            "",
            f"supply / sheltered demand   {self.ratio_sheltered:8.2f}x",
            f"supply / exposed demand     {self.ratio_exposed:8.2f}x",
        ]
        if self.ratio_sheltered < 1.0:
            lines.append("\nWARNING: even a fully sheltered population starves. "
                         "Nothing behavioural can be read off this world.")
        elif self.ratio_exposed >= 1.0:
            lines.append("\nNOTE: even an exposed population is fed, so shelter is "
                         "not load-bearing here and construction pays nothing.")
        return "\n".join(lines)


def subsistence(cfg: Config) -> Economy:
    """Demand and supply in berries for one episode of ``cfg``.

    Demand: an agent eats when hunger falls below ``eat_threshold`` and the meal
    restores ``eat_restore``, capped at ``hunger.max``. So the sustainable cycle
    is threshold -> threshold + restore -> threshold, i.e. ``restore`` points of
    hunger per meal, NOT a full tank. Dividing by the drain gives ticks per meal.

    Supply: every bush yields its initial berries plus one per completed regrowth
    interval. Regrowth is per bush and only runs below capacity, so this is an
    upper bound -- a bush nobody harvests stops at capacity and produces nothing
    further. Read it as "the most the island can give", which is how the M3 and
    M4 notes both use it.
    """
    h, b, w = cfg.hunger, cfg.bushes, cfg.world
    ticks = w.max_ticks
    drain = h.drain_per_tick

    # Night makes the average drain higher for anyone outside a shelter.
    if cfg.construction.enabled:
        night_share = cfg.construction.night_fraction
        exposed_drain = drain * (1.0 - night_share
                                 + night_share * cfg.construction.night_drain_multiplier)
    else:
        exposed_drain = drain

    ticks_per_meal = h.eat_restore / drain
    meals_sheltered = ticks / ticks_per_meal
    meals_exposed = ticks / (h.eat_restore / exposed_drain)

    bushes = w.num_agents and b.count
    regrowths = ticks // max(b.regrow_ticks, 1)
    # A bush cannot hold more than `capacity`, so its initial load is capped too.
    supply = bushes * (min(b.initial_berries, b.capacity) + regrowths)

    return Economy(
        ticks=ticks,
        num_agents=w.num_agents,
        ticks_per_meal=ticks_per_meal,
        meals_sheltered=meals_sheltered,
        meals_exposed=meals_exposed,
        demand_sheltered=meals_sheltered * w.num_agents,
        demand_exposed=meals_exposed * w.num_agents,
        supply=float(supply),
        bushes=bushes,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--target", type=float, default=None,
                    help="desired supply/sheltered-demand ratio; prints the bush "
                         "count that hits it")
    args = ap.parse_args()

    cfg = load_config(args.config)
    econ = subsistence(cfg)
    print(econ.report())

    if args.target is not None:
        per_bush = econ.supply / max(econ.bushes, 1)
        needed = args.target * econ.demand_sheltered / max(per_bush, 1e-9)
        print(f"\nfor {args.target:.2f}x sheltered subsistence: {needed:.1f} bushes "
              f"({per_bush:.1f} berries each), i.e. "
              f"{needed / max(cfg.bushes.num_clusters, 1):.1f} per cluster "
              f"across {cfg.bushes.num_clusters} clusters")


if __name__ == "__main__":
    main()
