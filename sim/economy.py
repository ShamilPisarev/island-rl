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
    # Island 2.0 stage 4: regrowth the island loses to blights. Reported
    # separately because `supply` is already net of it and a reader who wants to
    # know why a world got poorer cannot recover it from one number.
    blight_ticks: float = 0.0
    blight_loss: float = 0.0

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
        ]
        if self.blight_loss > 0.0:
            lines += [
                f"  ...after {self.blight_ticks:.0f} blighted ticks cost it "
                f"{self.blight_loss:.1f}",
            ]
        lines += [
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
    # A BLIGHT IS A CUT IN SUPPLY, so the sizing has to know about it -- CLAUDE.md
    # rule 5, applied to the mechanic that most obviously invalidates a previous
    # sizing. Shocks alternate blight/storm from the seed, so in expectation half
    # of them are blights, and each suspends regrowth (and its timer) for
    # `blight_ticks`. Overlapping blights are not modelled: at the shipped
    # interval they cannot overlap, and if they could this would read as an upper
    # bound on the loss, which is the safe direction.
    sc = cfg.society
    blight_ticks = 0.0
    if sc.enabled and sc.shock_interval > 0:
        # With the ramp, a shock at tick t lasts blight_ticks * (1 + ramp*t/T),
        # so the expectation is summed shock by shock rather than multiplied
        # once -- at ramp 2.0 the late blights dominate the loss.
        expected = sum(
            0.5 * sc.blight_ticks * (1.0 + sc.shock_ramp * t / ticks)
            for t in range(sc.shock_interval, ticks + 1, sc.shock_interval))
        blight_ticks = min(expected, float(ticks))
    growing = max(ticks - blight_ticks, 0.0)
    regrowths = growing // max(b.regrow_ticks, 1)
    lost = (ticks // max(b.regrow_ticks, 1)) - regrowths
    # A bush cannot hold more than `capacity`, so its initial load is capped too.
    supply = bushes * (min(b.initial_berries, b.capacity) + regrowths)

    return Economy(
        blight_ticks=blight_ticks,
        blight_loss=float(bushes * lost),
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


@dataclass(frozen=True)
class Materials:
    """Wood and stone: what the island holds against what the village needs.

    Rung 1 of the tech ladder is an axe that doubles chop YIELD, and this exists
    because the first thing to establish about it is which of two very different
    things it does. Trees hold a finite stock, so an axe never creates wood:

      * if the material STOCK is the binding constraint, an axe cannot help at
        all -- it empties the same trees sooner and the village still runs out.
      * if LABOUR is the constraint (the stock is slack and the cost of a shelter
        is the trips), an axe halves the trips and is worth real ticks.

    Sizing a tool world without knowing which regime it is in is rule 5's mistake
    with a new mechanic, so this is computed before the config is written rather
    than after the run disappoints.
    """

    trees: int
    rocks: int
    wood_supply: int
    stone_supply: int
    sites: int
    site_cost: int
    # Storms knock finished sites back down, so demand is not just the initial build.
    storm_rebuild: float
    chop_trips_bare: float
    chop_trips_axed: float
    axe_wood_cost: int
    axe_stone_cost: int
    axes_if_everyone: int

    @property
    def demand(self) -> float:
        return self.sites * self.site_cost + self.storm_rebuild

    @property
    def supply(self) -> float:
        return float(self.wood_supply + self.stone_supply)

    @property
    def ratio(self) -> float:
        return self.supply / max(self.demand, 1e-9)

    def report(self) -> str:
        lines = [
            f"material stock  {self.wood_supply:6d} wood ({self.trees} trees)"
            f" + {self.stone_supply:6d} stone ({self.rocks} rocks)"
            f" = {self.supply:.0f}",
            f"material demand {self.demand:6.0f}"
            f"   ({self.sites} sites x {self.site_cost}"
            f" + {self.storm_rebuild:.0f} storm rebuilds)",
            f"supply / demand {self.ratio:6.2f}x",
        ]
        if self.axes_if_everyone:
            axe_bill = self.axes_if_everyone * (self.axe_wood_cost + self.axe_stone_cost)
            lines += [
                "",
                f"an axe costs {self.axe_wood_cost}w + {self.axe_stone_cost}s;"
                f" arming all {self.axes_if_everyone} agents spends {axe_bill}"
                f" ({100.0 * axe_bill / max(self.supply, 1e-9):.1f}% of the island's stock)",
                f"chop trips for the wood bill: {self.chop_trips_bare:.0f} bare"
                f" -> {self.chop_trips_axed:.0f} with an axe",
            ]
            if self.ratio < 1.3:
                lines.append(
                    "\nWARNING: material STOCK is tight, so an axe cannot help -- it "
                    "empties the same trees sooner. A tool world needs slack stock, "
                    "or the thing being measured is scarcity, not technology.")
            else:
                lines.append(
                    "\nOK: stock is slack, so the constraint is LABOUR and an axe "
                    "buys trips. That is the regime rung 1 is asking about.")
        return "\n".join(lines)


def materials(cfg: Config) -> Materials:
    """The material side of the same arithmetic. See `Materials`."""
    cc = cfg.construction
    sc = cfg.society
    tc = cfg.tools
    site_cost = cc.site_wood_cost + cc.site_stone_cost
    # Storms damage every FINISHED site, clamped at a site's cost. In expectation
    # half the shocks are storms; each costs the village `storm_damage * ramp`
    # per finished site, which has to be rebuilt out of the same stock.
    rebuild = 0.0
    if sc.enabled and sc.shock_interval > 0 and cc.enabled:
        ticks = cfg.world.max_ticks
        rebuild = sum(
            0.5 * min(sc.storm_damage * (1.0 + sc.shock_ramp * t / ticks), site_cost)
            * cc.num_sites
            for t in range(sc.shock_interval, ticks + 1, sc.shock_interval))
    wood = cc.num_trees * cc.tree_wood
    per_chop = max(tc.chop_multiplier, 1) if tc.enabled else 1
    wood_bill = float(cc.num_sites * cc.site_wood_cost) + rebuild * 0.5
    return Materials(
        trees=cc.num_trees,
        rocks=cc.num_rocks,
        wood_supply=wood,
        stone_supply=cc.num_rocks * cc.rock_stone,
        sites=cc.num_sites,
        site_cost=site_cost,
        storm_rebuild=rebuild,
        chop_trips_bare=wood_bill,
        chop_trips_axed=wood_bill / per_chop,
        axe_wood_cost=tc.axe_wood_cost,
        axe_stone_cost=tc.axe_stone_cost,
        axes_if_everyone=cfg.world.num_agents if tc.enabled else 0,
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
    if cfg.construction.enabled:
        print()
        print(materials(cfg).report())

    if args.target is not None:
        per_bush = econ.supply / max(econ.bushes, 1)
        needed = args.target * econ.demand_sheltered / max(per_bush, 1e-9)
        print(f"\nfor {args.target:.2f}x sheltered subsistence: {needed:.1f} bushes "
              f"({per_bush:.1f} berries each), i.e. "
              f"{needed / max(cfg.bushes.num_clusters, 1):.1f} per cluster "
              f"across {cfg.bushes.num_clusters} clusters")


if __name__ == "__main__":
    main()
