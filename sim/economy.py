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
    # tech ladder rung 2. `hunted_share` is the share of an exposed night spent
    # inside a predator's reach -- a lower bound, see `subsistence`.
    hunted_share: float = 0.0
    predator_damage: float = 0.0
    # --- Island 3.0. Once the population is born rather than cast, "supply over
    # demand for N agents" stops being the question: N is an outcome. The number
    # that matters is the CARRYING CAPACITY -- how many agents the island's
    # regrowth rate can feed indefinitely -- and it is a rate calculation, not a
    # stock one. Section 15's collapse is exactly what happens when nobody asks
    # it (see `Materials`, where the collapsing resource actually was).
    carrying_sheltered: float = 0.0
    carrying_exposed: float = 0.0
    field_bushes: int = 0
    field_supply_rate: float = 0.0
    child_drain_frac: float = 1.0
    initial_agents: int = 0
    # --- stage 2: the granary. A store is a BUFFER, not supply -- it creates no
    # berries -- so the number that decides whether the technology is worth
    # anything is how long a full larder feeds the household that owns it,
    # against how long the island's worst shock lasts. Sized before the world is
    # written (rule 5), because a granary that bridges a blight and one that does
    # not are two different mechanics wearing one name.
    store_ticks: float = 0.0
    store_ticks_granary: float = 0.0
    worst_blight: float = 0.0

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
        if self.predator_damage > 0.0:
            lines.append(
                f"  ...the exposed figure includes predators: {self.hunted_share:.1%}"
                f" of an exposed night inside reach, at {self.predator_damage} hunger"
                f"/tick (a LOWER bound -- they hunt, they do not patrol)")
        if self.carrying_sheltered > 0.0:
            lines += [
                "",
                "--- Island 3.0: the population is an outcome, so read the RATE ---",
                f"carrying capacity, all sheltered  {self.carrying_sheltered:6.1f} agents",
                f"carrying capacity, all exposed    {self.carrying_exposed:6.1f} agents",
                f"...against {self.initial_agents} at tick 0 and "
                f"{self.num_agents} slots to grow into",
            ]
            if self.field_bushes:
                lines.append(
                    f"...plus {self.field_bushes} field slots at "
                    f"{self.field_supply_rate:.2f} berries/tick if every one is "
                    f"planted (+{self.field_supply_rate / max(1.0 / max(self.ticks_per_meal, 1e-9), 1e-9):.1f} agents)")
            if self.store_ticks_granary > 0.0:
                lines.append(
                    f"...a full larder feeds a FULL HOUSE {self.store_ticks:.0f} "
                    f"ticks, {self.store_ticks_granary:.0f} with a granary, "
                    f"against a worst blight of {self.worst_blight:.0f}"
                    + ("  <- the granary bridges it, the bare store does not"
                       if self.store_ticks < self.worst_blight
                       <= self.store_ticks_granary else ""))
            if self.child_drain_frac < 1.0:
                lines.append(
                    f"...and a child eats {self.child_drain_frac:.0%} of an adult, "
                    "so a growing population's true capacity sits above the "
                    "sheltered figure and falls toward it as the children grow")
            if self.carrying_sheltered < self.initial_agents:
                lines.append(
                    "\nWARNING: the island cannot feed the population it STARTS "
                    "with, so it will shrink whatever anyone does and no "
                    "reproduction result can be read off it.")
        if self.carrying_sheltered > 0.0:
            # The stock ratios above divide by `num_agents`, which in a 3.0 world
            # is a SLOT CAPACITY rather than a population -- so they read as a
            # famine in a world that is comfortably fed, and their warning would
            # be a lie. Rule 6, arriving in a new disguise: a denominator that
            # stopped meaning what it used to. The rate figures replace them.
            lines.append("\n(the two ratios above divide by SLOTS, not by "
                         "agents alive -- in a 3.0 world read the capacities, "
                         "not the ratios)")
        elif self.ratio_sheltered < 1.0:
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

    # A PREDATOR IS A DEMAND-SIDE CHANGE, so the sizing has to know about it --
    # rule 5, applied to rung 2 the way blights forced it on stage 4. An exposed
    # agent that is caught loses `damage` hunger on top of its night drain, so
    # the exposed figure rises and the sheltered one does not move at all (an
    # agent indoors is never hunted, which is the entire point of the mechanic).
    #
    # `hunted_share` is what this cannot know from the config: how much of a
    # night an exposed agent actually spends inside a predator's reach. Modelled
    # as the share of the island a hunting pack can cover, which is a LOWER
    # BOUND on the pressure -- predators walk at the exposed rather than
    # patrolling at random, so the real figure is higher. Read the exposed ratio
    # as optimistic and size with headroom.
    pc = cfg.predators
    hunted_share = 0.0
    if pc.enabled and cfg.construction.enabled:
        reach = pc.count * (pc.attack_radius ** 2)
        hunted_share = min(reach / max(cfg.world.island_radius ** 2, 1e-9), 1.0)
        exposed_drain += cfg.construction.night_fraction * hunted_share * pc.damage

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

    # --- Island 3.0. Supply per tick, not supply per episode: a world whose
    # population grows has no fixed demand to divide by, so the question becomes
    # how many mouths the island's REGROWTH can carry. One bush yields one berry
    # every `regrow_ticks` while below capacity; one agent eats one berry every
    # `ticks_per_meal`. The quotient is the capacity, and everything else in a
    # 3.0 sizing hangs off it.
    # Blights suspend regrowth, so the rate is net of them -- the same deduction
    # `supply` already makes on the stock figure. Leaving it out would have
    # over-stated a shocked island's capacity by a third.
    supply_rate = (bushes / max(b.regrow_ticks, 1)) * (growing / max(ticks, 1))
    ac = cfg.agriculture
    field_bushes = 0
    field_rate = 0.0
    if ac.enabled and sc.enabled:
        field_bushes = max(sc.num_households, 1) * ac.max_fields_per_household
        # A blight is a blight: it suspends a field's regrowth exactly as it
        # suspends a wild bush's (world.py has one regrowth phase, not two), so
        # the same deduction applies. Leaving it out over-states what farming
        # buys, in the one place a world would be sized on it.
        field_rate = (field_bushes / max(ac.field_regrow_ticks, 1)) * (growing / max(ticks, 1))
    eat_rate = 1.0 / max(ticks_per_meal, 1e-9)
    eat_rate_exposed = exposed_drain / max(h.eat_restore, 1e-9)
    rc = cfg.reproduction
    carrying_s = carrying_e = 0.0
    if rc.enabled:
        carrying_s = supply_rate / max(eat_rate, 1e-9)
        carrying_e = supply_rate / max(eat_rate_exposed, 1e-9)

    # A full larder against a household's own demand. `initial_agents` per
    # household is the founding size, which understates a grown household -- so
    # this is an UPPER bound on how long the store lasts, and it is the
    # optimistic direction, which is what a sizing check wants to be honest about.
    store_ticks = store_ticks_g = 0.0
    worst_blight = 0.0
    if rc.enabled and sc.enabled:
        # AGAINST A FULL HOUSE, not a founding pair. The first version of this
        # divided by `initial_agents / num_households` (2 here) and reported a
        # larder that lasts 420 ticks against a 60-tick blight -- so a granary
        # could never be worth anything and the read it was built for would have
        # measured noise. A household grows to its house's capacity, which is
        # what it eats at for most of a long run, and that is the denominator
        # this figure is about.
        hc = cfg.housing
        full_house = (hc.base_occupants + hc.max_rooms * hc.occupants_per_room
                      if hc.enabled else max(
                          w.num_agents / max(sc.num_households, 1), 1.0))
        household_eat = max(float(full_house), 1.0) * eat_rate
        store_ticks = sc.stockpile_food_capacity / max(household_eat, 1e-9)
        mult = cfg.tech.granary_multiplier if cfg.tech.enabled else 1
        store_ticks_g = store_ticks * mult
        if sc.shock_interval > 0:
            worst_blight = sc.blight_ticks * (1.0 + sc.shock_ramp)

    return Economy(
        store_ticks=store_ticks,
        store_ticks_granary=store_ticks_g,
        worst_blight=worst_blight,
        carrying_sheltered=carrying_s,
        carrying_exposed=carrying_e,
        field_bushes=field_bushes,
        field_supply_rate=field_rate,
        child_drain_frac=rc.child_drain_frac if rc.enabled else 1.0,
        initial_agents=(rc.initial_agents if (rc.enabled and rc.initial_agents)
                        else w.num_agents),
        hunted_share=hunted_share,
        predator_damage=pc.damage if pc.enabled else 0.0,
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
    # --- Island 3.0
    wood_regrow_ticks: int = 0
    rock_regrow_ticks: int = 0
    wood_regrown: float = 0.0
    stone_regrown: float = 0.0
    expansion_demand: float = 0.0
    field_demand: float = 0.0
    material_rate: float = 0.0     # units per tick the island puts back
    build_rate_needed: float = 0.0  # units per tick storms alone consume

    @property
    def demand(self) -> float:
        return (self.sites * self.site_cost + self.storm_rebuild
                + self.expansion_demand + self.field_demand)

    @property
    def supply(self) -> float:
        return float(self.wood_supply + self.stone_supply
                     + self.wood_regrown + self.stone_regrown)

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
            f" + {self.storm_rebuild:.0f} storm rebuilds"
            + (f" + {self.expansion_demand:.0f} rooms" if self.expansion_demand else "")
            + (f" + {self.field_demand:.0f} fields" if self.field_demand else "")
            + ")",
            f"supply / demand {self.ratio:6.2f}x",
        ]
        if self.wood_regrow_ticks or self.rock_regrow_ticks:
            # THE NUMBER SECTION 15 NEEDED AND NOBODY COMPUTED. Trees and rocks
            # never regrew, so a village whose storms keep levelling it consumes
            # material at a rate the island never replaces -- and over 600 ticks
            # that is invisible, because the initial stock covers it. Over 24,000
            # it is the whole story. These two rates are what decide whether a
            # world collapses or settles, and a stock ratio cannot say.
            lines += [
                "",
                f"...regrowth adds {self.wood_regrown:.0f} wood + "
                f"{self.stone_regrown:.0f} stone over the episode",
                f"island replaces  {self.material_rate:.3f} units/tick",
                f"storms consume   {self.build_rate_needed:.3f} units/tick",
            ]
            if self.material_rate < self.build_rate_needed:
                lines.append(
                    "WARNING: the village is consuming material faster than the "
                    "island replaces it. That is a collapse on a long enough run "
                    "(ISLAND2_DESIGN.md section 15), whatever a 600-tick episode says.")
            else:
                lines.append(
                    "OK: regrowth outpaces storm damage, so the material economy "
                    "is sustainable and a long run measures carrying capacity "
                    "rather than a countdown.")
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
    # --- Island 3.0. Regrowth is a RATE, and it is the rate that decides whether
    # a long run settles or collapses. An upper bound, like the berry supply: a
    # node stops at the stock it started with, so a forest nobody chops replaces
    # nothing. Read it as "the most the island can put back".
    ticks_total = cfg.world.max_ticks
    wood_regrown = stone_regrown = 0.0
    if cc.enabled and cc.tree_regrow_ticks > 0:
        wood_regrown = cc.num_trees * (ticks_total / cc.tree_regrow_ticks)
    if cc.enabled and cc.rock_regrow_ticks > 0:
        stone_regrown = cc.num_rocks * (ticks_total / cc.rock_regrow_ticks)
    material_rate = ((cc.num_trees / cc.tree_regrow_ticks if cc.tree_regrow_ticks else 0.0)
                     + (cc.num_rocks / cc.rock_regrow_ticks if cc.rock_regrow_ticks else 0.0))
    build_rate = rebuild / max(ticks_total, 1)
    hc = cfg.housing
    ac = cfg.agriculture
    expansion = (cc.num_sites * hc.max_rooms * hc.expand_units) if hc.enabled else 0.0
    fields = (max(sc.num_households, 1) * ac.max_fields_per_household
              * ac.plant_material_cost) if ac.enabled else 0.0
    return Materials(
        wood_regrow_ticks=cc.tree_regrow_ticks,
        rock_regrow_ticks=cc.rock_regrow_ticks,
        wood_regrown=wood_regrown,
        stone_regrown=stone_regrown,
        expansion_demand=float(expansion),
        field_demand=float(fields),
        material_rate=material_rate,
        build_rate_needed=build_rate,
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
