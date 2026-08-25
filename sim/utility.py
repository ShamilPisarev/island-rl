"""Island 2.0 stage 2: utility agents -- a needs arbiter over goal-level options.

THE SPLIT THIS MODULE EXISTS TO DRAW: script the muscles, learn the choices.
A GOAL (an option, in semi-MDP terms) is a multi-tick intention like "forage" or
"deliver material". Executing one is scripted here and always will be. CHOOSING
one is what stage 5 hands to PPO. The two halves meet at exactly one interface:

    goals = arbiter.choose(view, mask, rng)      # (n,) goal ids
    actions = execute_goals(goals, view, mask)   # (n,) primitive actions

`UtilityArbiter` is the scripted chooser (design doc option A). A learned chooser
(option B) implements the same `choose` and reuses `execute_goals` and
`OptionRunner` untouched, which is what makes the headline stage-5 experiment --
scripted arbiter vs learned arbiter, same world, same controllers -- a real
comparison rather than two different programs.

WHY THE OPTION LEVEL IS NOT JUST A CONVENIENCE. Island 1.0's one remaining open
problem is that PPO will not take a ~20-step trip worth +173 because every
one-step prefix is correctly priced <= 0 by an accurate critic. A committed
`FORAGE` goal IS that trip, taken as one decision. Note honestly what that does
and does not settle: it routes around the problem rather than solving it, and
what can emerge at this level is *when* to travel, never travelling itself.
Every write-up has to say which level it is claiming (design doc section 7).

WHAT IS DELIBERATELY NOT MODELLED YET. The design doc lists a `social` need.
There is no mechanic in the 1.0 world that satisfies one -- no households, no
shared stockpile, nothing that makes standing near someone pay -- so a social
need here would score goals against an appetite the world cannot feed, which is
decoration dressed as a drive. It arrives in stage 4 with households, which is
the thing that gives it teeth. The two giving goals are meanwhile driven by a
per-agent GENEROSITY TRAIT rather than by a need, which is the honest encoding:
handing a berry away restores nothing of the giver's own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agents import (BUILD, CHOP, GATHER, GIVE_FOOD, GIVE_MATERIAL, IDLE, MINE,
                     N_MOVE_ACTIONS, STEAL, num_actions)
from .config import Config
from .obsview import ObsView

# --- goals ------------------------------------------------------------------
# Appended, never inserted, for the same reason the action space is: a trait
# vector or a trained goal-head from an earlier run keeps its meaning.
GOAL_NAMES: tuple[str, ...] = (
    "forage",          # walk to the nearest berry-bearing bush and gather
    "harvest_wood",    # walk to the nearest tree with wood and chop
    "harvest_stone",   # walk to the nearest rock with stone and mine
    "deliver",         # carry material to the most-finished incomplete site
    "shelter",         # go to the nearest finished shelter and stay in it
    "steal",           # rob the nearest loaded neighbour
    "give_food",       # hand a berry to a hungry empty-handed neighbour
    "give_material",   # relay material to someone standing at a site
    "explore",         # commit to a heading and cover ground
    "rest",            # idle
)
(FORAGE, HARVEST_WOOD, HARVEST_STONE, DELIVER, SHELTER, GOAL_STEAL,
 GOAL_GIVE_FOOD, GOAL_GIVE_MATERIAL, EXPLORE, REST) = range(len(GOAL_NAMES))
N_GOALS = len(GOAL_NAMES)

# --- needs ------------------------------------------------------------------
NEED_NAMES: tuple[str, ...] = ("hunger", "food_stock", "safety", "shelter_stock", "wealth")
NEED_HUNGER, NEED_FOOD_STOCK, NEED_SAFETY, NEED_SHELTER_STOCK, NEED_WEALTH = range(5)
N_NEEDS = len(NEED_NAMES)

# How much each goal restores each need, in [0, 1]. Rows are goals, columns
# needs. This is the whole "what does this object advertise" table from a Sims
# needs system, and it is the file's main tuning surface.
RESTORE = np.zeros((N_GOALS, N_NEEDS))
RESTORE[FORAGE, NEED_FOOD_STOCK] = 1.0
RESTORE[FORAGE, NEED_HUNGER] = 0.9        # gathering feeds you, via auto-eat
RESTORE[GOAL_STEAL, NEED_FOOD_STOCK] = 1.0
RESTORE[GOAL_STEAL, NEED_HUNGER] = 0.9    # same payoff as foraging, different source
RESTORE[HARVEST_WOOD, NEED_WEALTH] = 1.0
RESTORE[HARVEST_STONE, NEED_WEALTH] = 1.0
RESTORE[DELIVER, NEED_SHELTER_STOCK] = 1.0
RESTORE[GOAL_GIVE_MATERIAL, NEED_SHELTER_STOCK] = 0.6   # a relay does build the shelter
RESTORE[SHELTER, NEED_SAFETY] = 1.0
# Exploring is how you FIND a bush you cannot currently see, so it restores the
# same needs foraging does -- just much less per tick, since a random heading is
# a far worse bet than a bush you can already see. Small enough that any visible
# target outranks it, large enough that an empty-handed agent with nothing in
# view goes looking instead of standing still.
RESTORE[EXPLORE, NEED_FOOD_STOCK] = 0.15
RESTORE[EXPLORE, NEED_HUNGER] = 0.15

# Maslow shaping: which tier each need sits in, lowest first. Only tiers that can
# actually kill gate anything -- hunger, then night exposure. A half-empty
# inventory is prudence, not an emergency, so food_stock/wealth/shelter_stock
# contribute to scores without suppressing anything.
NEED_TIER = np.array([0, 2, 1, 2, 2])
# EXPLORE sits at tier 0, i.e. ungated, and that placement is a correction worth
# recording. It was tier 4 first, which meant an agent whose inventory was empty
# had tier-2 urgency at 1.0, which zeroed the gate on every tier above it --
# including the search that was the only way to fix the shortage. Measured, that
# put 39.6% of all intentions into `rest`: hungry agents standing still because
# wanting food had suppressed looking for it. Searching is never a luxury, so it
# cannot sit above the needs it serves.
GOAL_TIER = np.array([0, 2, 2, 2, 1, 0, 3, 3, 0, 4])

# A goal serving no need at all still has to be choosable, or an agent with
# nothing visible would have no legal intention. These are the floors, and they
# are small enough that any real need outranks them.
BASE_APPEAL = np.zeros(N_GOALS)
BASE_APPEAL[EXPLORE] = 0.05
BASE_APPEAL[REST] = 0.02


@dataclass(frozen=True)
class ArbiterConfig:
    """Knobs for the scorer. Defaults chosen to be legible, not tuned.

    Nothing here has been fitted to an outcome yet; stage 2's exit condition is a
    watchable replay, and rule 1's warning about producing the behaviour you pay
    for applies just as much to a hand-tuned utility weight as to a shaped
    reward.
    """

    distance_scale: float = 25.0    # utility halves roughly every 17 world units
    critical: float = 0.75          # a lower-tier need at this deficit gates higher tiers
    deficit_power: float = 2.0      # convexity: urgent needs pull away from mild ones
    commit_ticks: int = 25          # how long an option runs before re-deciding
    softmax_temp: float = 0.0       # 0 = argmax; >0 samples, for visible variety
    generosity_scale: float = 0.35  # weight on the two giving goals (trait-scaled)
    trait_spread: float = 0.35      # per-agent lognormal sigma on goal preferences


def agent_traits(num_agents: int, seed: int, cfg: ArbiterConfig) -> np.ndarray:
    """Per-agent multiplicative preference over goals, shape (agents, goals).

    Design doc option D: individual character at 100 agents without 100 brains.
    Lognormal around 1.0 so a weight is never negative and the median agent is
    average -- one agent is 1.4x keener to build, another is a coward about
    night. Seeded from the world seed, so a replay is reproducible like
    everything else here.
    """
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(0.0, cfg.trait_spread, size=(num_agents, N_GOALS)))


def compute_needs(view: ObsView, cfg: Config) -> np.ndarray:
    """Current deficit per need, shape (agents, needs), each in [0, 1].

    A deficit of 1.0 means "maximally unsatisfied". These read only the
    observation, so an arbiter built on them is not psychic (design doc section
    7) -- the same discipline the scripted forager, thief, builder and trader all
    already keep.
    """
    n = view.n
    needs = np.zeros((n, N_NEEDS))

    # hunger: measured against the EAT THRESHOLD, not against zero. An agent at
    # 80% hunger is not 20% hungry in any way that matters -- it will not even
    # eat a berry it is carrying until it drops below the threshold. Scaling to
    # the threshold makes the deficit reach 1.0 exactly when the agent is dying.
    threshold = cfg.hunger.eat_threshold / cfg.hunger.max
    needs[:, NEED_HUNGER] = np.clip((threshold - view.hunger) / max(threshold, 1e-9), 0.0, 1.0)
    needs[:, NEED_FOOD_STOCK] = np.clip(1.0 - view.food, 0.0, 1.0)

    if cfg.construction.enabled:
        cc = cfg.construction
        # safety: how exposed am I to the coming night? Rises as dusk approaches
        # and falls to zero inside a finished shelter. Night is the last
        # `night_fraction` of the cycle, so `phase` is the clock the observation
        # already carries.
        dusk = 1.0 - cc.night_fraction
        # The ramp has to start EARLY ENOUGH TO WALK HOME, so it is derived from
        # the world rather than picked: crossing the observable range takes
        # `distance_scale / move_step` ticks, as a fraction of one day/night
        # cycle. On the 100-agent island that is 50/0.8 = 62 ticks of a 200-tick
        # cycle, i.e. a lead of 0.31 -- where the hardcoded 0.2 (40 ticks) had
        # agents setting off after dark and arriving during it. Capped at the
        # daylight available, so a world with very long nights cannot ask for a
        # lead longer than the day.
        cross_ticks = view.scale / max(cfg.world.move_step, 1e-9)
        lead = float(np.clip(cross_ticks / max(cc.night_cycle, 1), 0.05, dusk))
        urgency = np.clip((view.phase - (dusk - lead)) / max(lead, 1e-9), 0.0, 1.0)
        urgency = np.where(view.is_night, 1.0, urgency)
        # THE NEED IS "BE UNDER COVER TONIGHT", and it depends on the clock alone.
        # Two wrong versions came before this one, and both are worth recording
        # because they are the same confusion in different clothes:
        #
        #   * scaling the deficit by distance-to-shelter (`urgency * d_home/scale`)
        #     made an agent 9.7 units from cover at night only 19% unsafe. It is
        #     100% exposed; the drain is 3x whatever the distance is. Distance is
        #     the COST OF FIXING the need, never its SIZE -- and the cost is
        #     already handled, by the derived lead time above.
        #   * zeroing it once inside looks right and empties the shelter halfway
        #     through the night: satisfied need -> shelter scores 0 -> the agent
        #     wanders back out. Being under cover is not a state that discharges
        #     the need, it is how the need goes on being met, tick by tick. So the
        #     deficit stays up all night and the goal's own executor is what keeps
        #     an arrived agent sitting still.
        #
        # Trait spread does the rest: a coward heads home early, a night owl
        # forages on. That variety is option D's individual character, not noise.
        needs[:, NEED_SAFETY] = urgency

        # shelter_stock: the share of visible sites that are unfinished. A
        # population that has built everything nearby stops wanting to build.
        visible = view.sites.present.sum(axis=1)
        unfinished = (view.sites.present & ~view.site_complete).sum(axis=1)
        needs[:, NEED_SHELTER_STOCK] = np.where(visible > 0, unfinished / np.maximum(visible, 1), 0.0)

        needs[:, NEED_WEALTH] = np.clip(1.0 - view.material_carried, 0.0, 1.0)
    return needs


def score_goals(view: ObsView, cfg: Config, needs: np.ndarray, traits: np.ndarray,
                acfg: ArbiterConfig) -> np.ndarray:
    """Utility of every goal for every agent, shape (agents, goals).

    The Sims-style scorer, with a Maslow gate on top:

        score(g) = trait[g] * gate(tier(g)) * discount(distance to g)
                   * sum_n deficit(n)^p * restore(g, n)

    plus a base appeal so `explore` and `rest` are always choosable. Goals whose
    target does not exist score exactly zero and are never chosen -- that is the
    "advertised action" half of a utility system: the world offers what it can
    support, and an agent that can see no bush cannot want to forage at one.
    """
    n = view.n
    weighted = needs ** acfg.deficit_power              # (n, needs)
    base = weighted @ RESTORE.T                          # (n, goals)
    base = base + BASE_APPEAL[None, :]

    # --- Maslow gate. urgency[t] is the worst deficit in tier t; a goal in tier
    # T is scaled by the product over every LOWER tier of how satisfied that tier
    # is. A starving agent's construction score goes to zero, which is the whole
    # point of the hierarchy.
    tiers = int(GOAL_TIER.max()) + 1
    urgency = np.zeros((n, tiers))
    for t in range(tiers):
        cols = np.flatnonzero(NEED_TIER == t)
        if cols.size:
            urgency[:, t] = needs[:, cols].max(axis=1)
    satisfied = np.clip(1.0 - urgency / max(acfg.critical, 1e-9), 0.0, 1.0)
    gate = np.ones((n, N_GOALS))
    for g in range(N_GOALS):
        lower = GOAL_TIER[g]
        if lower > 0:
            gate[:, g] = np.prod(satisfied[:, :lower], axis=1)

    # --- distance discount and target existence, per goal
    discount = np.ones((n, N_GOALS))
    available = np.ones((n, N_GOALS), dtype=bool)

    def set_target(goal: int, distance: np.ndarray) -> None:
        ok = np.isfinite(distance)
        available[:, goal] = ok
        discount[:, goal] = np.where(ok, np.exp(-np.where(ok, distance, 0.0)
                                               / max(acfg.distance_scale, 1e-9)), 0.0)

    d_bush, _, _, _ = view.bushes.nearest(view.loaded_bushes)
    set_target(FORAGE, d_bush)

    room_for_food = view.food < 1.0 - 1e-6
    available[:, FORAGE] &= room_for_food

    if cfg.competition.enable_steal:
        loaded = view.neighbour_food > 0.0
        d_victim, _, _, _ = view.neighbours.nearest(loaded & view.neighbours.present)
        set_target(GOAL_STEAL, d_victim)
        # OPPORTUNISTIC ONLY: a steal is available when a victim is ALREADY in
        # reach, never as somewhere to walk to. This is not a taste call, it is
        # 1.0's measurement. The scripted thief -- which never chases -- beats the
        # forager by +87.8, while forcing every legal steal on the learned policy
        # is worth -8.3 +- 4.1 ticks: "opportunistic theft targeted at loaded
        # neighbours is useful, indiscriminate theft-for-reward is not."
        #
        # It is also what stopped this population eating its own tail. Pursued as
        # a travelling goal it produced 6932 steals an episode and the
        # pre-registered PERMANENT WAR: a loaded victim sits at a median 2.8 units
        # against a loaded bush at 4.2, so with 100 agents packed together the
        # distance discount handed theft every contest, foraging fell to 10.9% of
        # intentions and the population harvested 685 berries against a demand of
        # 857. Theft redistributes and never creates (the M3 lesson) and a utility
        # scorer over selfish needs cannot see that -- so the mechanic, not the
        # weight, is where the correction belongs (rule 2).
        available[:, GOAL_STEAL] &= room_for_food & (d_victim <= cfg.competition.steal_radius)
    else:
        available[:, GOAL_STEAL] = False

    if cfg.construction.enabled:
        cc = cfg.construction
        room_for_material = view.material_carried < 1.0 - 1e-6
        d_tree, _, _, _ = view.trees.nearest(view.trees.present)
        set_target(HARVEST_WOOD, d_tree)
        available[:, HARVEST_WOOD] &= room_for_material
        d_rock, _, _, _ = view.rocks.nearest(view.rocks.present)
        set_target(HARVEST_STONE, d_rock)
        available[:, HARVEST_STONE] &= room_for_material

        incomplete = view.sites.present & ~view.site_complete
        d_site, _, _, _ = view.sites.nearest(incomplete)
        set_target(DELIVER, d_site)
        available[:, DELIVER] &= view.material_carried > 0.0

        d_home, _, _, _ = view.sites.nearest(view.site_complete)
        set_target(SHELTER, d_home)
        # SHELTER IS A DEADLINE, NOT AN OPPORTUNITY, so it does not get discounted
        # for being far away. The distance discount encodes "a nearer satisfier is
        # a better deal", which is right for optional goals and exactly backwards
        # for a curfew: a shelter 40 units off is MORE urgent to set out for, not
        # less. Discounting it measured 22.7% of nights indoors on a world sized
        # to starve an exposed population (supply is 0.87x exposed demand), with
        # agents declining the walk home because home was far -- the discount
        # talking them out of the one thing that keeps the night affordable.
        discount[:, SHELTER] = np.where(np.isfinite(d_home), 1.0, 0.0)
    else:
        available[:, [HARVEST_WOOD, HARVEST_STONE, DELIVER, SHELTER]] = False

    # --- giving. Driven by a trait rather than a need, and gated on the receiver
    # being about to USE the unit -- the scripted trader's hard-won lesson, that
    # a floor made of churn is not a floor. The recipient is whoever the WORLD
    # would pick (the nearest in reach with room), never whoever looks neediest,
    # because that is what `give` actually does.
    if cfg.exchange.enabled:
        in_reach = view.neighbours.present & (view.neighbours.distance
                                              <= cfg.exchange.give_radius)
        # food: the nearest reachable neighbour with room, if it is hungry and empty
        room = view.neighbour_food < 1.0 - 1e-6
        eligible = in_reach & room
        slot = eligible.argmax(axis=1)
        rows = np.arange(n)
        exists = eligible.any(axis=1)
        threshold = cfg.hunger.eat_threshold / cfg.hunger.max
        starving = (exists & (view.neighbour_hunger[rows, slot] < threshold)
                    & (view.neighbour_food[rows, slot] <= 0.0))
        fed = view.hunger > threshold
        available[:, GOAL_GIVE_FOOD] = starving & fed & (view.food > 0.0)
        base[:, GOAL_GIVE_FOOD] += acfg.generosity_scale

        if cfg.construction.enabled:
            room_m = view.neighbour_material < 1.0 - 1e-6
            eligible_m = in_reach & room_m
            slot_m = eligible_m.argmax(axis=1)
            exists_m = eligible_m.any(axis=1)
            incomplete = view.sites.present & ~view.site_complete
            d_site, sx, sz, _ = view.sites.nearest(incomplete)
            # Can they deliver it this tick where I cannot? Both offsets share an
            # origin, so their distance to the site is just the difference.
            their_d = np.hypot(sx - view.neighbours.dx[rows, slot_m],
                               sz - view.neighbours.dz[rows, slot_m])
            available[:, GOAL_GIVE_MATERIAL] = (
                exists_m & np.isfinite(d_site)
                & (their_d <= cfg.construction.build_radius)
                & (d_site > cfg.construction.build_radius)
                & (view.material_carried > 0.0)
            )
            base[:, GOAL_GIVE_MATERIAL] += acfg.generosity_scale
        else:
            available[:, GOAL_GIVE_MATERIAL] = False
    else:
        available[:, [GOAL_GIVE_FOOD, GOAL_GIVE_MATERIAL]] = False

    scores = base * gate * discount * traits
    return np.where(available, scores, 0.0)


def _heading(dx: np.ndarray, dz: np.ndarray) -> np.ndarray:
    """Compass action pointing along (dx, dz); action i is i*45deg from +z."""
    return (np.round(np.arctan2(dx, dz) / (np.pi / 4.0)) % N_MOVE_ACTIONS).astype(np.int64)


def goal_viable(view: ObsView, cfg: Config, goals: np.ndarray) -> np.ndarray:
    """Can each agent still pursue the goal it currently holds?

    An option terminates when its target vanishes (the bush was emptied, the site
    was finished by somebody else) or when its purpose is served (inventory full).
    This is the termination half of the semi-MDP contract, and it is the one place
    the bookkeeping can silently rot -- so it is one function, used by both the
    scripted and (in stage 5) the learned path.
    """
    n = view.n
    ok = np.ones(n, dtype=bool)
    d_bush, _, _, _ = view.bushes.nearest(view.loaded_bushes)
    room_food = view.food < 1.0 - 1e-6

    def when(goal: int, condition: np.ndarray) -> None:
        rows = goals == goal
        ok[rows] = condition[rows]

    when(FORAGE, np.isfinite(d_bush) & room_food)

    if cfg.competition.enable_steal:
        loaded = view.neighbour_food > 0.0
        d_victim, _, _, _ = view.neighbours.nearest(loaded & view.neighbours.present)
        # Terminates the moment the victim moves out of reach -- an opportunistic
        # steal is not something you follow someone around for. See score_goals.
        when(GOAL_STEAL, (d_victim <= cfg.competition.steal_radius) & room_food)

    if cfg.construction.enabled:
        room_material = view.material_carried < 1.0 - 1e-6
        d_tree, _, _, _ = view.trees.nearest(view.trees.present)
        d_rock, _, _, _ = view.rocks.nearest(view.rocks.present)
        when(HARVEST_WOOD, np.isfinite(d_tree) & room_material)
        when(HARVEST_STONE, np.isfinite(d_rock) & room_material)
        incomplete = view.sites.present & ~view.site_complete
        d_site, _, _, _ = view.sites.nearest(incomplete)
        when(DELIVER, np.isfinite(d_site) & (view.material_carried > 0.0))
        d_home, _, _, _ = view.sites.nearest(view.site_complete)
        # Shelter is only worth holding while the night lasts.
        when(SHELTER, np.isfinite(d_home) & (view.is_night | (view.phase > 0.5)))

    # The gifts are single-tick by nature and `explore`/`rest` never fail.
    when(GOAL_GIVE_FOOD, np.zeros(n, dtype=bool))
    when(GOAL_GIVE_MATERIAL, np.zeros(n, dtype=bool))
    return ok


def execute_goals(goals: np.ndarray, view: ObsView, cfg: Config,
                  mask: np.ndarray, explore_heading: np.ndarray) -> np.ndarray:
    """Turn goal ids into primitive actions. The scripted "muscles".

    Every branch is walk-there-then-act: head for the target while out of range,
    take the action once the mask says it can succeed. Reading the mask is the
    same allowance the 1.0 scripted builder and trader take, and the learned
    policy receives it too, so nothing is being smuggled in.
    """
    n = view.n
    actions = np.full(n, IDLE, dtype=np.int64)

    def walk_then(goal: int, distance, dx, dz, action: int, radius: float) -> None:
        """Move toward the target, or take `action` when it is legal here."""
        rows = goals == goal
        if not rows.any():
            return
        arrived = (distance <= radius) & mask[:, action]
        chosen = np.where(arrived, action, _heading(dx, dz))
        # A target we can see but cannot act on yet is still worth walking at;
        # one we cannot see at all leaves the agent idle, and the arbiter will
        # pick something else next decision because the goal is not viable.
        chosen = np.where(np.isfinite(distance), chosen, IDLE)
        actions[rows] = chosen[rows]

    d_bush, bx, bz, _ = view.bushes.nearest(view.loaded_bushes)
    walk_then(FORAGE, d_bush, bx, bz, GATHER, cfg.bushes.gather_radius)

    if cfg.competition.enable_steal:
        loaded = view.neighbours.present & (view.neighbour_food > 0.0)
        d_victim, vx, vz, _ = view.neighbours.nearest(loaded)
        walk_then(GOAL_STEAL, d_victim, vx, vz, STEAL, cfg.competition.steal_radius)

    if cfg.construction.enabled:
        cc = cfg.construction
        d_tree, tx, tz, _ = view.trees.nearest(view.trees.present)
        walk_then(HARVEST_WOOD, d_tree, tx, tz, CHOP, cc.harvest_radius)
        d_rock, rx, rz, _ = view.rocks.nearest(view.rocks.present)
        walk_then(HARVEST_STONE, d_rock, rx, rz, MINE, cc.harvest_radius)

        # Deliver to the MOST FINISHED incomplete site rather than the nearest.
        # Progress is objective, so every agent picks the same focal site and
        # material lands in one place -- the 1.0 builder's hardest-won lesson
        # (six agents each feeding their own nearest site completed 0.4 shelters
        # an episode; a focal site completes before the first nightfall).
        incomplete = view.sites.present & ~view.site_complete
        remaining = np.where(incomplete, view.site_remaining, np.inf)
        j = np.argmin(remaining, axis=1)
        rows = np.arange(n)
        has_site = np.isfinite(remaining[rows, j])
        sx, sz = view.sites.dx[rows, j], view.sites.dz[rows, j]
        d_site = np.where(has_site, np.hypot(sx, sz), np.inf)
        walk_then(DELIVER, d_site, sx, sz, BUILD, cc.build_radius)

        d_home, hx, hz, _ = view.sites.nearest(view.site_complete)
        shelter_rows = goals == SHELTER
        if shelter_rows.any():
            # "Inside" is well within the radius, not on its lip: an agent that
            # stops at the boundary drifts out again on the next jitter.
            inside = d_home <= cc.shelter_radius * 0.6
            chosen = np.where(inside, IDLE, _heading(hx, hz))
            chosen = np.where(np.isfinite(d_home), chosen, IDLE)
            actions[shelter_rows] = chosen[shelter_rows]

    if cfg.exchange.enabled:
        for goal, action in ((GOAL_GIVE_FOOD, GIVE_FOOD), (GOAL_GIVE_MATERIAL, GIVE_MATERIAL)):
            rows = (goals == goal) & mask[:, action]
            actions[rows] = action

    # explore: hold a heading. Ballistic travel rather than a fresh random step
    # each tick, which is the one thing `nav-commit` showed is worth real ticks
    # (+50 zero-shot in a spread world) even with no direction learned.
    explore_rows = goals == EXPLORE
    actions[explore_rows] = explore_heading[explore_rows]
    # ...but never straight into the sea: at the shoreline, turn inward.
    if explore_rows.any():
        at_edge = view.edge_room < 0.06
        inward = _heading(-view.outward_x, -view.outward_z)
        actions = np.where(explore_rows & at_edge, inward, actions)

    actions[goals == REST] = IDLE
    # Last line of defence: never emit an action the world would refuse. A goal
    # whose action went illegal between scoring and execution falls back to idle
    # rather than wasting the tick on something the mask already forbade.
    illegal = ~np.take_along_axis(mask, actions[:, None], axis=1).ravel()
    actions[illegal] = IDLE
    return actions


class UtilityArbiter:
    """The scripted chooser: score every goal from needs, take the best.

    Holds no world state -- only per-agent traits -- so it is a pure function of
    the observation plus the agent's identity. That is what lets stage 5 drop a
    learned chooser in beside it without touching anything else.
    """

    def __init__(self, cfg: Config, acfg: ArbiterConfig | None = None,
                 seed: int = 0) -> None:
        self.cfg = cfg
        self.acfg = acfg or ArbiterConfig()
        self.traits = agent_traits(cfg.world.num_agents, seed, self.acfg)

    def choose(self, view: ObsView, mask: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
        needs = compute_needs(view, self.cfg)
        scores = score_goals(view, self.cfg, needs, self.traits, self.acfg)
        if self.acfg.softmax_temp > 0.0:
            # Softmax over AVAILABLE goals only, so variety never picks an
            # impossible intention. Scores are already zero where unavailable.
            logits = np.where(scores > 0.0, scores / self.acfg.softmax_temp, -np.inf)
            all_zero = ~np.isfinite(logits).any(axis=1)
            logits[all_zero, REST] = 0.0
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            # One categorical draw per agent, vectorised via the inverse CDF.
            u = rng.random((view.n, 1))
            return (p.cumsum(axis=1) < u).sum(axis=1).clip(0, N_GOALS - 1)
        # argmax, with `rest` as the tie-break when literally nothing is on offer
        best = scores.argmax(axis=1)
        return np.where(scores.max(axis=1) > 0.0, best, REST)


class OptionRunner:
    """Runs an arbiter's goals as committed multi-tick options.

    A goal is re-decided only when it TERMINATES, TIMES OUT, or is INTERRUPTED by
    a tier-0 emergency. Everything in between executes the same intention, which
    is what makes a 20-tick crossing a single decision.

    The interruption rule is the piece worth being careful about: without it a
    committed forager walks past its own death, and with too loose a rule every
    tick becomes a decision again and the commitment is decorative. Here a
    running option is dropped only when hunger crosses `critical` while the agent
    is not already pursuing a hunger goal.

    `decisions` and `goal_ticks` accumulate the bookkeeping stage 5 needs (one
    PPO transition per decision, rewards summed in between) and stage 2 reads for
    its own diagnostics.
    """

    def __init__(self, arbiter, cfg: Config, seed: int = 0) -> None:
        self.arbiter = arbiter
        self.cfg = cfg
        self.acfg = getattr(arbiter, "acfg", ArbiterConfig())
        n = cfg.world.num_agents
        self.rng = np.random.default_rng(seed)
        self.goals = np.full(n, REST, dtype=np.int64)
        self.ticks_left = np.zeros(n, dtype=np.int64)
        self.explore_heading = self.rng.integers(0, N_MOVE_ACTIONS, size=n)
        self.decisions = np.zeros(n, dtype=np.int64)
        self.goal_ticks = np.zeros(N_GOALS, dtype=np.int64)
        self.last_decided = np.zeros(n, dtype=bool)

    def reset(self) -> None:
        self.goals[:] = REST
        self.ticks_left[:] = 0
        self.decisions[:] = 0
        self.goal_ticks[:] = 0

    def act(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """One tick: keep or re-decide each agent's goal, then execute it."""
        view = ObsView(obs, self.cfg)
        needs = compute_needs(view, self.cfg)
        viable = goal_viable(view, self.cfg, self.goals)

        emergency = needs[:, NEED_HUNGER] >= self.acfg.critical
        pursuing_food = np.isin(self.goals, (FORAGE, GOAL_STEAL))
        interrupted = emergency & ~pursuing_food

        redecide = (~viable) | (self.ticks_left <= 0) | interrupted
        if redecide.any():
            fresh = self.arbiter.choose(view, mask, self.rng)
            self.goals = np.where(redecide, fresh, self.goals)
            self.ticks_left = np.where(redecide, self.acfg.commit_ticks, self.ticks_left)
            self.decisions += redecide
            # A new explore option gets a new heading; re-rolling it every tick
            # would turn ballistic travel back into a random walk.
            new_explore = redecide & (self.goals == EXPLORE)
            if new_explore.any():
                roll = self.rng.integers(0, N_MOVE_ACTIONS, size=view.n)
                self.explore_heading = np.where(new_explore, roll, self.explore_heading)
        self.last_decided = redecide

        self.ticks_left -= 1
        np.add.at(self.goal_ticks, self.goals, 1)
        return execute_goals(self.goals, view, self.cfg, mask, self.explore_heading)


def utility_runner(cfg: Config, seed: int = 0,
                   acfg: ArbiterConfig | None = None) -> OptionRunner:
    """The stage-2 population: scripted arbiter, scripted controllers."""
    return OptionRunner(UtilityArbiter(cfg, acfg, seed=seed), cfg, seed=seed)
