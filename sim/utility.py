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

from .agents import (BUILD, CHOP, CRAFT, DEPOSIT_FOOD, DEPOSIT_MATERIAL, GATHER,
                     GIVE_FOOD, GIVE_MATERIAL, IDLE, MINE, N_MOVE_ACTIONS, PLANT,
                     RAID, STEAL, WITHDRAW_FOOD, WITHDRAW_MATERIAL, num_actions)
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
    # --- stage 4
    "store_food",      # carry surplus food home to the household stockpile
    "store_material",  # ...and surplus material
    "draw_food",       # eat out of the household store
    "draw_material",   # take material out of the store to rebuild with
    "raid",            # take from ANOTHER household's stockpile
    # --- tech ladder rung 1
    "craft",           # make an axe at a site, out of wood and stone carried
    # --- Island 3.0
    "expand",          # add a room to my own finished house
    "plant",           # put a field in, next to my own house
)
(FORAGE, HARVEST_WOOD, HARVEST_STONE, DELIVER, SHELTER, GOAL_STEAL,
 GOAL_GIVE_FOOD, GOAL_GIVE_MATERIAL, EXPLORE, REST,
 STORE_FOOD, STORE_MATERIAL, DRAW_FOOD, DRAW_MATERIAL, GOAL_RAID,
 CRAFT_AXE, EXPAND, PLANT_FIELD) = range(len(GOAL_NAMES))
N_GOALS = len(GOAL_NAMES)
# The goal count at each frozen rung. `agent_traits` draws each block in its own
# call so appending a goal cannot perturb an existing world's traits, and
# `goal_width` slices a learned head to the rung its world is at; see the comment
# in `agent_traits` for the failure this prevents.
N_GOALS_STAGE4 = 15
N_GOALS_RUNG1 = 16   # + craft

# --- needs ------------------------------------------------------------------
NEED_NAMES: tuple[str, ...] = ("hunger", "food_stock", "safety", "shelter_stock", "wealth",
                               # stage 4: the household's larder and its material store.
                               # These are the first needs in the project that are
                               # not about the agent's own body, which is what gives
                               # a group something to be a group ABOUT.
                               "house_food", "house_material",
                               # tech ladder rung 1: a capability I do not have.
                               # Every other need is about a STOCK running low;
                               # this one is about being unable to do something
                               # well, and it stays at full deficit until the tool
                               # exists, then goes to zero forever.
                               "tool",
                               # Island 3.0: room in the family house, and land
                               # to farm. Both are household needs like
                               # house_food -- the first two needs in the project
                               # that are about the family's PROPERTY rather than
                               # its stores.
                               "house_room", "land")
(NEED_HUNGER, NEED_FOOD_STOCK, NEED_SAFETY, NEED_SHELTER_STOCK, NEED_WEALTH,
 NEED_HOUSE_FOOD, NEED_HOUSE_MATERIAL, NEED_TOOL,
 NEED_HOUSE_ROOM, NEED_LAND) = range(10)
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
# stage 4. Note what `store_*` restores: a HOUSEHOLD need, never the depositor's
# own. Depositing is a pure personal cost, and encoding it any other way would be
# paying the agent for the behaviour stage 5 is supposed to ask about (rule 1).
RESTORE[STORE_FOOD, NEED_HOUSE_FOOD] = 1.0
RESTORE[STORE_MATERIAL, NEED_HOUSE_MATERIAL] = 1.0
RESTORE[DRAW_FOOD, NEED_FOOD_STOCK] = 1.0
RESTORE[DRAW_FOOD, NEED_HUNGER] = 0.9
RESTORE[DRAW_MATERIAL, NEED_WEALTH] = 1.0
RESTORE[GOAL_RAID, NEED_FOOD_STOCK] = 1.0
RESTORE[GOAL_RAID, NEED_HUNGER] = 0.9
# Tech ladder rung 1. An axe serves WEALTH -- the material need -- because that is
# what it is for: every later chop brings back twice the wood. Half of what a
# harvest trip restores, not more, and the halving is the honest bit: crafting
# spends two units NOW for a rate that only pays off over the trips that follow,
# so it must not outbid the trip it is competing with while an agent is short.
# What it must not be is a reward (rule 1): nothing in world.py pays for a craft,
# and if axes get made it is because the wood they bring back keeps agents alive.
# ...and it serves the TOOL need, which is the only reason it can ever be
# chosen. Restoring `wealth` alone was the first version and it can never fire:
# `craft` is available exactly when the agent is carrying a wood and a stone,
# and on a 2-unit inventory that means FULL -- so the wealth deficit is zero at
# the precise moment the goal becomes possible, and the goal scores zero. The
# lesson is the file's own, arriving in new clothes: a need is what you LACK, and
# what an unarmed agent lacks is the tool, not the material in its hands.
RESTORE[CRAFT_AXE, NEED_TOOL] = 1.0
RESTORE[CRAFT_AXE, NEED_WEALTH] = 0.5
# Island 3.0. `expand` serves the family's need for ROOM and nothing else: it is
# not a shelter for the builder (the builder already has one, that is what makes
# the house expandable) and it is not wealth. Same shape as `store_*` -- a
# household need, paid for by an individual -- which is deliberately the chain
# stage 5 could never climb, now offered again with a recurring prize.
RESTORE[EXPAND, NEED_HOUSE_ROOM] = 1.0
# `plant` serves LAND, and it needs its own need for exactly the reason `craft`
# did: planting costs a carried unit, so the wealth deficit is near zero at the
# moment the goal becomes possible, and a field yields nothing at all on the tick
# it goes in (`field_initial` is 0). A need is what you LACK, and what a hungry
# household without a field lacks is the field.
RESTORE[PLANT_FIELD, NEED_LAND] = 1.0

# Maslow shaping: which tier each need sits in, lowest first. Only tiers that can
# actually kill gate anything -- hunger, then night exposure. A half-empty
# inventory is prudence, not an emergency, so food_stock/wealth/shelter_stock
# contribute to scores without suppressing anything.
NEED_TIER = np.array([0, 2, 1, 2, 2, 3, 3,
                      2,      # tool: prudence, like every other stock need
                      3, 2])  # house_room with the other household needs; land
                              # with the stock needs, because a field is how you
                              # stop being hungry LATER and prudence is exactly
                              # what that is
# EXPLORE sits at tier 0, i.e. ungated, and that placement is a correction worth
# recording. It was tier 4 first, which meant an agent whose inventory was empty
# had tier-2 urgency at 1.0, which zeroed the gate on every tier above it --
# including the search that was the only way to fix the shortage. Measured, that
# put 39.6% of all intentions into `rest`: hungry agents standing still because
# wanting food had suppressed looking for it. Searching is never a luxury, so it
# cannot sit above the needs it serves.
#
# The stage-4 tiers apply that same correction rather than rediscovering it.
# `draw_food` and `raid` both SERVE hunger, so neither may sit above it -- a
# starving agent whose own store is full must not have the tier gate zero out the
# one goal that empties it. `store_*` and `draw_material` are prudence and sit
# with the other tier-2 goals; a household larder is never an emergency.
GOAL_TIER = np.array([0, 2, 2, 2, 1, 0, 3, 3, 0, 4,
                      2, 2, 0, 2, 0,
                      2,      # craft: prudence, exactly like harvesting
                      2, 2])  # expand and plant: prudence too. Neither may sit
                              # above the needs it serves -- the correction
                              # EXPLORE's placement records, applied ahead of
                              # time rather than after a measurement

# A goal serving no need at all still has to be choosable, or an agent with
# nothing visible would have no legal intention. These are the floors, and they
# are small enough that any real need outranks them.
BASE_APPEAL = np.zeros(N_GOALS)
BASE_APPEAL[EXPLORE] = 0.05
BASE_APPEAL[REST] = 0.02

# Which goals may run to their goal state under `persist_until_goal`. A goal
# qualifies on two counts, and the second one was learned by running it: its
# `goal_viable` test must encode something the WORLD reaches (inventory full,
# site fed, night over, pile emptied) AND that state must be REACHABLE while the
# option runs. Five goals fail one test or the other, and all five keep
# `commit_ticks` even when the flag is on:
#
#   * `rest` never fails its test at all -- an unbounded rest is paralysis, not
#     persistence.
#   * the two gifts are single-tick by nature.
#   * `steal` is opportunistic by construction, never something you follow
#     somebody around for (stage 2's permanent-war correction). `raid` is the
#     same correction on a fatter prize, and persisting it measured raids rising
#     from 8.3% to 13.4% of all intentions -- re-opening at the option level the
#     mechanic that correction closed at the world level.
#   * `explore` looked like the clearest case and is the sharpest counterexample.
#     Its goal state is a PERCEPTION -- "a loaded bush is in view and I have room
#     for it" -- and an agent with a full inventory can never reach it, so a
#     persisted explore is a 150-tick wander. Measured, its share went 27% ->
#     36% of all intentions: stage 2's own "serving out a commitment whose reason
#     had expired" finding, rebuilt by the lever meant to fix a different one.
PERSIST_GOALS = np.zeros(N_GOALS, dtype=bool)
PERSIST_GOALS[[FORAGE, HARVEST_WOOD, HARVEST_STONE, DELIVER, SHELTER,
               STORE_FOOD, STORE_MATERIAL, DRAW_FOOD, DRAW_MATERIAL]] = True
PERSIST_GOALS[EXPAND] = True
# `craft` is deliberately NOT persistent: its goal state is reached in the single
# tick the axe is made, so a persisted craft is a commitment with nothing left to
# commit to -- the `rest` failure mode above, at a workbench. `plant` is the same
# shape and is left out for the same reason; `expand` is a delivery and persists
# exactly as `deliver` does.


def inherit_traits(traits: np.ndarray, rng: np.random.Generator, births,
                   cfg: Config) -> None:
    """Give each newborn its parents' traits, mutated. In place.

    GEOMETRIC mean, not arithmetic: the traits are lognormal about 1.0, so
    averaging them arithmetically walks the whole population upward for free and
    would look like selection when it is only algebra.

    One function, used by every arbiter that holds a trait vector, for the same
    reason `goal_viable` is one function: a learned chooser carries the scripted
    traits as an INPUT, and two implementations of the blend would let the two
    halves of a mixed population disagree about who the same agent is.

    WHAT CAN EVOLVE HERE IS BOUNDED, and every write-up says so. A trait is a
    multiplier on a GOAL's score, so a lineage can become keener or cooler on
    each of the goals that already exist -- never acquire a new one, and never
    find a new way of pursuing one. "The village evolved a strategy" would be a
    claim about weights inside a scorer somebody wrote.
    """
    rc = cfg.reproduction
    if not (rc.enabled and rc.heritable_traits) or not births:
        return
    width = traits.shape[1]
    for slot, pa, pb in births:
        if slot >= traits.shape[0]:
            continue
        blend = np.sqrt(traits[pa] * traits[pb])
        traits[slot] = blend * np.exp(rng.normal(0.0, rc.trait_mutation, size=width))


def commit_budget(acfg: ArbiterConfig, goals: np.ndarray) -> np.ndarray:
    """Ticks each freshly-decided goal is committed for.

    One function, used by `OptionRunner`, the trainer and the imitation loop, so
    the three cannot drift on what a commitment is worth -- the same reason
    `goal_viable` is one function.
    """
    if not acfg.persist_until_goal:
        return np.full(np.shape(goals), acfg.commit_ticks, dtype=np.int64)
    return np.where(PERSIST_GOALS[goals], acfg.persist_timeout,
                    acfg.commit_ticks).astype(np.int64)


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
    # stage 4: the two conditions under which a raid becomes thinkable at all.
    # Kept as thresholds rather than as weights on purpose -- stage 2 showed a
    # theft goal that competes on score alone wins on proximity and produces
    # permanent war, so the correction belongs at availability, not in the number.
    raid_hunger: float = 0.5        # hunger deficit at which desperation qualifies
    raid_grudge: float = 0.5        # remembered theft at which revenge qualifies
    # --- stage 5 lever 3: persist-until-goal options. OFF by default, so every
    # result in this project stays bit-identical (pinned by a test).
    #
    # WHAT IT CHANGES. With it off, an option runs `commit_ticks` (25) and is
    # then re-decided whether or not it achieved anything -- so a build
    # programme, which costs six carried units at a capacity of two, is at least
    # three separate decisions with two uncreditable harvest legs between them.
    # That is 1.0's compound-prize wall rebuilt one level up, and it is the one
    # thing the measured 0.0% construction share of every learned population is
    # consistent with. With it on, a goal whose viability test encodes a real
    # GOAL STATE runs until that state is reached, with `persist_timeout` as a
    # backstop -- so a whole shelter is one semi-MDP decision.
    #
    # THE HONEST COST, and it goes in every write-up: what can emerge at this
    # level is WHEN to build, never building. The programme itself is scripted,
    # exactly as `execute_goals` scripts the walk. See design doc section 7.
    persist_until_goal: bool = False
    persist_timeout: int = 150      # backstop only; ~2 shelters' worth of ticks


def agent_traits(num_agents: int, seed: int, cfg: ArbiterConfig) -> np.ndarray:
    """Per-agent multiplicative preference over goals, shape (agents, goals).

    Design doc option D: individual character at 100 agents without 100 brains.
    Lognormal around 1.0 so a weight is never negative and the median agent is
    average -- one agent is 1.4x keener to build, another is a coward about
    night. Seeded from the world seed, so a replay is reproducible like
    everything else here.
    """
    rng = np.random.default_rng(seed)
    # DRAWN IN TWO CALLS, AND THE FIRST SHAPE IS FROZEN. `rng.normal` fills
    # row-major, so a single draw of (agents, N_GOALS) reshuffles EVERY agent's
    # every trait the moment a goal is appended -- which is how rung 1 first broke
    # the persist-off golden checksum, in a world with no tools in it. The
    # append-never-insert rule applies to the random stream as well as to the
    # index: the stage-4 block keeps the draw it always had, and each later rung
    # takes fresh numbers off the end.
    blocks = [np.exp(rng.normal(0.0, cfg.trait_spread,
                                size=(num_agents, N_GOALS_STAGE4)))]
    # One call per RUNG, not one call for "everything after stage 4". Merging the
    # later rungs into a single draw would reshuffle rung 1's column the moment
    # Island 3.0 appended two more -- the identical row-major failure this
    # splitting exists to prevent, one rung further along.
    for width in (N_GOALS_RUNG1 - N_GOALS_STAGE4, N_GOALS - N_GOALS_RUNG1):
        if width > 0:
            blocks.append(np.exp(rng.normal(0.0, cfg.trait_spread,
                                            size=(num_agents, width))))
    return np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]


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

    # tech ladder rung 1. Full deficit while unarmed, zero once the axe exists.
    # Note what this makes the adoption number MEAN, stated here rather than
    # discovered later: the scripted arbiter's adoption rate is a consequence of
    # this line, not an emergent finding. What the axe world can honestly measure
    # is whether the tool PAYS (against the no-axe control on the same seeds) and
    # whether its owners SPECIALISE -- neither of which anything here scores.
    # "A tool is adopted because agents worked out it was worth it" is a claim
    # only a learned chooser can earn.
    if cfg.tools.enabled:
        needs[:, NEED_TOOL] = (~view.axe).astype(np.float64)

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

    if cfg.society.enabled:
        # The household's larder and material store, read off my own observation
        # -- I know what is in my household's pile because I live there, and the
        # channel is there whether or not I am standing next to it. Note these are
        # deficits of a SHARED thing: every member of a household reads the same
        # number, which is what makes a run on the store, or a collective effort to
        # fill it, something the population does together without being told to.
        needs[:, NEED_HOUSE_FOOD] = np.clip(1.0 - view.stock_food, 0.0, 1.0)
        needs[:, NEED_HOUSE_MATERIAL] = np.clip(1.0 - view.stock_material, 0.0, 1.0)

    # --- Island 3.0. Both read the family's PROPERTY off my own observation, the
    # same allowance the stockpile channels take: I know how big my house is and
    # whether we farm, because I live there.
    if cfg.housing.enabled:
        # How far over capacity the family is. Zero when everyone has a bed, and
        # zero again when nothing can be done about it (the house is at
        # `max_rooms`) -- wanting a room that cannot exist is the doomed action
        # the M3 mask deletes, wearing a need's clothes.
        # A FAMILY EXPANDS WHEN THE HOUSE IS FULL, NOT WHEN IT IS OVER-FULL, and
        # that correction was forced by the mechanic rather than chosen. A birth
        # needs a free bed, so a household can never EXCEED its capacity by
        # breeding -- it stops at it. Keying the need on overflow (the obvious
        # version, and the first one written) made `expand` unavailable forever:
        # the only path to overflow is a storm knocking the roof off. So the
        # deficit rises as the beds fill and saturates when the last one goes.
        needs[:, NEED_HOUSE_ROOM] = np.where(
            view.home_expandable,
            np.clip(1.0 - view.beds_free + view.home_overflow, 0.0, 1.0), 0.0)
    if cfg.agriculture.enabled:
        # Full deficit while the household has farming and no fields, zero once
        # it has all it may have. Note what this makes the adoption number MEAN,
        # stated here rather than discovered later, exactly as the axe's row is:
        # the scripted arbiter's planting rate is a consequence of this line. What
        # the world can honestly measure is whether the UNLOCK fires more where
        # food is short -- which nothing here scores, because the unlock is a
        # fact about hunger in world.py -- and whether a field pays against a
        # control on the same seeds.
        needs[:, NEED_LAND] = np.where(view.farming,
                                       np.clip(view.field_room, 0.0, 1.0), 0.0)
    return needs


def goal_availability(view: ObsView, cfg: Config, needs: np.ndarray,
                      acfg: ArbiterConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(available, discount, bonus), each (agents, goals).

    THE OPTION MENU, factored out of the scripted scorer so the stage-5 learned
    chooser and the utility arbiter face the IDENTICAL set of options and differ
    only in how they choose among them -- otherwise the headline comparison is
    two different games wearing one name. `available` is the goal-level analogue
    of the M3 action mask (a goal whose target does not exist cannot be wanted);
    `discount` and `bonus` are scripted-scorer inputs that a learned chooser
    ignores, returned from here only because they are computed in the same pass
    over the same targets.

    Note what this means for the comparison's honesty, stated rather than hidden:
    some availability tests encode corrections, not just possibility -- the raid
    motive gate, the store/draw surplus rules, opportunistic-only theft. Those
    corrections were made at the MECHANIC level precisely because a needs scorer
    cannot learn its way out of a degenerate loop; a learned chooser inherits
    them as shared constraints. What it gets to learn is everything the scorer
    decides with weights: which available option, when, for whom.
    """
    n = view.n
    discount = np.ones((n, N_GOALS))
    available = np.ones((n, N_GOALS), dtype=bool)
    bonus = np.zeros((n, N_GOALS))
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
        if cfg.society.enabled and cfg.society.household_theft_immunity:
            loaded = loaded & ~view.same_household
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

        d_site, _, _, _ = view.sites.nearest(programme_sites(view, cfg, acfg))
        set_target(DELIVER, d_site)
        if acfg.persist_until_goal:
            # A persisted `deliver` is the whole programme, so it is available to
            # an agent with empty hands PROVIDED it can restock: the option only
            # means something if the harvest leg has somewhere to go.
            d_supply, _, _, _ = resupply_leg(view, cfg, acfg)
            available[:, DELIVER] &= ((view.material_carried > 0.0)
                                      | np.isfinite(d_supply))
        else:
            available[:, DELIVER] &= view.material_carried > 0.0

        d_home, _, _ = shelter_target(view, cfg)
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
        bonus[:, GOAL_GIVE_FOOD] = acfg.generosity_scale

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
            bonus[:, GOAL_GIVE_MATERIAL] = acfg.generosity_scale
        else:
            available[:, GOAL_GIVE_MATERIAL] = False
    else:
        available[:, [GOAL_GIVE_FOOD, GOAL_GIVE_MATERIAL]] = False

    # --- stage 4: the household store, and raiding somebody else's
    if cfg.society.enabled:
        sc = cfg.society
        threshold = cfg.hunger.eat_threshold / cfg.hunger.max
        d_home = view.home_distance
        room_for_food = view.food < 1.0 - 1e-6
        room_for_material = view.material_carried < 1.0 - 1e-6

        # THE STORE AND THE DRAW MUST NOT BOTH BE AVAILABLE TO THE SAME AGENT, and
        # the first version of this let them be. Measured: 5858 deposits and 5475
        # withdrawals an episode, with the pile never rising above 1.14 of 12 --
        # an agent deposited its surplus, which raised its own food_stock deficit,
        # which made drawing the best goal, at the same location, forever. That is
        # exactly M5's gift farm in new clothes (42x the transfers, nothing used),
        # and the fix is the same shape: not a smaller weight, a mechanic that
        # cannot loop. A deposit needs a real SURPLUS (keep one unit back); a draw
        # needs a real SHORTAGE (be empty-handed and actually getting hungry). The
        # two conditions are now mutually exclusive by construction.
        capacity = max(cfg.food.capacity, 1)
        surplus_food = view.food * capacity >= 2.0 - 1e-6
        set_target(STORE_FOOD, d_home)
        available[:, STORE_FOOD] &= (surplus_food & (view.hunger > threshold)
                                     & (view.stock_food < 1.0 - 1e-6))
        set_target(STORE_MATERIAL, d_home)
        # Material is surplus only when there is nowhere to USE it: banking a unit
        # a visible site wants, then drawing it back out, is the treadmill in its
        # third disguise (M5's gift farm, then this store's food loop, then --
        # measured on society4_trade -- 5377 material deposits against 4678
        # withdrawals an episode). `deliverable_sites` is the same test `deliver`
        # uses, so the two goals partition the carried unit's fates instead of
        # competing for it at the same doorstep.
        can_use = deliverable_sites(view, cfg).any(axis=1)
        available[:, STORE_MATERIAL] &= ((view.material_carried > 0.0) & ~can_use
                                         & (view.stock_material < 1.0 - 1e-6))

        set_target(DRAW_FOOD, d_home)
        available[:, DRAW_FOOD] &= ((view.stock_food > 0.0) & (view.food <= 0.0)
                                    & (needs[:, NEED_HUNGER] > 0.0))
        set_target(DRAW_MATERIAL, d_home)
        # Only worth drawing if the store holds a KIND some visible incomplete
        # site actually wants -- which is why the observation carries the pantry's
        # composition. "There is material in the store and building to do" is not
        # enough: in a non-fungible world the store can be full of the wrong kind,
        # and drawing it out just re-arms the deposit half of the treadmill.
        incomplete = view.sites.present & ~view.site_complete
        useful = ((incomplete & (view.sites.extra[0] > 0.0)).any(axis=1)
                  & (view.stock_wood > 0.0)) | \
                 ((incomplete & (view.sites.extra[1] > 0.0)).any(axis=1)
                  & (view.stock_stone > 0.0))
        available[:, DRAW_MATERIAL] &= useful & (view.material_carried <= 0.0)

        d_raid = view.raid_distance
        set_target(GOAL_RAID, d_raid)
        # RAIDING IS GATED ON MOTIVE, NOT ON PROXIMITY, and that is the whole
        # lesson of stage 2's correction 1 applied to a bigger target. Theft as a
        # simple travelling goal produced the pre-registered permanent war: with
        # 100 agents packed together, the distance discount handed it every
        # contest. A stockpile is a fatter prize than a pocket, so scored the same
        # way it would be worse. So a raid needs a REASON -- either the raider is
        # in real trouble (its own larder is empty and it is hungry) or it is
        # settling a score (a grudge against someone whose household this is).
        # Both are things a watcher can see coming, which is the point.
        desperate = ((needs[:, NEED_HUNGER] >= acfg.raid_hunger)
                     & (view.stock_food <= 0.0))
        vengeful = view.grudge.max(axis=1) >= acfg.raid_grudge if view.grudge.size \
            else np.zeros(n, dtype=bool)
        has_loot = (view.raid_food > 0.0) | (view.raid_material > 0.0)
        available[:, GOAL_RAID] &= (desperate | vengeful) & has_loot & (
            room_for_food | room_for_material)
    else:
        available[:, [STORE_FOOD, STORE_MATERIAL, DRAW_FOOD, DRAW_MATERIAL,
                      GOAL_RAID]] = False


    # --- tech ladder rung 1. Available only where the mask would allow the act:
    # at a site, with a haft and a head in hand, and no axe already. Same rule as
    # every other availability test -- a goal whose target does not exist cannot
    # be wanted -- and the "no axe already" clause is what stops a tool-owner
    # spending the rest of its life wanting a second one.
    if cfg.tools.enabled:
        tc = cfg.tools
        d_site, _, _, _ = view.sites.nearest(view.sites.present)
        set_target(CRAFT_AXE, d_site)
        available[:, CRAFT_AXE] &= (
            ~view.axe
            & (view.wood * cfg.construction.material_capacity >= tc.axe_wood_cost - 1e-6)
            & (view.stone * cfg.construction.material_capacity >= tc.axe_stone_cost - 1e-6)
        )
    else:
        available[:, CRAFT_AXE] = False

    # --- Island 3.0. Both are trips to my OWN house, so both take the home
    # offset the stage-4 block already carries rather than a k-nearest slot: an
    # extension goes on the family house or it is not an extension, and a field
    # next to somebody else's larder is not this family's field.
    if cfg.housing.enabled and cfg.society.enabled:
        set_target(EXPAND, view.home_distance)
        available[:, EXPAND] &= (view.home_expandable & (view.material_carried > 0.0)
                                 & (view.beds_free <= 1e-6))
    else:
        available[:, EXPAND] = False

    if cfg.agriculture.enabled and cfg.society.enabled:
        set_target(PLANT_FIELD, view.home_distance)
        cap_m = max(cfg.construction.material_capacity, 1)
        available[:, PLANT_FIELD] &= (
            view.farming & (view.field_room > 0.0)
            & (view.material_carried * cap_m >= cfg.agriculture.plant_material_cost - 1e-6))
    else:
        available[:, PLANT_FIELD] = False

    return available, discount, bonus


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

    available, discount, bonus = goal_availability(view, cfg, needs, acfg)
    base = base + bonus

    scores = base * gate * discount * traits
    return np.where(available, scores, 0.0)


def deliverable_sites(view: ObsView, cfg: Config) -> np.ndarray:
    """Per (agent, site slot): is this an incomplete site my load can advance?

    With fungible sites (the m4h lineage) any material advances any site, so the
    test is just "incomplete". Without them a site wants its literal composition,
    and an agent carrying only stone must not walk to -- or stand refreshing a
    commitment at -- a site that needs only wood. That is the m4h deadlock
    ("all wood in, one stone missing, agents carrying 3.7 stone") reappearing at
    the GOAL level: measured on `society4_trade.yaml` before this filter, 27.1%
    of all intentions were `deliver` while completions fell 52 -> 14 and half the
    population died -- agents camped at sites the mask would never let them feed.
    """
    incomplete = view.sites.present & ~view.site_complete
    if cfg.construction.fungible_materials:
        return incomplete
    need_w = view.sites.extra[0] > 0.0
    need_s = view.sites.extra[1] > 0.0
    wants_mine = ((need_w & (view.wood[:, None] > 0.0))
                  | (need_s & (view.stone[:, None] > 0.0)))
    return incomplete & wants_mine


def programme_sites(view: ObsView, cfg: Config, acfg: ArbiterConfig) -> np.ndarray:
    """`deliverable_sites`, widened for an empty-handed agent under persistence.

    With `persist_until_goal` off this IS `deliverable_sites` and nothing
    changes. With it on, `deliver` is a whole build programme rather than a
    single drop-off, so an agent carrying nothing is still pursuing a site --
    it just has a harvest leg to do first. Without this widening the programme
    could never START empty-handed, which is the entire point of the lever:
    `deliverable_sites` asks "can my CURRENT load advance this site", and in a
    non-fungible world an empty pocket advances nothing.
    """
    sites = deliverable_sites(view, cfg)
    if not acfg.persist_until_goal:
        return sites
    incomplete = view.sites.present & ~view.site_complete
    return np.where((view.material_carried <= 0.0)[:, None], incomplete, sites)


def resupply_leg(view: ObsView, cfg: Config,
                 acfg: ArbiterConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(distance, dx, dz, action) of the harvest an empty-handed builder needs.

    The supply half of a persisted build programme: walk to the nearer of the
    nearest tree and the nearest rock and take a unit. In a NON-FUNGIBLE world
    the choice is narrowed to a kind some visible incomplete site actually
    wants, which is `deliverable_sites`' own lesson (m4h's one-stone-short
    deadlock) applied to the harvest rather than to the drop-off.
    """
    d_tree, tx, tz, _ = view.trees.nearest(view.trees.present)
    d_rock, rx, rz, _ = view.rocks.nearest(view.rocks.present)
    if not cfg.construction.fungible_materials:
        incomplete = view.sites.present & ~view.site_complete
        want_w = (incomplete & (view.sites.extra[0] > 0.0)).any(axis=1)
        want_s = (incomplete & (view.sites.extra[1] > 0.0)).any(axis=1)
        d_tree = np.where(want_w, d_tree, np.inf)
        d_rock = np.where(want_s, d_rock, np.inf)
    take_wood = d_tree <= d_rock
    return (np.where(take_wood, d_tree, d_rock),
            np.where(take_wood, tx, rx),
            np.where(take_wood, tz, rz),
            np.where(take_wood, CHOP, MINE).astype(np.int64))


def shelter_target(view: ObsView, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(distance, dx, dz) of the shelter this agent should sleep in.

    Without households that is simply the nearest finished shelter. WITH them it
    is the agent's OWN home whenever its roof is on, and the nearest finished
    shelter otherwise (a storm can take your roof off, and standing in the rain on
    principle is not loyalty).

    This is what makes a household a place rather than a label. Targeting the
    nearest finished shelter in a stage-4 world measured 55.7% of agents ending up
    closer to a foreign home than their own -- twenty stockpiles with nobody
    reliably living at any of them, so "my group is who sleeps where I sleep"
    quietly became false and every household statistic was describing a
    round-robin index rather than a group.
    """
    d_home, hx, hz, _ = view.sites.nearest(view.site_complete)
    if not cfg.society.enabled:
        return d_home, hx, hz
    own = view.home_complete
    return (np.where(own, view.home_distance, d_home),
            np.where(own, view.home_dx, hx),
            np.where(own, view.home_dz, hz))


def _heading(dx: np.ndarray, dz: np.ndarray) -> np.ndarray:
    """Compass action pointing along (dx, dz); action i is i*45deg from +z."""
    return (np.round(np.arctan2(dx, dz) / (np.pi / 4.0)) % N_MOVE_ACTIONS).astype(np.int64)


def goal_viable(view: ObsView, cfg: Config, goals: np.ndarray,
                acfg: ArbiterConfig | None = None) -> np.ndarray:
    """Can each agent still pursue the goal it currently holds?

    An option terminates when its target vanishes (the bush was emptied, the site
    was finished by somebody else) or when its purpose is served (inventory full).
    This is the termination half of the semi-MDP contract, and it is the one place
    the bookkeeping can silently rot -- so it is one function, used by both the
    scripted and (in stage 5) the learned path.
    """
    n = view.n
    acfg = acfg or ArbiterConfig()
    ok = np.ones(n, dtype=bool)
    d_bush, _, _, _ = view.bushes.nearest(view.loaded_bushes)
    room_food = view.food < 1.0 - 1e-6

    def when(goal: int, condition: np.ndarray) -> None:
        rows = goals == goal
        ok[rows] = condition[rows]

    when(FORAGE, np.isfinite(d_bush) & room_food)

    if cfg.competition.enable_steal:
        loaded = view.neighbour_food > 0.0
        if cfg.society.enabled and cfg.society.household_theft_immunity:
            loaded = loaded & ~view.same_household
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
        d_site, _, _, _ = view.sites.nearest(programme_sites(view, cfg, acfg))
        if acfg.persist_until_goal:
            # THE GOAL STATE, stated once: a persisted `deliver` ends when there
            # is no site left in view that wants material, or when the agent can
            # neither carry nor fetch one. Running out of carried material is
            # NOT the end of it any more -- that is the harvest leg, and cutting
            # the option there is exactly the compound-prize wall this lever
            # exists to remove.
            d_supply, _, _, _ = resupply_leg(view, cfg, acfg)
            when(DELIVER, np.isfinite(d_site)
                 & ((view.material_carried > 0.0) | np.isfinite(d_supply)))
        else:
            when(DELIVER, np.isfinite(d_site) & (view.material_carried > 0.0))
        d_home, _, _ = shelter_target(view, cfg)
        # Shelter is only worth holding while the night lasts.
        when(SHELTER, np.isfinite(d_home) & (view.is_night | (view.phase > 0.5)))

    if cfg.society.enabled:
        threshold = cfg.hunger.eat_threshold / cfg.hunger.max
        # A trip to the store terminates when the thing that justified it is gone
        # -- the surplus was eaten, the pile filled up, somebody else emptied it.
        # Every one of these is a state another agent can change while I walk,
        # which is the whole reason the semi-MDP contract has a termination test
        # rather than just a timeout.
        capacity = max(cfg.food.capacity, 1)
        when(STORE_FOOD, (view.food * capacity >= 2.0 - 1e-6) & (view.hunger > threshold)
             & (view.stock_food < 1.0 - 1e-6))
        when(STORE_MATERIAL, (view.material_carried > 0.0)
             & ~deliverable_sites(view, cfg).any(axis=1)
             & (view.stock_material < 1.0 - 1e-6))
        # These mirror score_goals' availability tests deliberately. A termination
        # test looser than the availability test is how a committed option outlives
        # the reason it was chosen -- and here that would have re-opened the
        # deposit/withdraw treadmill one tick at a time.
        when(DRAW_FOOD, (view.stock_food > 0.0) & (view.food <= 0.0))
        when(DRAW_MATERIAL, (view.stock_material > 0.0) & (view.material_carried <= 0.0))
        d_raid = view.raid_distance
        when(GOAL_RAID, np.isfinite(d_raid)
             & ((view.raid_food > 0.0) | (view.raid_material > 0.0)))

    # EXPLORE ENDS WHEN THE SEARCH SUCCEEDS. It used to be unconditionally viable,
    # which meant an agent that set off looking for food kept walking for the full
    # 25-tick commitment even if a loaded bush came into view on tick three.
    # Measured, that put 28.3% of all goal-ticks into `explore` while the mean
    # forage score among the explorers was 0.40 against explore's 0.17 -- they
    # were not choosing to wander, they were serving out a commitment whose reason
    # had expired. Searching is the one option whose purpose is a perception, so
    # its termination test is a perception too.
    if cfg.tools.enabled:
        # A craft ends the instant the axe exists -- there is nothing further to
        # pursue. Also ends if the materials go (raided, or spent on a wall),
        # which is the same "another agent changed the state while I walked" case
        # the stockpile goals guard against.
        cap_m = max(cfg.construction.material_capacity, 1)
        when(CRAFT_AXE, ~view.axe
             & (view.wood * cap_m >= cfg.tools.axe_wood_cost - 1e-6)
             & (view.stone * cap_m >= cfg.tools.axe_stone_cost - 1e-6))

    if cfg.housing.enabled:
        # Ends when the family is housed, when the house can take no more rooms,
        # or when the material is gone (spent, given, raided out of the pocket).
        # Mirrors the availability test deliberately: a termination test looser
        # than the availability test is how a committed option outlives its
        # reason, which is the correction `explore` and the store goals both
        # record.
        when(EXPAND, view.home_expandable & (view.beds_free <= 1e-6)
             & (view.material_carried > 0.0))
    if cfg.agriculture.enabled:
        when(PLANT_FIELD, view.farming & (view.field_room > 0.0)
             & (view.material_carried > 0.0))

    when(EXPLORE, ~(view.loaded_bushes.any(axis=1) & room_food))
    # The gifts are single-tick by nature and `rest` never fails.
    when(GOAL_GIVE_FOOD, np.zeros(n, dtype=bool))
    when(GOAL_GIVE_MATERIAL, np.zeros(n, dtype=bool))
    return ok


def execute_goals(goals: np.ndarray, view: ObsView, cfg: Config,
                  mask: np.ndarray, explore_heading: np.ndarray,
                  acfg: ArbiterConfig | None = None) -> np.ndarray:
    """Turn goal ids into primitive actions. The scripted "muscles".

    Every branch is walk-there-then-act: head for the target while out of range,
    take the action once the mask says it can succeed. Reading the mask is the
    same allowance the 1.0 scripted builder and trader take, and the learned
    policy receives it too, so nothing is being smuggled in.
    """
    n = view.n
    acfg = acfg or ArbiterConfig()
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
        if cfg.society.enabled and cfg.society.household_theft_immunity:
            loaded = loaded & ~view.same_household
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
        remaining = np.where(programme_sites(view, cfg, acfg), view.site_remaining, np.inf)
        j = np.argmin(remaining, axis=1)
        rows = np.arange(n)
        has_site = np.isfinite(remaining[rows, j])
        sx, sz = view.sites.dx[rows, j], view.sites.dz[rows, j]
        d_site = np.where(has_site, np.hypot(sx, sz), np.inf)
        walk_then(DELIVER, d_site, sx, sz, BUILD, cc.build_radius)

        if acfg.persist_until_goal:
            # The programme's supply leg. Same walk-there-then-act shape as every
            # other controller -- it is one more scripted muscle, not a decision:
            # an agent that chose "build that shelter" and is holding nothing
            # goes and gets something. This is where the honest cost of the lever
            # lives, and every write-up says so.
            restock = (goals == DELIVER) & (view.material_carried <= 0.0)
            if restock.any():
                d_sup, ux, uz, act = resupply_leg(view, cfg, acfg)
                legal = np.take_along_axis(mask, act[:, None], axis=1).ravel()
                arrived = (d_sup <= cc.harvest_radius) & legal
                chosen = np.where(arrived, act, _heading(ux, uz))
                chosen = np.where(np.isfinite(d_sup), chosen, IDLE)
                actions = np.where(restock, chosen, actions)

        d_home, hx, hz = shelter_target(view, cfg)
        shelter_rows = goals == SHELTER
        if shelter_rows.any():
            # "Inside" is well within the radius, not on its lip: an agent that
            # stops at the boundary drifts out again on the next jitter.
            inside = d_home <= cc.shelter_radius * 0.6
            chosen = np.where(inside, IDLE, _heading(hx, hz))
            chosen = np.where(np.isfinite(d_home), chosen, IDLE)
            actions[shelter_rows] = chosen[shelter_rows]

        if cfg.tools.enabled:
            # Walk to the nearest site and make the axe there. Same
            # walk-there-then-act muscle as every other goal; nothing here
            # decides WHETHER to craft, which is the arbiter's job and, one day,
            # a learned chooser's.
            d_shop, wx, wz, _ = view.sites.nearest(view.sites.present)
            walk_then(CRAFT_AXE, d_shop, wx, wz, CRAFT, cfg.tools.craft_radius)

    if cfg.exchange.enabled:
        for goal, action in ((GOAL_GIVE_FOOD, GIVE_FOOD), (GOAL_GIVE_MATERIAL, GIVE_MATERIAL)):
            rows = (goals == goal) & mask[:, action]
            actions[rows] = action

    if cfg.society.enabled:
        sc = cfg.society
        walk_then(STORE_FOOD, view.home_distance, view.home_dx, view.home_dz,
                  DEPOSIT_FOOD, sc.stockpile_radius)
        walk_then(STORE_MATERIAL, view.home_distance, view.home_dx, view.home_dz,
                  DEPOSIT_MATERIAL, sc.stockpile_radius)
        walk_then(DRAW_FOOD, view.home_distance, view.home_dx, view.home_dz,
                  WITHDRAW_FOOD, sc.stockpile_radius)
        walk_then(DRAW_MATERIAL, view.home_distance, view.home_dx, view.home_dz,
                  WITHDRAW_MATERIAL, sc.stockpile_radius)
        walk_then(GOAL_RAID, view.raid_distance, view.raid_dx, view.raid_dz,
                  RAID, sc.stockpile_radius)
        # --- Island 3.0. Both are walk-home-then-act, like the store goals.
        # `expand` uses BUILD at the family site and `plant` uses PLANT near it,
        # and neither decides WHETHER -- that is the arbiter's job, and one day a
        # learned chooser's.
        if cfg.housing.enabled:
            walk_then(EXPAND, view.home_distance, view.home_dx, view.home_dz,
                      BUILD, cfg.construction.build_radius)
        if cfg.agriculture.enabled:
            walk_then(PLANT_FIELD, view.home_distance, view.home_dx, view.home_dz,
                      PLANT, cfg.agriculture.plant_radius)

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
        # HEREDITY GETS ITS OWN STREAM, for the reason shocks and predators have
        # theirs: mutation draws must not shift the layout, the spawn positions
        # or a softmax draw of an otherwise identical world.
        self._birth_rng = np.random.default_rng(seed + 40507)

    def on_births(self, births, cfg: Config | None = None) -> None:
        inherit_traits(self.traits, self._birth_rng, births, cfg or self.cfg)

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


def option_interrupted(needs: np.ndarray, goals: np.ndarray, acfg: ArbiterConfig,
                       decided_safe: np.ndarray | None = None) -> np.ndarray:
    """Which running options a lower-tier emergency drops. One function, because
    `OptionRunner` and the stage-5 trainer both need it and a drift between them
    would change what a decision IS without changing any output shape.

    TIER 0, always: hunger past `critical`, unless the option already serves
    hunger. Without it a committed forager walks past its own death; with too
    loose a rule every tick becomes a decision and the commitment is decorative.

    THE CURFEW, only under `persist_until_goal`, and it is a correction the first
    run of this lever forced. A 25-tick budget re-opened every agent's choice at
    least twice per dusk lead for free; a 150-tick one does not, so a raid or an
    explore decided at noon runs straight through the night. Measured on the
    scripted population, that cost 585.8 -> 525.3 ticks of life, took nights
    indoors from 86.4% to 60.9% and flattened the commute (day 9.5/night 5.3
    became 12.0/11.9). Safety is tier 1 -- the second thing in this world that
    can kill -- so it gets the same treatment hunger already had.

    It fires ONCE per option, on `decided_safe`: an option chosen before the dusk
    ramp began is re-opened when it begins, and whatever is chosen then stands.
    A night owl that looks at the sky and forages on keeps its commitment, which
    is the difference between a curfew and a veto -- re-testing every tick would
    hand every agent a decision per tick for a third of the day.
    """
    emergency = needs[:, NEED_HUNGER] >= acfg.critical
    pursuing_food = np.isin(goals, (FORAGE, GOAL_STEAL, DRAW_FOOD, GOAL_RAID))
    interrupted = emergency & ~pursuing_food
    if acfg.persist_until_goal and decided_safe is not None:
        # `> 0` is the START of the ramp, not `critical`: the ramp's lead is
        # derived from how long crossing the observable range takes (see
        # compute_needs), so it is already the moment "you would have to set off
        # now" -- where `critical` lands ~15 ticks before dark on this island.
        curfew = (needs[:, NEED_SAFETY] > 0.0) & (goals != SHELTER)
        interrupted = interrupted | (curfew & decided_safe)
    return interrupted


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

    def __init__(self, arbiter, cfg: Config, seed: int = 0, world=None) -> None:
        self.arbiter = arbiter
        self.cfg = cfg
        self.acfg = getattr(arbiter, "acfg", ArbiterConfig())
        # HEREDITY NEEDS A PEDIGREE, and the pedigree lives in the world: the
        # world records who the parents were, the arbiter owns the traits. The
        # runner is the only object that sees both, so it is where the two meet
        # -- and a driver that forgets to hand the world over would silently give
        # every newborn its slot's founding traits, which is the failure this
        # raise exists to make impossible rather than subtle.
        self.world = world
        rc = cfg.reproduction
        if rc.enabled and rc.heritable_traits and world is None:
            raise ValueError(
                "reproduction.heritable_traits is on but OptionRunner was given "
                "no world: newborns would silently inherit nothing. Pass "
                "world=... (sim.society.make_runner does).")
        n = cfg.world.num_agents
        self.rng = np.random.default_rng(seed)
        self.goals = np.full(n, REST, dtype=np.int64)
        self.ticks_left = np.zeros(n, dtype=np.int64)
        self.explore_heading = self.rng.integers(0, N_MOVE_ACTIONS, size=n)
        self.decisions = np.zeros(n, dtype=np.int64)
        self.goal_ticks = np.zeros(N_GOALS, dtype=np.int64)
        self.last_decided = np.zeros(n, dtype=bool)
        # "this option was chosen before the dusk ramp began", so the curfew
        # fires once per option rather than every tick of every evening.
        self.decided_safe = np.ones(n, dtype=bool)

    def reset(self) -> None:
        self.goals[:] = REST
        self.ticks_left[:] = 0
        self.decisions[:] = 0
        self.goal_ticks[:] = 0
        self.decided_safe[:] = True

    def act(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Births from the PREVIOUS step, before anything is scored. A newborn is
        # created inside `World.step`, so the first tick it can act on is this
        # one -- and its traits have to be in place before its first goal is
        # chosen, not after.
        if self.world is not None:
            births = getattr(self.world, "last_births", None)
            if births:
                on_births = getattr(self.arbiter, "on_births", None)
                if on_births is not None:
                    on_births(births, self.cfg)
        """One tick: keep or re-decide each agent's goal, then execute it."""
        view = ObsView(obs, self.cfg)
        needs = compute_needs(view, self.cfg)
        viable = goal_viable(view, self.cfg, self.goals, self.acfg)

        interrupted = option_interrupted(needs, self.goals, self.acfg,
                                         self.decided_safe)

        redecide = (~viable) | (self.ticks_left <= 0) | interrupted
        if redecide.any():
            fresh = self.arbiter.choose(view, mask, self.rng)
            self.goals = np.where(redecide, fresh, self.goals)
            self.ticks_left = np.where(redecide, commit_budget(self.acfg, self.goals),
                                       self.ticks_left)
            self.decided_safe = np.where(redecide, needs[:, NEED_SAFETY] <= 0.0,
                                         self.decided_safe)
            self.decisions += redecide
            # A new explore option gets a new heading; re-rolling it every tick
            # would turn ballistic travel back into a random walk.
            new_explore = redecide & (self.goals == EXPLORE)
            if new_explore.any():
                roll = self.rng.integers(0, N_MOVE_ACTIONS, size=view.n)
                self.explore_heading = np.where(new_explore, roll, self.explore_heading)
        self.last_decided = redecide

        self.ticks_left -= 1
        # COUNT THE LIVING ONLY. A dead agent's row is all-False except `idle`,
        # so "can I move" is an exact aliveness test and needs no new plumbing.
        # This was harmless while the cast was fixed and a run lost three agents;
        # Island 3.0 breaks it outright, because a world with 200 slots and 40
        # agents alive spends 80% of its goal-ticks in rows that do not exist.
        # Rule 6 again: a rate is dominated by whichever states dominate the
        # denominator, and here most of them were nobody.
        active = mask[:, :N_MOVE_ACTIONS].any(axis=1)
        np.add.at(self.goal_ticks, self.goals[active], 1)
        return execute_goals(self.goals, view, self.cfg, mask, self.explore_heading,
                             self.acfg)


def utility_runner(cfg: Config, seed: int = 0,
                   acfg: ArbiterConfig | None = None, world=None) -> OptionRunner:
    """The stage-2 population: scripted arbiter, scripted controllers."""
    return OptionRunner(UtilityArbiter(cfg, acfg, seed=seed), cfg, seed=seed,
                        world=world)
