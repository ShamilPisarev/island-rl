"""Typed configuration loaded from YAML.

Everything tunable lives in ``config/default.yaml``; this module turns it into
frozen dataclasses so typos surface at load time rather than as an AttributeError
2000 updates into a training run. ``Config.to_dict`` round-trips back to plain
data so a run's exact configuration can be embedded in its replay file.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


@dataclass(frozen=True)
class WorldConfig:
    island_radius: float = 40.0
    num_agents: int = 6
    max_ticks: int = 600
    move_step: float = 0.8
    spawn_radius_frac: float = 0.6
    # An action, once chosen, persists for this many ticks; agents decide only on
    # ticks where tick % decision_interval == 0. Enforced by World.step itself, so
    # every driver (trainer, evaluation, replay, scripted baselines) lives under
    # the same commitment and none of them can drift.
    #
    # The temporal-resolution lever from the nav-spread postmortem: the per-step
    # advantage of walking toward food is ~0.01-0.05 against ~0.9 of advantage
    # noise (sim.advantage), so a single-tick direction choice is invisible to
    # PPO. Committing for k ticks multiplies the per-decision slope by k without
    # touching the reward. 1 reproduces every earlier world bit-identically.
    decision_interval: int = 1


@dataclass(frozen=True)
class HungerConfig:
    max: float = 100.0
    drain_per_tick: float = 0.5
    eat_threshold: float = 60.0
    eat_restore: float = 35.0


@dataclass(frozen=True)
class FoodConfig:
    capacity: int = 3


@dataclass(frozen=True)
class BushConfig:
    num_clusters: int = 5
    bushes_per_cluster: int = 4
    cluster_std: float = 5.0
    cluster_radius_frac: float = 0.75
    capacity: int = 6
    initial_berries: int = 6
    regrow_ticks: int = 50
    gather_radius: float = 2.0
    resample_each_episode: bool = True

    @property
    def count(self) -> int:
        return self.num_clusters * self.bushes_per_cluster


@dataclass(frozen=True)
class CompetitionConfig:
    """Milestone 3. All off by default, so M1/M2 worlds are bit-identical."""

    contest_bushes: bool = False   # only one agent may take from a bush per tick
    exclusive_bushes: bool = False  # ...and only the closest agent may take at all
    enable_steal: bool = False     # adds an 11th action: take a berry from a neighbour
    steal_radius: float = 2.5
    observe_neighbour_food: bool = False  # neighbours' carried food enters the observation
    observe_bush_contested: bool = False  # per bush: is a living rival closer than me?
    mask_invalid_actions: bool = False    # hide gather/steal when they cannot succeed


@dataclass(frozen=True)
class ConstructionConfig:
    """Milestone 4. All off by default, so M1-M3 worlds are bit-identical."""

    enabled: bool = False
    # material nodes, placed like bushes (clustered)
    num_trees: int = 8
    tree_wood: int = 6              # units per tree; no regrowth within an episode
    num_rocks: int = 5
    rock_stone: int = 4
    # Island 3.0: material nodes grow back. 0 is "never", which is every world
    # before 3.0 -- and that asymmetry (bushes regrow forever, trees never) is
    # what ISLAND2_DESIGN.md section 15 measured killing a society over 24,000
    # ticks. A tree refills one unit every `tree_regrow_ticks` up to the stock
    # it started with, exactly as a bush does. Deliberately NOT suspended by a
    # blight: a blight is a failure of the berry crop, and having it stop the
    # forest too would make one shock do two jobs and make the ramp world
    # unreadable.
    tree_regrow_ticks: int = 0
    rock_regrow_ticks: int = 0
    harvest_radius: float = 2.0
    material_capacity: int = 2      # carried wood+stone combined
    # shelter sites
    num_sites: int = 3
    sites_at_clusters: bool = False  # place shelter sites on the berry clusters
    # Place trees and rocks on the berry clusters too.
    #
    # m4b moved the sites onto the clusters and stopped there, which relocated
    # the uncreditable walk to the harvest leg instead of deleting it. Measured
    # on the m4c-anneal policy: mean distance to the nearest tree 10.2 and to the
    # nearest rock 15.5, against 2.4 to the nearest bush, and agents stand within
    # harvest_radius on 2.7% of ticks -- so `chop` is *reachable* on 0.2% of
    # ticks, and it is taken on 82-93% of those. The policy is not declining to
    # harvest; it is almost never in a position to.
    materials_at_clusters: bool = False
    # Let either material count toward a site's remaining cost.
    #
    # With separate wood and stone costs, sites deadlock on COMPOSITION rather
    # than on volume. Measured on m4g: 22.5% of sites end an episode with all
    # their wood in and one stone missing, 10% the reverse, and agents finish
    # holding 3.7 stone -- the material exists, it is simply in the wrong hands,
    # because each agent feeds its nearest site whatever it happens to carry and
    # nothing routes the last stone to the site that wants it. The scripted
    # builder dodges this by targeting one focal site globally; six independent
    # brains have no such coordination.
    #
    # Fungible deliveries remove the mismatch without removing either economy:
    # both wood and stone still have to be harvested and carried, and `mine` is
    # still worth taking, but a site takes whatever arrives.
    fungible_materials: bool = False
    site_wood_cost: int = 4         # delivered units to complete a site
    site_stone_cost: int = 2
    build_radius: float = 2.5
    shelter_radius: float = 6.0     # protection range of a COMPLETED shelter
    # the hazard shelter protects from
    night_cycle: int = 200          # ticks per full day
    night_fraction: float = 0.25    # last quarter of each cycle is night
    night_drain_multiplier: float = 3.0
    partial_shelter: bool = False    # half-built walls give half the protection
    # How much of a shelter's protection is withheld until it is FINISHED.
    #
    # partial_shelter alone makes protection linear in build progress, which
    # fixed the credit-assignment cliff by removing the reason to complete: the
    # last unit buys exactly what the first one did. Measured on m4c, every unit
    # of a 4-unit site is worth 0.25 of the night drain, final unit included.
    # That is why nobody finishes -- not a perception failure, correct play.
    #
    # With a premium p, a site at progress q protects q*(1-p) and a finished one
    # protects 1.0. Every unit still buys something (the gradient partial_shelter
    # was introduced for survives) and the last one additionally buys p. 0.0
    # reproduces every earlier M4 result exactly.
    completion_premium: float = 0.0
    # observation channels
    k_trees: int = 2
    k_rocks: int = 2
    k_sites: int = 2
    # One extra site channel: "one more unit, of the kind I am already carrying,
    # finishes this". Derivable in principle from own.wood/own.stone and the
    # site's need channels, but only as a conjunction across distant parts of a
    # 55-dim vector, which is exactly the shape of thing a small MLP does badly.
    # Same call as observe_neighbour_food before theft and the M3 action mask:
    # a mechanic the policy cannot see is one it cannot respond to.
    observe_final_unit: bool = False


@dataclass(frozen=True)
class ExchangeConfig:
    """Milestone 5. Off by default, so M1-M4 worlds are bit-identical.

    Enabling exchange appends two actions, ``give_food`` and ``give_material``.
    Food and materials are split because they are two different economies -- one
    keeps an agent alive, the other builds shelter -- and an agent carrying both
    would otherwise be unable to choose which it is participating in. Wood and
    stone share one action: both are construction inputs, the receiver's ``build``
    already resolves which the site needs, and a third give action would be a
    third thing PPO has to discover the value of.
    """

    enabled: bool = False
    give_radius: float = 2.5          # same reach as a steal, so giving is not cheaper
    observe_neighbour_materials: bool = False  # neighbours' carried wood/stone
    log_transfers: bool = False       # accumulate a per-episode transfer ledger


@dataclass(frozen=True)
class SocietyConfig:
    """Island 2.0 stage 4: regions, households, stockpiles, reputation, shocks.

    All off by default, so every 1.0 world (and stages 1-3) stays bit-identical.
    Enabling this appends five actions -- deposit/withdraw food and material, and
    raid -- and widens the observation; the append-never-insert rule that carried
    an M1 checkpoint into M3 applies here too.

    WHY THESE FIVE MECHANICS AND NOT MORE. Stage 2 measured what the 100-agent
    utility population was missing, and every item here answers one of those
    findings rather than being a feature someone fancied:

      * `region_split` -- stage 2's world had wood and stone on every cluster
        (the m4f lineage's `materials_at_clusters`), so nobody ever needed
        anything from anybody. M5's own postmortem says a relay needs its chain
        shortened by GEOGRAPHY, not its deliveries made fungible; splitting the
        island is that.
      * households + stockpiles -- "construction is over by the first nightfall"
        (20 sites x 4 units against 100 agents carrying one each), and lifespan
        Gini was 0.058 because there was nothing for inequality to accumulate
        IN. A stockpile is a store, a sustained demand, and a raid target.
      * reputation -- retaliation and guarding without any scripted war logic.
      * shocks -- populations that never get stressed never visibly cooperate.
        Deterministic from the seed, like everything else here.
    """

    enabled: bool = False

    # --- regions. Trees to one side of the island, rocks to the other, berries
    # everywhere. `region_axis` is the compass angle (radians) of the wood half's
    # outward normal; 0 puts wood at +z ("north") and stone at -z.
    region_split: bool = False
    region_axis: float = 0.0

    # --- households. Agents are dealt round-robin, so a household is a stable
    # group from tick 0 and "my group" needs no learning to identify. Each
    # household owns the site of the same index (num_sites must be >= this), and
    # that site's position is also its stockpile's.
    num_households: int = 1
    observe_household: bool = False   # per-neighbour same-household flag

    # --- stockpiles. Capacity is per household, not per agent, which is the
    # whole point: it is the first thing in this project bigger than a pocket.
    stockpile_food_capacity: int = 12
    stockpile_material_capacity: int = 12
    stockpile_radius: float = 2.5
    # May an agent rob a member of its OWN household?
    #
    # This is a correction, not a taste call, and it is stage 2's correction 1
    # arriving in a new world exactly as rule 5 says to expect. Stage 4 spawns a
    # household together at its own site, which packs five agents inside
    # `steal_radius` of each other permanently -- and an opportunistic steal goal
    # that only needs a loaded victim in reach then fires every tick. Measured:
    # 8105 steals an episode against stage 2's 5.8% of goal-ticks becoming 13.8%,
    # tripping the pre-registered PERMANENT WAR check, and cascading into raids
    # because every housemate held a saturated grudge against every other.
    #
    # Immunity is also the coherent reading of the mechanic: a household shares a
    # stockpile, so taking from a housemate's pocket is not how you get food out
    # of your own group -- `withdraw` is. Theft stays available against everyone
    # else, which is what keeps competition between households real.
    household_theft_immunity: bool = False

    # --- reputation. grudge[i, j] in [0, 1] is how much i remembers j taking
    # from it (a steal from i, or a raid on i's household stockpile). Decays
    # geometrically so an old robbery stops mattering, which is what lets a
    # feud end.
    reputation: bool = False
    grudge_per_theft: float = 0.34    # three thefts saturate the memory
    grudge_decay: float = 0.995       # half-life ~138 ticks
    observe_grudge: bool = False      # per-neighbour grudge channel

    # --- shocks. Every `shock_interval` ticks one fires, alternating
    # deterministically between a blight (berry regrowth stops) and a storm
    # (every finished shelter loses units and needs rebuilding). 0 disables.
    shock_interval: int = 0
    blight_ticks: int = 60            # how long a blight suspends regrowth
    storm_damage: int = 2             # units knocked out of each finished site
    observe_shock: bool = False       # one channel: is a blight running
    # Escalation: shock SEVERITY grows linearly with episode progress. A shock
    # at tick t is scaled by 1 + shock_ramp * t/max_ticks, so at ramp 2.0 the
    # last shock of an episode is ~3x the first (blights ~3x longer, storms
    # ~3x the damage, clamped so a site never needs more than it costs). The
    # cadence is untouched -- same clock, same seed stream -- so 0.0 reproduces
    # every earlier world bit-identically. sim.economy models the ramp; re-size
    # before reading anything behavioural off a ramped world (rule 5).
    shock_ramp: float = 0.0


@dataclass(frozen=True)
class ToolsConfig:
    """Island 2.0 tech ladder, rung 1: a craftable axe.

    Off by default, so every 1.0 world and every island2 stage stays
    bit-identical. Enabling it appends ONE action (`craft`, index 21) and ONE
    observation channel (`own.axe`), by the append-never-insert rule that carried
    an M1 checkpoint into M3 -- and, like `society`, enabling it implies the whole
    block below it is present, so `craft` is a fixed index in every tool world.

    WHY AN AXE AND WHY THIS SHAPE. It is the cheapest thing in this project that
    is honestly a *technology*: a durable object an agent makes out of two things
    it already gathers, which then changes the rate at which it can gather one of
    them. The chain is short enough to be creditable (craft once, benefit every
    chop afterwards) -- which matters, because every long chain this project has
    tried has failed, and the point of rung 1 is to establish the ladder, not to
    re-run the credit-assignment wall.

    IT DOES NOT CREATE WOOD. Trees hold a finite stock, so an axe buys TICKS, not
    supply: the same wood arrives in half the trips. `sim.economy` says which of
    those two the world is actually short of, and it says so before a config is
    sized (rule 5) -- if material stock is the binding constraint, the axe cannot
    help and the run would measure nothing.

    An axe is permanent once made. Durability was left out deliberately: it is a
    second mechanic (a decay rate to tune) wearing the same name, and rung 1 has
    to answer "does a tool get adopted at all" before anything is tuned.
    """

    enabled: bool = False
    axe_wood_cost: int = 1
    axe_stone_cost: int = 1
    # Yield per successful chop while holding an axe. 2 is the rung-1 setting;
    # the multiplier is a config knob so the adoption question can be asked at a
    # price the agents can actually see.
    chop_multiplier: int = 2
    # An axe is made AT A SITE -- the household's own building spot doubles as the
    # workshop. Somewhere rather than anywhere, because a tool you can make while
    # standing in a berry patch is not a technology, it is an inventory slot.
    craft_radius: float = 2.0
    observe_axe: bool = True


@dataclass(frozen=True)
class PredatorConfig:
    """Island 2.0 tech ladder, rung 2: a night predator.

    Off by default, so every world before it stays bit-identical. Unlike rung 1
    this appends NO action and NO goal -- only observation channels. That is
    deliberate on two counts. There is nothing to do about a predator that the
    world does not already offer (be indoors, or be somewhere else), and adding
    a `flee` goal would score the response and then measure the score, which is
    rule 1 wearing a scripted arbiter's clothes. The honest question a rung-2
    world asks is whether the night behaviour a population ALREADY has is enough.

    DETERMINISTIC FROM THE SEED, and with no per-tick randomness at all: a
    predator's den is drawn once at reset and every move afterwards is a function
    of positions. So a replay reproduces exactly, and adding predators cannot
    shift the bush layout or the spawn positions of an otherwise identical world.

    IT IS A DEMAND-SIDE CHANGE, so `sim.economy` had to be taught about it before
    any config was sized (rule 5): a hunted night costs hunger the same way an
    exposed one does, and the exposed demand rises accordingly.
    """

    enabled: bool = False
    count: int = 4
    # Faster than an agent (move_step 0.8), or nothing is ever caught -- but not
    # so fast that reaching a shelter is hopeless, which would make the mechanic
    # a tax rather than a hazard.
    speed: float = 1.0
    attack_radius: float = 2.0
    # Hunger taken per tick of contact. Comparable to a night's exposure drain
    # (0.5 x 3.0 = 1.5/tick) so being caught is serious without being instant.
    damage: float = 1.5
    # A predator only hunts agents whose shelter protection is below this. An
    # agent indoors is safe: the shelter has to be the answer, or the mechanic
    # teaches nothing about shelter.
    protection_safe: float = 0.5
    # By day predators walk back to their den and wait. Somewhere to be, so the
    # island is not permanently patrolled and a daytime forager is free.
    den_radius_frac: float = 0.75
    observe_predator: bool = True


@dataclass(frozen=True)
class ReproductionConfig:
    """Island 3.0: the population is born rather than cast.

    Off by default, so every 1.0 and 2.0 world stays bit-identical -- and note
    what "off" means here, because it is not the usual inert-block story. With
    reproduction off, `world.num_agents` is the population, every slot is alive
    at tick 0, and every statistic divides by it exactly as it always has. With
    it on, `world.num_agents` becomes a SLOT CAPACITY: `initial_agents` are alive
    at tick 0 and the rest are unborn, occupying inert rows so every array in the
    engine stays rectangular (the same trick that lets a dead agent keep its
    slot). Lifespan and death counts then divide by the agents that were actually
    BORN, or a world with room to grow would report a lifespan halved by rows
    that never lived.

    A BIRTH IS NOT AN ACTION AND NOT A GOAL. It happens when two grown, well-fed
    members of one household are at their own house, the house has a free bed,
    the family larder can pay for the child, and the household's cooldown has
    elapsed. This is the auto-eat decision applied to the thing the stage is
    about: if `reproduce` were a goal, the utility scorer would set how badly a
    family wants children and "the strongest tribe reproduces" would restate a
    weight we typed. Automatic and conditioned on prosperity, a birth becomes a
    MEASUREMENT of how well a household feeds and houses itself.

    Deterministic, like everything else here: candidate parents are taken in
    agent order, so a replay reproduces exactly and no rng stream is consumed.
    """

    enabled: bool = False
    # How many of `world.num_agents` slots start alive. None = all of them, which
    # is the pre-3.0 world and leaves no room to grow.
    initial_agents: int | None = None
    # Both parents must be above this. Well above `eat_threshold` (60) on
    # purpose: a family that is merely not starving should not be breeding, or
    # the mechanic stops discriminating between households.
    birth_hunger: float = 80.0
    birth_food_cost: int = 4        # taken from the household stockpile
    birth_cooldown: int = 150       # ticks before that household may do it again
    birth_radius: float = 3.0       # both parents within this of the home site
    # A child is a mouth before it is a pair of hands. It cannot chop, mine,
    # build, craft, plant, steal or raid until it grows up -- it can walk,
    # gather, eat, and use the family store.
    maturity_ticks: int = 200
    child_drain_frac: float = 0.6   # ...and it eats less while it is one
    max_age: int = 0                # 0 = nobody dies of old age
    observe_age: bool = True

    # --- stage 2: the generational horizon, and heredity.
    #
    # A SLOT IS USED ONCE unless this is on, and that is a deliberate contract
    # rather than an oversight: a dead agent keeps its row forever, which is what
    # makes every per-agent statistic in this project mean something. The cost is
    # a hard ceiling on how many LIVES an episode can hold -- 40 founders plus 160
    # births exhausts 200 rows -- and R6 measured a village going extinct against
    # that ceiling and nearly reported it as mortality.
    #
    # With reuse on, a birth may take the row of an agent that died at least
    # `reuse_delay` ticks ago (oldest death first, so the choice is
    # deterministic). Per-LIFE bookkeeping moves into a ledger: `alive_ticks` is
    # pushed to it and zeroed, `age` resets, and the grudge matrix's ROW AND
    # COLUMN are cleared, because a new person is owed nothing and owes nothing.
    # Counters that mean "this row's contribution to the episode" -- berries,
    # builds, steals -- are not reset, because they are population totals.
    # EVERY FOUNDER IS THE SAME AGE unless this is on, and with `max_age` that
    # means the entire first generation dies on the SAME TICK -- a synchronised
    # die-off that a real population never has, and a second cause of R6's
    # extinction that nobody named at the time. Staggering spreads the founding
    # ages evenly over [maturity, max_age), deterministically and without
    # touching any rng stream. Off by default, so every stage-1 world (R6
    # included) reproduces exactly, and it does nothing at all when `max_age` is
    # 0, because then nobody dies of age and a founder's age never matters.
    stagger_founders: bool = False
    reuse_slots: bool = False
    # A row stays a corpse for this long before anyone can be born into it. Not
    # cosmetic: the replay draws a corpse folding forward over several ticks, and
    # a row that flips straight back to a walking newborn reads as a resurrection.
    reuse_delay: int = 50
    # A child's arbiter traits are the GEOMETRIC mean of its parents' times a
    # lognormal mutation -- geometric because the traits are lognormal about 1.0,
    # so the arithmetic mean would drift the population upward for free.
    #
    # This is the one mechanic in the stage that can produce behaviour nobody
    # wrote. What it can produce is bounded and every write-up says so: a trait is
    # a multiplier on a GOAL's score, so what evolves is how much a lineage wants
    # each of the goals that already exist -- never a new goal and never a new way
    # of pursuing one.
    heritable_traits: bool = False
    trait_mutation: float = 0.15


@dataclass(frozen=True)
class HousingConfig:
    """Island 3.0: a house holds a family, and can be made bigger.

    Until now `shelter_radius` was a disc that protected everyone inside it, so
    one hut could shelter a hundred agents and a house was a place rather than a
    thing with room in it. A finished house now shelters `base_occupants`, plus
    `occupants_per_room` for each room built onto it, and when more agents stand
    in range than there are beds the household that OWNS the site is admitted
    first (each household owns the site of its own index) and the rest sleep out.

    Expansion needs no new action: `build` at a finished site adds a unit to
    `site_extra`, and every `expand_units` of those is a room. So a family whose
    children have outgrown the house has somewhere for its labour to go, and the
    material economy gains a recurring demand that stage 4 had to invent storms
    to create.

    Admission is greedy per site in a fixed order (kin first, then by distance),
    which is deterministic and cheap. It is not a global optimal assignment, and
    it does not need to be: an agent takes the best protection any site admits it
    to, so a greedy pass can only under-house, never mis-report.
    """

    enabled: bool = False
    base_occupants: int = 4
    expand_units: int = 2          # delivered units per extra room
    occupants_per_room: int = 2
    max_rooms: int = 6
    kin_priority: bool = True
    observe_house: bool = True     # home.beds_free, home.occupancy


@dataclass(frozen=True)
class AgricultureConfig:
    """Island 3.0 tech rung 3: a field, unlocked by going hungry.

    A FIELD IS A BUSH. Planting appends nothing: field slots are preallocated
    inactive at reset (position (0,0), zero berries, `bush_active` false) and
    planting switches one on where the planter stands. Everything downstream --
    the observation, the action mask, `gather`, the `forage` goal,
    `sim.navigation` -- already tests `berries > 0`, so an unplanted slot is
    invisible and a planted one is just a bush that refills faster. That is why
    agriculture costs one action and no new machinery, and why the block being
    off is bit-identical rather than merely inert.

    THE INVENTION IS SCORED, NOT DISCOVERED, and every write-up has to say so. A
    household unlocks farming once its living members have spent
    `unlock_hunger_ticks` agent-ticks below `unlock_hunger` -- necessity is the
    mother of invention because we wrote that down. What is genuinely measured is
    whether the unlock fires more in a hungry world than a fed one (a fact about
    the world, not the rule), and whether a field once available gets used and
    pays. Same honesty the axe rung carries.
    """

    enabled: bool = False
    unlock_hunger: float = 35.0        # "hungry" for the purposes of inventing
    unlock_hunger_ticks: int = 300     # cumulative agent-ticks, per household
    plant_material_cost: int = 1       # a unit of wood or stone becomes a field
    plant_radius: float = 8.0          # ...within this of the household's own site
    max_fields_per_household: int = 3
    field_capacity: int = 8
    field_regrow_ticks: int = 25       # much faster than a wild bush
    field_initial: int = 0             # a new field starts empty and has to grow
    observe_agriculture: bool = True

    # --- stage 2: technology spreads by CONTACT. A household that lacks a tech
    # accumulates one tick of learning for every tick one of its living members
    # stands within `teach_radius` of a member of a household that has it; at
    # `teach_ticks` it adopts. 0 disables diffusion entirely, which is stage 1
    # and is bit-identical.
    #
    # Invention stays hunger-driven and adoption becomes social, which is how
    # technology actually moves and is also the only version that can be told
    # apart from invention in the data: `taught` and `invented` are counted
    # separately, so "it spread" is a measurement rather than a story.
    teach_ticks: int = 0
    teach_radius: float = 4.0


@dataclass(frozen=True)
class TechConfig:
    """Island 3.0 stage 2: a technology that needs another technology first.

    The granary. It requires FARMING and a larder that has been full for
    `full_ticks` household-ticks, and it doubles that household's food store.
    Off by default, so every world before it is bit-identical.

    WHY A PREREQUISITE IS THE POINT. Every technology in this project so far --
    the axe, farming -- is reachable from a standing start: gather two materials,
    or go hungry. A granary cannot be reached at all until something else has
    been, which is the first time the ladder is a ladder rather than a list. It
    is also the realistic shape: storage is what a surplus is FOR, and a surplus
    is what farming makes.

    IT CHANGES A HOUSEHOLD'S FOOD CAPACITY, so it is a supply-side change to the
    thing `sim.economy` sizes, and the tool is told about it before any world is
    written (rule 5). Note what it does NOT do: it creates no berries. A granary
    lets a household hold a bigger buffer across a blight, which is worth
    something exactly when the seasons are hard and nothing at all when they are
    not -- a prediction the ramp world can check.

    It spreads by the same teaching rule farming does (`agriculture.teach_*`),
    because a technology whose diffusion rule differed from its neighbour's would
    make the two incomparable.
    """

    enabled: bool = False
    granary_full_ticks: int = 200     # household-ticks with a full larder
    granary_multiplier: int = 2       # stockpile_food_capacity x this
    observe_tech: bool = True         # own.granary


@dataclass(frozen=True)
class TribeConfig:
    """Island 3.0: households group into tribes, and a tribe is a PLACE.

    Households are assigned to tribes by the ANGLE of their site around the
    island, so a tribe is a contiguous arc of coast and inter-tribe raiding is a
    border phenomenon. Round-robin tribes -- the obvious implementation, and the
    one households themselves use -- would have produced tribes that are
    everywhere and therefore nowhere, and "the strongest tribe" would have meant
    nothing spatial at all.

    Nothing new is added to stealing or raiding. What a tribe changes is who is
    immune and who remembers: theft immunity widens from the household to the
    tribe, and a raid is remembered by every living member of the victim's tribe
    rather than only the victim's household. A war that appears here is the
    stage-4 grudge economy at a larger grain, not a new capability.
    """

    enabled: bool = False
    num_tribes: int = 4
    tribe_theft_immunity: bool = True
    collective_grudge: bool = True
    observe_tribe: bool = True    # per-neighbour same_tribe flag


@dataclass(frozen=True)
class MixConfig:
    """Train on TWO worlds at once: a share of the envs run a second config.

    The lever the subset-steering result points at. Navigation fails here because
    every sub-~20-tick deviation toward far food is correctly priced <= 0 under a
    wandering policy, so one-step improvement never proposes the excursion that
    pays +173. Sequencing worlds does not fix it -- `spread-nav` and `nav-refork`
    both show the scarce world erasing a navigator it was handed. Interleaving
    puts states where crossings COMPLETE into every batch instead, so a positive
    navigation gradient is present in the same update as the scarce world's
    negative one.

    Both configs must agree on the observation layout, the action set and the
    agent count -- one policy trains on both, and a mismatch would feed trained
    weights the wrong features. VecWorld raises rather than reshaping anything.

    Only the PRIMARY config's episodes reach the metrics log, and evaluation
    builds a primary World, so every reported number stays a statement about the
    world the run is named for. Blending two worlds' lifespans into one mean
    would describe neither (rule 6).
    """

    config: str | None = None   # path to the second world config
    fraction: float = 0.0       # share of envs running it


@dataclass(frozen=True)
class ObservationConfig:
    k_bushes: int = 4
    k_agents: int = 3
    distance_scale: float = 20.0


@dataclass(frozen=True)
class RewardConfig:
    alive_per_tick: float = 0.01
    gather: float = 1.0
    eat: float = 2.0
    death: float = -10.0
    steal: float = 0.0   # 0.0 is the brief-faithful default: theft earns nothing
                         # directly and must pay for itself through the food. Only
                         # config/m3_shaped.yaml raises it, as a labelled ablation.
    # Milestone 4 shaping. The brief calls M4 the milestone that NEEDS shaping --
    # the terminal chain (chop -> carry -> build -> survive the night) is far too
    # long for the survival signal alone. These are documented bootstraps: m4.yaml
    # sets them non-zero, m4_unshaped.yaml is the control, and the annealing test
    # retrains with them returned to zero.
    wood: float = 0.0
    stone: float = 0.0
    build: float = 0.0
    complete: float = 0.0  # split among contributors when a shelter completes
    # Milestone 5. 0.0 is the brief-faithful default, and the same call as
    # reward.steal: a gift has to pay for itself through what the receiver then
    # does with it, or it is not exchange, it is us paying agents to hand things
    # over. config/m5_shaped.yaml raises it as a labelled ablation.
    give: float = 0.0


@dataclass(frozen=True)
class PolicyConfig:
    hidden_sizes: tuple[int, ...] = (128, 128)
    mode: str = "shared"          # "shared" (M1) or "individual" (M2)
    init_from: str | None = None  # checkpoint to fork individual brains from


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 32
    rollout_ticks: int = 128
    total_updates: int = 300
    epochs: int = 4
    num_minibatches: int = 4
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    ent_coef_final: float | None = None  # anneal ent_coef to this by the last update
    max_grad_norm: float = 0.5
    device: str = "cpu"
    threads: int | None = 4   # CPU threads torch may use; null = all cores.
                              # 4 measured as fast as 8 -- these nets are small
                              # enough that the env step dominates, so extra
                              # cores buy heat and nothing else. Bit-identical
                              # results either way (tested).


@dataclass(frozen=True)
class LoggingConfig:
    run_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    replay_dir: str = "viewer/replays"
    log_every: int = 1
    checkpoint_every: int = 25
    replay_every: int = 50
    baseline_episodes: int = 20


@dataclass(frozen=True)
class Config:
    seed: int = 0
    world: WorldConfig = field(default_factory=WorldConfig)
    hunger: HungerConfig = field(default_factory=HungerConfig)
    food: FoodConfig = field(default_factory=FoodConfig)
    bushes: BushConfig = field(default_factory=BushConfig)
    competition: CompetitionConfig = field(default_factory=CompetitionConfig)
    construction: ConstructionConfig = field(default_factory=ConstructionConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    society: SocietyConfig = field(default_factory=SocietyConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    predators: PredatorConfig = field(default_factory=PredatorConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)
    housing: HousingConfig = field(default_factory=HousingConfig)
    agriculture: AgricultureConfig = field(default_factory=AgricultureConfig)
    tech: TechConfig = field(default_factory=TechConfig)
    tribes: TribeConfig = field(default_factory=TribeConfig)
    mix: MixConfig = field(default_factory=MixConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **overrides: Any) -> "Config":
        """Return a copy with dotted-path overrides applied, e.g. ``ppo.num_envs=4``.

        Used by tests and CLI flags to shrink a run without maintaining a second
        YAML file that will inevitably drift from the real one.
        """
        nested: dict[str, dict[str, Any]] = {}
        top: dict[str, Any] = {}
        for key, value in overrides.items():
            if "." in key:
                section, _, leaf = key.partition(".")
                nested.setdefault(section, {})[leaf] = value
            else:
                top[key] = value
        for section, leaves in nested.items():
            current = getattr(self, section)
            top[section] = dataclasses.replace(current, **leaves)
        return dataclasses.replace(self, **top)


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    """Instantiate a (flat) config dataclass from a mapping, rejecting unknown keys.

    Silently ignoring a typo'd key is the classic way to spend an afternoon
    wondering why a config change did nothing, so unknown keys are fatal.
    """
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    if "hidden_sizes" in data:
        data["hidden_sizes"] = tuple(int(v) for v in data["hidden_sizes"])
    return cls(**data)


_SECTIONS: dict[str, type] = {
    "world": WorldConfig,
    "hunger": HungerConfig,
    "food": FoodConfig,
    "bushes": BushConfig,
    "competition": CompetitionConfig,
    "construction": ConstructionConfig,
    "exchange": ExchangeConfig,
    "society": SocietyConfig,
    "tools": ToolsConfig,
    "predators": PredatorConfig,
    "reproduction": ReproductionConfig,
    "housing": HousingConfig,
    "agriculture": AgricultureConfig,
    "tech": TechConfig,
    "tribes": TribeConfig,
    "mix": MixConfig,
    "observation": ObservationConfig,
    "reward": RewardConfig,
    "policy": PolicyConfig,
    "ppo": PPOConfig,
    "logging": LoggingConfig,
}


def config_from_dict(data: dict[str, Any]) -> Config:
    data = dict(data)
    data.pop("extends", None)  # resolved by load_config before we get here
    sections = {name: _build(cls, data.pop(name, None)) for name, cls in _SECTIONS.items()}
    unknown = set(data) - {"seed"}
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    return Config(seed=int(data.get("seed", 0)), **sections)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, leaving both untouched."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a YAML config, resolving a chain of ``extends`` parents.

    A config may name a parent to inherit from::

        extends: default.yaml
        bushes:
          capacity: 3

    Only the keys it restates are overridden, so a milestone that changes four
    numbers says exactly those four and cannot silently drift from the base
    config. Parent paths are relative to the child's own directory.
    """
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in (*seen, path))
        raise ValueError(f"circular config extends: {chain}")
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.get("extends")
    if not parent:
        return raw
    return _deep_merge(_load_raw(path.parent / parent, (*seen, path)), raw)


def load_config(path: str | Path | None = None) -> Config:
    return config_from_dict(_load_raw(Path(path) if path is not None else DEFAULT_CONFIG_PATH))
