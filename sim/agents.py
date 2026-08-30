"""Agent state, the discrete action space, and egocentric observation construction.

State is stored struct-of-arrays (one numpy array per field, indexed by agent) so
the whole population can be stepped and observed with vectorised operations
instead of a Python loop over agent objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config

# --- Action space -----------------------------------------------------------
# 0-7: move one `move_step` in a compass direction. 8: idle. 9: gather.
# 10: steal, present only when `competition.enable_steal` is on (Milestone 3).
# 11-13: chop/mine/build (Milestone 4). 14-15: give_food/give_material (M5).
# Compass convention: index 0 is +z ("north"), angle increases clockwise through
# +x ("east"), matching the viewer's world axes. All eight directions are unit
# vectors, so diagonal movement is not secretly faster.
#
# Steal is appended rather than inserted so every earlier action keeps its index:
# an M1 or M2 checkpoint means the same thing in an M3 world.
BASE_ACTION_NAMES: tuple[str, ...] = (
    "N", "NE", "E", "SE", "S", "SW", "W", "NW", "idle", "gather",
)
ACTION_NAMES: tuple[str, ...] = BASE_ACTION_NAMES
STEAL_ACTION_NAMES: tuple[str, ...] = BASE_ACTION_NAMES + ("steal",)
# Milestone 4 appends three more. When construction is enabled the steal slot is
# always present (inert if enable_steal is false -- an inert steal is already
# tested behaviour), so CHOP/MINE/BUILD are stable indices that never shift.
CONSTRUCTION_ACTION_NAMES: tuple[str, ...] = STEAL_ACTION_NAMES + ("chop", "mine", "build")
# Milestone 5 appends two more, by the same rule: enabling exchange implies the
# whole construction block is present (inert if construction is off), so
# give_food and give_material are fixed indices in every exchange world.
EXCHANGE_ACTION_NAMES: tuple[str, ...] = CONSTRUCTION_ACTION_NAMES + ("give_food", "give_material")
# Island 2.0 stage 4 appends five more, by the same append-never-insert rule:
# enabling `society` implies the whole exchange block is present (inert if
# exchange is off), so these are fixed indices in every stage-4 world.
#
# Deposit and withdraw are SPLIT BY ECONOMY for the reason M5 split giving: food
# keeps an agent alive and material builds shelter, and an agent carrying both
# would otherwise be unable to say which economy it is taking part in. Raid is
# ONE action rather than a foreign-stockpile variant of each withdraw, because a
# raid is a distinct social act -- it is the thing that creates a grudge -- and
# keeping the reputation bookkeeping behind a single action means there is one
# place it can be got wrong. A raider takes whatever is there, food first.
SOCIETY_ACTION_NAMES: tuple[str, ...] = EXCHANGE_ACTION_NAMES + (
    "deposit_food", "deposit_material", "withdraw_food", "withdraw_material", "raid")
# Island 2.0 tech ladder rung 1 appends one more, same rule again: enabling
# `tools` implies the whole society block is present, so `craft` is a fixed index
# in every tool world and no earlier checkpoint's action head is disturbed.
TOOL_ACTION_NAMES: tuple[str, ...] = SOCIETY_ACTION_NAMES + ("craft",)
# Island 3.0 appends ONE action, and only one. Reproduction is automatic (see
# ReproductionConfig) and expanding a house reuses `build`, so agriculture is the
# only thing in the whole stage that an agent has to be able to *choose* to do.
AGRICULTURE_ACTION_NAMES: tuple[str, ...] = TOOL_ACTION_NAMES + ("plant",)
N_MOVE_ACTIONS = 8
IDLE = 8
GATHER = 9
STEAL = 10
CHOP = 11
MINE = 12
BUILD = 13
GIVE_FOOD = 14
GIVE_MATERIAL = 15
DEPOSIT_FOOD = 16
DEPOSIT_MATERIAL = 17
WITHDRAW_FOOD = 18
WITHDRAW_MATERIAL = 19
RAID = 20
CRAFT = 21
PLANT = 22
N_ACTIONS = len(BASE_ACTION_NAMES)

# Item codes for the transfer ledger and the replay's per-tick transfer list.
ITEM_FOOD, ITEM_WOOD, ITEM_STONE = 0, 1, 2
ITEM_NAMES: tuple[str, ...] = ("food", "wood", "stone")

# Island 3.0 stage 2: the technologies a HOUSEHOLD can hold. Appended, never
# inserted, like everything else here. They live next to the item codes rather
# than in world.py because the observation, the action mask and the engine all
# need the same indices and there must be exactly one place they are defined.
TECH_NAMES: tuple[str, ...] = ("farming", "granary")
TECH_FARMING, TECH_GRANARY = 0, 1
N_TECHS = len(TECH_NAMES)
# How a household came by a technology, for the diffusion measurement. A single
# has-it flag cannot tell invention from adoption, and telling them apart is the
# whole of read S3. `TECH_SETTLED` is the third way a technology travels, added
# with village fission: settlers take what they know with them, so a daughter
# household holds its parent's technologies without inventing or being taught
# anything -- MIGRATION, which is how most technology actually moved.
TECH_NONE, TECH_INVENTED, TECH_TAUGHT, TECH_SETTLED = 0, 1, 2, 3


def action_names(cfg: Config) -> tuple[str, ...]:
    if cfg.agriculture.enabled:
        return AGRICULTURE_ACTION_NAMES
    if cfg.tools.enabled:
        return TOOL_ACTION_NAMES
    if cfg.society.enabled:
        return SOCIETY_ACTION_NAMES
    if cfg.exchange.enabled:
        return EXCHANGE_ACTION_NAMES
    if cfg.construction.enabled:
        return CONSTRUCTION_ACTION_NAMES
    return STEAL_ACTION_NAMES if cfg.competition.enable_steal else BASE_ACTION_NAMES


def num_actions(cfg: Config) -> int:
    return len(action_names(cfg))

_ANGLES = np.arange(N_MOVE_ACTIONS) * (np.pi / 4.0)
MOVE_VECTORS: np.ndarray = np.stack([np.sin(_ANGLES), np.cos(_ANGLES)], axis=1)
MOVE_VECTORS[np.abs(MOVE_VECTORS) < 1e-12] = 0.0  # kill float dust at the axes


@dataclass
class AgentPool:
    """Struct-of-arrays state for every agent in one world.

    ``hunger`` runs from ``hunger.max`` down to 0 and death is ``hunger <= 0``:
    it is a satiety meter despite the name (kept to match the project brief).
    """

    x: np.ndarray
    z: np.ndarray
    hunger: np.ndarray
    food: np.ndarray
    wood: np.ndarray
    stone: np.ndarray
    alive: np.ndarray
    last_action: np.ndarray
    # Tech ladder rung 1. Always allocated, never read unless `tools.enabled` --
    # one array of zeros costs nothing and keeps every world's pool one shape,
    # which is what stops the construction/exchange/society branches multiplying.
    axe: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # Island 3.0. `born` separates "has not been born yet" from "is dead": both
    # are `alive == False` and both must be inert, but only the second one lived,
    # and every per-agent statistic (lifespan, deaths, the Gini) divides by the
    # agents that lived. Always allocated, all-True and all-zero in a world
    # without reproduction, for the same reason `axe` is: one shape of pool.
    age: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    born: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    def adult(self, cfg: Config) -> np.ndarray:
        """Who may do adult work. Everyone, in a world without reproduction."""
        if not cfg.reproduction.enabled:
            return np.ones(self.n, dtype=bool)
        return self.age >= cfg.reproduction.maturity_ticks

    @staticmethod
    def create(num_agents: int, max_hunger: float) -> "AgentPool":
        return AgentPool(
            x=np.zeros(num_agents, dtype=np.float64),
            z=np.zeros(num_agents, dtype=np.float64),
            hunger=np.full(num_agents, max_hunger, dtype=np.float64),
            food=np.zeros(num_agents, dtype=np.int64),
            wood=np.zeros(num_agents, dtype=np.int64),
            stone=np.zeros(num_agents, dtype=np.int64),
            alive=np.ones(num_agents, dtype=bool),
            last_action=np.full(num_agents, IDLE, dtype=np.int64),
            axe=np.zeros(num_agents, dtype=np.int64),
            age=np.zeros(num_agents, dtype=np.int64),
            born=np.ones(num_agents, dtype=bool),
        )


@dataclass
class ConstructionView:
    """The slice of world state that observations and masks need for Milestone 4.

    A plain data bundle rather than the World itself, so agents.py keeps no
    dependency on world.py and the tests can fabricate one in three lines.
    """

    tree_x: np.ndarray
    tree_z: np.ndarray
    tree_wood: np.ndarray
    rock_x: np.ndarray
    rock_z: np.ndarray
    rock_stone: np.ndarray
    site_x: np.ndarray
    site_z: np.ndarray
    site_wood_needed: np.ndarray   # remaining, not total
    site_stone_needed: np.ndarray
    tick: int


@dataclass
class SocietyView:
    """The slice of world state stage 4's observations and masks need.

    Same shape of bundle as ``ConstructionView``, for the same reason: agents.py
    keeps no dependency on world.py and a test can fabricate one in three lines.

    ``household`` is per agent; the stockpile arrays are per household, and a
    household's stockpile sits at its own shelter site, so ``stock_x[h]`` is also
    where household h sleeps. ``grudge[i, j]`` is how much i remembers j taking
    from it.
    """

    household: np.ndarray          # (A,) household index per agent
    stock_x: np.ndarray            # (H,)
    stock_z: np.ndarray
    stock_food: np.ndarray         # (H,)
    stock_wood: np.ndarray         # (H,) -- the store tracks composition; see
    stock_stone: np.ndarray        #        World.reset for why that is load-bearing
    grudge: np.ndarray             # (A, A)
    blight: bool

    @property
    def stock_material(self) -> np.ndarray:
        return self.stock_wood + self.stock_stone


@dataclass
class PredatorView:
    """Where the predators are, and whether they are hunting tonight.

    Same shape of bundle as `ConstructionView` and `SocietyView`, for the same
    reason: agents.py keeps no dependency on world.py, and a test can build one
    in two lines.
    """

    x: np.ndarray
    z: np.ndarray
    hunting: bool


@dataclass
class Island3View:
    """The slice of Island 3.0 state observations and masks read.

    Same bundle-of-arrays shape as the three views above, and for the same
    reason: agents.py never imports world.py, and a test builds one in two lines.
    Every field is sized so a world with the relevant block off can pass zeros
    rather than None -- branching on four more Nones inside the mask was the
    alternative and it is how the M4 observation bug happened.
    """

    tribe: np.ndarray            # (A,) tribe index per agent
    tribe_of_household: np.ndarray   # (H,)
    site_extra: np.ndarray       # (S,) units delivered beyond completion
    site_capacity: np.ndarray    # (S,) beds, base + rooms
    household_size: np.ndarray   # (H,) living members
    farming: np.ndarray          # (H,) has this household invented agriculture
    fields: np.ndarray           # (H,) fields planted so far
    granary: np.ndarray          # (H,) stage 2: does it hold the second tech
    stock_food_capacity: np.ndarray  # (H,) per household -- a granary doubles it


def neighbour_society_channels(cfg: Config) -> int:
    """Stage-4 additions to a neighbour slot: same-household, and a grudge.

    Island 3.0 adds a third: is this neighbour in my TRIBE. Same argument as
    same-household before it -- a tribe is who you do not rob and who avenges
    you, and an agent has no way to infer that from offsets.
    """
    tribe = int(cfg.tribes.enabled and cfg.tribes.observe_tribe)
    if not cfg.society.enabled:
        return tribe
    return (int(cfg.society.observe_household) + int(cfg.society.observe_grudge)
            + tribe)


def reproduction_channels(cfg: Config) -> int:
    """Island 3.0: how old am I, and am I grown."""
    return 2 * int(cfg.reproduction.enabled and cfg.reproduction.observe_age)


def housing_channels(cfg: Config) -> int:
    """Island 3.0: free beds at my own house, and how far it is over capacity.

    Three channels rather than one occupancy number, because they answer three
    different questions and a single ratio clipped to [0, 1] would throw away the
    two that drive behaviour: `beds_free` says the family is housed, `overflow`
    says how much bigger the house needs to be, and `expandable` says whether
    making it bigger is a thing that can be done at all (the house is finished
    and under `max_rooms`). Without the third, an agent at a maxed-out house
    would keep wanting a room it can never add, which is precisely the doomed
    action the M3 mask exists to delete.
    """
    return 3 * int(cfg.housing.enabled and cfg.housing.observe_house)


def agriculture_channels(cfg: Config) -> int:
    """Island 3.0: has my household invented farming, and may it plant again."""
    return 2 * int(cfg.agriculture.enabled and cfg.agriculture.observe_agriculture)


def tech_channels(cfg: Config) -> int:
    """Island 3.0 stage 2: does my household hold the second technology.

    One channel, and it is separate from `own.stock_food` for the same reason
    `own.axe` is separate from `own.wood`: the magnitude says how full the store
    is, and nothing in it says what KIND of store it is.
    """
    return int(cfg.tech.enabled and cfg.tech.observe_tech)


def society_channels(cfg: Config) -> int:
    """Fixed-width stage-4 block: own stockpile, home offset, nearest foreign pile.

    Ten channels, plus one if blights are observed. It does NOT grow with the
    number of households -- the nearest foreign stockpile is one k-nearest slot,
    exactly as bushes and sites are, which is what keeps the observation width
    independent of population (design doc section 4).
    """
    if not cfg.society.enabled:
        return 0
    return 10 + int(cfg.society.observe_shock)


def night_phase(tick: int, cfg: Config) -> tuple[float, bool]:
    """(cycle phase in [0,1), is it night). Night is the last `night_fraction`."""
    cc = cfg.construction
    phase = (tick % cc.night_cycle) / cc.night_cycle
    return phase, phase >= 1.0 - cc.night_fraction


def neighbour_channels(cfg: Config) -> int:
    """3 per neighbour (dx, dz, hunger), +1 for carried food, +2 for wood/stone.

    Milestone 3 needs the food channel: stealing from a neighbour who is carrying
    nothing is a wasted tick, and a policy that cannot see who has food cannot
    learn to rob selectively -- it could only learn "rob at random", which would
    look like the behaviour without being it.

    Milestone 5 adds the material channels for the mirror-image reason. A gift is
    only worth making to someone who can use it, and "who can use wood" is
    exactly "who is carrying none and is standing at a site". Without these the
    policy could only learn to give indiscriminately, which resembles exchange
    without being it.
    """
    return (3 + int(cfg.competition.observe_neighbour_food)
            + 2 * int(cfg.exchange.observe_neighbour_materials)
            + neighbour_society_channels(cfg))


def berry_scale(cfg: Config) -> int:
    """What a bush's berry count is normalised by.

    A planted field holds more than a wild bush, so with agriculture on the
    divisor is the larger of the two -- otherwise a full field reads above 1.0
    and breaks the [-1, 1] invariant every observation test enforces. With
    agriculture off it is `bushes.capacity` exactly, so no earlier world moves.
    """
    cap = max(cfg.bushes.capacity, 1)
    if cfg.agriculture.enabled:
        cap = max(cap, cfg.agriculture.field_capacity)
    return cap


def bush_channels(cfg: Config) -> int:
    """3 per bush (dx, dz, berries), or 4 with "a rival is closer than me".

    Same principle as ``neighbour_channels``: with ``exclusive_bushes`` on, whether
    an agent may harvest a bush depends on whether any living rival stands nearer
    to it. That is *derivable* from the neighbour and bush offsets already in the
    observation, but only via a comparison of distances the network would have to
    discover for itself -- and until it does, a blocked bush is indistinguishable
    from a free one, so the only learnable policy is "gather and hope".

    Making the mechanic directly perceivable is the same call that was made for
    theft. A mechanic the policy cannot see is a mechanic it cannot respond to.
    """
    return 4 if cfg.competition.observe_bush_contested else 3


def site_channels(cfg: Config) -> int:
    """dx, dz, need_wood, need_stone, complete, and optionally ``finishes``.

    The sixth channel is agent-relative where the other five are not: whether
    *this* agent could complete *this* site with what it is carrying right now.
    """
    return 6 if cfg.construction.observe_final_unit else 5


def observation_dim(cfg: Config) -> int:
    """2 own scalars + K_b bushes x (3 or 4) + K_a agents x (3 or 4) + 3 edge,
    plus the Milestone 4 block when construction is enabled."""
    dim = (2 + bush_channels(cfg) * cfg.observation.k_bushes
           + neighbour_channels(cfg) * cfg.observation.k_agents + 3)
    if cfg.construction.enabled:
        cc = cfg.construction
        dim += 2                    # own wood, own stone
        dim += 3 * cc.k_trees       # dx, dz, wood left
        dim += 3 * cc.k_rocks       # dx, dz, stone left
        dim += site_channels(cfg) * cc.k_sites
        dim += 2                    # cycle phase, is_night
    dim += society_channels(cfg)
    dim += tool_channels(cfg)
    dim += predator_channels(cfg)
    dim += reproduction_channels(cfg)
    dim += housing_channels(cfg)
    dim += agriculture_channels(cfg)
    dim += tech_channels(cfg)
    return dim


def tool_channels(cfg: Config) -> int:
    """Tech ladder rung 1: one channel, `own.axe`."""
    return int(cfg.tools.enabled and cfg.tools.observe_axe)


def predator_channels(cfg: Config) -> int:
    """Tech ladder rung 2: the nearest predator's offset, and whether it hunts."""
    return 3 * int(cfg.predators.enabled and cfg.predators.observe_predator)


def action_mask(
    pool: AgentPool,
    bush_x: np.ndarray,
    bush_z: np.ndarray,
    bush_berries: np.ndarray,
    cfg: Config,
    construction: "ConstructionView | None" = None,
    society: "SocietyView | None" = None,
    island3: "Island3View | None" = None,
) -> np.ndarray:
    """Which actions can possibly do anything, per agent. Shape ``(A, n_actions)``.

    Moving and idling are always available. ``gather`` is available only with a
    free inventory slot and a berry-bearing bush in range; ``steal`` only with a
    free slot and a living neighbour in range who is carrying something.

    This is not a reward change and not a hint about what is *best* -- it is the
    same kind of information the observation already carries, one step further:
    the observation says what is there, the mask says what is reachable. Without
    it, "stand still and mash gather" is a local optimum PPO does not escape,
    because a doomed gather costs nothing and occasionally a real one pays +1.
    Measured on the unmasked policy: 30% of every tick it lived went on actions
    that could not succeed.

    Dead agents get ``idle`` only. A fully-masked row would make the action
    distribution undefined, and NaN logits propagate silently.
    """
    n, n_act = pool.n, num_actions(cfg)
    mask = np.zeros((n, n_act), dtype=bool)
    mask[:, :N_MOVE_ACTIONS + 1] = True          # the 8 moves and idle

    has_room = pool.food < cfg.food.capacity
    if bush_x.size:
        bush_d2 = ((bush_x[None, :] - pool.x[:, None]) ** 2
                   + (bush_z[None, :] - pool.z[:, None]) ** 2)
        reachable = (bush_d2 <= cfg.bushes.gather_radius ** 2) & (bush_berries[None, :] > 0)
        mask[:, GATHER] = has_room & reachable.any(axis=1)

    if cfg.competition.enable_steal:
        agent_d2 = ((pool.x[None, :] - pool.x[:, None]) ** 2
                    + (pool.z[None, :] - pool.z[:, None]) ** 2)
        np.fill_diagonal(agent_d2, np.inf)
        victims = ((agent_d2 <= cfg.competition.steal_radius ** 2)
                   & pool.alive[None, :] & (pool.food[None, :] > 0))
        if cfg.society.enabled and cfg.society.household_theft_immunity:
            # Mirrors World.step's rule exactly. A mask that promised a steal the
            # world then refuses is the doomed action masking exists to delete.
            assert society is not None, "society world state missing"
            victims &= society.household[None, :] != society.household[:, None]
        if cfg.tribes.enabled and cfg.tribes.tribe_theft_immunity:
            assert island3 is not None, "island3 world state missing"
            victims &= island3.tribe[None, :] != island3.tribe[:, None]
        mask[:, STEAL] = has_room & victims.any(axis=1)

    if cfg.construction.enabled and construction is not None:
        cc = cfg.construction
        room = (pool.wood + pool.stone) < cc.material_capacity

        def in_reach(ex, ez, stock, radius):
            if ex.size == 0:
                return np.zeros(n, dtype=bool)
            d2 = (ex[None, :] - pool.x[:, None]) ** 2 + (ez[None, :] - pool.z[:, None]) ** 2
            return ((d2 <= radius ** 2) & (stock[None, :] > 0)).any(axis=1)

        mask[:, CHOP] = room & in_reach(construction.tree_x, construction.tree_z,
                                        construction.tree_wood, cc.harvest_radius)
        mask[:, MINE] = room & in_reach(construction.rock_x, construction.rock_z,
                                        construction.rock_stone, cc.harvest_radius)
        # build: an incomplete site in reach that needs a material this agent carries
        if construction.site_x.size:
            d2 = ((construction.site_x[None, :] - pool.x[:, None]) ** 2
                  + (construction.site_z[None, :] - pool.z[:, None]) ** 2)
            near = d2 <= cc.build_radius ** 2
            # Island 3.0: a FINISHED house still takes material, as an
            # extension, until it has `max_rooms`. So `build` stops being an
            # action that expires the moment the village is up, and the material
            # economy acquires the recurring demand stage 4 needed storms for.
            expandable = np.zeros(construction.site_x.shape[0], dtype=bool)
            if cfg.housing.enabled:
                assert island3 is not None, "island3 world state missing"
                complete = ((construction.site_wood_needed == 0)
                            & (construction.site_stone_needed == 0))
                rooms = island3.site_extra // max(cfg.housing.expand_units, 1)
                expandable = complete & (rooms < cfg.housing.max_rooms)
            if cc.fungible_materials:
                # Must mirror World.step's build rule exactly: any outstanding
                # unit takes any carried material. A mask that promised more
                # than the rule delivers would reintroduce the doomed actions
                # masking exists to remove.
                outstanding = ((construction.site_wood_needed
                                + construction.site_stone_needed) > 0) | expandable
                can = near & outstanding[None, :] & ((pool.wood + pool.stone)[:, None] > 0)
                mask[:, BUILD] = can.any(axis=1)
            else:
                can_wood = near & ((construction.site_wood_needed[None, :] > 0)
                                   | expandable[None, :]) & (pool.wood[:, None] > 0)
                can_stone = near & ((construction.site_stone_needed[None, :] > 0)
                                    | expandable[None, :]) & (pool.stone[:, None] > 0)
                mask[:, BUILD] = (can_wood | can_stone).any(axis=1)

    if cfg.exchange.enabled:
        # A gift needs something to give and someone with room to take it. Note
        # what is deliberately NOT in here: whether the receiver *needs* it. The
        # mask says what is possible, never what is worthwhile -- "give to the
        # hungry" is the behaviour this milestone is trying to observe emerging,
        # so encoding it in the mask would be answering our own question.
        d2 = ((pool.x[None, :] - pool.x[:, None]) ** 2
              + (pool.z[None, :] - pool.z[:, None]) ** 2)
        np.fill_diagonal(d2, np.inf)
        near = (d2 <= cfg.exchange.give_radius ** 2) & pool.alive[None, :]
        mask[:, GIVE_FOOD] = (pool.food > 0) & (near & (pool.food[None, :]
                                                        < cfg.food.capacity)).any(axis=1)
        if cfg.construction.enabled:
            room = (pool.wood + pool.stone) < cfg.construction.material_capacity
            mask[:, GIVE_MATERIAL] = ((pool.wood + pool.stone) > 0) & (near & room[None, :]).any(axis=1)

    if cfg.society.enabled:
        assert society is not None, "society world state missing"
        sc = cfg.society
        mine = society.household
        rows = np.arange(n)
        # Distance to my OWN stockpile, and to the nearest foreign one. Both are
        # world-state reads the mask is entitled to make -- the mask says what is
        # reachable, and reachability is a fact about the world (the M3 note).
        home_d2 = ((society.stock_x[mine] - pool.x) ** 2
                   + (society.stock_z[mine] - pool.z) ** 2)
        at_home = home_d2 <= sc.stockpile_radius ** 2
        # THE HOUSEHOLD'S OWN capacity, not the config's: a granary makes two
        # households on the same island differ, and a mask that promised a
        # deposit the world then refuses is the doomed action masking exists to
        # delete. `island3` carries it so a stage-1 world (where every household
        # has the same larder) computes exactly the number it always did.
        if island3 is not None and island3.stock_food_capacity.size:
            food_cap_h = island3.stock_food_capacity[mine]
        else:
            food_cap_h = np.full(n, sc.stockpile_food_capacity, dtype=np.int64)
        food_room = society.stock_food[mine] < food_cap_h
        mat_room = society.stock_material[mine] < sc.stockpile_material_capacity
        mask[:, DEPOSIT_FOOD] = at_home & (pool.food > 0) & food_room
        carrying = (pool.wood + pool.stone) > 0
        mask[:, DEPOSIT_MATERIAL] = at_home & carrying & mat_room
        mask[:, WITHDRAW_FOOD] = (at_home & (society.stock_food[mine] > 0)
                                  & (pool.food < cfg.food.capacity))
        if cfg.construction.enabled:
            room_m = (pool.wood + pool.stone) < cfg.construction.material_capacity
        else:
            room_m = np.zeros(n, dtype=bool)
        mask[:, WITHDRAW_MATERIAL] = (at_home & (society.stock_material[mine] > 0) & room_m)

        h = society.stock_x.shape[0]
        if h > 1:
            dx = society.stock_x[None, :] - pool.x[:, None]
            dz = society.stock_z[None, :] - pool.z[:, None]
            d2 = dx ** 2 + dz ** 2
            d2[rows, mine] = np.inf
            # A raid can take food or material, so it is legal at a foreign pile
            # holding either, as long as the raider has room for what is there.
            has_food = society.stock_food[None, :] > 0
            has_mat = society.stock_material[None, :] > 0
            in_reach = d2 <= sc.stockpile_radius ** 2
            takeable = ((has_food & (pool.food < cfg.food.capacity)[:, None])
                        | (has_mat & room_m[:, None]))
            mask[:, RAID] = (in_reach & takeable).any(axis=1)

    if cfg.tools.enabled and construction is not None:
        # Craft: at a site, with the materials, and not already holding an axe.
        # The "already holding" clause is what makes this a mask rule rather than
        # a preference -- a second axe does nothing, so offering one would be a
        # doomed action, which is the entire thing masking exists to delete.
        tc = cfg.tools
        if construction.site_x.size:
            d2 = ((construction.site_x[None, :] - pool.x[:, None]) ** 2
                  + (construction.site_z[None, :] - pool.z[:, None]) ** 2)
            at_site = (d2 <= tc.craft_radius ** 2).any(axis=1)
        else:
            at_site = np.zeros(n, dtype=bool)
        # Wood AND stone, never fungible, even in a `fungible_materials` world: a
        # site takes whatever arrives because a wall is a wall, and an axe is a
        # haft and a head. Keeping the composition here is also the only reason
        # `mine` stays worth doing once a village is built.
        mask[:, CRAFT] = (at_site & (pool.axe == 0)
                          & (pool.wood >= tc.axe_wood_cost)
                          & (pool.stone >= tc.axe_stone_cost))

    if cfg.agriculture.enabled:
        # Plant: my household has invented farming, has a field slot left, I am
        # carrying a unit to spend, and I am near my OWN house. Near home rather
        # than anywhere, for the reason an axe is made at a site: a field you can
        # put down while standing in someone else's berry patch is not
        # agriculture, it is a bush dispenser.
        assert island3 is not None and society is not None, "island3 world state missing"
        ac = cfg.agriculture
        mine = society.household
        home_d2 = ((society.stock_x[mine] - pool.x) ** 2
                   + (society.stock_z[mine] - pool.z) ** 2)
        mask[:, PLANT] = (island3.farming[mine]
                          & (island3.fields[mine] < ac.max_fields_per_household)
                          & (home_d2 <= ac.plant_radius ** 2)
                          & ((pool.wood + pool.stone) >= ac.plant_material_cost))

    if cfg.reproduction.enabled:
        # A child is a mouth before it is a pair of hands. It walks, gathers,
        # eats and uses the family store; it does not chop, mine, build, craft,
        # plant, steal or raid. Masked rather than merely scored, because the
        # mask is where "what is reachable" lives and a child cannot swing an
        # axe -- and because a scored version would be a weight we typed.
        child = ~pool.adult(cfg)
        if child.any():
            for a in (CHOP, MINE, BUILD, STEAL, RAID, CRAFT, PLANT):
                if a < n_act:
                    mask[child, a] = False

    mask[~pool.alive] = False
    mask[~pool.alive, IDLE] = True
    return mask


def observation_layout(cfg: Config) -> tuple[str, ...]:
    """One name per observation column, in order.

    This exists so a policy can be carried across a milestone boundary *by
    feature* rather than by position. Optional channels are inserted in the
    middle of the vector (a neighbour's food sits inside the neighbour block),
    so widening the observation shifts every column after it. Copying weights
    into the top-left corner of a bigger matrix therefore feeds trained weights
    the wrong inputs -- which is exactly the bug this replaced: growing an M2
    brain into an M3 world silently fed its shoreline weights neighbour data.

    Keep this in sync with ``build_observations``; the tests compare the two.
    """
    names: list[str] = ["own.hunger", "own.food"]
    if cfg.construction.enabled:
        names += ["own.wood", "own.stone"]
    for j in range(cfg.observation.k_bushes):
        names += [f"bush{j}.dx", f"bush{j}.dz", f"bush{j}.berries"]
        if bush_channels(cfg) == 4:
            names.append(f"bush{j}.blocked")
    for j in range(cfg.observation.k_agents):
        names += [f"neighbour{j}.dx", f"neighbour{j}.dz", f"neighbour{j}.hunger"]
        if cfg.competition.observe_neighbour_food:
            names.append(f"neighbour{j}.food")
        if cfg.exchange.observe_neighbour_materials:
            names += [f"neighbour{j}.wood", f"neighbour{j}.stone"]
        if cfg.society.enabled and cfg.society.observe_household:
            names.append(f"neighbour{j}.same_household")
        if cfg.society.enabled and cfg.society.observe_grudge:
            names.append(f"neighbour{j}.grudge")
        if cfg.tribes.enabled and cfg.tribes.observe_tribe:
            names.append(f"neighbour{j}.same_tribe")
    if cfg.construction.enabled:
        cc = cfg.construction
        for j in range(cc.k_trees):
            names += [f"tree{j}.dx", f"tree{j}.dz", f"tree{j}.wood"]
        for j in range(cc.k_rocks):
            names += [f"rock{j}.dx", f"rock{j}.dz", f"rock{j}.stone"]
        for j in range(cc.k_sites):
            names += [f"site{j}.dx", f"site{j}.dz",
                      f"site{j}.need_wood", f"site{j}.need_stone", f"site{j}.complete"]
            if cc.observe_final_unit:
                names.append(f"site{j}.finishes")
        names += ["night.phase", "night.is_night"]
    if cfg.society.enabled:
        names += ["own.stock_food", "own.stock_wood", "own.stock_stone",
                  "home.dx", "home.dz", "home.complete",
                  "raid.dx", "raid.dz", "raid.food", "raid.material"]
        if cfg.society.observe_shock:
            names.append("shock.blight")
    if cfg.tools.enabled and cfg.tools.observe_axe:
        # One channel, not two. "Can I craft right now" is already the mask's
        # job, and duplicating it in the observation would be paying twice for
        # the same fact -- what the policy cannot otherwise know is whether it is
        # ALREADY carrying an axe, because nothing else in the vector implies it.
        names.append("own.axe")
    if cfg.predators.enabled and cfg.predators.observe_predator:
        # One k-nearest slot, like the foreign stockpile: an agent only ever
        # needs the closest one, and the fixed-width rule is what keeps the
        # observation independent of how many predators a world has.
        names += ["predator.dx", "predator.dz", "predator.hunting"]
    # --- Island 3.0, appended in block order after everything before it, so a
    # 2.0 checkpoint carries into a 3.0 world with the new columns zeroed.
    if cfg.reproduction.enabled and cfg.reproduction.observe_age:
        names += ["own.age", "own.adult"]
    if cfg.housing.enabled and cfg.housing.observe_house:
        names += ["home.beds_free", "home.overflow", "home.expandable"]
    if cfg.agriculture.enabled and cfg.agriculture.observe_agriculture:
        names += ["own.farming", "home.field_room"]
    if cfg.tech.enabled and cfg.tech.observe_tech:
        names.append("own.granary")
    names += ["edge.room", "edge.outward_x", "edge.outward_z"]
    return tuple(names)


def _k_nearest(dist2: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the k smallest entries per row, plus a validity mask.

    Entries set to ``inf`` by the caller (self, dead agents) are treated as
    absent, as are missing columns when there are fewer than k candidates; both
    come back with ``valid == False`` so the caller can zero-pad them.
    """
    rows, cols = dist2.shape
    if cols == 0:
        return np.zeros((rows, k), dtype=np.int64), np.zeros((rows, k), dtype=bool)
    if cols > k:
        # argpartition + a stable sort of just the k winners, instead of a full
        # argsort of every entity per agent. Profiled at 100 agents with every
        # mechanic on, the full sort was the single largest cost in the engine
        # (~32% of the tick). Results are bit-identical to the stable argsort:
        # sorting the partitioned indices ascending first makes the final stable
        # sort break distance ties by original index, exactly as before. The one
        # tie the partition itself could split differently is a tie ACROSS the
        # k-boundary, which for these inputs is either impossible (continuous
        # random positions) or invisible (inf-masked entries are zero-padded via
        # `valid` whichever index is taken).
        part = np.argpartition(dist2, k - 1, axis=1)[:, :k]
        part.sort(axis=1)
        vals = np.take_along_axis(dist2, part, axis=1)
        order = np.take_along_axis(part, np.argsort(vals, axis=1, kind="stable"), axis=1)
    else:
        order = np.argsort(dist2, axis=1, kind="stable")[:, :k]
    valid = np.isfinite(np.take_along_axis(dist2, order, axis=1))
    if order.shape[1] < k:
        pad = k - order.shape[1]
        order = np.concatenate([order, np.zeros((rows, pad), dtype=np.int64)], axis=1)
        valid = np.concatenate([valid, np.zeros((rows, pad), dtype=bool)], axis=1)
    return order, valid


def _entity_block(out: np.ndarray, col: int, pool: AgentPool, ex: np.ndarray,
                  ez: np.ndarray, extra: list[np.ndarray], k: int, scale: float) -> int:
    """Write k-nearest (dx, dz, *extra) triples-or-more for one entity type."""
    n = pool.n
    dx = ex[None, :] - pool.x[:, None]
    dz = ez[None, :] - pool.z[:, None]
    idx, valid = _k_nearest(dx ** 2 + dz ** 2, k)
    rows = np.arange(n)
    for j in range(k):
        take, ok = idx[:, j], valid[:, j]
        out[:, col + 0] = np.where(ok, np.clip(dx[rows, take] / scale, -1.0, 1.0), 0.0)
        out[:, col + 1] = np.where(ok, np.clip(dz[rows, take] / scale, -1.0, 1.0), 0.0)
        for c, channel in enumerate(extra):
            # 1-D channels are per entity ("berries left"); 2-D ones are per
            # (agent, entity) and have to be gathered on both axes ("could I
            # finish this site").
            vals = channel[rows, take] if channel.ndim == 2 else channel[take]
            out[:, col + 2 + c] = np.where(ok, vals, 0.0)
        col += 2 + len(extra)
    return col


def build_observations(
    pool: AgentPool,
    bush_x: np.ndarray,
    bush_z: np.ndarray,
    bush_berries: np.ndarray,
    cfg: Config,
    construction: "ConstructionView | None" = None,
    society: "SocietyView | None" = None,
    predators: "PredatorView | None" = None,
    island3: "Island3View | None" = None,
) -> np.ndarray:
    """Egocentric fixed-size observation for every agent, shape ``(A, obs_dim)``.

    Layout (all components land in [-1, 1]; a test enforces that):
      [0:2]   own hunger / max, own food / capacity                    -> [0, 1]
      [2:2+3K_b]  per nearest bush: dx, dz (scaled+clipped), berries/capacity
      [...]       per nearest living other agent: dx, dz, hunger/max
      [-3:]   distance to the shoreline (as a fraction of the radius) and the
              outward radial unit vector, i.e. "how much room is left, and which
              way is the sea".

    Absent entities (fewer bushes than K, fewer living neighbours than K) are
    zero-padded. Zero is unambiguous as padding because signed offsets are
    centred at zero *and* the magnitude channels (berries, hunger) are in [0, 1]:
    a padded slot reads as "a thing with nothing in it, at zero distance", which
    no real entity of interest ever is.

    Dead agents get an all-zero observation; they do not act and are masked out
    of every loss term, this just keeps the batch rectangular.
    """
    obs_cfg = cfg.observation
    scale = obs_cfg.distance_scale
    radius = cfg.world.island_radius
    n = pool.n
    out = np.zeros((n, observation_dim(cfg)), dtype=np.float32)

    alive = pool.alive
    if not alive.any():
        return out

    # --- own state
    out[:, 0] = pool.hunger / cfg.hunger.max
    out[:, 1] = pool.food / max(cfg.food.capacity, 1)
    own_cols = 2
    if cfg.construction.enabled:
        cap = max(cfg.construction.material_capacity, 1)
        out[:, 2] = pool.wood / cap
        out[:, 3] = pool.stone / cap
        own_cols = 4

    # --- K nearest bushes (regardless of whether they still hold berries; the
    # berry channel tells the policy whether it is worth walking to)
    col = own_cols
    bush_dx = bush_x[None, :] - pool.x[:, None]
    bush_dz = bush_z[None, :] - pool.z[:, None]
    bush_d2 = bush_dx**2 + bush_dz**2
    idx, valid = _k_nearest(bush_d2, obs_cfg.k_bushes)
    bush_ch = bush_channels(cfg)
    if bush_ch == 4:
        # For every (agent, bush) pair: is some *other* living agent nearer to that
        # bush than this one is? Computed once as a matrix rather than per slot.
        rival_d2 = np.where(alive[:, None], bush_d2, np.inf)
        best_rival = np.full_like(bush_d2, np.inf)
        for i in range(n):
            others = np.ones(n, dtype=bool)
            others[i] = False
            if others.any():
                best_rival[i] = rival_d2[others].min(axis=0)
        blocked = (best_rival < bush_d2).astype(np.float32)
    for j in range(obs_cfg.k_bushes):
        take = idx[:, j]
        ok = valid[:, j]
        rows = np.arange(n)
        out[:, col + 0] = np.where(ok, np.clip(bush_dx[rows, take] / scale, -1.0, 1.0), 0.0)
        out[:, col + 1] = np.where(ok, np.clip(bush_dz[rows, take] / scale, -1.0, 1.0), 0.0)
        out[:, col + 2] = np.where(ok, bush_berries[take] / berry_scale(cfg), 0.0)
        if bush_ch == 4:
            out[:, col + 3] = np.where(ok, blocked[rows, take], 0.0)
        col += bush_ch

    # --- K nearest living other agents
    agent_dx = pool.x[None, :] - pool.x[:, None]
    agent_dz = pool.z[None, :] - pool.z[:, None]
    agent_d2 = agent_dx**2 + agent_dz**2
    agent_d2[:, ~alive] = np.inf          # the dead are not neighbours
    np.fill_diagonal(agent_d2, np.inf)    # nor is oneself
    idx, valid = _k_nearest(agent_d2, obs_cfg.k_agents)
    channels = neighbour_channels(cfg)
    mat_cap = max(cfg.construction.material_capacity, 1)
    for j in range(obs_cfg.k_agents):
        take = idx[:, j]
        ok = valid[:, j]
        out[:, col + 0] = np.where(ok, np.clip(agent_dx[np.arange(n), take] / scale, -1.0, 1.0), 0.0)
        out[:, col + 1] = np.where(ok, np.clip(agent_dz[np.arange(n), take] / scale, -1.0, 1.0), 0.0)
        out[:, col + 2] = np.where(ok, pool.hunger[take] / cfg.hunger.max, 0.0)
        extra = col + 3
        if cfg.competition.observe_neighbour_food:
            out[:, extra] = np.where(ok, pool.food[take] / max(cfg.food.capacity, 1), 0.0)
            extra += 1
        if cfg.exchange.observe_neighbour_materials:
            out[:, extra + 0] = np.where(ok, pool.wood[take] / mat_cap, 0.0)
            out[:, extra + 1] = np.where(ok, pool.stone[take] / mat_cap, 0.0)
            extra += 2
        if cfg.society.enabled:
            assert society is not None, "society world state missing"
            if cfg.society.observe_household:
                # "My group" is a flag, not something to be inferred: households
                # are stable from tick 0 and an agent has no way to work out who
                # sleeps where from offsets alone. Same call as neighbours' food
                # before theft -- a mechanic the policy cannot see is one it
                # cannot respond to.
                same = society.household[take] == society.household
                out[:, extra] = np.where(ok, same.astype(np.float32), 0.0)
                extra += 1
            if cfg.society.observe_grudge:
                out[:, extra] = np.where(ok, society.grudge[np.arange(n), take], 0.0)
                extra += 1
        if cfg.tribes.enabled and cfg.tribes.observe_tribe:
            assert island3 is not None, "island3 world state missing"
            same_tribe = island3.tribe[take] == island3.tribe
            out[:, extra] = np.where(ok, same_tribe.astype(np.float32), 0.0)
            extra += 1
        col += channels

    # --- Milestone 4: material nodes, shelter sites, and the clock
    if cfg.construction.enabled:
        cc = cfg.construction
        assert construction is not None, "construction world state missing"
        col = _entity_block(out, col, pool, construction.tree_x, construction.tree_z,
                            [construction.tree_wood / max(cc.tree_wood, 1)],
                            cc.k_trees, scale)
        col = _entity_block(out, col, pool, construction.rock_x, construction.rock_z,
                            [construction.rock_stone / max(cc.rock_stone, 1)],
                            cc.k_rocks, scale)
        # Per-material remaining need, not a blended progress number: whether to
        # bring wood or stone is a decision the policy has to make, and a policy
        # that cannot see which material is missing can only guess -- the same
        # argument that put neighbours' food in the M3 observation.
        need_w = construction.site_wood_needed / max(cc.site_wood_cost, 1)
        need_s = construction.site_stone_needed / max(cc.site_stone_cost, 1)
        complete = ((construction.site_wood_needed == 0)
                    & (construction.site_stone_needed == 0)).astype(np.float64)
        site_extra = [need_w, need_s, complete]
        if cc.observe_final_unit:
            # "One more unit, of the kind I already carry, finishes this site."
            # Agent-relative, hence (A, S) rather than (S,). A site is one short
            # when exactly one unit of either material remains, and the build
            # rule spends wood first, so the carrier of that specific material is
            # the one who can close it.
            need_w_raw = construction.site_wood_needed
            need_s_raw = construction.site_stone_needed
            one_short = (need_w_raw + need_s_raw) == 1
            finishes = one_short[None, :] & (
                ((need_w_raw[None, :] == 1) & (pool.wood[:, None] > 0))
                | ((need_s_raw[None, :] == 1) & (pool.stone[:, None] > 0))
            )
            site_extra.append(finishes.astype(np.float64))
        col = _entity_block(out, col, pool, construction.site_x, construction.site_z,
                            site_extra, cc.k_sites, scale)
        phase, is_night = night_phase(construction.tick, cfg)
        out[:, col + 0] = phase
        out[:, col + 1] = float(is_night)
        col += 2

    # --- Island 2.0 stage 4: my household's stockpile, and the nearest foreign one
    if cfg.society.enabled:
        assert society is not None, "society world state missing"
        sc = cfg.society
        mine = society.household
        rows = np.arange(n)
        # Normalised by the household's OWN larder, so "my store is full" reads
        # 1.0 whether the household has a granary or not -- which is the fact the
        # scorer needs. That a granary exists at all is a separate channel
        # (`own.granary`), for the same reason `own.axe` is separate: what the
        # magnitude cannot say is what KIND of thing is holding it.
        if island3 is not None and island3.stock_food_capacity.size:
            food_cap = np.maximum(island3.stock_food_capacity[society.household], 1)
        else:
            food_cap = max(sc.stockpile_food_capacity, 1)
        mat_cap_s = max(sc.stockpile_material_capacity, 1)
        out[:, col + 0] = society.stock_food[mine] / food_cap
        # Wood and stone separately, not a blended count. In a fungible world the
        # split is redundant; in a non-fungible one an agent that cannot tell a
        # pantry full of stone from one full of wood cannot know whether drawing
        # will help its site -- which is exactly the deposit/draw treadmill
        # measured on society4_trade (5377 deposits, 4678 withdrawals an episode,
        # the same unit going in and out). Same argument as the site's own
        # per-material need channels.
        out[:, col + 1] = society.stock_wood[mine] / mat_cap_s
        out[:, col + 2] = society.stock_stone[mine] / mat_cap_s
        home_dx = society.stock_x[mine] - pool.x
        home_dz = society.stock_z[mine] - pool.z
        out[:, col + 3] = np.clip(home_dx / scale, -1.0, 1.0)
        out[:, col + 4] = np.clip(home_dz / scale, -1.0, 1.0)
        # DOES MY OWN HOUSE HAVE A ROOF ON IT? Derivable in principle from the
        # k-nearest site block, but only when the home site happens to rank inside
        # k -- and after a storm, standing anywhere else, it may not. An agent
        # certainly knows whether its own shelter is finished, and a household
        # that cannot tell has no reason to prefer home over the nearest hut,
        # which is measurably what dissolves the household as a place: without
        # this channel 55.7% of agents ended a run nearer a foreign home than
        # their own, and "my group is who sleeps where I sleep" stopped being true.
        h = society.stock_x.shape[0]
        if construction is not None:
            home_done = ((construction.site_wood_needed[:h] == 0)
                         & (construction.site_stone_needed[:h] == 0))
            out[:, col + 5] = home_done[mine].astype(np.float32)
        col += 6
        # The nearest stockpile that is NOT mine. One slot, so the width does not
        # grow with the number of households; a raider only ever needs the
        # closest target, and the design doc's fixed-width rule (section 4) is
        # what keeps a stage-5 policy trainable at any population.
        if h > 1:
            dx = society.stock_x[None, :] - pool.x[:, None]
            dz = society.stock_z[None, :] - pool.z[:, None]
            d2 = dx ** 2 + dz ** 2
            d2[rows, mine] = np.inf     # never raid your own pantry
            j = np.argmin(d2, axis=1)
            out[:, col + 0] = np.clip(dx[rows, j] / scale, -1.0, 1.0)
            out[:, col + 1] = np.clip(dz[rows, j] / scale, -1.0, 1.0)
            # The foreign pile against ITS OWN larder, not mine. With granaries
            # the two differ, and "that store is full" is a fact about that
            # store -- dividing it by my capacity would report a full small pile
            # as half empty to a raider who has a granary.
            if island3 is not None and island3.stock_food_capacity.size:
                out[:, col + 2] = society.stock_food[j] / np.maximum(
                    island3.stock_food_capacity[j], 1)
            else:
                out[:, col + 2] = society.stock_food[j] / food_cap
            out[:, col + 3] = society.stock_material[j] / mat_cap_s
        col += 4
        if sc.observe_shock:
            out[:, col] = float(society.blight)
            col += 1

    # --- tech ladder rung 1: am I carrying an axe?
    if cfg.tools.enabled and cfg.tools.observe_axe:
        out[:, col] = pool.axe.astype(np.float64)
        col += 1

    # --- tech ladder rung 2: where is the nearest predator, and is it hunting?
    if cfg.predators.enabled and cfg.predators.observe_predator:
        if predators is not None and predators.x.size:
            dx = predators.x[None, :] - pool.x[:, None]
            dz = predators.z[None, :] - pool.z[:, None]
            j = np.argmin(dx ** 2 + dz ** 2, axis=1)
            rows_p = np.arange(n)
            out[:, col + 0] = np.clip(dx[rows_p, j] / scale, -1.0, 1.0)
            out[:, col + 1] = np.clip(dz[rows_p, j] / scale, -1.0, 1.0)
            # One flag rather than a per-predator state: what an agent needs to
            # know is whether the thing out there is hunting tonight or sleeping
            # off the day, and that is the same answer for all of them.
            out[:, col + 2] = float(predators.hunting)
        col += 3

    # --- Island 3.0: my age, my house, my household's farming
    rc = cfg.reproduction
    if rc.enabled and rc.observe_age:
        assert island3 is not None, "island3 world state missing"
        # Age against MATURITY, not against max_age: what an agent needs to know
        # is whether it is grown, and in a world with no old age max_age is 0 and
        # a ratio against it is undefined. Clipped, so a long-lived adult reads 1.
        out[:, col + 0] = np.clip(pool.age / max(rc.maturity_ticks, 1), 0.0, 1.0)
        out[:, col + 1] = pool.adult(cfg).astype(np.float32)
        col += 2
    if cfg.housing.enabled and cfg.housing.observe_house:
        assert island3 is not None and society is not None, "island3 world state missing"
        mine_h = society.household
        cap = np.maximum(island3.site_capacity[mine_h], 1)
        size = island3.household_size[mine_h]
        out[:, col + 0] = np.clip((cap - size) / cap, 0.0, 1.0)
        out[:, col + 1] = np.clip((size - cap) / cap, 0.0, 1.0)
        if construction is not None and construction.site_x.size:
            h = society.stock_x.shape[0]
            done = ((construction.site_wood_needed[:h] == 0)
                    & (construction.site_stone_needed[:h] == 0))
            rooms = island3.site_extra[:h] // max(cfg.housing.expand_units, 1)
            out[:, col + 2] = (done & (rooms < cfg.housing.max_rooms))[mine_h].astype(np.float32)
        col += 3
    if cfg.agriculture.enabled and cfg.agriculture.observe_agriculture:
        assert island3 is not None and society is not None, "island3 world state missing"
        mine_h = society.household
        out[:, col + 0] = island3.farming[mine_h].astype(np.float32)
        room = cfg.agriculture.max_fields_per_household - island3.fields[mine_h]
        out[:, col + 1] = np.clip(room / max(cfg.agriculture.max_fields_per_household, 1),
                                  0.0, 1.0)
        col += 2
    if cfg.tech.enabled and cfg.tech.observe_tech:
        assert island3 is not None and society is not None, "island3 world state missing"
        out[:, col] = island3.granary[society.household].astype(np.float32)
        col += 1

    # --- shoreline
    r = np.sqrt(pool.x**2 + pool.z**2)
    safe = np.maximum(r, 1e-9)
    out[:, col + 0] = np.clip((radius - r) / radius, 0.0, 1.0)
    out[:, col + 1] = np.where(r > 1e-9, pool.x / safe, 0.0)
    out[:, col + 2] = np.where(r > 1e-9, pool.z / safe, 0.0)

    out[~alive] = 0.0
    return out
