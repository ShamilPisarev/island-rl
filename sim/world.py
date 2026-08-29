"""The island environment: state, resources, and the tick.

One ``World`` is one island with one population. ``VecWorld`` runs many of them
side by side for PPO throughput and auto-resets finished episodes.

Tick order (fixed, and the tests depend on it):
  1. decode actions -> move, gather, steal, build, or give
  2. hunger drain
  3. auto-eat
  4. death check
  5. survival bonus
  6. bush regrowth
  7. advance the clock, test for termination

Gathering happens before the drain, so a berry picked on an agent's last tick can
still save it. That is deliberate: it makes the gather -> eat -> survive chain one
tick shorter, which is exactly the chain PPO has to discover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .agents import (
    BUILD,
    CHOP,
    CRAFT,
    GATHER,
    GIVE_FOOD,
    GIVE_MATERIAL,
    IDLE,
    ITEM_FOOD,
    ITEM_STONE,
    ITEM_WOOD,
    MINE,
    MOVE_VECTORS,
    N_MOVE_ACTIONS,
    RAID,
    DEPOSIT_FOOD,
    DEPOSIT_MATERIAL,
    WITHDRAW_FOOD,
    WITHDRAW_MATERIAL,
    STEAL,
    AgentPool,
    ConstructionView,
    PredatorView,
    SocietyView,
    action_mask,
    build_observations,
    night_phase,
    num_actions,
    observation_dim,
)
from .config import Config


@dataclass
class StepResult:
    """Outcome of one tick for one world.

    ``acted`` marks agents that were alive at the *start* of the tick, i.e. the
    transitions PPO is allowed to learn from. ``terminated`` marks agents that
    died during it.
    """

    obs: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    acted: np.ndarray
    episode_done: bool
    truncated: bool
    gathered: np.ndarray
    ate: np.ndarray
    stole: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    robbed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    contested: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    harvested: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    built: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    gave: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    received: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # (giver, receiver, item) for every transfer that happened this tick, in the
    # order they resolved. Always populated -- it is at most one row per agent --
    # while the per-episode ledger in EpisodeStats is opt-in.
    transfers: list[tuple[int, int, int]] = field(default_factory=list)
    # Island 2.0 stage 4. `raided` counts a successful raid per raider this tick;
    # `raids` is (raider, victim household, item) for the reputation ledger and
    # the replay, since a raid -- like a transfer -- leaves no trace in the
    # post-step state that a recorder could read back.
    deposited: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    withdrew: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    raided: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    raids: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass
class EpisodeStats:
    ticks: int = 0
    deaths: int = 0
    berries_gathered: int = 0
    meals: int = 0
    mean_lifespan: float = 0.0
    mean_final_hunger: float = 0.0
    survivors: int = 0
    steals: int = 0
    contests_lost: int = 0
    wood_gathered: int = 0
    stone_gathered: int = 0
    builds: int = 0
    shelters_completed: int = 0
    night_ticks_sheltered: int = 0
    night_ticks_exposed: int = 0
    gifts: int = 0
    food_given: int = 0
    materials_given: int = 0
    # (tick, giver, receiver, item) for the whole episode. Empty unless
    # exchange.log_transfers is on: training runs 32 worlds at once and does not
    # need the ledger, the analysis tools do.
    transfers: list[tuple[int, int, int, int]] = field(default_factory=list)
    # stage 4
    deposits: int = 0
    withdrawals: int = 0
    raids: int = 0
    blight_ticks: int = 0
    storms: int = 0
    shelters_damaged: int = 0
    stock_food_final: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    stock_material_final: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # (tick, raider household, victim household, item) for the whole episode --
    # small enough to always keep, unlike the gift ledger: a raid is rare where a
    # gift can be thousands.
    raid_ledger: list[tuple[int, int, int, int]] = field(default_factory=list)
    # tech ladder rung 1: axes made, and who is holding one at the end. Per agent,
    # because "who became the village lumberjack" is the M2 specialisation
    # question asked of a tool, and a population total cannot answer it.
    axes_crafted: int = 0
    axes_per_agent: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    axe_holders: int = 0
    # tech ladder rung 2: agent-ticks spent in a predator's jaws, and the hunger
    # that cost. Per agent as well as summed, because "was it the same unlucky
    # few every night" is the question a mean cannot answer.
    attacks: int = 0
    attacks_per_agent: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    hunger_lost_to_predators: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class World:
    """A single island episode.

    Bush layout is resampled on every reset by default (``bushes.resample_each_episode``).
    Observations are purely egocentric, so a fixed layout could not be memorised
    anyway, and resampling stops the policy overfitting to one arrangement of
    clusters. Set it false if you want a stable map for eyeballing replays.
    """

    def __init__(self, cfg: Config, seed: int) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self.obs_dim = observation_dim(cfg)
        self._bush_layout: tuple[np.ndarray, np.ndarray] | None = None
        self.reset()

    # --- setup ------------------------------------------------------------

    def _sample_in_disc(self, radius: float, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Uniform samples inside a disc (sqrt on the radius, or you get a bullseye)."""
        theta = self.rng.uniform(0.0, 2.0 * np.pi, size=n)
        r = radius * np.sqrt(self.rng.uniform(0.0, 1.0, size=n))
        return r * np.sin(theta), r * np.cos(theta)

    def _place_bushes(self) -> tuple[np.ndarray, np.ndarray]:
        """Clustered bush positions: gaussian blobs around a few centres.

        Clustering (rather than uniform scatter) is the point -- it creates
        hotspots several agents want at once, which is where competition can
        emerge without anyone rewarding it.
        """
        bc = self.cfg.bushes
        limit = self.cfg.world.island_radius * 0.97
        cx, cz = self._sample_in_disc(self.cfg.world.island_radius * bc.cluster_radius_frac,
                                      bc.num_clusters)
        self._cluster_centres = (cx.copy(), cz.copy())
        xs, zs = [], []
        for c in range(bc.num_clusters):
            for _ in range(bc.bushes_per_cluster):
                for _attempt in range(10):
                    x = cx[c] + self.rng.normal(0.0, bc.cluster_std)
                    z = cz[c] + self.rng.normal(0.0, bc.cluster_std)
                    if x * x + z * z <= limit * limit:
                        break
                else:
                    # pathological cluster std; pull the bush back onto the island
                    norm = np.hypot(x, z)
                    x, z = x * limit / norm, z * limit / norm
                xs.append(x)
                zs.append(z)
        return np.array(xs, dtype=np.float64), np.array(zs, dtype=np.float64)

    def _scatter(self, count: int, margin: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """Uniformly scattered static entities (trees, rocks, shelter sites)."""
        return self._sample_in_disc(self.cfg.world.island_radius * margin, count)

    def _at_clusters(self, count: int, spread: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
        """Static entities dealt round-robin onto the berry clusters.

        Agents live at the clusters because that is where food is, so anything
        placed here needs no uncreditable approach walk to reach. Used for
        shelter sites (``sites_at_clusters``) and material nodes
        (``materials_at_clusters``).
        """
        ccx, ccz = self._cluster_centres
        pick = np.arange(count) % len(ccx)
        jitter = self.rng.normal(0.0, spread, size=(2, count))
        return ccx[pick] + jitter[0], ccz[pick] + jitter[1]

    def _at_clusters_in_region(self, count: int, wood_side: bool,
                               spread: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
        """Entities dealt onto the clusters on ONE SIDE of the island only.

        Stage 4's resource asymmetry. `materials_at_clusters` put wood and stone
        on every cluster, which is why stage 2's population never needed anything
        from anybody: a household could harvest both at home. Splitting the island
        makes one of the two a journey -- and therefore makes a relay, or a raid,
        the cheap alternative to the walk. M5's own postmortem asked for exactly
        this ("a relay needs its chain shortened by GEOGRAPHY").

        Falls back to every cluster if one side happens to hold none, so a
        pathological layout degrades to the stage-2 behaviour rather than placing
        nothing at all.
        """
        ccx, ccz = self._cluster_centres
        axis = self.cfg.society.region_axis
        side = ccx * np.sin(axis) + ccz * np.cos(axis)
        keep = np.flatnonzero(side > 0 if wood_side else side <= 0)
        if keep.size == 0:
            keep = np.arange(ccx.shape[0])
        pick = keep[np.arange(count) % keep.size]
        jitter = self.rng.normal(0.0, spread, size=(2, count))
        return ccx[pick] + jitter[0], ccz[pick] + jitter[1]

    def reset(self) -> np.ndarray:
        cfg = self.cfg
        if self._bush_layout is None or cfg.bushes.resample_each_episode:
            self._bush_layout = self._place_bushes()
        self.bush_x, self.bush_z = (arr.copy() for arr in self._bush_layout)
        n_bushes = self.bush_x.shape[0]
        self.bush_berries = np.full(n_bushes, cfg.bushes.initial_berries, dtype=np.int64)
        self.bush_timer = np.zeros(n_bushes, dtype=np.int64)

        self.pool = AgentPool.create(cfg.world.num_agents, cfg.hunger.max)
        self.pool.x, self.pool.z = self._sample_in_disc(
            cfg.world.island_radius * cfg.world.spawn_radius_frac, cfg.world.num_agents
        )

        cc = cfg.construction
        if cc.enabled:
            # Same resample policy as bushes: layouts follow the same seed stream,
            # so determinism holds and a fixed map pins everything at once.
            # m4b moved the SITES onto the clusters and left the material nodes
            # scattered, which moved the uncreditable walk upstream rather than
            # deleting it: measured on the m4c policy, mean distance to the
            # nearest tree is 10.2 and to the nearest rock 15.5, against 2.4 to
            # the nearest bush, and agents are within harvest range on 2.7% of
            # ticks. materials_at_clusters finishes the job m4b started.
            regions = cfg.society.enabled and cfg.society.region_split
            if regions:
                self.tree_x, self.tree_z = self._at_clusters_in_region(cc.num_trees, True)
            elif cc.materials_at_clusters:
                self.tree_x, self.tree_z = self._at_clusters(cc.num_trees)
            else:
                self.tree_x, self.tree_z = self._scatter(cc.num_trees)
            self.tree_wood = np.full(cc.num_trees, cc.tree_wood, dtype=np.int64)
            if regions:
                self.rock_x, self.rock_z = self._at_clusters_in_region(cc.num_rocks, False)
            elif cc.materials_at_clusters:
                self.rock_x, self.rock_z = self._at_clusters(cc.num_rocks)
            else:
                self.rock_x, self.rock_z = self._scatter(cc.num_rocks)
            self.rock_stone = np.full(cc.num_rocks, cc.rock_stone, dtype=np.int64)
            if cc.sites_at_clusters:
                # Put the shelters where the agents already are. The approach walk
                # to a scattered site is the part of the build chain nothing pays
                # for and PPO cannot credit; agents live at the bush clusters, so
                # siting there removes that leg entirely. Same shape of fix as the
                # M3 action mask -- delete the uncreditable step rather than pay
                # more for it.
                self.site_x, self.site_z = self._at_clusters(cc.num_sites)
            else:
                self.site_x, self.site_z = self._scatter(cc.num_sites, margin=0.7)
            self.site_wood_needed = np.full(cc.num_sites, cc.site_wood_cost, dtype=np.int64)
            self.site_stone_needed = np.full(cc.num_sites, cc.site_stone_cost, dtype=np.int64)
        else:
            empty = np.zeros(0, dtype=np.float64)
            empty_i = np.zeros(0, dtype=np.int64)
            self.tree_x = self.tree_z = self.rock_x = self.rock_z = empty
            self.site_x = self.site_z = empty
            self.tree_wood = self.rock_stone = empty_i
            self.site_wood_needed = self.site_stone_needed = empty_i

        self.tick = 0
        self._alive_ticks = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._gathered = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._meals = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._stole = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._robbed = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._contested = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._wood = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._stone = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._builds = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._gave = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._received = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._given_by_item = np.zeros(3, dtype=np.int64)
        self.transfers: list[tuple[int, int, int, int]] = []
        # Transfers from the most recent step only. The replay recorder reads the
        # world rather than being handed diffs, and a transfer is the one event
        # that leaves no trace in the post-step state.
        self.last_transfers: list[tuple[int, int, int]] = []
        self._completions = 0
        self._night_sheltered = 0
        self._night_exposed = 0
        # Per-agent mirrors of the two night counters. Pure bookkeeping (no RNG,
        # no dynamics), added for the mixed-population comparison: a subset
        # statistic like "the learned 20's nights indoors" cannot be recovered
        # from the aggregate.
        self.night_sheltered_agent = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self.night_exposed_agent = np.zeros(cfg.world.num_agents, dtype=np.int64)

        # --- Island 2.0 stage 4: households, stockpiles, reputation, shocks
        sc = cfg.society
        n_house = max(sc.num_households, 1) if sc.enabled else 1
        if sc.enabled:
            if cc.enabled and cc.num_sites < n_house:
                raise ValueError(
                    f"society.num_households={n_house} needs at least that many "
                    f"construction.num_sites (have {cc.num_sites}): a household's "
                    f"stockpile sits at its own shelter site.")
            # Round-robin, so households are equal-sized and stable from tick 0.
            # "My group is who sleeps where I sleep" needs no learning to identify
            # and no new abstract channel (design doc section 3).
            self.household = np.arange(cfg.world.num_agents) % n_house
            self.stock_x = self.site_x[:n_house].copy()
            self.stock_z = self.site_z[:n_house].copy()
            # Agents start at their household's site rather than scattered: a
            # household that begins as a crowd of strangers on opposite shores is
            # a household in name only, and the whole point of stage 4 is that
            # the group is the unit the drama happens between.
            self.pool.x = self.stock_x[self.household] + self.rng.normal(0.0, 2.0,
                                                                        size=cfg.world.num_agents)
            self.pool.z = self.stock_z[self.household] + self.rng.normal(0.0, 2.0,
                                                                        size=cfg.world.num_agents)
        else:
            self.household = np.zeros(cfg.world.num_agents, dtype=np.int64)
            self.stock_x = np.zeros(1, dtype=np.float64)
            self.stock_z = np.zeros(1, dtype=np.float64)
        self.stock_food = np.zeros(n_house, dtype=np.int64)
        # Material is tracked BY KIND even though the observation reports the sum.
        # With fungible sites the split is cosmetic; without them it is
        # load-bearing, because "deposit stone, withdraw wood" would otherwise be
        # a transmutation loophole that substitutes for the cross-region trade the
        # non-fungible world exists to force. A withdrawal returns wood first --
        # the majority need (sites want 3 wood + 1 stone) -- but only wood that
        # was actually put in.
        self.stock_wood = np.zeros(n_house, dtype=np.int64)
        self.stock_stone = np.zeros(n_house, dtype=np.int64)
        self.grudge = np.zeros((cfg.world.num_agents, cfg.world.num_agents))
        self.blight_until = -1
        # Tech ladder rung 1. Per agent, so "who became a lumberjack" can be
        # asked of the same array that says who owns a tool.
        self._crafted = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._shocks_fired = 0
        self._blight_ticks = 0
        self._storms = 0
        self._damaged = 0
        self._deposits = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._withdrawals = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._raids = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self.raid_ledger: list[tuple[int, int, int, int]] = []
        self.last_raids: list[tuple[int, int, int]] = []
        # Shocks from the most recent step only, for the replay recorder. Same
        # reason as `last_raids`: a storm leaves its mark on the site counters but
        # nothing says WHEN it landed, and a blight leaves no mark at all -- the
        # regrowth that did not happen is invisible in the state either side.
        # Pure bookkeeping: nothing in the dynamics reads either field.
        self.last_storm = 0
        # Shocks come off their own stream so adding or removing one cannot shift
        # the bush layout or the spawn positions of an otherwise identical world.
        self.shock_rng = np.random.default_rng(self._seed + 991)
        # --- tech ladder rung 2: predators. Dens are drawn once, off their OWN
        # stream for the same reason shocks have one, and every move afterwards
        # is a deterministic function of positions -- so a predator world has no
        # per-tick randomness in it at all and a replay reproduces exactly.
        pc = cfg.predators
        n_pred = pc.count if pc.enabled else 0
        self.den_x = np.zeros(n_pred)
        self.den_z = np.zeros(n_pred)
        if n_pred:
            pred_rng = np.random.default_rng(self._seed + 7717)
            angles = np.arange(n_pred) * (2.0 * np.pi / n_pred)
            angles = angles + pred_rng.uniform(0.0, 2.0 * np.pi)
            r = cfg.world.island_radius * pc.den_radius_frac
            self.den_x = r * np.cos(angles)
            self.den_z = r * np.sin(angles)
        self.predator_x = self.den_x.copy()
        self.predator_z = self.den_z.copy()
        self._attacks = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._hunger_lost_to_predators = 0.0
        # Who was under cover when the drain was applied. By day nobody is, which
        # is right: a predator walking home does not care.
        self._last_sheltered = np.zeros(cfg.world.num_agents, dtype=bool)
        return self.observations()

    def construction_view(self) -> ConstructionView | None:
        if not self.cfg.construction.enabled:
            return None
        return ConstructionView(
            tree_x=self.tree_x, tree_z=self.tree_z, tree_wood=self.tree_wood,
            rock_x=self.rock_x, rock_z=self.rock_z, rock_stone=self.rock_stone,
            site_x=self.site_x, site_z=self.site_z,
            site_wood_needed=self.site_wood_needed,
            site_stone_needed=self.site_stone_needed,
            tick=self.tick,
        )

    @property
    def stock_material(self) -> np.ndarray:
        """Combined material per household -- what the observation and mask read."""
        return self.stock_wood + self.stock_stone

    @property
    def blight_active(self) -> bool:
        return self.tick < self.blight_until

    def society_view(self) -> SocietyView | None:
        if not self.cfg.society.enabled:
            return None
        return SocietyView(
            household=self.household,
            stock_x=self.stock_x, stock_z=self.stock_z,
            stock_food=self.stock_food,
            stock_wood=self.stock_wood, stock_stone=self.stock_stone,
            grudge=self.grudge, blight=self.blight_active,
        )

    def predator_view(self) -> PredatorView | None:
        if not self.cfg.predators.enabled:
            return None
        return PredatorView(x=self.predator_x, z=self.predator_z,
                            hunting=self.predators_hunting)

    @property
    def predators_hunting(self) -> bool:
        """Predators hunt at night and go home by day."""
        if not (self.cfg.predators.enabled and self.cfg.construction.enabled):
            return False
        _, is_night = night_phase(self.tick, self.cfg)
        return bool(is_night)

    # --- stepping ---------------------------------------------------------

    def observations(self) -> np.ndarray:
        return build_observations(
            self.pool, self.bush_x, self.bush_z, self.bush_berries, self.cfg,
            self.construction_view(), self.society_view(), self.predator_view(),
        )

    def action_mask(self) -> np.ndarray:
        """Which actions could do anything right now, per agent (A, n_actions).

        All-true when masking is disabled, so callers never branch on the flag.
        """
        if not self.cfg.competition.mask_invalid_actions:
            return np.ones((self.pool.n, num_actions(self.cfg)), dtype=bool)
        return action_mask(self.pool, self.bush_x, self.bush_z, self.bush_berries,
                           self.cfg, self.construction_view(), self.society_view())

    def step(self, actions: np.ndarray) -> StepResult:
        cfg = self.cfg
        pool = self.pool
        n = pool.n
        acted = pool.alive.copy()
        actions = np.asarray(actions, dtype=np.int64).reshape(n)
        if cfg.world.decision_interval > 1 and self.tick % cfg.world.decision_interval:
            # Not a decision tick: the action chosen at the last decision persists,
            # whatever the caller passed. Tick 0 is always a decision tick and reset
            # zeroes the clock, so an episode can never start mid-commitment.
            actions = pool.last_action
        actions = np.where(acted, actions, IDLE)
        pool.last_action = actions

        rewards = np.zeros(n, dtype=np.float64)
        gathered = np.zeros(n, dtype=np.int64)
        ate = np.zeros(n, dtype=np.int64)

        # 1a. movement, then projection back onto the island
        moving = acted & (actions < N_MOVE_ACTIONS)
        if moving.any():
            delta = MOVE_VECTORS[np.clip(actions, 0, N_MOVE_ACTIONS - 1)] * cfg.world.move_step
            pool.x = np.where(moving, pool.x + delta[:, 0], pool.x)
            pool.z = np.where(moving, pool.z + delta[:, 1], pool.z)
            r = np.hypot(pool.x, pool.z)
            over = r > cfg.world.island_radius
            if over.any():
                # Walking into the sea is not fatal, it just wastes the tick: the
                # agent slides along the shoreline. The edge observation still has
                # to be learned because those ticks buy no food.
                shrink = np.where(over, cfg.world.island_radius / np.maximum(r, 1e-12), 1.0)
                pool.x *= shrink
                pool.z *= shrink

        # 1b. gathering (few agents, and the branchy logic is clearer as a loop)
        claimed: set[int] = set()
        contested = np.zeros(n, dtype=np.int64)
        for i in np.flatnonzero(acted & (actions == GATHER)):
            if pool.food[i] >= cfg.food.capacity:
                continue
            d2 = (self.bush_x - pool.x[i]) ** 2 + (self.bush_z - pool.z[i]) ** 2
            in_reach = (d2 <= cfg.bushes.gather_radius ** 2) & (self.bush_berries > 0)
            candidates = np.flatnonzero(in_reach)
            if candidates.size == 0:
                continue
            target = int(candidates[np.argmin(d2[candidates])])
            if cfg.competition.exclusive_bushes:
                # Blocking with teeth: only the agent closest to a bush may take
                # from it, so standing on one denies it to everyone else.
                #
                # The same-tick tie-break below is not enough on its own. In a
                # scarce world bushes are empty most of the time, so two agents
                # rarely manage a *successful* gather on the same tick even when
                # they are both parked on the bush -- measured contention was ~0.
                # What agents actually compete over is who is standing there when
                # a berry regrows, and that needs exclusion by distance.
                others = pool.alive.copy()
                others[i] = False
                if others.any():
                    rival_d2 = ((self.bush_x[target] - pool.x) ** 2
                                + (self.bush_z[target] - pool.z) ** 2)
                    rival_d2[~others] = np.inf
                    if rival_d2.min() < d2[target]:
                        contested[i] = 1
                        continue
            if cfg.competition.contest_bushes and target in claimed:
                # Someone earlier in agent order already took this bush's berry
                # this tick. Losing the race costs the tick, which is what makes a
                # bush worth holding rather than merely visiting.
                contested[i] = 1
                continue
            claimed.add(target)
            self.bush_berries[target] -= 1
            pool.food[i] += 1
            rewards[i] += cfg.reward.gather
            gathered[i] = 1

        # 1c. stealing (Milestone 3). Deliberately pays NO reward: the brief asks
        #     for no reward terms beyond survival, so theft has to earn its place
        #     through the food it yields and the eating that food enables. Paying
        #     the gather bonus for a successful robbery would be rewarding
        #     aggression directly, which is the thing we want to avoid asserting.
        sc = cfg.society
        stole = np.zeros(n, dtype=np.int64)
        robbed = np.zeros(n, dtype=np.int64)
        # (thief, victim) pairs this tick. Needed because `stole` and `robbed` are
        # per-agent counts and a grudge is a relation -- reconstructing who robbed
        # whom from two count vectors is not possible once two thefts land in the
        # same tick.
        theft_pairs: list[tuple[int, int]] = []
        if cfg.competition.enable_steal:
            for i in np.flatnonzero(acted & (actions == STEAL)):
                if pool.food[i] >= cfg.food.capacity:
                    continue
                d2 = (pool.x - pool.x[i]) ** 2 + (pool.z - pool.z[i]) ** 2
                eligible = ((d2 <= cfg.competition.steal_radius ** 2) & pool.alive
                            & (pool.food > 0))
                if sc.enabled and sc.household_theft_immunity:
                    eligible &= self.household != self.household[i]
                victims = np.flatnonzero(eligible)
                victims = victims[victims != i]
                if victims.size == 0:
                    continue
                victim = int(victims[np.argmin(d2[victims])])
                pool.food[victim] -= 1
                pool.food[i] += 1
                rewards[i] += cfg.reward.steal   # 0.0 unless deliberately shaped
                stole[i] = 1
                robbed[victim] = 1
                theft_pairs.append((int(i), victim))

        # 1d. construction (Milestone 4): harvest materials, deliver to sites.
        wood_got = np.zeros(n, dtype=np.int64)
        stone_got = np.zeros(n, dtype=np.int64)
        built = np.zeros(n, dtype=np.int64)
        cc = cfg.construction
        if cc.enabled:
            def harvest(action, ex, ez, stock, per_take=None):
                out = np.zeros(n, dtype=np.int64)
                for i in np.flatnonzero(acted & (actions == action)):
                    room = cc.material_capacity - (pool.wood[i] + pool.stone[i])
                    if room <= 0:
                        continue
                    d2 = (ex - pool.x[i]) ** 2 + (ez - pool.z[i]) ** 2
                    cand = np.flatnonzero((d2 <= cc.harvest_radius ** 2) & (stock > 0))
                    if cand.size == 0:
                        continue
                    target = int(cand[np.argmin(d2[cand])])
                    # An axe takes more per swing, bounded by what the agent can
                    # carry and what the tree still holds. Both bounds matter: an
                    # unbounded multiplier would let one chop empty a tree, and a
                    # tool that overfills an inventory is a capacity bug wearing a
                    # technology's name. `per_take` defaults to 1, so mining and
                    # every tool-free world take exactly the old path.
                    take = 1 if per_take is None else int(per_take[i])
                    take = min(take, int(room), int(stock[target]))
                    if take <= 0:
                        continue
                    stock[target] -= take
                    out[i] = take
                return out

            chop_take = None
            if cfg.tools.enabled:
                # 1 without an axe, `chop_multiplier` with one. Note this is a
                # RATE, not a supply: the trees hold what they hold, so an axe
                # buys ticks rather than wood (see ToolsConfig and sim.economy).
                chop_take = 1 + pool.axe * (cfg.tools.chop_multiplier - 1)
            wood_got = harvest(CHOP, self.tree_x, self.tree_z, self.tree_wood,
                               chop_take)
            pool.wood += wood_got
            rewards += wood_got * cfg.reward.wood
            stone_got = harvest(MINE, self.rock_x, self.rock_z, self.rock_stone)
            pool.stone += stone_got
            rewards += stone_got * cfg.reward.stone

            for i in np.flatnonzero(acted & (actions == BUILD)):
                d2 = (self.site_x - pool.x[i]) ** 2 + (self.site_z - pool.z[i]) ** 2
                near = np.flatnonzero(d2 <= cc.build_radius ** 2)
                delivered = False
                for s in near[np.argsort(d2[near])]:
                    if cc.fungible_materials:
                        # A site takes whatever arrives. Spend the agent's wood
                        # first and retire the wood counter first, so the pair
                        # (wood_needed, stone_needed) still sums to the units
                        # outstanding -- every progress and protection
                        # calculation reads that sum and needs no special case.
                        if (self.site_wood_needed[s] + self.site_stone_needed[s]) > 0 and (
                                pool.wood[i] > 0 or pool.stone[i] > 0):
                            if pool.wood[i] > 0:
                                pool.wood[i] -= 1
                            else:
                                pool.stone[i] -= 1
                            if self.site_wood_needed[s] > 0:
                                self.site_wood_needed[s] -= 1
                            else:
                                self.site_stone_needed[s] -= 1
                            delivered = True
                    elif self.site_wood_needed[s] > 0 and pool.wood[i] > 0:
                        self.site_wood_needed[s] -= 1
                        pool.wood[i] -= 1
                        delivered = True
                    elif self.site_stone_needed[s] > 0 and pool.stone[i] > 0:
                        self.site_stone_needed[s] -= 1
                        pool.stone[i] -= 1
                        delivered = True
                    if delivered:
                        built[i] = 1
                        rewards[i] += cfg.reward.build
                        if self.site_wood_needed[s] == 0 and self.site_stone_needed[s] == 0:
                            # completion bonus goes to whoever laid the last unit;
                            # spreading it over past contributors would need a
                            # ledger and reward agents for work already paid for
                            rewards[i] += cfg.reward.complete
                            self._completions += 1
                        break

        # 1e. exchange (Milestone 5). One unit to the nearest neighbour in reach
        #     who has room for it. Like theft it pays nothing by default: a gift
        #     costs the giver now and can only repay through what the receiver
        #     does with it, which is a longer and weaker credit chain than theft
        #     -- and theft needed action masking before PPO would touch it.
        #
        #     Deliberately NOT targeted at whoever needs it most. "Give to the
        #     hungry" is the behaviour this milestone exists to look for; putting
        #     it in the transfer rule would mean the world does the trading and
        #     the policy merely presses a button. The giver chooses when, and
        #     which economy (food or materials); the receiver is simply whoever
        #     is standing closest with a free slot.
        #
        #     Placed before the drain for the same reason gathering is: a berry
        #     handed over on an agent's last tick can still save it, which makes
        #     the give -> eat -> survive chain one tick shorter.
        gave = np.zeros(n, dtype=np.int64)
        received = np.zeros(n, dtype=np.int64)
        transfers: list[tuple[int, int, int]] = []
        self.last_transfers = transfers
        if cfg.exchange.enabled:
            giving = acted & ((actions == GIVE_FOOD) | (actions == GIVE_MATERIAL))
            for i in (int(v) for v in np.flatnonzero(giving)):
                if actions[i] == GIVE_FOOD:
                    if pool.food[i] <= 0:
                        continue
                    room = pool.food < cfg.food.capacity
                elif cc.enabled and pool.wood[i] + pool.stone[i] > 0:
                    room = (pool.wood + pool.stone) < cc.material_capacity
                else:
                    continue
                d2 = (pool.x - pool.x[i]) ** 2 + (pool.z - pool.z[i]) ** 2
                cand = np.flatnonzero((d2 <= cfg.exchange.give_radius ** 2) & pool.alive & room)
                cand = cand[cand != i]
                if cand.size == 0:
                    continue
                j = int(cand[np.argmin(d2[cand])])
                if actions[i] == GIVE_FOOD:
                    item = ITEM_FOOD
                    pool.food[i] -= 1
                    pool.food[j] += 1
                elif pool.wood[i] > 0:
                    item = ITEM_WOOD
                    pool.wood[i] -= 1
                    pool.wood[j] += 1
                else:
                    item = ITEM_STONE
                    pool.stone[i] -= 1
                    pool.stone[j] += 1
                rewards[i] += cfg.reward.give   # 0.0 unless deliberately shaped
                gave[i] += 1
                received[j] += 1
                self._given_by_item[item] += 1
                transfers.append((i, j, item))
            if transfers and cfg.exchange.log_transfers:
                self.transfers.extend((self.tick, g, r, it) for g, r, it in transfers)

        # 1f. stockpiles and raids (Island 2.0 stage 4). Placed here, before the
        #     drain, for the same reason gathering and giving are: a berry drawn
        #     from the household store on an agent's last tick can still save it.
        #
        #     Nothing here pays a reward. A deposit is a pure cost to the
        #     depositor -- it hands a unit to a store somebody else may draw from
        #     -- which is deliberately the same shape of chain that defeated M5's
        #     giving, and the honest question stage 5 gets to ask at the option
        #     level. The one difference from a gift, and the reason a stockpile is
        #     not just a slower give: the depositor can draw it back out, so the
        #     chain is a LOAN to the group rather than a donation, and hoarding is
        #     a strategy rather than an accident.
        deposited = np.zeros(n, dtype=np.int64)
        withdrew = np.zeros(n, dtype=np.int64)
        raided = np.zeros(n, dtype=np.int64)
        raids: list[tuple[int, int, int]] = []
        self.last_raids = raids
        if sc.enabled:
            store_actions = ((actions == DEPOSIT_FOOD) | (actions == DEPOSIT_MATERIAL)
                             | (actions == WITHDRAW_FOOD) | (actions == WITHDRAW_MATERIAL))
            for i in (int(v) for v in np.flatnonzero(acted & store_actions)):
                h = int(self.household[i])
                d2 = (self.stock_x[h] - pool.x[i]) ** 2 + (self.stock_z[h] - pool.z[i]) ** 2
                if d2 > sc.stockpile_radius ** 2:
                    continue
                a = actions[i]
                if a == DEPOSIT_FOOD:
                    if pool.food[i] > 0 and self.stock_food[h] < sc.stockpile_food_capacity:
                        pool.food[i] -= 1
                        self.stock_food[h] += 1
                        deposited[i] = 1
                elif a == DEPOSIT_MATERIAL:
                    if (pool.wood[i] + pool.stone[i] > 0
                            and self.stock_material[h] < sc.stockpile_material_capacity):
                        # Wood is spent from the pocket first, mirroring the build
                        # rule -- and the store records WHICH kind arrived, so a
                        # withdrawal can only return what was really put in.
                        if pool.wood[i] > 0:
                            pool.wood[i] -= 1
                            self.stock_wood[h] += 1
                        else:
                            pool.stone[i] -= 1
                            self.stock_stone[h] += 1
                        deposited[i] = 1
                elif a == WITHDRAW_FOOD:
                    if self.stock_food[h] > 0 and pool.food[i] < cfg.food.capacity:
                        self.stock_food[h] -= 1
                        pool.food[i] += 1
                        withdrew[i] = 1
                else:
                    room = (cc.enabled
                            and pool.wood[i] + pool.stone[i] < cc.material_capacity)
                    if self.stock_material[h] > 0 and room:
                        # You take out what the house needs. Each household owns
                        # the site of its own index, so "what the house needs" is
                        # well-defined and local: prefer the kind the home site is
                        # still short of, fall back to wood-first when it is
                        # finished or the store lacks that kind. A blind
                        # wood-first rule handed agents back the kind they had
                        # just banked as useless, which is where half the
                        # deposit/draw treadmill came from.
                        want_wood = self.site_wood_needed[h] > 0
                        want_stone = self.site_stone_needed[h] > 0
                        if want_wood and self.stock_wood[h] > 0:
                            take_wood = True
                        elif want_stone and self.stock_stone[h] > 0:
                            take_wood = False
                        else:
                            take_wood = self.stock_wood[h] > 0
                        if take_wood:
                            self.stock_wood[h] -= 1
                            pool.wood[i] += 1
                        else:
                            self.stock_stone[h] -= 1
                            pool.stone[i] += 1
                        withdrew[i] = 1

            for i in (int(v) for v in np.flatnonzero(acted & (actions == RAID))):
                d2 = (self.stock_x - pool.x[i]) ** 2 + (self.stock_z - pool.z[i]) ** 2
                d2[self.household[i]] = np.inf
                cand = np.flatnonzero(d2 <= sc.stockpile_radius ** 2)
                if cand.size == 0:
                    continue
                room_m = cc.enabled and pool.wood[i] + pool.stone[i] < cc.material_capacity
                took = False
                for h in cand[np.argsort(d2[cand])]:
                    h = int(h)
                    # Food first: a raider takes what keeps it alive.
                    if self.stock_food[h] > 0 and pool.food[i] < cfg.food.capacity:
                        self.stock_food[h] -= 1
                        pool.food[i] += 1
                        item = ITEM_FOOD
                    elif self.stock_material[h] > 0 and room_m:
                        if self.stock_wood[h] > 0:
                            self.stock_wood[h] -= 1
                            pool.wood[i] += 1
                            item = ITEM_WOOD
                        else:
                            self.stock_stone[h] -= 1
                            pool.stone[i] += 1
                            item = ITEM_STONE
                    else:
                        continue
                    raided[i] = 1
                    raids.append((i, h, item))
                    self.raid_ledger.append((self.tick, int(self.household[i]), h, item))
                    took = True
                    break
                if took and sc.reputation:
                    # THE GRUDGE IS HELD BY THE HOUSEHOLD, NOT THE PANTRY. Every
                    # living member of the victim household remembers the raider,
                    # which is what lets retaliation be collective without any
                    # scripted "war" logic -- the design doc's minimal reputation.
                    victims = (self.household == raids[-1][1]) & pool.alive
                    self.grudge[victims, i] = np.minimum(
                        self.grudge[victims, i] + sc.grudge_per_theft, 1.0)

        if sc.enabled and sc.reputation:
            # A theft is remembered by its individual victim, where a raid is
            # remembered by the whole household -- the difference between being
            # robbed and being invaded. The decay is what lets a feud end rather
            # than accumulate monotonically for 600 ticks.
            for thief, victim in theft_pairs:
                self.grudge[victim, thief] = min(
                    self.grudge[victim, thief] + sc.grudge_per_theft, 1.0)
            self.grudge *= sc.grudge_decay

        # 1g. crafting (tech ladder rung 1). Last of the action phases, and after
        #     the raids on purpose: material raided this tick is in the pocket, so
        #     an axe can be made out of it. The reverse order would make a raid
        #     silently useless for exactly one tick, which is the kind of hidden
        #     ordering rule the M5 gotcha note exists to warn about.
        #
        #     Nothing pays for crafting. Its whole return is the extra wood every
        #     later chop brings in -- deliberately the same unpaid shape the rest
        #     of the project uses, so an axe that gets adopted was adopted because
        #     it works (rule 1).
        crafted = np.zeros(n, dtype=np.int64)
        if cfg.tools.enabled:
            tc = cfg.tools
            for i in (int(v) for v in np.flatnonzero(acted & (actions == CRAFT))):
                if pool.axe[i]:
                    continue
                if pool.wood[i] < tc.axe_wood_cost or pool.stone[i] < tc.axe_stone_cost:
                    continue
                if self.site_x.size:
                    d2 = (self.site_x - pool.x[i]) ** 2 + (self.site_z - pool.z[i]) ** 2
                    if float(d2.min()) > tc.craft_radius ** 2:
                        continue
                else:
                    continue
                pool.wood[i] -= tc.axe_wood_cost
                pool.stone[i] -= tc.axe_stone_cost
                pool.axe[i] = 1
                crafted[i] = 1
            self._crafted += crafted

        # 2. hunger drain -- multiplied at night for anyone not near a completed
        #    shelter. This is the hazard that makes shelter worth its materials.
        drain = np.full(n, cfg.hunger.drain_per_tick)
        if cc.enabled:
            _, is_night = night_phase(self.tick, cfg)
            if is_night:
                # Protection from the best site in range. With partial_shelter on
                # it scales with build progress, so every delivered unit buys a
                # little less night drain immediately; off, only a finished
                # shelter counts and the value is a cliff at the final unit.
                #
                # The cliff is what stalled M4: three of four units bought
                # nothing, so PPO saw no gradient to climb until a completion it
                # almost never reached by chance (0.056 an episode). This is the
                # same move as siting shelters on the clusters -- make the reward
                # landscape continuous rather than paying more at the summit.
                total = max(cc.site_wood_cost + cc.site_stone_cost, 1)
                progress = 1.0 - (self.site_wood_needed + self.site_stone_needed) / total
                if not cc.partial_shelter:
                    progress = (progress >= 1.0).astype(np.float64)
                elif cc.completion_premium > 0.0:
                    # partial_shelter made protection LINEAR in progress, which
                    # removed the cliff and with it any reason to lay the last
                    # unit -- measured on m4c, every unit of a 4-unit site is
                    # worth the same 0.25 of the drain. The premium withholds a
                    # slice until the site is finished, so the gradient survives
                    # and finishing is worth more than the units it took.
                    progress = np.where(progress >= 1.0, 1.0,
                                        progress * (1.0 - cc.completion_premium))
                protection = np.zeros(n)
                if self.site_x.size:
                    d2 = ((self.site_x[None, :] - pool.x[:, None]) ** 2
                          + (self.site_z[None, :] - pool.z[:, None]) ** 2)
                    protection = np.where(d2 <= cc.shelter_radius ** 2,
                                          progress[None, :], 0.0).max(axis=1)
                drain = drain * (1.0 + (cc.night_drain_multiplier - 1.0) * (1.0 - protection))
                sheltered = protection >= 0.5   # "indoors" for the stats
                self._last_sheltered = protection >= cfg.predators.protection_safe
                # --- tech ladder rung 2: the hunt. Here, not in its own phase,
                # because `protection` is what says who is exposed and it is
                # computed exactly once. A predator only takes from an agent
                # below `protection_safe`: the shelter has to be the answer, or
                # the mechanic teaches nothing about shelter.
                pc = cfg.predators
                if pc.enabled and self.predator_x.size:
                    prey = pool.alive & (protection < pc.protection_safe)
                    d2p = ((self.predator_x[None, :] - pool.x[:, None]) ** 2
                           + (self.predator_z[None, :] - pool.z[:, None]) ** 2)
                    caught = (d2p <= pc.attack_radius ** 2).any(axis=1) & prey
                    if caught.any():
                        drain = drain + np.where(caught, pc.damage, 0.0)
                        self._attacks += caught
                        self._hunger_lost_to_predators += float(
                            caught.sum()) * pc.damage
                self._night_sheltered += int((acted & sheltered).sum())
                self._night_exposed += int((acted & ~sheltered).sum())
                self.night_sheltered_agent += acted & sheltered
                self.night_exposed_agent += acted & ~sheltered
        pool.hunger = np.where(acted, pool.hunger - drain, pool.hunger)

        # 3. auto-eat. Eating is automatic rather than an 11th action for two
        #    reasons: the brief pins the action space at 10, and the eat reward
        #    scales with hunger deficit -- an explicit action would pay an agent
        #    to starve itself closer to death before eating.
        eating = acted & (pool.hunger < cfg.hunger.eat_threshold) & (pool.food > 0)
        if eating.any():
            deficit = np.clip(1.0 - pool.hunger / cfg.hunger.eat_threshold, 0.0, 1.0)
            rewards += np.where(eating, cfg.reward.eat * deficit, 0.0)
            pool.hunger = np.where(
                eating, np.minimum(pool.hunger + cfg.hunger.eat_restore, cfg.hunger.max), pool.hunger
            )
            pool.food = np.where(eating, pool.food - 1, pool.food)
            ate = eating.astype(np.int64)
            self._meals += ate

        # 4. death
        died = acted & (pool.hunger <= 0.0)
        if died.any():
            pool.alive = pool.alive & ~died
            pool.hunger = np.where(died, 0.0, pool.hunger)
            rewards += np.where(died, cfg.reward.death, 0.0)

        # 5. survival bonus for anyone who made it through the tick
        survived = acted & pool.alive
        rewards += np.where(survived, cfg.reward.alive_per_tick, 0.0)
        self._alive_ticks += survived.astype(np.int64)
        self._gathered += gathered
        self._stole += stole
        self._robbed += robbed
        self._contested += contested
        self._wood += wood_got
        self._stone += stone_got
        self._builds += built
        self._gave += gave
        self._received += received

        # 6. bush regrowth: a depleted bush ticks back up one berry at a time.
        #    A BLIGHT SUSPENDS IT -- the timer stops too, rather than continuing to
        #    accumulate, so a blight costs the island its full duration of income
        #    instead of being repaid in a burst the moment it lifts. A shock that
        #    the world silently makes up afterwards is not a shock.
        if sc.enabled and self.blight_active:
            self._blight_ticks += 1
            ready = np.zeros_like(self.bush_berries, dtype=bool)
        else:
            below = self.bush_berries < cfg.bushes.capacity
            self.bush_timer = np.where(below, self.bush_timer + 1, 0)
            ready = below & (self.bush_timer >= cfg.bushes.regrow_ticks)
        if ready.any():
            self.bush_berries = np.where(ready, self.bush_berries + 1, self.bush_berries)
            self.bush_timer = np.where(ready, 0, self.bush_timer)

        self._deposits += deposited
        self._withdrawals += withdrew
        self._raids += raided

        # 7. clock and termination
        self.tick += 1

        # 7b. shocks (Island 2.0 stage 4). Fired on the clock, deterministically
        #     from the seed like everything else -- a "storyteller" whose script is
        #     reproducible. They exist because stage 2 measured a population that
        #     was never stressed: construction was over by the first nightfall (20
        #     sites x 4 units against 100 agents carrying one each), so
        #     `shelter_stock` sat at zero and there was nothing left to cooperate
        #     about. A storm restores the demand and a blight restores the scarcity.
        # 7c. predators move (tech ladder rung 2). AFTER the tick counter, so the
        #     positions a replay records are the ones that will hunt next tick,
        #     and DETERMINISTIC: every move is a function of positions, so a
        #     predator world has no per-tick randomness in it.
        #
        #     At night each predator walks at the nearest EXPOSED living agent;
        #     by day it walks home to its den. Chasing only the exposed is what
        #     makes a shelter a refuge rather than a delay -- a predator that
        #     besieged a hut would turn the mechanic into a tax on everyone.
        pc = cfg.predators
        if pc.enabled and self.predator_x.size:
            if self.predators_hunting:
                prey = np.flatnonzero(pool.alive & ~self._last_sheltered)
            else:
                prey = np.zeros(0, dtype=np.int64)
            if prey.size:
                dx = pool.x[prey][None, :] - self.predator_x[:, None]
                dz = pool.z[prey][None, :] - self.predator_z[:, None]
                j = np.argmin(dx ** 2 + dz ** 2, axis=1)
                rows_p = np.arange(self.predator_x.size)
                tx, tz = dx[rows_p, j], dz[rows_p, j]
            else:
                tx = self.den_x - self.predator_x
                tz = self.den_z - self.predator_z
            dist = np.hypot(tx, tz)
            step = np.minimum(pc.speed, dist)
            move = np.where(dist > 1e-9, step / np.maximum(dist, 1e-9), 0.0)
            self.predator_x = self.predator_x + tx * move
            self.predator_z = self.predator_z + tz * move

        self.last_storm = 0
        if sc.enabled and sc.shock_interval > 0 and self.tick % sc.shock_interval == 0:
            kind = int(self.shock_rng.integers(0, 2))
            self._shocks_fired += 1
            # The ramp scales severity with episode progress, never cadence --
            # the shock clock and its rng stream are untouched, so ramp 0.0 is
            # bit-identical to the pre-ramp world.
            ramp = 1.0 + sc.shock_ramp * self.tick / cfg.world.max_ticks
            if kind == 0:
                self.blight_until = self.tick + int(round(sc.blight_ticks * ramp))
            elif cc.enabled and self.site_x.size:
                done = np.flatnonzero((self.site_wood_needed == 0)
                                      & (self.site_stone_needed == 0))
                if done.size:
                    # Damage lands on the WOOD counter, so (wood_needed +
                    # stone_needed) still sums to the units outstanding -- every
                    # protection and progress calculation reads that sum, and a
                    # storm must not be the one place the invariant breaks.
                    # Clamped at the site's total cost: a ramped storm must not
                    # push `needed` past it, or build progress goes NEGATIVE and
                    # night protection with it.
                    total = cc.site_wood_cost + cc.site_stone_cost
                    damage = int(round(sc.storm_damage * ramp))
                    self.site_wood_needed[done] = np.minimum(
                        self.site_wood_needed[done] + damage, total)
                    self._damaged += int(done.size)
                    self.last_storm = int(done.size)
                self._storms += 1
        truncated = self.tick >= cfg.world.max_ticks
        all_dead = not bool(pool.alive.any())
        return StepResult(
            obs=self.observations(),
            rewards=rewards,
            terminated=died,
            acted=acted,
            episode_done=truncated or all_dead,
            truncated=truncated and not all_dead,
            gathered=gathered,
            ate=ate,
            stole=stole,
            robbed=robbed,
            contested=contested,
            harvested=wood_got + stone_got,
            built=built,
            gave=gave,
            received=received,
            transfers=transfers,
            deposited=deposited,
            withdrew=withdrew,
            raided=raided,
            raids=raids,
        )

    # --- reporting --------------------------------------------------------

    @property
    def alive_ticks(self) -> np.ndarray:
        """Ticks each agent survived this episode, per agent."""
        return self._alive_ticks.copy()

    def stats(self) -> EpisodeStats:
        pool = self.pool
        return EpisodeStats(
            ticks=self.tick,
            deaths=int((~pool.alive).sum()),
            berries_gathered=int(self._gathered.sum()),
            meals=int(self._meals.sum()),
            mean_lifespan=float(self._alive_ticks.mean()),
            mean_final_hunger=float(pool.hunger.mean()),
            survivors=int(pool.alive.sum()),
            steals=int(self._stole.sum()),
            contests_lost=int(self._contested.sum()),
            wood_gathered=int(self._wood.sum()),
            stone_gathered=int(self._stone.sum()),
            builds=int(self._builds.sum()),
            shelters_completed=self._completions,
            night_ticks_sheltered=self._night_sheltered,
            night_ticks_exposed=self._night_exposed,
            gifts=int(self._gave.sum()),
            food_given=int(self._given_by_item[ITEM_FOOD]),
            materials_given=int(self._given_by_item[ITEM_WOOD] + self._given_by_item[ITEM_STONE]),
            transfers=list(self.transfers),
            deposits=int(self._deposits.sum()),
            withdrawals=int(self._withdrawals.sum()),
            raids=int(self._raids.sum()),
            blight_ticks=self._blight_ticks,
            storms=self._storms,
            shelters_damaged=self._damaged,
            stock_food_final=self.stock_food.copy(),
            stock_material_final=self.stock_material.copy(),
            raid_ledger=list(self.raid_ledger),
            axes_crafted=int(self._crafted.sum()),
            axes_per_agent=self._crafted.copy(),
            axe_holders=int((self.pool.axe > 0).sum()),
            attacks=int(self._attacks.sum()),
            attacks_per_agent=self._attacks.copy(),
            hunger_lost_to_predators=float(self._hunger_lost_to_predators),
        )


class VecWorld:
    """``num_envs`` independent islands stepped together, with auto-reset.

    Envs desynchronise (an episode can end early when every agent starves), so
    resets are per-env. The observation from the final tick of a finished episode
    is returned in ``final_obs`` because PPO needs it to bootstrap the value of a
    time-limit truncation -- without it, hitting ``max_ticks`` looks like death.
    """

    def __init__(self, cfg: Config, seed: int, num_envs: int | None = None) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs if num_envs is not None else cfg.ppo.num_envs)
        self.num_agents = cfg.world.num_agents
        self.obs_dim = observation_dim(cfg)
        # Distinct, reproducible stream per env; SeedSequence avoids the
        # correlated-stream trap of seeding with seed+0, seed+1, ...
        seeds = np.random.SeedSequence(seed).generate_state(self.num_envs, dtype=np.uint32)

        # cfg.mix: run a share of the envs on a SECOND world config, so one policy
        # trains on both distributions in the same update. See MixConfig for why.
        # The two configs must agree on everything the policy sees, or trained
        # weights would read off the wrong features with nothing printed -- the
        # same failure --init-from's column_map exists to prevent.
        self.mix_cfg: Config | None = None
        n_mix = 0
        if cfg.mix.config and cfg.mix.fraction > 0.0:
            from .config import load_config
            self.mix_cfg = load_config(cfg.mix.config)
            checks = (
                ("observation dim", observation_dim(self.mix_cfg), self.obs_dim),
                ("action count", num_actions(self.mix_cfg), num_actions(cfg)),
                ("agent count", self.mix_cfg.world.num_agents, self.num_agents),
            )
            for what, got, want in checks:
                if got != want:
                    raise ValueError(
                        f"mix config {cfg.mix.config!r} disagrees on {what}: "
                        f"{got} vs the primary world's {want}. One policy trains on "
                        f"both, so these must match exactly."
                    )
            n_mix = int(round(self.num_envs * cfg.mix.fraction))
        # Mixed envs occupy the low indices, which keeps the split reproducible
        # and lets a test name exactly which worlds should be which.
        self.is_mix = [i < n_mix for i in range(self.num_envs)]
        self.worlds = [World(self.mix_cfg if self.is_mix[i] else cfg, int(s))
                       for i, s in enumerate(seeds)]
        self.finished_episodes: list[EpisodeStats] = []
        self.mix_episodes = 0

    def reset(self) -> np.ndarray:
        return np.stack([w.reset() for w in self.worlds])

    def observations(self) -> np.ndarray:
        return np.stack([w.observations() for w in self.worlds])

    def action_masks(self) -> np.ndarray:
        return np.stack([w.action_mask() for w in self.worlds])

    def step(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        n_env, n_agent = self.num_envs, self.num_agents
        obs = np.zeros((n_env, n_agent, self.obs_dim), dtype=np.float32)
        final_obs = np.zeros_like(obs)
        masks = np.ones((n_env, n_agent, num_actions(self.cfg)), dtype=bool)
        rewards = np.zeros((n_env, n_agent), dtype=np.float32)
        terminated = np.zeros((n_env, n_agent), dtype=bool)
        acted = np.zeros((n_env, n_agent), dtype=bool)
        truncated = np.zeros((n_env, n_agent), dtype=bool)
        episode_done = np.zeros(n_env, dtype=bool)
        gathered = np.zeros((n_env, n_agent), dtype=np.int64)

        # One call = one DECISION. With decision_interval k > 1 each world advances
        # up to k ticks under the same action, and the rewards those ticks earn are
        # summed into the single transition PPO stores -- gamma then discounts per
        # decision, exactly as frame-skip is normally trained. A world whose episode
        # ends mid-commitment stops there (its fresh episode starts at the next
        # decision, at tick 0), so no reward, termination or stats can leak across
        # the boundary.
        k = max(self.cfg.world.decision_interval, 1)
        for e, world in enumerate(self.worlds):
            for sub in range(k):
                res = world.step(actions[e])
                rewards[e] += res.rewards
                terminated[e] |= res.terminated
                gathered[e] += res.gathered
                if sub == 0:
                    acted[e] = res.acted
                if res.episode_done:
                    episode_done[e] = True
                    # Survivors at max_ticks are truncated, not terminated.
                    truncated[e] = res.truncated & world.pool.alive
                    final_obs[e] = res.obs
                    # A mixed env's episode still TRAINS the policy -- its
                    # transitions are in the rollout like any other -- but it does
                    # not get logged: a lifespan averaged over an easy probe world
                    # and a scarce one describes neither, and this project has
                    # been wrong twice from exactly that (rule 6).
                    if self.is_mix[e]:
                        self.mix_episodes += 1
                    else:
                        self.finished_episodes.append(world.stats())
                    obs[e] = world.reset()
                    break
            else:
                obs[e] = res.obs
            masks[e] = world.action_mask()

        return {
            "obs": obs,
            "action_mask": masks,
            "final_obs": final_obs,
            "rewards": rewards,
            "terminated": terminated,
            "truncated": truncated,
            "acted": acted,
            "episode_done": episode_done,
            "gathered": gathered,
        }

    def drain_episode_stats(self) -> list[EpisodeStats]:
        out = self.finished_episodes
        self.finished_episodes = []
        return out
