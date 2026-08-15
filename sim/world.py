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
    STEAL,
    AgentPool,
    ConstructionView,
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
            self.tree_x, self.tree_z = self._scatter(cc.num_trees)
            self.tree_wood = np.full(cc.num_trees, cc.tree_wood, dtype=np.int64)
            self.rock_x, self.rock_z = self._scatter(cc.num_rocks)
            self.rock_stone = np.full(cc.num_rocks, cc.rock_stone, dtype=np.int64)
            if cc.sites_at_clusters:
                # Put the shelters where the agents already are. The approach walk
                # to a scattered site is the part of the build chain nothing pays
                # for and PPO cannot credit; agents live at the bush clusters, so
                # siting there removes that leg entirely. Same shape of fix as the
                # M3 action mask -- delete the uncreditable step rather than pay
                # more for it.
                ccx, ccz = self._cluster_centres
                pick = np.arange(cc.num_sites) % len(ccx)
                jitter = self.rng.normal(0.0, 1.5, size=(2, cc.num_sites))
                self.site_x = ccx[pick] + jitter[0]
                self.site_z = ccz[pick] + jitter[1]
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

    # --- stepping ---------------------------------------------------------

    def observations(self) -> np.ndarray:
        return build_observations(
            self.pool, self.bush_x, self.bush_z, self.bush_berries, self.cfg,
            self.construction_view(),
        )

    def action_mask(self) -> np.ndarray:
        """Which actions could do anything right now, per agent (A, n_actions).

        All-true when masking is disabled, so callers never branch on the flag.
        """
        if not self.cfg.competition.mask_invalid_actions:
            return np.ones((self.pool.n, num_actions(self.cfg)), dtype=bool)
        return action_mask(self.pool, self.bush_x, self.bush_z, self.bush_berries,
                           self.cfg, self.construction_view())

    def step(self, actions: np.ndarray) -> StepResult:
        cfg = self.cfg
        pool = self.pool
        n = pool.n
        acted = pool.alive.copy()
        actions = np.asarray(actions, dtype=np.int64).reshape(n)
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
        stole = np.zeros(n, dtype=np.int64)
        robbed = np.zeros(n, dtype=np.int64)
        if cfg.competition.enable_steal:
            for i in np.flatnonzero(acted & (actions == STEAL)):
                if pool.food[i] >= cfg.food.capacity:
                    continue
                d2 = (pool.x - pool.x[i]) ** 2 + (pool.z - pool.z[i]) ** 2
                victims = np.flatnonzero(
                    (d2 <= cfg.competition.steal_radius ** 2) & pool.alive & (pool.food > 0)
                )
                victims = victims[victims != i]
                if victims.size == 0:
                    continue
                victim = int(victims[np.argmin(d2[victims])])
                pool.food[victim] -= 1
                pool.food[i] += 1
                rewards[i] += cfg.reward.steal   # 0.0 unless deliberately shaped
                stole[i] = 1
                robbed[victim] = 1

        # 1d. construction (Milestone 4): harvest materials, deliver to sites.
        wood_got = np.zeros(n, dtype=np.int64)
        stone_got = np.zeros(n, dtype=np.int64)
        built = np.zeros(n, dtype=np.int64)
        cc = cfg.construction
        if cc.enabled:
            def harvest(action, ex, ez, stock):
                out = np.zeros(n, dtype=np.int64)
                for i in np.flatnonzero(acted & (actions == action)):
                    if pool.wood[i] + pool.stone[i] >= cc.material_capacity:
                        continue
                    d2 = (ex - pool.x[i]) ** 2 + (ez - pool.z[i]) ** 2
                    cand = np.flatnonzero((d2 <= cc.harvest_radius ** 2) & (stock > 0))
                    if cand.size == 0:
                        continue
                    target = int(cand[np.argmin(d2[cand])])
                    stock[target] -= 1
                    out[i] = 1
                return out

            wood_got = harvest(CHOP, self.tree_x, self.tree_z, self.tree_wood)
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
                    if self.site_wood_needed[s] > 0 and pool.wood[i] > 0:
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
                self._night_sheltered += int((acted & sheltered).sum())
                self._night_exposed += int((acted & ~sheltered).sum())
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

        # 6. bush regrowth: a depleted bush ticks back up one berry at a time
        below = self.bush_berries < cfg.bushes.capacity
        self.bush_timer = np.where(below, self.bush_timer + 1, 0)
        ready = below & (self.bush_timer >= cfg.bushes.regrow_ticks)
        if ready.any():
            self.bush_berries = np.where(ready, self.bush_berries + 1, self.bush_berries)
            self.bush_timer = np.where(ready, 0, self.bush_timer)

        # 7. clock and termination
        self.tick += 1
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
        self.worlds = [World(cfg, int(s)) for s in seeds]
        self.finished_episodes: list[EpisodeStats] = []

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

        for e, world in enumerate(self.worlds):
            res = world.step(actions[e])
            rewards[e] = res.rewards
            terminated[e] = res.terminated
            acted[e] = res.acted
            gathered[e] = res.gathered
            episode_done[e] = res.episode_done
            if res.episode_done:
                # Survivors at max_ticks are truncated, not terminated.
                truncated[e] = res.truncated & world.pool.alive
                final_obs[e] = res.obs
                self.finished_episodes.append(world.stats())
                obs[e] = world.reset()
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
