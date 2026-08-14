"""The island environment: state, resources, and the tick.

One ``World`` is one island with one population. ``VecWorld`` runs many of them
side by side for PPO throughput and auto-resets finished episodes.

Tick order (fixed, and the tests depend on it):
  1. decode actions -> move or gather
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
    GATHER,
    IDLE,
    MOVE_VECTORS,
    N_MOVE_ACTIONS,
    STEAL,
    AgentPool,
    build_observations,
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

        self.tick = 0
        self._alive_ticks = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._gathered = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._meals = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._stole = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._robbed = np.zeros(cfg.world.num_agents, dtype=np.int64)
        self._contested = np.zeros(cfg.world.num_agents, dtype=np.int64)
        return self.observations()

    # --- stepping ---------------------------------------------------------

    def observations(self) -> np.ndarray:
        return build_observations(
            self.pool, self.bush_x, self.bush_z, self.bush_berries, self.cfg
        )

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
                stole[i] = 1
                robbed[victim] = 1

        # 2. hunger drain
        pool.hunger = np.where(acted, pool.hunger - cfg.hunger.drain_per_tick, pool.hunger)

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

    def step(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        n_env, n_agent = self.num_envs, self.num_agents
        obs = np.zeros((n_env, n_agent, self.obs_dim), dtype=np.float32)
        final_obs = np.zeros_like(obs)
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

        return {
            "obs": obs,
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
