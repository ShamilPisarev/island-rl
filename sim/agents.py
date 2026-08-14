"""Agent state, the discrete action space, and egocentric observation construction.

State is stored struct-of-arrays (one numpy array per field, indexed by agent) so
the whole population can be stepped and observed with vectorised operations
instead of a Python loop over agent objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config

# --- Action space -----------------------------------------------------------
# 0-7: move one `move_step` in a compass direction. 8: idle. 9: gather.
# 10: steal, present only when `competition.enable_steal` is on (Milestone 3).
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
N_MOVE_ACTIONS = 8
IDLE = 8
GATHER = 9
STEAL = 10
N_ACTIONS = len(BASE_ACTION_NAMES)


def action_names(cfg: Config) -> tuple[str, ...]:
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
    alive: np.ndarray
    last_action: np.ndarray

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @staticmethod
    def create(num_agents: int, max_hunger: float) -> "AgentPool":
        return AgentPool(
            x=np.zeros(num_agents, dtype=np.float64),
            z=np.zeros(num_agents, dtype=np.float64),
            hunger=np.full(num_agents, max_hunger, dtype=np.float64),
            food=np.zeros(num_agents, dtype=np.int64),
            alive=np.ones(num_agents, dtype=bool),
            last_action=np.full(num_agents, IDLE, dtype=np.int64),
        )


def neighbour_channels(cfg: Config) -> int:
    """3 per neighbour (dx, dz, hunger), or 4 with their carried food.

    Milestone 3 needs the fourth: stealing from a neighbour who is carrying
    nothing is a wasted tick, and a policy that cannot see who has food cannot
    learn to rob selectively -- it could only learn "rob at random", which would
    look like the behaviour without being it.
    """
    return 4 if cfg.competition.observe_neighbour_food else 3


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


def observation_dim(cfg: Config) -> int:
    """2 own scalars + K_b bushes x (3 or 4) + K_a agents x (3 or 4) + 3 edge."""
    return (2 + bush_channels(cfg) * cfg.observation.k_bushes
            + neighbour_channels(cfg) * cfg.observation.k_agents + 3)


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
    for j in range(cfg.observation.k_bushes):
        names += [f"bush{j}.dx", f"bush{j}.dz", f"bush{j}.berries"]
        if bush_channels(cfg) == 4:
            names.append(f"bush{j}.blocked")
    for j in range(cfg.observation.k_agents):
        names += [f"neighbour{j}.dx", f"neighbour{j}.dz", f"neighbour{j}.hunger"]
        if neighbour_channels(cfg) == 4:
            names.append(f"neighbour{j}.food")
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
    order = np.argsort(dist2, axis=1, kind="stable")[:, :k]
    valid = np.isfinite(np.take_along_axis(dist2, order, axis=1))
    if order.shape[1] < k:
        pad = k - order.shape[1]
        order = np.concatenate([order, np.zeros((rows, pad), dtype=np.int64)], axis=1)
        valid = np.concatenate([valid, np.zeros((rows, pad), dtype=bool)], axis=1)
    return order, valid


def build_observations(
    pool: AgentPool,
    bush_x: np.ndarray,
    bush_z: np.ndarray,
    bush_berries: np.ndarray,
    cfg: Config,
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

    # --- K nearest bushes (regardless of whether they still hold berries; the
    # berry channel tells the policy whether it is worth walking to)
    col = 2
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
        out[:, col + 2] = np.where(ok, bush_berries[take] / cfg.bushes.capacity, 0.0)
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
    for j in range(obs_cfg.k_agents):
        take = idx[:, j]
        ok = valid[:, j]
        out[:, col + 0] = np.where(ok, np.clip(agent_dx[np.arange(n), take] / scale, -1.0, 1.0), 0.0)
        out[:, col + 1] = np.where(ok, np.clip(agent_dz[np.arange(n), take] / scale, -1.0, 1.0), 0.0)
        out[:, col + 2] = np.where(ok, pool.hunger[take] / cfg.hunger.max, 0.0)
        if channels == 4:
            out[:, col + 3] = np.where(ok, pool.food[take] / max(cfg.food.capacity, 1), 0.0)
        col += channels

    # --- shoreline
    r = np.sqrt(pool.x**2 + pool.z**2)
    safe = np.maximum(r, 1e-9)
    out[:, col + 0] = np.clip((radius - r) / radius, 0.0, 1.0)
    out[:, col + 1] = np.where(r > 1e-9, pool.x / safe, 0.0)
    out[:, col + 2] = np.where(r > 1e-9, pool.z / safe, 0.0)

    out[~alive] = 0.0
    return out
