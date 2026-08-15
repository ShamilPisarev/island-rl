"""The policy networks, plus the two reference policies we measure them against.

``ActorCritic`` is the learned brain: a small tanh MLP trunk with separate policy
and value heads.

* **Milestone 1** shares one instance across all agents (parameter sharing) --
  each agent still acts on its own observation, they merely share weights while
  learning. It is a training shortcut, not the destination.
* **Milestone 2** wraps one instance per agent in a ``PolicyGroup``, forked from
  the shared checkpoint and then trained independently, so distinct strategies
  can diverge.

Both expose the same ``act``/``value``/``evaluate_actions`` surface, taking an
``agent_ids`` argument that ``ActorCritic`` ignores and ``PolicyGroup`` dispatches
on. That is what lets ``ppo.py`` stay one training loop instead of two that drift
apart.

The two hand-written policies are here rather than in a corner of the test suite
because they are the yardstick the definition of done is measured against:
``random_actions`` is the floor, ``greedy_forager_actions`` is a rough ceiling for
"reactive foraging with no memory".
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .agents import (BUILD, CHOP, GATHER, GIVE_FOOD, GIVE_MATERIAL, IDLE, MINE,
                     N_ACTIONS, N_MOVE_ACTIONS, STEAL, neighbour_channels,
                     num_actions, observation_layout, site_channels)
from .config import Config


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    """Orthogonal init. The standard PPO recipe: gain sqrt(2) in the trunk, 0.01
    on the policy head so the initial distribution is near-uniform (otherwise the
    first updates fight an arbitrary early preference), 1.0 on the value head."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS,
                 hidden_sizes: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_sizes = tuple(hidden_sizes)

        layers: list[nn.Module] = []
        last = obs_dim
        for size in self.hidden_sizes:
            layers += [layer_init(nn.Linear(last, size)), nn.Tanh()]
            last = size
        self.trunk = nn.Sequential(*layers)
        self.policy_head = layer_init(nn.Linear(last, n_actions), std=0.01)
        self.value_head = layer_init(nn.Linear(last, 1), std=1.0)

    # Every method takes `agent_ids` and ignores it. One shared brain has no use
    # for the caller's agent identities, but accepting them keeps this class
    # drop-in interchangeable with PolicyGroup inside ppo.py.
    def forward(self, obs: torch.Tensor, agent_ids: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @staticmethod
    def _masked(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Drive unavailable actions to zero probability.

        A large negative constant rather than -inf: -inf produces NaN gradients
        if a row ever ends up fully masked, and a silent NaN is far worse to debug
        than a merely improbable action. World.action_mask guarantees at least
        `idle` survives, so this is belt and braces.
        """
        if mask is None:
            return logits
        return logits.masked_fill(~mask, -1e8)

    def value(self, obs: torch.Tensor, agent_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.value_head(self.trunk(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, agent_ids: torch.Tensor | None = None,
            deterministic: bool = False, mask: torch.Tensor | None = None
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or argmax) actions. Returns (action, log_prob, value)."""
        logits, value = self(obs)
        logits = self._masked(logits, mask)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                         agent_ids: torch.Tensor | None = None,
                         mask: torch.Tensor | None = None
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-probs, entropy and values for stored transitions (the PPO update).

        The mask must be reapplied here: the ratio compares the current policy
        against the behaviour policy, and the behaviour policy was masked. Scoring
        against unmasked logits would make the ratio meaningless.
        """
        logits, value = self(obs)
        dist = Categorical(logits=self._masked(logits, mask))
        return dist.log_prob(actions), dist.entropy(), value

    def clip_grad_norm(self, max_norm: float) -> None:
        nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    def config_dict(self) -> dict:
        return {"mode": "shared", "obs_dim": self.obs_dim, "n_actions": self.n_actions,
                "hidden_sizes": list(self.hidden_sizes)}


class PolicyGroup(nn.Module):
    """One ``ActorCritic`` per agent, dispatched by agent id (Milestone 2).

    Conceptually this is what the project always wanted -- one brain per agent --
    with Milestone 1's shared policy as the shortcut that gets them all off the
    ground first. Forking from a trained shared checkpoint rather than from
    scratch means every agent starts competent and diverges from there, instead of
    six agents independently rediscovering how to walk to a bush.

    The dispatch is a plain loop over ``num_agents`` sub-batches. With six agents
    that is six small matmuls where the shared policy did one large one -- slower
    per step, and worth it for a training loop that stays readable.
    """

    def __init__(self, policies: Sequence[ActorCritic]) -> None:
        super().__init__()
        if not policies:
            raise ValueError("PolicyGroup needs at least one policy")
        self.policies = nn.ModuleList(policies)
        self.obs_dim = policies[0].obs_dim
        self.n_actions = policies[0].n_actions
        self.hidden_sizes = policies[0].hidden_sizes

    @property
    def num_agents(self) -> int:
        return len(self.policies)

    @staticmethod
    def from_shared(shared: ActorCritic, num_agents: int) -> "PolicyGroup":
        """Fork a trained shared policy into ``num_agents`` identical copies."""
        return PolicyGroup([copy.deepcopy(shared) for _ in range(num_agents)])

    def forward(self, obs: torch.Tensor, agent_ids: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = obs.new_zeros((obs.shape[0], self.n_actions))
        values = obs.new_zeros(obs.shape[0])
        for i, policy in enumerate(self.policies):
            rows = torch.nonzero(agent_ids == i, as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            head, value = policy(obs[rows])
            # Out-of-place scatter back into the original row order, so the
            # caller's batch layout is preserved and autograd routes each row's
            # gradient to exactly one brain.
            logits = logits.index_copy(0, rows, head)
            values = values.index_copy(0, rows, value)
        return logits, values

    def value(self, obs: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        return self(obs, agent_ids)[1]

    @torch.no_grad()
    def act(self, obs: torch.Tensor, agent_ids: torch.Tensor, deterministic: bool = False,
            mask: torch.Tensor | None = None
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs, agent_ids)
        logits = ActorCritic._masked(logits, mask)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                         agent_ids: torch.Tensor, mask: torch.Tensor | None = None
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs, agent_ids)
        dist = Categorical(logits=ActorCritic._masked(logits, mask))
        return dist.log_prob(actions), dist.entropy(), value

    def clip_grad_norm(self, max_norm: float) -> None:
        """Clip each brain separately.

        Clipping the union would couple them: one agent having a bad update would
        scale down every other agent's gradient that step. These policies are
        meant to be independent, so their gradient norms must be too.
        """
        for policy in self.policies:
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm)

    def config_dict(self) -> dict:
        return {"mode": "individual", "obs_dim": self.obs_dim, "n_actions": self.n_actions,
                "hidden_sizes": list(self.hidden_sizes), "num_agents": self.num_agents}


Brain = ActorCritic | PolicyGroup


def column_map(source_names: Sequence[str], target_names: Sequence[str]) -> list[int]:
    """For each source column, the index of the same-named target column.

    Raises if a source feature has vanished from the target: dropping a trained
    input silently is never what anyone meant.
    """
    lookup = {name: i for i, name in enumerate(target_names)}
    missing = [n for n in source_names if n not in lookup]
    if missing:
        raise ValueError(f"target layout is missing source features: {missing}")
    return [lookup[name] for name in source_names]


def grow_actor_critic(source: ActorCritic, obs_dim: int, n_actions: int,
                      obs_map: Sequence[int] | None = None,
                      action_map: Sequence[int] | None = None) -> ActorCritic:
    """Copy a trained brain into a wider observation and/or action space.

    Milestone 3 adds an observation channel and an action, so an M2 checkpoint no
    longer fits. Retraining from scratch would work but throws away competence
    that took millions of steps to acquire.

    Every trained weight is copied to the column/row holding *the same feature* in
    the target, and everything new is initialised to **zero**. A zeroed input
    column contributes nothing and a zeroed action row gives the new action a
    logit of 0 beside trained logits, so the grown policy starts out behaving
    exactly as it did and then learns to use what it has been given. The new
    action is reachable rather than masked, which is what lets PPO find out
    whether it is worth taking.

    ``obs_map``/``action_map`` map source index -> target index; without them the
    mapping is positional, which is only correct when every new feature is
    *appended*. Optional observation channels are inserted mid-vector, so callers
    crossing a milestone boundary must pass a real map built by ``column_map``
    from ``observation_layout``. Getting this wrong is silent: the policy runs
    fine and quietly reads its shoreline weights off neighbour data.
    """
    if obs_dim < source.obs_dim or n_actions < source.n_actions:
        raise ValueError(
            f"cannot shrink a policy: source is obs_dim={source.obs_dim}/"
            f"n_actions={source.n_actions}, target is {obs_dim}/{n_actions}"
        )
    obs_map = list(obs_map) if obs_map is not None else list(range(source.obs_dim))
    action_map = list(action_map) if action_map is not None else list(range(source.n_actions))
    if len(obs_map) != source.obs_dim or len(action_map) != source.n_actions:
        raise ValueError("obs_map/action_map must have one entry per source column/row")

    grown = ActorCritic(obs_dim, n_actions, source.hidden_sizes)
    src, dst = source.state_dict(), grown.state_dict()
    with torch.no_grad():
        for name, param in dst.items():
            old = src[name]
            param.zero_()
            if name == "trunk.0.weight":                 # (hidden, obs_dim)
                param[:, obs_map] = old
            elif name == "policy_head.weight":           # (n_actions, hidden)
                param[action_map, :] = old
            elif name == "policy_head.bias":             # (n_actions,)
                param[action_map] = old
            else:                                        # unchanged shapes
                param.copy_(old)
    grown.load_state_dict(dst)
    return grown


def grow_policy(source: Brain, obs_dim: int, n_actions: int,
                obs_map: Sequence[int] | None = None,
                action_map: Sequence[int] | None = None) -> Brain:
    """``grow_actor_critic`` for either brain type, preserving the mode."""
    if isinstance(source, PolicyGroup):
        return PolicyGroup([
            grow_actor_critic(p, obs_dim, n_actions, obs_map, action_map)
            for p in source.policies
        ])
    return grow_actor_critic(source, obs_dim, n_actions, obs_map, action_map)


def build_policy(cfg: Config, obs_dim: int, mode: str | None = None) -> Brain:
    """Build the brain the config asks for. ``mode`` overrides ``cfg.policy.mode``."""
    mode = mode or cfg.policy.mode
    n_actions = num_actions(cfg)   # 11 when stealing is enabled (M3), else 10
    if mode == "shared":
        return ActorCritic(obs_dim, n_actions, cfg.policy.hidden_sizes)
    if mode == "individual":
        return PolicyGroup([
            ActorCritic(obs_dim, n_actions, cfg.policy.hidden_sizes)
            for _ in range(cfg.world.num_agents)
        ])
    raise ValueError(f"unknown policy mode {mode!r} (expected 'shared' or 'individual')")


def policy_from_config_dict(cfg: Config, policy_config: dict) -> Brain:
    """Rebuild the right brain type from a checkpoint's stored policy config.

    Checkpoints written before Milestone 2 have no ``mode`` key; they are shared.
    """
    mode = policy_config.get("mode", "shared")
    obs_dim = policy_config["obs_dim"]
    hidden = tuple(policy_config.get("hidden_sizes", cfg.policy.hidden_sizes))
    if mode == "shared":
        return ActorCritic(obs_dim, policy_config["n_actions"], hidden)
    return PolicyGroup([
        ActorCritic(obs_dim, policy_config["n_actions"], hidden)
        for _ in range(policy_config["num_agents"])
    ])


# --- reference policies -----------------------------------------------------


def random_actions(obs: np.ndarray, rng: np.random.Generator,
                   n_actions: int = N_ACTIONS) -> np.ndarray:
    """Uniform over the action space. The baseline the brief asks us to report."""
    return rng.integers(0, n_actions, size=obs.shape[0])


def greedy_forager_actions(obs: np.ndarray, cfg: Config) -> np.ndarray:
    """Hand-written reactive forager, driven purely by the observation vector.

    Head for the nearest bush that still has berries; gather once within range;
    stop gathering at carrying capacity and idle instead. No memory, no planning.

    It reads only the observation -- not world state -- on purpose: if a greedy
    controller can survive on these numbers, the observation is sufficient for the
    task, and any failure to learn is the algorithm's fault, not the sensor's.
    """
    kb = cfg.observation.k_bushes
    scale = cfg.observation.distance_scale
    n = obs.shape[0]
    actions = np.full(n, N_MOVE_ACTIONS, dtype=np.int64)  # idle

    # Columns located by NAME, not position: optional channels (neighbour food,
    # bush contested, the whole M4 block) shift the layout, and a hardcoded
    # offset fails silently -- the M4 world's own-materials columns landed where
    # this function expected bushes, and the forager quietly starved.
    layout = observation_layout(cfg)
    col = {name: i for i, name in enumerate(layout)}
    bush_stride = col["bush1.dx"] - col["bush0.dx"] if kb > 1 else 3
    bush = obs[:, col["bush0.dx"]:col["bush0.dx"] + kb * bush_stride]
    bush = bush.reshape(n, kb, bush_stride)
    has_berries = bush[:, :, 2] > 0.0
    dist = np.hypot(bush[:, :, 0], bush[:, :, 1]) * scale
    dist = np.where(has_berries, dist, np.inf)
    nearest = np.argmin(dist, axis=1)
    rows = np.arange(n)
    best = dist[rows, nearest]

    full = obs[:, col["own.food"]] >= 1.0 - 1e-6
    reachable = np.isfinite(best) & ~full
    in_range = reachable & (best <= cfg.bushes.gather_radius)

    dx = bush[rows, nearest, 0]
    dz = bush[rows, nearest, 1]
    # action i points at angle i*45 degrees from +z, measured towards +x
    heading = (np.round(np.arctan2(dx, dz) / (np.pi / 4.0)) % N_MOVE_ACTIONS).astype(np.int64)
    actions = np.where(reachable, heading, actions)
    actions = np.where(in_range, GATHER, actions)
    return actions


def greedy_thief_actions(obs: np.ndarray, cfg: Config) -> np.ndarray:
    """The forager, plus: rob a loaded neighbour when one is already in reach.

    The Milestone 3 reference policy. It only ever steals opportunistically -- if a
    neighbour carrying food is within ``steal_radius`` and the agent has room, take
    from them instead of walking to a bush. It never chases a victim, so this is a
    floor on how much theft is worth, not a ceiling.

    Requires ``competition.observe_neighbour_food``: without that channel there is
    no way to tell a loaded neighbour from an empty one, and "steal at random" is
    a different behaviour wearing the same name.
    """
    if not cfg.competition.enable_steal:
        return greedy_forager_actions(obs, cfg)
    if not cfg.competition.observe_neighbour_food:
        raise ValueError("greedy_thief needs competition.observe_neighbour_food")

    actions = greedy_forager_actions(obs, cfg)
    n = obs.shape[0]
    ka = cfg.observation.k_agents
    scale = cfg.observation.distance_scale

    layout = observation_layout(cfg)
    col = {name: i for i, name in enumerate(layout)}
    start = col["neighbour0.dx"]
    neighbours = obs[:, start:start + 4 * ka].reshape(n, ka, 4)
    loaded = neighbours[:, :, 3] > 0.0
    reach = np.hypot(neighbours[:, :, 0], neighbours[:, :, 1]) * scale
    reach = np.where(loaded, reach, np.inf)

    full = obs[:, col["own.food"]] >= 1.0 - 1e-6
    can_rob = ~full & (reach.min(axis=1) <= cfg.competition.steal_radius)
    return np.where(can_rob, STEAL, actions)


def _heading(dx: np.ndarray, dz: np.ndarray) -> np.ndarray:
    """Compass action pointing along (dx, dz); action i is i*45deg from +z."""
    return (np.round(np.arctan2(dx, dz) / (np.pi / 4.0)) % N_MOVE_ACTIONS).astype(np.int64)


def greedy_builder_actions(obs: np.ndarray, cfg: Config,
                           mask: np.ndarray | None = None) -> np.ndarray:
    """The Milestone 4 reference: forage first, build shelter, sleep indoors.

    Priorities per agent, top first:
      1. keep food topped up (delegate to the greedy forager when hungry or
         under-stocked) -- a dead builder builds nothing
      2. at night, or just before nightfall, run to the nearest completed
         shelter and stay inside it
      3. otherwise work on construction: deliver carried material to the
         nearest incomplete site, else harvest whichever material that site
         still needs more of

    Observation-driven like the forager and thief, with one honest exception:
    it also reads the action mask for "could chop/mine/build succeed right
    here", which is information the learned policy receives too. It never
    plans routes and never coordinates -- a floor on what construction is
    worth, not a ceiling.
    """
    cc = cfg.construction
    if not cc.enabled:
        return greedy_forager_actions(obs, cfg)

    layout = observation_layout(cfg)
    col = {name: i for i, name in enumerate(layout)}
    scale = cfg.observation.distance_scale
    n = obs.shape[0]
    actions = greedy_forager_actions(obs, cfg)   # default: keep food coming

    def block(prefix: str, k: int, per: int) -> np.ndarray:
        start = col[f"{prefix}0.dx"]
        return obs[:, start:start + k * per].reshape(n, k, per)

    trees = block("tree", cc.k_trees, 3)
    rocks = block("rock", cc.k_rocks, 3)
    sites = block("site", cc.k_sites, site_channels(cfg))
    phase = obs[:, col["night.phase"]]
    is_night = obs[:, col["night.is_night"]] > 0.5
    # head home a little before dusk: crossing the island takes ~50 ticks
    dusk_soon = phase > (1.0 - cc.night_fraction) - 0.2

    hungry = obs[:, col["own.hunger"]] < 0.55
    low_food = obs[:, col["own.food"]] < 0.5
    forage_mode = hungry | low_food
    carrying = (obs[:, col["own.wood"]] + obs[:, col["own.stone"]]) > 0.0

    def nearest(entities: np.ndarray, want: np.ndarray) -> tuple[np.ndarray, ...]:
        d = np.hypot(entities[:, :, 0], entities[:, :, 1]) * scale
        d = np.where(want, d, np.inf)
        j = np.argmin(d, axis=1)
        r = np.arange(n)
        return d[r, j], entities[r, j, 0], entities[r, j, 1]

    # --- 2. shelter at night (complete sites only)
    complete = sites[:, :, 4] > 0.5
    d_home, hx, hz = nearest(sites, complete)
    inside = d_home <= cc.shelter_radius * 0.6
    seek_shelter = (is_night | dusk_soon) & np.isfinite(d_home)
    actions = np.where(seek_shelter & ~inside, _heading(hx, hz), actions)
    # once home at night, stay put unless the forager found food in arm's reach
    stay = seek_shelter & inside & (actions < N_MOVE_ACTIONS)
    actions = np.where(stay, IDLE, actions)

    # --- 3. construction work, only when fed and it is broad daylight.
    # Target the MOST FINISHED incomplete site, not the nearest one: progress is
    # objective, so every agent picks the same focal site and material lands in
    # one place instead of being smeared across all three. (Six agents each
    # feeding their own nearest site completed 0.4 shelters an episode; a focal
    # site completes before the first nightfall.)
    work = ~forage_mode & ~seek_shelter
    present = np.hypot(sites[:, :, 0], sites[:, :, 1]) > 0
    incomplete = (sites[:, :, 4] < 0.5) & present
    remaining = sites[:, :, 2] + sites[:, :, 3]          # need_wood + need_stone
    remaining = np.where(incomplete, remaining, np.inf)
    j = np.argmin(remaining, axis=1)
    r = np.arange(n)
    has_site = np.isfinite(remaining[r, j])
    sx, sz = sites[r, j, 0], sites[r, j, 1]
    site_needs_wood = sites[r, j, 2] > 0.0
    site_needs_stone = sites[r, j, 3] > 0.0

    if mask is None:
        mask = np.ones((n, num_actions(cfg)), dtype=bool)

    deliver = work & carrying & has_site
    actions = np.where(deliver & mask[:, BUILD], BUILD, actions)
    actions = np.where(deliver & ~mask[:, BUILD], _heading(sx, sz), actions)

    fetch = work & ~carrying & has_site
    d_tree, tx, tz = nearest(trees, trees[:, :, 2] > 0)
    d_rock, rx, rz = nearest(rocks, rocks[:, :, 2] > 0)
    # fetch what the focal site actually still needs; stone when wood is covered
    want_stone = (~site_needs_wood & site_needs_stone) | ~np.isfinite(d_tree)
    actions = np.where(fetch & mask[:, CHOP] & ~want_stone, CHOP, actions)
    actions = np.where(fetch & mask[:, MINE] & want_stone, MINE, actions)
    walk_material = fetch & ~mask[:, CHOP] & ~mask[:, MINE]
    goal_x = np.where(want_stone, rx, tx)
    goal_z = np.where(want_stone, rz, tz)
    ok = walk_material & (np.isfinite(d_tree) | np.isfinite(d_rock))
    actions = np.where(ok, _heading(goal_x, goal_z), actions)
    return actions


def greedy_trader_actions(obs: np.ndarray, cfg: Config,
                          mask: np.ndarray | None = None) -> np.ndarray:
    """The Milestone 5 reference: the builder, plus two opportunistic gifts.

    On top of everything ``greedy_builder_actions`` does:

      * **food to the hungry.** Carrying a berry, fed myself, and a neighbour in
        reach is below the eating threshold *and carrying nothing* -- hand it
        over. Redistribution, the scripted thief's economics in reverse: it
        moves a berry to whoever is closest to needing it.
      * **material to whoever is standing on the site.** Carrying wood or stone,
        and a neighbour in reach is within build range of the focal site while I
        am not -- hand it over and go back for more. A relay, which is the
        simplest thing that deserves the name specialisation-plus-exchange: I
        harvest, you deliver.

    Both conditions are deliberately narrow. The first draft gave to anyone
    marginally hungrier and to anyone marginally closer, and produced 107
    transfers an episode of which ~11% were ever used -- two agents passing a
    berry back and forth because each was momentarily the hungrier one. A floor
    made of churn is not a floor. Requiring the receiver to be about to *use* the
    unit is what makes this a reference worth losing to.

    Both are strictly opportunistic -- it never walks to a recipient, never
    negotiates, never remembers who gave it anything. So this is a **floor** on
    what exchange is worth in this world, not a ceiling, exactly like the thief.
    That floor is the point: if this policy cannot beat the plain builder, then
    a learned policy failing to trade is telling us about the world, not about
    PPO.

    Reads only the observation (plus the action mask, as the builder does).
    A neighbour's distance to a site is computable because both are egocentric
    offsets from the same origin: ``site_offset - neighbour_offset``.
    """
    if not cfg.exchange.enabled:
        return greedy_builder_actions(obs, cfg, mask)
    if not cfg.competition.observe_neighbour_food:
        raise ValueError("greedy_trader needs competition.observe_neighbour_food")
    if not cfg.exchange.observe_neighbour_materials:
        raise ValueError("greedy_trader needs exchange.observe_neighbour_materials")

    actions = greedy_builder_actions(obs, cfg, mask)
    n = obs.shape[0]
    ka = cfg.observation.k_agents
    stride = neighbour_channels(cfg)
    scale = cfg.observation.distance_scale
    col = {name: i for i, name in enumerate(observation_layout(cfg))}
    if mask is None:
        mask = np.ones((n, num_actions(cfg)), dtype=bool)

    start = col["neighbour0.dx"]
    nb = obs[:, start:start + ka * stride].reshape(n, ka, stride)
    ndx, ndz = nb[:, :, 0], nb[:, :, 1]
    n_hunger, n_food = nb[:, :, 2], nb[:, :, 3]
    n_material = nb[:, :, 4] + nb[:, :, 5]
    # A zero-padded neighbour slot is all zeros, which would read as "starving,
    # empty-handed, standing on top of me" -- i.e. the ideal recipient. Living
    # neighbours always have hunger > 0, so that is the presence test.
    present = n_hunger > 0.0
    reach = np.hypot(ndx, ndz) * scale
    in_reach = present & (reach <= cfg.exchange.give_radius)

    own_hunger = obs[:, col["own.hunger"]]
    fed = own_hunger > cfg.hunger.eat_threshold / cfg.hunger.max

    def nearest_eligible(room: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(slot, exists) for the neighbour the world would actually hand it to.

        The giver does not choose a recipient: ``give`` transfers to the NEAREST
        neighbour in reach with room, exactly as ``steal`` takes from the nearest
        loaded victim. So a script that asks "is anyone here hungry?" mostly ends
        up feeding somebody else -- measured, 80 food gifts an episode of which 3
        were eaten. What the giver actually controls is where it stands, so the
        condition has to be evaluated on that one recipient. Neighbour slots are
        already ordered by distance, so the first eligible slot is the one.
        """
        eligible = in_reach & room
        return eligible.argmax(axis=1), eligible.any(axis=1)

    # --- material to a neighbour standing nearer the focal site
    if cfg.construction.enabled:
        per = site_channels(cfg)
        sites = obs[:, col["site0.dx"]:col["site0.dx"] + cfg.construction.k_sites * per]
        sites = sites.reshape(n, cfg.construction.k_sites, per)
        incomplete = (sites[:, :, 4] < 0.5) & (np.hypot(sites[:, :, 0], sites[:, :, 1]) > 0)
        remaining = np.where(incomplete, sites[:, :, 2] + sites[:, :, 3], np.inf)
        j = np.argmin(remaining, axis=1)
        rows = np.arange(n)
        has_site = np.isfinite(remaining[rows, j])
        sx, sz = sites[rows, j, 0:1], sites[rows, j, 1:2]
        my_d = (np.hypot(sx, sz) * scale).ravel()
        their_d = np.hypot(sx - ndx, sz - ndz) * scale
        slot, exists = nearest_eligible(n_material < 1.0 - 1e-6)
        # They can deliver it this tick and I cannot: that is the whole case for
        # handing it over rather than walking the last stretch myself.
        relay = (exists & (their_d[rows, slot] <= cfg.construction.build_radius)
                 & (my_d > cfg.construction.build_radius))
        actions = np.where(has_site & relay & mask[:, GIVE_MATERIAL], GIVE_MATERIAL, actions)

    # --- food to someone about to eat it (last, so survival outranks logistics)
    slot, exists = nearest_eligible(n_food < 1.0 - 1e-6)
    r = np.arange(n)
    starving = (exists & (n_hunger[r, slot] < cfg.hunger.eat_threshold / cfg.hunger.max)
                & (n_food[r, slot] <= 0.0))
    actions = np.where(fed & starving & mask[:, GIVE_FOOD], GIVE_FOOD, actions)
    return actions
