"""The policy network, plus the two reference policies we measure it against.

``ActorCritic`` is the learned brain: a small tanh MLP trunk with separate policy
and value heads. Milestone 1 shares one instance across all agents (parameter
sharing) -- each agent still acts on its own observation, they merely share
weights while learning. Nothing in here knows how many agents exist, so
Milestone 2 can fork one instance per agent without touching this file.

The two hand-written policies are here rather than in a corner of the test suite
because they are the yardstick the definition of done is measured against:
``random_actions`` is the floor, ``greedy_forager_actions`` is a rough ceiling for
"reactive foraging with no memory".
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .agents import GATHER, N_ACTIONS, N_MOVE_ACTIONS
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

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or argmax) actions. Returns (action, log_prob, value)."""
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-probs, entropy and values for stored transitions (the PPO update)."""
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    def config_dict(self) -> dict:
        return {"obs_dim": self.obs_dim, "n_actions": self.n_actions,
                "hidden_sizes": list(self.hidden_sizes)}


def build_policy(cfg: Config, obs_dim: int) -> ActorCritic:
    return ActorCritic(obs_dim, N_ACTIONS, cfg.policy.hidden_sizes)


# --- reference policies -----------------------------------------------------


def random_actions(obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform over all 10 actions. The baseline the brief asks us to report."""
    return rng.integers(0, N_ACTIONS, size=obs.shape[0])


def greedy_forager_actions(obs: np.ndarray, cfg: Config) -> np.ndarray:
    """Hand-written reactive forager, driven purely by the observation vector.

    Head for the nearest bush that still has berries; gather once within range;
    stop gathering at carrying capacity and idle instead. No memory, no planning.

    It reads only the observation -- not world state -- on purpose: if a greedy
    controller can survive on these 26 numbers, the observation is sufficient for
    the task, and any failure to learn is the algorithm's fault, not the sensor's.
    """
    kb = cfg.observation.k_bushes
    scale = cfg.observation.distance_scale
    n = obs.shape[0]
    actions = np.full(n, N_MOVE_ACTIONS, dtype=np.int64)  # idle

    bush = obs[:, 2:2 + 3 * kb].reshape(n, kb, 3)
    has_berries = bush[:, :, 2] > 0.0
    dist = np.hypot(bush[:, :, 0], bush[:, :, 1]) * scale
    dist = np.where(has_berries, dist, np.inf)
    nearest = np.argmin(dist, axis=1)
    rows = np.arange(n)
    best = dist[rows, nearest]

    full = obs[:, 1] >= 1.0 - 1e-6
    reachable = np.isfinite(best) & ~full
    in_range = reachable & (best <= cfg.bushes.gather_radius)

    dx = bush[rows, nearest, 0]
    dz = bush[rows, nearest, 1]
    # action i points at angle i*45 degrees from +z, measured towards +x
    heading = (np.round(np.arctan2(dx, dz) / (np.pi / 4.0)) % N_MOVE_ACTIONS).astype(np.int64)
    actions = np.where(reachable, heading, actions)
    actions = np.where(in_range, GATHER, actions)
    return actions
