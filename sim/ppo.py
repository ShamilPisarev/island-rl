"""PPO, hand-rolled, for a multi-agent world with agents that die mid-episode.

Standard clipped-surrogate PPO with GAE. The two things that are not standard,
and that everything else here is arranged around:

**Dead agents.** Agents die at different ticks, so the set of active learners
changes every step. Rather than ragged batching, every slot is stepped every tick
and an ``active`` mask records which transitions were real. Dead slots get a
zero observation, a forced ``idle``, and are dropped from GAE and from every loss
term. The masks are applied to the *loss*, never to the buffer shape, so the
tensors stay rectangular and easy to reason about.

**Two kinds of ending.** An agent that starves is *terminated*: its future value
is genuinely zero. An episode that hits ``max_ticks`` is *truncated*: the agents
in it were alive and would have carried on, so their future value is not zero and
must be bootstrapped from V(final observation). Treating truncation as
termination teaches the policy that the world ends at tick 600, which quietly
poisons every value estimate near the horizon.

Both are folded into a single backwards GAE pass: bootstrap value is added into
the reward at a truncation boundary, and the boundary is then marked terminal so
the recursion does not run across an episode edge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .agents import IDLE
from .config import Config
from .metrics import UpdateMetrics, explained_variance, summarise_episodes
from .policy import Brain
from .world import VecWorld


@dataclass
class Rollout:
    """One rollout, flattened over (time, env, agent) with an activity mask."""

    obs: torch.Tensor          # (T, N, A, D)
    actions: torch.Tensor      # (T, N, A)
    log_probs: torch.Tensor    # (T, N, A)
    values: torch.Tensor       # (T, N, A)
    advantages: torch.Tensor   # (T, N, A)
    returns: torch.Tensor      # (T, N, A)
    active: torch.Tensor       # (T, N, A) bool
    mean_reward: float
    episodes: list


class PPOTrainer:
    """Trains either a shared brain or a per-agent group -- the loop is identical.

    Every call into the policy carries an ``agent_ids`` tensor saying which agent
    each row belongs to. A shared ``ActorCritic`` ignores it; a ``PolicyGroup``
    dispatches on it. That one argument is what keeps Milestone 1 and Milestone 2
    on the same training loop instead of two copies that drift apart.
    """

    def __init__(self, cfg: Config, policy: Brain, envs: VecWorld,
                 device: str | torch.device = "cpu") -> None:
        self.cfg = cfg
        self.p = cfg.ppo
        self.policy = policy
        self.envs = envs
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.p.lr, eps=1e-5)
        # Own generator: minibatch shuffling must not depend on global numpy state,
        # or the determinism guarantee holds only until something else draws.
        self.rng = np.random.default_rng(cfg.seed)

        self.num_envs = envs.num_envs
        self.num_agents = envs.num_agents
        self.obs_dim = envs.obs_dim
        self.obs = torch.as_tensor(envs.reset(), dtype=torch.float32, device=self.device)
        self.agent_alive = torch.ones((self.num_envs, self.num_agents),
                                      dtype=torch.bool, device=self.device)
        self.global_step = 0

        # Which agent each row of a flattened (..., num_agents) batch belongs to.
        # Flattening puts agents in the fastest-varying position, so the pattern
        # is just arange(A) tiled -- precomputed here because it never changes.
        self.step_agent_ids = torch.arange(self.num_agents, device=self.device).repeat(self.num_envs)
        self.rollout_agent_ids = torch.arange(self.num_agents, device=self.device).repeat(
            self.p.rollout_ticks * self.num_envs
        )

    # --- rollout ----------------------------------------------------------

    @torch.no_grad()
    def collect(self) -> Rollout:
        T, N, A = self.p.rollout_ticks, self.num_envs, self.num_agents
        dev = self.device

        obs_buf = torch.zeros((T, N, A, self.obs_dim), device=dev)
        act_buf = torch.zeros((T, N, A), dtype=torch.long, device=dev)
        logp_buf = torch.zeros((T, N, A), device=dev)
        val_buf = torch.zeros((T, N, A), device=dev)
        rew_buf = torch.zeros((T, N, A), device=dev)
        # The GAE reward has the truncation bootstrap folded in; the raw copy is
        # what gets reported, so "rew/step" stays a statement about the world.
        raw_rew_buf = torch.zeros((T, N, A), device=dev)
        done_buf = torch.zeros((T, N, A), device=dev)     # 1 = do not bootstrap past here
        active_buf = torch.zeros((T, N, A), dtype=torch.bool, device=dev)

        for t in range(T):
            active = self.agent_alive.clone()
            obs_buf[t] = self.obs
            active_buf[t] = active

            action, log_prob, value = self.policy.act(
                self.obs.reshape(N * A, -1), self.step_agent_ids
            )
            action = action.reshape(N, A)
            act_buf[t] = action
            logp_buf[t] = log_prob.reshape(N, A)
            val_buf[t] = value.reshape(N, A)

            # Dead slots still went through the network to keep the batch square;
            # the world ignores their action anyway, but forcing idle keeps the
            # recorded action honest.
            env_actions = torch.where(active, action, torch.full_like(action, IDLE))
            step = self.envs.step(env_actions.cpu().numpy())

            rew_buf[t] = torch.as_tensor(step["rewards"], dtype=torch.float32, device=dev)
            raw_rew_buf[t] = rew_buf[t]

            terminated = torch.as_tensor(step["terminated"], device=dev)
            truncated = torch.as_tensor(step["truncated"], device=dev)
            episode_done = torch.as_tensor(step["episode_done"], device=dev)

            if truncated.any():
                # Bootstrap the survivors of a time-limit ending. One extra forward
                # pass per rollout at most, on the pre-reset observations.
                final_obs = torch.as_tensor(step["final_obs"], dtype=torch.float32, device=dev)
                final_value = self.policy.value(
                    final_obs.reshape(N * A, -1), self.step_agent_ids
                ).reshape(N, A)
                rew_buf[t] = rew_buf[t] + self.p.gamma * final_value * truncated

            # A slot stops bootstrapping at death, at an episode boundary, or
            # while it is inactive (a corpse waiting for the episode to end).
            done_buf[t] = (terminated | truncated | episode_done.unsqueeze(1) | ~active).float()

            self.obs = torch.as_tensor(step["obs"], dtype=torch.float32, device=dev)
            still_alive = active & ~terminated
            reset = episode_done.unsqueeze(1).expand_as(still_alive)
            self.agent_alive = torch.where(reset, torch.ones_like(still_alive), still_alive)

        self.global_step += T * N * A

        with torch.no_grad():
            last_value = self.policy.value(
                self.obs.reshape(N * A, -1), self.step_agent_ids
            ).reshape(N, A)
        advantages, returns = self._gae(rew_buf, val_buf, done_buf, last_value)

        active_count = int(active_buf.sum().item())
        mean_reward = float(raw_rew_buf[active_buf].mean().item()) if active_count else 0.0

        return Rollout(
            obs=obs_buf, actions=act_buf, log_probs=logp_buf, values=val_buf,
            advantages=advantages, returns=returns, active=active_buf,
            mean_reward=mean_reward, episodes=self.envs.drain_episode_stats(),
        )

    def _gae(self, rewards: torch.Tensor, values: torch.Tensor,
             dones: torch.Tensor, last_value: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generalised advantage estimation, per (env, agent) slot.

        ``dones[t] == 1`` cuts the recursion after step t, which covers death,
        episode end, and inactive stretches in one condition.
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(last_value)
        next_value = last_value
        for t in reversed(range(T)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.p.gamma * next_value * non_terminal - values[t]
            gae = delta + self.p.gamma * self.p.gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values[t]
        return advantages, advantages + values

    # --- update -----------------------------------------------------------

    def update(self, rollout: Rollout) -> dict[str, float]:
        active = rollout.active.reshape(-1)
        n_active = int(active.sum().item())
        if n_active == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                    "approx_kl": 0.0, "clip_fraction": 0.0, "explained_variance": float("nan")}

        # Drop inactive transitions once, here, rather than masking in every term.
        # agent_ids rides along through the same mask and the same shuffle, so a
        # row can never be scored by another agent's brain.
        obs = rollout.obs.reshape(-1, self.obs_dim)[active]
        actions = rollout.actions.reshape(-1)[active]
        old_log_probs = rollout.log_probs.reshape(-1)[active]
        advantages = rollout.advantages.reshape(-1)[active]
        returns = rollout.returns.reshape(-1)[active]
        old_values = rollout.values.reshape(-1)[active]
        agent_ids = self.rollout_agent_ids[active]

        batch_size = n_active
        minibatch_size = max(batch_size // self.p.num_minibatches, 1)
        indices = np.arange(batch_size)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_fraction": 0.0}
        n_batches = 0

        for _ in range(self.p.epochs):
            self.rng.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb = torch.as_tensor(indices[start:start + minibatch_size], device=self.device)
                if mb.numel() < 2:
                    continue

                log_probs, entropy, values = self.policy.evaluate_actions(
                    obs[mb], actions[mb], agent_ids[mb]
                )
                ratio = (log_probs - old_log_probs[mb]).exp()

                # Normalising per minibatch (not per batch) is the standard recipe
                # and matters more here than usual: the reward scale swings hard
                # between an ordinary tick (+0.01) and a death (-10).
                mb_adv = advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                policy_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - self.p.clip_coef, 1 + self.p.clip_coef),
                ).mean()

                # Clipped value loss, for the same reason the policy is clipped:
                # one -10 return should not yank the critic across the batch.
                v_clipped = old_values[mb] + torch.clamp(
                    values - old_values[mb], -self.p.clip_coef, self.p.clip_coef
                )
                value_loss = 0.5 * torch.max(
                    (values - returns[mb]) ** 2, (v_clipped - returns[mb]) ** 2
                ).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + self.p.vf_coef * value_loss - self.p.ent_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Delegated to the policy so a PolicyGroup can clip each brain
                # separately; a single global norm would couple six agents that
                # are supposed to be learning independently.
                self.policy.clip_grad_norm(self.p.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs[mb]
                    stats["approx_kl"] += float(((ratio - 1) - log_ratio).mean())
                    stats["clip_fraction"] += float(
                        ((ratio - 1).abs() > self.p.clip_coef).float().mean()
                    )
                stats["policy_loss"] += policy_loss.detach().item()
                stats["value_loss"] += value_loss.detach().item()
                stats["entropy"] += entropy_loss.detach().item()
                n_batches += 1

        for key in stats:
            stats[key] /= max(n_batches, 1)
        stats["explained_variance"] = explained_variance(
            old_values.cpu().numpy(), returns.cpu().numpy()
        )
        return stats

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    # --- one full iteration ----------------------------------------------

    def train_update(self, update_index: int) -> UpdateMetrics:
        if self.p.anneal_lr and self.p.total_updates > 0:
            frac = 1.0 - update_index / self.p.total_updates
            self.set_lr(self.p.lr * max(frac, 0.0))
        lr = self.optimizer.param_groups[0]["lr"]

        started = time.perf_counter()
        rollout = self.collect()
        stats = self.update(rollout)
        elapsed = time.perf_counter() - started
        steps = self.p.rollout_ticks * self.num_envs * self.num_agents

        return UpdateMetrics(
            update=update_index,
            agent_steps=self.global_step,
            elapsed_s=elapsed,
            steps_per_s=steps / max(elapsed, 1e-9),
            lr=lr,
            mean_reward=rollout.mean_reward,
            **summarise_episodes(rollout.episodes),
            **stats,
        )
