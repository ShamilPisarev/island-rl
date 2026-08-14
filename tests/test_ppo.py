"""PPO: shapes, masking, GAE correctness, and a learning smoke test."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sim.agents import IDLE, N_ACTIONS, observation_dim
from sim.metrics import explained_variance
from sim.policy import ActorCritic, build_policy
from sim.ppo import PPOTrainer
from sim.world import VecWorld


@pytest.fixture
def tiny(cfg):
    """A config small enough to run a few real updates in a test."""
    return cfg.replace(**{
        "ppo.num_envs": 4,
        "ppo.rollout_ticks": 24,
        "ppo.num_minibatches": 2,
        "ppo.epochs": 2,
        "ppo.total_updates": 4,
        "world.max_ticks": 40,
    })


def make_trainer(config, seed: int = 0) -> PPOTrainer:
    torch.manual_seed(seed)
    envs = VecWorld(config, seed=seed, num_envs=config.ppo.num_envs)
    policy = build_policy(config, envs.obs_dim)
    return PPOTrainer(config, policy, envs, device="cpu")


# --- network ----------------------------------------------------------------


def test_policy_shapes(cfg):
    policy = build_policy(cfg, observation_dim(cfg))
    obs = torch.randn(11, observation_dim(cfg))
    logits, value = policy(obs)
    assert logits.shape == (11, N_ACTIONS)
    assert value.shape == (11,)

    action, log_prob, v = policy.act(obs)
    assert action.shape == (11,) and log_prob.shape == (11,) and v.shape == (11,)
    assert action.min() >= 0 and action.max() < N_ACTIONS


def test_policy_head_starts_near_uniform(cfg):
    """The 0.01-gain init exists so early updates are not fighting an arbitrary
    initial preference; if that regresses, exploration silently degrades."""
    policy = build_policy(cfg, observation_dim(cfg))
    probs = torch.softmax(policy(torch.randn(256, observation_dim(cfg)))[0], dim=-1)
    assert probs.mean(0).max() < 1.0 / N_ACTIONS + 0.02


def test_deterministic_action_is_the_argmax(cfg):
    policy = build_policy(cfg, observation_dim(cfg))
    obs = torch.randn(16, observation_dim(cfg))
    logits, _ = policy(obs)
    action, _, _ = policy.act(obs, deterministic=True)
    assert torch.equal(action, logits.argmax(dim=-1))


# --- rollout bookkeeping ----------------------------------------------------


def test_rollout_shapes_and_finiteness(tiny):
    trainer = make_trainer(tiny)
    r = trainer.collect()
    T, N, A = tiny.ppo.rollout_ticks, tiny.ppo.num_envs, tiny.world.num_agents
    assert r.obs.shape == (T, N, A, trainer.obs_dim)
    assert r.actions.shape == (T, N, A)
    assert r.advantages.shape == (T, N, A)
    assert r.active.shape == (T, N, A)
    for tensor in (r.obs, r.log_probs, r.values, r.advantages, r.returns):
        assert torch.isfinite(tensor).all()
    assert trainer.global_step == T * N * A


def _collect_with_spy(trainer) -> tuple:
    """Collect a rollout while recording what the env reported each tick."""
    seen: list[dict] = []
    original = trainer.envs.step

    def spy(actions):
        out = original(actions)
        seen.append({k: np.copy(v) for k, v in out.items()})
        return out

    trainer.envs.step = spy
    rollout = trainer.collect()
    trainer.envs.step = original
    return rollout, seen


def test_agents_only_come_back_to_life_at_an_episode_boundary(cfg):
    """The whole masking story: once an agent starves it stays inactive for the
    rest of its episode, and the only thing that may flip it back on is a reset.

    The trainer tracks liveness itself (to avoid a round trip through the env
    every tick), so this checks that bookkeeping against what the world actually
    reported -- a drift between the two would silently train on corpses.
    """
    quick = cfg.replace(**{
        "ppo.num_envs": 3, "ppo.rollout_ticks": 80, "ppo.num_minibatches": 1,
        "ppo.epochs": 1, "world.max_ticks": 10_000, "hunger.drain_per_tick": 4.0,
    })
    trainer = make_trainer(quick)
    rollout, seen = _collect_with_spy(trainer)
    active = rollout.active.numpy()

    assert not active.all(), "nobody died; the test is not testing anything"

    # The trainer's mask must equal the world's own view of who acted.
    for t, step in enumerate(seen):
        assert np.array_equal(active[t], step["acted"]), f"liveness drift at tick {t}"

    resurrections = 0
    for t in range(active.shape[0] - 1):
        came_back = ~active[t] & active[t + 1]
        if came_back.any():
            assert seen[t]["episode_done"][came_back.any(axis=1)].all(), (
                f"agent revived at tick {t} without an episode reset"
            )
            resurrections += 1
    assert resurrections > 0, "no episode ended; reset path untested"


def test_inactive_transitions_are_dropped_from_the_update(cfg):
    """The update must consume exactly the active transitions -- no more (dead
    agents would poison the gradient) and no fewer."""
    quick = cfg.replace(**{
        "ppo.num_envs": 2, "ppo.rollout_ticks": 40, "ppo.num_minibatches": 1,
        "ppo.epochs": 1, "world.max_ticks": 10_000, "hunger.drain_per_tick": 5.0,
    })
    trainer = make_trainer(quick)
    r = trainer.collect()
    n_active = int(r.active.sum())
    assert n_active < r.active.numel(), "no dead agents in this rollout"

    seen = {}
    original = trainer.policy.evaluate_actions

    def spy(obs, actions):
        seen["rows"] = seen.get("rows", 0) + obs.shape[0]
        return original(obs, actions)

    trainer.policy.evaluate_actions = spy
    trainer.update(r)
    assert seen["rows"] == n_active  # 1 epoch x 1 minibatch


def test_dead_agents_are_forced_to_idle(cfg):
    """A corpse's sampled action must never reach the world."""
    quick = cfg.replace(**{
        "ppo.num_envs": 2, "ppo.rollout_ticks": 30, "ppo.num_minibatches": 1,
        "ppo.epochs": 1, "world.max_ticks": 10_000, "hunger.drain_per_tick": 6.0,
    })
    trainer = make_trainer(quick)
    seen: list[np.ndarray] = []
    original = trainer.envs.step
    trainer.envs.step = lambda actions: (seen.append(actions.copy()), original(actions))[1]

    r = trainer.collect()
    active = r.active.numpy()
    assert not active.all()
    for t, actions in enumerate(seen):
        assert (actions[~active[t]] == IDLE).all()


# --- GAE --------------------------------------------------------------------


def test_gae_matches_a_hand_computation(cfg):
    """No dones: advantages must equal the explicit backwards recursion."""
    trainer = make_trainer(cfg.replace(**{"ppo.num_envs": 1, "ppo.rollout_ticks": 5}))
    g, lam = trainer.p.gamma, trainer.p.gae_lambda
    rewards = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]])
    values = torch.tensor([[[0.5]], [[0.6]], [[0.7]], [[0.8]], [[0.9]]])
    dones = torch.zeros(5, 1, 1)
    last = torch.tensor([[1.0]])

    adv, ret = trainer._gae(rewards, values, dones, last)

    expected = [0.0] * 5
    gae = 0.0
    next_value = 1.0
    for t in reversed(range(5)):
        delta = rewards[t, 0, 0].item() + g * next_value - values[t, 0, 0].item()
        gae = delta + g * lam * gae
        expected[t] = gae
        next_value = values[t, 0, 0].item()
    assert np.allclose(adv.squeeze().numpy(), expected, atol=1e-6)
    assert np.allclose(ret.squeeze().numpy(), np.array(expected) + values.squeeze().numpy(), atol=1e-6)


def test_gae_does_not_leak_across_a_done(cfg):
    """A terminal step's advantage is r - V, and nothing after it contributes."""
    trainer = make_trainer(cfg.replace(**{"ppo.num_envs": 1, "ppo.rollout_ticks": 3}))
    rewards = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    values = torch.tensor([[[0.5]], [[0.6]], [[0.7]]])
    dones = torch.tensor([[[0.0]], [[1.0]], [[0.0]]])
    adv, _ = trainer._gae(rewards, values, dones, torch.tensor([[9.0]]))

    # step 1 is terminal: advantage is exactly r1 - V1
    assert adv[1, 0, 0].item() == pytest.approx(2.0 - 0.6, abs=1e-6)
    # step 0 bootstraps from V1 only, and inherits nothing past the boundary
    g, lam = trainer.p.gamma, trainer.p.gae_lambda
    expected0 = (1.0 + g * 0.6 - 0.5) + g * lam * (2.0 - 0.6)
    assert adv[0, 0, 0].item() == pytest.approx(expected0, abs=1e-6)


def test_truncation_bootstraps_but_death_does_not(cfg):
    """A time-limit ending must be worth more than starving to death in the same
    state; if truncation were treated as terminal these would be equal."""
    short = cfg.replace(**{
        "ppo.num_envs": 8, "ppo.rollout_ticks": 12, "world.max_ticks": 10,
        "ppo.num_minibatches": 1, "ppo.epochs": 1,
    })
    trainer = make_trainer(short)
    # Force a positive value estimate so the bootstrap is visibly non-zero.
    with torch.no_grad():
        trainer.policy.value_head.bias.fill_(5.0)
        trainer.policy.value_head.weight.zero_()
    r = trainer.collect()

    # tick 10 of 10 truncates every env; those returns must include gamma * 5.
    truncation_step = short.world.max_ticks - 1
    returns_at_truncation = r.returns[truncation_step]
    assert returns_at_truncation.mean().item() > 4.0


def test_explained_variance_edges():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(y, y) == pytest.approx(1.0)
    assert explained_variance(np.full_like(y, y.mean()), y) == pytest.approx(0.0)
    assert explained_variance(-y, y) < 0.0
    assert np.isnan(explained_variance(np.ones(4), np.ones(4)))


# --- learning ---------------------------------------------------------------


class BanditEnv:
    """Trivial contextual bandit: the observation says which action pays.

    Deliberately not the island. The point of a smoke test is to fail loudly when
    the *optimiser* is broken, and on a task where the correct answer is written
    on the tin, a working PPO must solve it in a handful of updates. If this test
    fails, the bug is in ppo.py, not in the reward design.

    The target is one-hot encoded rather than packed into one scalar, which makes
    the optimal policy linearly separable. That matters: with a scalar cue the net
    has to carve one axis into ten bins, and a slow climb would be ambiguous
    between "PPO is broken" and "the task is fiddly". Here it cannot be.
    """

    def __init__(self, num_envs: int, num_agents: int, obs_dim: int, seed: int = 0) -> None:
        assert obs_dim >= N_ACTIONS
        self.num_envs, self.num_agents, self.obs_dim = num_envs, num_agents, obs_dim
        self.rng = np.random.default_rng(seed)
        self.target = self.rng.integers(0, N_ACTIONS, size=(num_envs, num_agents))
        self.finished_episodes: list = []

    def _obs(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, self.num_agents, self.obs_dim), dtype=np.float32)
        np.put_along_axis(obs, self.target[:, :, None], 1.0, axis=2)
        return obs

    def reset(self) -> np.ndarray:
        return self._obs()

    def step(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        rewards = (actions == self.target).astype(np.float32)
        self.target = self.rng.integers(0, N_ACTIONS, size=(self.num_envs, self.num_agents))
        shape = (self.num_envs, self.num_agents)
        return {
            "obs": self._obs(),
            "final_obs": np.zeros((*shape, self.obs_dim), dtype=np.float32),
            "rewards": rewards,
            "terminated": np.zeros(shape, dtype=bool),
            "truncated": np.zeros(shape, dtype=bool),
            "acted": np.ones(shape, dtype=bool),
            "episode_done": np.zeros(self.num_envs, dtype=bool),
            "gathered": np.zeros(shape, dtype=np.int64),
        }

    def drain_episode_stats(self) -> list:
        return []


def test_ppo_learns_a_trivial_task(cfg):
    """The smoke test: on a task with a known optimum, reward must rise from the
    random floor to near-perfect and the policy must sharpen.

    Note what is *not* asserted, and why:

    * **Value loss falling from the start.** It does not, and should not. As the
      policy improves, reward per step goes from 0.1 to ~1.0, so the discounted
      returns the critic is chasing grow by an order of magnitude. Absolute value
      loss therefore climbs while learning is going well, and only falls once the
      return scale settles. The assertion below is made over the second half for
      exactly that reason -- comparing against update 0 would be measuring the
      reward scale, not the critic.
    * **Explained variance.** This bandit resamples its target i.i.d. every step,
      so the return genuinely does not depend on the observation and EV pins to 0
      no matter how good the critic is. Asserting on it here would be asserting
      on noise.
    """
    smoke = cfg.replace(**{
        "ppo.num_envs": 8, "ppo.rollout_ticks": 64, "ppo.epochs": 4,
        "ppo.num_minibatches": 4, "ppo.lr": 3e-3, "ppo.anneal_lr": False,
        "ppo.ent_coef": 0.003, "ppo.total_updates": 40,
    })
    torch.manual_seed(0)
    envs = BanditEnv(8, smoke.world.num_agents, observation_dim(smoke), seed=0)
    policy = build_policy(smoke, envs.obs_dim)
    trainer = PPOTrainer(smoke, policy, envs, device="cpu")

    rewards, value_losses, entropies = [], [], []
    for i in range(smoke.ppo.total_updates):
        m = trainer.train_update(i)
        rewards.append(m.mean_reward)
        value_losses.append(m.value_loss)
        entropies.append(m.entropy)

    chance = 1.0 / N_ACTIONS
    first, last = rewards[0], float(np.mean(rewards[-5:]))
    assert first < 2 * chance, f"task was already solved at init ({first:.3f})"
    assert last > 0.9, f"policy did not learn: {first:.3f} -> {last:.3f}"

    # The policy started uniform (entropy ln 10) and committed.
    assert entropies[0] > np.log(N_ACTIONS) - 0.05
    assert entropies[-1] < 0.5

    half = len(value_losses) // 2
    assert np.mean(value_losses[-5:]) < np.mean(value_losses[half:half + 5])
    assert trainer.policy.state_dict()["policy_head.weight"].isfinite().all()


def test_island_update_runs_and_reports_metrics(tiny):
    """Two real updates on the island: everything wires together and the metrics
    are populated rather than silently NaN."""
    trainer = make_trainer(tiny)
    metrics = [trainer.train_update(i) for i in range(2)]
    for m in metrics:
        assert np.isfinite(m.policy_loss) and np.isfinite(m.value_loss)
        assert m.entropy > 0.0
        assert m.agent_steps > 0
        assert m.steps_per_s > 0
    assert metrics[-1].agent_steps == 2 * tiny.ppo.rollout_ticks * tiny.ppo.num_envs * tiny.world.num_agents
    # 40-tick episodes inside a 24-tick rollout: the second update must have seen
    # some finish, which exercises the episode-stat plumbing.
    assert metrics[-1].episodes > 0
    assert metrics[-1].mean_lifespan is not None


def test_lr_annealing_reaches_zero(tiny):
    trainer = make_trainer(tiny)
    trainer.train_update(0)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(tiny.ppo.lr)
    trainer.train_update(tiny.ppo.total_updates)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_training_is_deterministic_under_a_fixed_seed(tiny):
    """Same seed, same config, same weights -- the guarantee the brief asks for,
    extended past the world into the optimiser."""
    def run() -> list[float]:
        trainer = make_trainer(tiny, seed=17)
        return [trainer.train_update(i).mean_reward for i in range(3)]

    a, b = run(), run()
    assert a == b
