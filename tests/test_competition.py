"""Milestone 3: bush contention, stealing, and the config layering behind them."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sim.agents import (
    BASE_ACTION_NAMES,
    GATHER,
    IDLE,
    STEAL,
    action_names,
    num_actions,
    observation_dim,
)
from sim.config import load_config
from sim.policy import (
    ActorCritic,
    build_policy,
    greedy_thief_actions,
    grow_actor_critic,
    grow_policy,
    PolicyGroup,
)
from sim.ppo import PPOTrainer
from sim.world import VecWorld, World


@pytest.fixture
def m3() -> object:
    return load_config("config/m3.yaml")


@pytest.fixture
def scarce() -> object:
    return load_config("config/scarce.yaml")


# --- config layering --------------------------------------------------------


def test_extends_inherits_and_overrides(cfg, scarce):
    assert scarce.world.island_radius == cfg.world.island_radius     # inherited
    assert scarce.hunger.drain_per_tick == cfg.hunger.drain_per_tick  # inherited
    assert scarce.bushes.capacity == 2 and cfg.bushes.capacity == 6   # overridden
    assert scarce.bushes.count == 6 and cfg.bushes.count == 20


def test_extends_chains_two_deep(m3, cfg):
    """m3 -> scarce -> default: values must come from the nearest one that sets them."""
    assert m3.bushes.capacity == 2                     # from scarce
    assert m3.world.island_radius == cfg.world.island_radius  # from default
    assert m3.competition.enable_steal is True         # from m3


def test_default_config_has_competition_off(cfg):
    """M1 and M2 worlds must be untouched by anything added for M3."""
    assert cfg.competition.contest_bushes is False
    assert cfg.competition.enable_steal is False
    assert cfg.competition.observe_neighbour_food is False
    assert num_actions(cfg) == len(BASE_ACTION_NAMES) == 10
    assert observation_dim(cfg) == 26


def test_circular_extends_is_caught(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="circular config extends"):
        load_config(tmp_path / "a.yaml")


# --- action and observation space -------------------------------------------


def test_steal_is_appended_not_inserted(m3):
    """Every pre-M3 action must keep its index, or old checkpoints change meaning."""
    names = action_names(m3)
    assert names[:10] == BASE_ACTION_NAMES
    assert names[STEAL] == "steal"
    assert num_actions(m3) == 11


def test_neighbour_food_widens_the_observation(m3, cfg):
    assert observation_dim(m3) == observation_dim(cfg) + m3.observation.k_agents
    assert observation_dim(m3) == 29


def test_neighbour_food_channel_reports_carried_food(m3):
    w = World(m3, seed=1)
    w.pool.alive[:] = False
    w.pool.alive[0] = w.pool.alive[1] = True
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1], w.pool.z[1] = 3.0, 0.0
    w.pool.food[1] = m3.food.capacity

    start = 2 + 3 * m3.observation.k_bushes
    neighbours = w.observations()[0, start:start + 4 * m3.observation.k_agents]
    assert neighbours.reshape(m3.observation.k_agents, 4)[0, 3] == pytest.approx(1.0)

    w.pool.food[1] = 0
    neighbours = w.observations()[0, start:start + 4 * m3.observation.k_agents]
    assert neighbours.reshape(m3.observation.k_agents, 4)[0, 3] == pytest.approx(0.0)


def test_observation_bounds_still_hold_with_competition_on(m3):
    w = World(m3.replace(**{"world.max_ticks": 10_000}), seed=2)
    rng = np.random.default_rng(0)
    for _ in range(400):
        res = w.step(rng.integers(0, num_actions(m3), size=m3.world.num_agents))
        assert res.obs.min() >= -1.0 and res.obs.max() <= 1.0
        if not w.pool.alive.any():
            break


# --- bush contention --------------------------------------------------------


def _park_on_bush(world, agents, bush=0):
    for i in agents:
        world.pool.x[i], world.pool.z[i] = world.bush_x[bush], world.bush_z[bush]
        world.pool.hunger[i] = world.cfg.hunger.max


def test_contested_bush_yields_one_berry_per_tick(m3):
    """Two agents on one bush: exactly one berry moves, and the loser is recorded."""
    w = World(m3, seed=3)
    w.bush_berries[:] = m3.bushes.capacity
    _park_on_bush(w, [0, 1])
    actions = np.full(m3.world.num_agents, IDLE)
    actions[0] = actions[1] = GATHER

    before = int(w.bush_berries[0])
    res = w.step(actions)

    assert int(w.bush_berries[0]) == before - 1
    assert int(w.pool.food[0]) + int(w.pool.food[1]) == 1
    assert res.gathered.sum() == 1
    assert res.contested.sum() == 1


def test_without_contention_both_agents_take_a_berry(scarce):
    """The control: on the same setup with contest_bushes off, both succeed."""
    w = World(scarce, seed=3)
    w.bush_berries[:] = scarce.bushes.capacity
    _park_on_bush(w, [0, 1])
    actions = np.full(scarce.world.num_agents, IDLE)
    actions[0] = actions[1] = GATHER

    before = int(w.bush_berries[0])
    res = w.step(actions)
    assert int(w.bush_berries[0]) == before - 2
    assert res.gathered.sum() == 2
    assert res.contested.sum() == 0


def test_contention_never_overdraws_a_bush(m3):
    """Three agents, one berry: the bush must not go negative."""
    w = World(m3, seed=4)
    w.bush_berries[:] = 0
    w.bush_berries[0] = 1
    _park_on_bush(w, [0, 1, 2])
    actions = np.full(m3.world.num_agents, IDLE)
    actions[:3] = GATHER
    res = w.step(actions)
    assert int(w.bush_berries[0]) == 0
    assert res.gathered.sum() == 1
    assert (w.bush_berries >= 0).all()


# --- stealing ---------------------------------------------------------------


def test_steal_moves_one_berry_between_agents(m3):
    w = World(m3, seed=5)
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1], w.pool.z[1] = 1.0, 0.0
    w.pool.food[1] = 2
    w.pool.hunger[:] = m3.hunger.max

    actions = np.full(m3.world.num_agents, IDLE)
    actions[0] = STEAL
    res = w.step(actions)

    assert int(w.pool.food[0]) == 1
    assert int(w.pool.food[1]) == 1
    assert res.stole[0] == 1 and res.robbed[1] == 1


def test_steal_pays_no_reward(m3):
    """The brief allows no reward terms beyond survival, so theft must be worth
    only the food. If this starts paying out, we are rewarding aggression."""
    w = World(m3, seed=6)
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1], w.pool.z[1] = 1.0, 0.0
    w.pool.food[1] = 1
    w.pool.hunger[:] = m3.hunger.max

    actions = np.full(m3.world.num_agents, IDLE)
    actions[0] = STEAL
    res = w.step(actions)
    assert res.stole[0] == 1
    assert res.rewards[0] == pytest.approx(m3.reward.alive_per_tick)


def test_steal_is_conserving(m3):
    """Total food in the world must not change: theft moves berries, never mints
    or destroys them."""
    w = World(m3.replace(**{"hunger.eat_threshold": 0.0}), seed=7)  # no eating
    w.pool.food[:] = 1
    w.pool.x[:] = 0.0
    w.pool.z[:] = 0.0
    total = int(w.pool.food.sum())
    for _ in range(20):
        w.step(np.full(m3.world.num_agents, STEAL))
        assert int(w.pool.food.sum()) == total
        assert (w.pool.food >= 0).all()


def test_steal_out_of_range_does_nothing(m3):
    w = World(m3, seed=8)
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1:], w.pool.z[1:] = 30.0, 0.0
    w.pool.food[1:] = 2
    actions = np.full(m3.world.num_agents, IDLE)
    actions[0] = STEAL
    res = w.step(actions)
    assert int(w.pool.food[0]) == 0
    assert res.stole.sum() == 0


def test_cannot_steal_from_an_empty_neighbour(m3):
    w = World(m3, seed=9)
    w.pool.x[:] = 0.0
    w.pool.z[:] = 0.0
    w.pool.food[:] = 0
    res = w.step(np.full(m3.world.num_agents, STEAL))
    assert res.stole.sum() == 0
    assert int(w.pool.food.sum()) == 0


def test_cannot_steal_from_the_dead(m3):
    w = World(m3, seed=10)
    w.pool.x[:] = 0.0
    w.pool.z[:] = 0.0
    w.pool.food[1] = 2
    w.pool.alive[1] = False
    actions = np.full(m3.world.num_agents, IDLE)
    actions[0] = STEAL
    res = w.step(actions)
    assert res.stole[0] == 0
    assert int(w.pool.food[1]) == 2


def test_full_agent_does_not_steal(m3):
    w = World(m3, seed=11)
    w.pool.x[:] = 0.0
    w.pool.z[:] = 0.0
    w.pool.hunger[:] = m3.hunger.max
    w.pool.food[0] = m3.food.capacity
    w.pool.food[1] = 1
    actions = np.full(m3.world.num_agents, IDLE)
    actions[0] = STEAL
    w.step(actions)
    assert int(w.pool.food[1]) == 1


def test_steal_action_is_inert_when_disabled(scarce):
    """Action 10 does not exist in a non-competition world; passing it must not
    steal, and must not crash either."""
    w = World(scarce, seed=12)
    w.pool.x[:] = 0.0
    w.pool.z[:] = 0.0
    w.pool.food[1] = 2
    actions = np.full(scarce.world.num_agents, IDLE)
    actions[0] = STEAL
    res = w.step(actions)
    assert int(w.pool.food[0]) == 0
    assert int(w.pool.food[1]) == 2
    assert res.stole.sum() == 0


def test_episode_stats_tally_theft_and_contests(m3):
    w = World(m3, seed=13)
    w.pool.x[:] = 0.0
    w.pool.z[:] = 0.0
    w.pool.food[:] = 1
    w.pool.hunger[:] = m3.hunger.max
    w.step(np.full(m3.world.num_agents, STEAL))
    assert w.stats().steals >= 1


# --- scripted thief ---------------------------------------------------------


def test_scripted_thief_robs_a_loaded_neighbour_in_reach(m3):
    w = World(m3, seed=14)
    w.pool.alive[:] = False
    w.pool.alive[0] = w.pool.alive[1] = True
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1], w.pool.z[1] = 1.0, 0.0
    w.pool.food[1] = 2
    assert greedy_thief_actions(w.observations(), m3)[0] == STEAL


def test_scripted_thief_ignores_an_empty_neighbour(m3):
    w = World(m3, seed=15)
    w.pool.alive[:] = False
    w.pool.alive[0] = w.pool.alive[1] = True
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1], w.pool.z[1] = 1.0, 0.0
    w.pool.food[1] = 0
    assert greedy_thief_actions(w.observations(), m3)[0] != STEAL


def test_scripted_thief_needs_the_food_channel(scarce):
    steal_only = scarce.replace(**{"competition.enable_steal": True})
    with pytest.raises(ValueError, match="observe_neighbour_food"):
        greedy_thief_actions(np.zeros((6, observation_dim(steal_only)), dtype=np.float32),
                             steal_only)


# --- growing a policy across a milestone boundary ---------------------------


def test_grown_policy_behaves_like_its_source_on_shared_inputs(cfg, m3):
    """The whole point of zero-initialising the new weights: an M2 brain dropped
    into an M3 world must start out doing exactly what it used to."""
    torch.manual_seed(0)
    old = ActorCritic(observation_dim(cfg), num_actions(cfg), (32,))
    new = grow_actor_critic(old, observation_dim(m3), num_actions(m3))

    obs_old = torch.randn(8, observation_dim(cfg))
    obs_new = torch.zeros(8, observation_dim(m3))
    obs_new[:, :observation_dim(cfg)] = obs_old

    old_logits, old_value = old(obs_old)
    new_logits, new_value = new(obs_new)
    assert torch.allclose(old_logits, new_logits[:, :num_actions(cfg)], atol=1e-6)
    assert torch.allclose(old_value, new_value, atol=1e-6)


def test_grown_policy_starts_the_new_action_at_logit_zero(cfg, m3):
    """Reachable but unpreferred: PPO gets to find out if stealing is worth it."""
    old = ActorCritic(observation_dim(cfg), num_actions(cfg), (32,))
    new = grow_actor_critic(old, observation_dim(m3), num_actions(m3))
    logits, _ = new(torch.randn(4, observation_dim(m3)))
    assert torch.allclose(logits[:, STEAL], torch.zeros(4), atol=1e-6)


def test_growing_ignores_the_new_observation_channels_at_first(cfg, m3):
    old = ActorCritic(observation_dim(cfg), num_actions(cfg), (32,))
    new = grow_actor_critic(old, observation_dim(m3), num_actions(m3))
    base = torch.randn(6, observation_dim(m3))
    perturbed = base.clone()
    perturbed[:, observation_dim(cfg):] = 1.0   # scribble on the new channels only
    assert torch.allclose(new(base)[0], new(perturbed)[0], atol=1e-6)


def test_grow_refuses_to_shrink(cfg, m3):
    big = ActorCritic(observation_dim(m3), num_actions(m3), (32,))
    with pytest.raises(ValueError, match="cannot shrink"):
        grow_actor_critic(big, observation_dim(cfg), num_actions(cfg))


def test_grow_preserves_policy_group_mode(cfg, m3):
    group = PolicyGroup([ActorCritic(observation_dim(cfg), num_actions(cfg), (16,))
                         for _ in range(3)])
    grown = grow_policy(group, observation_dim(m3), num_actions(m3), 3)
    assert isinstance(grown, PolicyGroup)
    assert grown.num_agents == 3
    assert grown.obs_dim == observation_dim(m3)
    assert grown.n_actions == num_actions(m3)


# --- training ---------------------------------------------------------------


def test_ppo_runs_in_a_competition_world(m3):
    tiny = m3.replace(**{
        "ppo.num_envs": 4, "ppo.rollout_ticks": 24, "ppo.num_minibatches": 2,
        "ppo.epochs": 2, "ppo.total_updates": 3, "world.max_ticks": 60,
        "policy.mode": "individual",
    })
    torch.manual_seed(0)
    envs = VecWorld(tiny, seed=0, num_envs=tiny.ppo.num_envs)
    assert envs.obs_dim == observation_dim(tiny)
    trainer = PPOTrainer(tiny, build_policy(tiny, envs.obs_dim), envs, device="cpu")
    for i in range(2):
        m = trainer.train_update(i)
        assert np.isfinite(m.policy_loss) and np.isfinite(m.value_loss)
    assert trainer.policy.n_actions == 11


def test_competition_world_is_deterministic(m3):
    tiny = m3.replace(**{"world.max_ticks": 60})

    def run() -> list:
        w = World(tiny, seed=21)
        rng = np.random.default_rng(5)
        out = []
        for _ in range(60):
            res = w.step(rng.integers(0, num_actions(tiny), size=tiny.world.num_agents))
            out.append((res.rewards.copy(), res.stole.copy(), res.contested.copy()))
        return out

    a, b = run(), run()
    assert all(np.array_equal(x[i], y[i]) for x, y in zip(a, b) for i in range(3))
