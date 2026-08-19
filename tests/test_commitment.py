"""decision_interval: an action persists k ticks, enforced by the world itself."""

from __future__ import annotations

import numpy as np

from sim.agents import IDLE
from sim.world import VecWorld, World


def make_world(cfg, k: int, **overrides) -> World:
    return World(cfg.replace(**{"world.decision_interval": k, **overrides}), seed=7)


def test_sticky_ticks_ignore_the_caller(cfg):
    """Pass east at the decision tick, then north on the sticky ticks: the agent
    keeps walking east, because the world repeats the decision whatever arrives."""
    w = make_world(cfg, k=4)
    n = cfg.world.num_agents
    east = np.full(n, 2, dtype=np.int64)   # +x
    north = np.full(n, 0, dtype=np.int64)  # +z
    x0, z0 = w.pool.x.copy(), w.pool.z.copy()
    w.step(east)
    for _ in range(3):
        w.step(north)
    assert np.allclose(w.pool.x, x0 + 4 * cfg.world.move_step)
    assert np.allclose(w.pool.z, z0)
    # tick 4 is a decision tick again: north now takes effect
    w.step(north)
    assert np.allclose(w.pool.z, z0 + cfg.world.move_step)


def test_interval_one_is_bit_identical(cfg):
    """decision_interval=1 must not change a single position or reward."""
    rng = np.random.default_rng(3)
    actions = rng.integers(0, 9, size=(20, cfg.world.num_agents))
    a = World(cfg, seed=11)
    b = World(cfg.replace(**{"world.decision_interval": 1}), seed=11)
    for t in range(20):
        ra = a.step(actions[t])
        rb = b.step(actions[t])
        assert np.allclose(ra.rewards, rb.rewards)
    assert np.allclose(a.pool.x, b.pool.x)


def test_vecworld_advances_k_ticks_per_decision(cfg):
    v = VecWorld(cfg.replace(**{"world.decision_interval": 3}), seed=5, num_envs=2)
    actions = np.full((2, cfg.world.num_agents), IDLE, dtype=np.int64)
    v.step(actions)
    assert all(w.tick == 3 for w in v.worlds)


def test_vecworld_accumulates_rewards_over_the_commitment(cfg):
    """Idle agents earn alive_per_tick every tick, so one decision under k=3
    must return exactly three ticks' worth."""
    v = VecWorld(cfg.replace(**{"world.decision_interval": 3}), seed=5, num_envs=1)
    actions = np.full((1, cfg.world.num_agents), IDLE, dtype=np.int64)
    out = v.step(actions)
    assert np.allclose(out["rewards"], 3 * cfg.reward.alive_per_tick)


def test_episode_end_mid_commitment_stops_the_skip(cfg):
    """max_ticks=5 with k=3: the second decision covers only ticks 3 and 4, ends
    the episode, and nothing from the fresh episode leaks into its reward."""
    small = cfg.replace(**{"world.decision_interval": 3, "world.max_ticks": 5})
    v = VecWorld(small, seed=5, num_envs=1)
    actions = np.full((1, cfg.world.num_agents), IDLE, dtype=np.int64)
    v.step(actions)
    out = v.step(actions)
    assert out["episode_done"][0]
    assert np.allclose(out["rewards"], 2 * cfg.reward.alive_per_tick)
    # auto-reset: the new episode starts at tick 0, i.e. at a decision boundary
    assert v.worlds[0].tick == 0
    stats = v.drain_episode_stats()
    assert len(stats) == 1 and stats[0].ticks == 5
