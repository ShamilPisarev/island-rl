"""Observation shape, bounds, egocentricity and padding."""

from __future__ import annotations

import numpy as np
import pytest

from sim.agents import IDLE, AgentPool, build_observations, observation_dim
from sim.world import World


def obs_slices(cfg):
    kb, ka = cfg.observation.k_bushes, cfg.observation.k_agents
    own = slice(0, 2)
    bushes = slice(2, 2 + 3 * kb)
    others = slice(2 + 3 * kb, 2 + 3 * kb + 3 * ka)
    edge = slice(2 + 3 * kb + 3 * ka, None)
    return own, bushes, others, edge


def test_observation_dim_matches_the_layout(cfg):
    kb, ka = cfg.observation.k_bushes, cfg.observation.k_agents
    assert observation_dim(cfg) == 2 + 3 * kb + 3 * ka + 3


def test_shape_and_dtype(cfg):
    w = World(cfg, seed=1)
    obs = w.observations()
    assert obs.shape == (cfg.world.num_agents, observation_dim(cfg))
    assert obs.dtype == np.float32


def test_bounds_hold_over_a_long_random_rollout(cfg):
    """Every component must stay inside [-1, 1] whatever the world does."""
    w = World(cfg.replace(**{"world.max_ticks": 10_000}), seed=2)
    rng = np.random.default_rng(0)
    lo, hi = 1.0, -1.0
    for _ in range(800):
        res = w.step(rng.integers(0, 10, size=cfg.world.num_agents))
        lo = min(lo, float(res.obs.min()))
        hi = max(hi, float(res.obs.max()))
        assert np.isfinite(res.obs).all()
        if not w.pool.alive.any():
            break
    assert lo >= -1.0
    assert hi <= 1.0


def test_bounds_hold_at_the_extremes(cfg):
    """Agents piled on the shoreline, bushes at the far edge: still in range."""
    w = World(cfg, seed=3)
    w.pool.x[:] = cfg.world.island_radius
    w.pool.z[:] = 0.0
    w.bush_x[:] = -cfg.world.island_radius
    w.bush_z[:] = 0.0
    w.bush_berries[:] = cfg.bushes.capacity
    w.pool.hunger[:] = cfg.hunger.max
    w.pool.food[:] = cfg.food.capacity
    obs = w.observations()
    assert (obs >= -1.0).all() and (obs <= 1.0).all()


def test_own_state_channels(cfg):
    w = World(cfg, seed=4)
    w.pool.hunger[0] = cfg.hunger.max / 2
    w.pool.food[0] = cfg.food.capacity
    obs = w.observations()
    own, _, _, _ = obs_slices(cfg)
    assert obs[0, 0] == pytest.approx(0.5, abs=1e-6)
    assert obs[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_bush_offsets_are_egocentric_and_signed(cfg):
    """A bush due north of the agent must show +dz and dx == 0."""
    w = World(cfg, seed=5)
    scale = cfg.observation.distance_scale
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.bush_x[:] = 100.0  # push the rest away
    w.bush_z[:] = 100.0
    w.bush_x[0], w.bush_z[0] = 0.0, 5.0
    w.bush_berries[0] = cfg.bushes.capacity
    obs = w.observations()
    _, bushes, _, _ = obs_slices(cfg)
    b = obs[0, bushes]
    assert b[0] == pytest.approx(0.0, abs=1e-6)
    assert b[1] == pytest.approx(5.0 / scale, abs=1e-5)
    assert b[2] == pytest.approx(1.0, abs=1e-6)


def test_bushes_are_ordered_nearest_first(cfg):
    w = World(cfg, seed=6)
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    obs = w.observations()
    _, bushes, _, _ = obs_slices(cfg)
    b = obs[0, bushes].reshape(cfg.observation.k_bushes, 3)
    d = np.hypot(b[:, 0], b[:, 1])
    assert np.all(np.diff(d) >= -1e-6)


def test_far_offsets_saturate_rather_than_wrap(cfg):
    w = World(cfg, seed=7)
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.bush_x[:] = cfg.observation.distance_scale * 3.0
    w.bush_z[:] = 0.0
    obs = w.observations()
    _, bushes, _, _ = obs_slices(cfg)
    assert obs[0, bushes][0] == pytest.approx(1.0)


def test_missing_bushes_are_zero_padded(cfg):
    """With one bush and K=4, the last three slots must be exactly zero."""
    w = World(cfg, seed=8)
    w.bush_x = w.bush_x[:1].copy()
    w.bush_z = w.bush_z[:1].copy()
    w.bush_berries = w.bush_berries[:1].copy()
    obs = w.observations()
    _, bushes, _, _ = obs_slices(cfg)
    b = obs[0, bushes].reshape(cfg.observation.k_bushes, 3)
    assert np.allclose(b[1:], 0.0)


def test_dead_neighbours_are_excluded_and_padded(cfg):
    w = World(cfg, seed=9)
    w.pool.alive[1:] = False
    obs = w.observations()
    _, _, others, _ = obs_slices(cfg)
    assert np.allclose(obs[0, others], 0.0)  # agent 0 has no living neighbours


def test_neighbour_channels_report_relative_position_and_hunger(cfg):
    w = World(cfg, seed=10)
    w.pool.alive[:] = False
    w.pool.alive[0] = w.pool.alive[1] = True
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1], w.pool.z[1] = 4.0, 0.0
    w.pool.hunger[1] = cfg.hunger.max / 4
    obs = w.observations()
    _, _, others, _ = obs_slices(cfg)
    o = obs[0, others].reshape(cfg.observation.k_agents, 3)
    assert o[0, 0] == pytest.approx(4.0 / cfg.observation.distance_scale, abs=1e-6)
    assert o[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert o[0, 2] == pytest.approx(0.25, abs=1e-6)
    assert np.allclose(o[1:], 0.0)


def test_agent_never_observes_itself(cfg):
    """A lone survivor must see empty neighbour slots, not a copy of itself."""
    w = World(cfg, seed=11)
    w.pool.alive[:] = False
    w.pool.alive[0] = True
    _, _, others, _ = obs_slices(cfg)
    assert np.allclose(w.observations()[0, others], 0.0)


def test_edge_features(cfg):
    w = World(cfg, seed=12)
    r = cfg.world.island_radius
    w.pool.x[0], w.pool.z[0] = r, 0.0        # on the eastern shore
    w.pool.x[1], w.pool.z[1] = 0.0, 0.0      # dead centre
    obs = w.observations()
    _, _, _, edge = obs_slices(cfg)
    shore, centre = obs[0, edge], obs[1, edge]
    assert shore[0] == pytest.approx(0.0, abs=1e-6)   # no room left
    assert shore[1] == pytest.approx(1.0, abs=1e-6)   # sea is due east
    assert shore[2] == pytest.approx(0.0, abs=1e-6)
    assert centre[0] == pytest.approx(1.0, abs=1e-6)  # maximum room
    assert centre[1] == pytest.approx(0.0, abs=1e-6)  # direction undefined at r=0
    assert centre[2] == pytest.approx(0.0, abs=1e-6)


def test_dead_agents_observe_nothing(cfg):
    w = World(cfg, seed=13)
    w.pool.alive[0] = False
    assert np.allclose(w.observations()[0], 0.0)


def test_no_living_agents_yields_all_zeros(cfg):
    w = World(cfg, seed=14)
    w.pool.alive[:] = False
    assert np.allclose(w.observations(), 0.0)


def test_observations_are_deterministic(cfg):
    a = World(cfg, seed=15).observations()
    b = World(cfg, seed=15).observations()
    assert np.array_equal(a, b)
