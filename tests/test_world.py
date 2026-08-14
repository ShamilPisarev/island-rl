"""World mechanics: movement, hunger, death, gathering, regrowth, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from sim.agents import GATHER, IDLE, MOVE_VECTORS, N_ACTIONS
from sim.world import VecWorld, World


def make_world(cfg, **overrides) -> World:
    return World(cfg.replace(**overrides), seed=7)


# --- geometry ---------------------------------------------------------------


def test_move_vectors_are_unit_length():
    assert np.allclose(np.linalg.norm(MOVE_VECTORS, axis=1), 1.0)
    assert MOVE_VECTORS.shape == (8, 2)
    assert np.allclose(MOVE_VECTORS[0], [0.0, 1.0])  # action 0 is north (+z)
    assert np.allclose(MOVE_VECTORS[2], [1.0, 0.0])  # action 2 is east (+x)


def test_agents_start_inside_the_island(cfg):
    w = make_world(cfg)
    r = np.hypot(w.pool.x, w.pool.z)
    assert (r <= cfg.world.island_radius * cfg.world.spawn_radius_frac + 1e-9).all()


def test_bushes_are_on_the_island(cfg):
    w = make_world(cfg)
    assert w.bush_x.shape[0] == cfg.bushes.count
    r = np.hypot(w.bush_x, w.bush_z)
    assert (r <= cfg.world.island_radius).all()


def test_bushes_are_clustered_not_uniform(cfg):
    """A clustered layout has a much shorter mean nearest-neighbour distance
    than a uniform scatter of the same count over the same disc."""
    w = make_world(cfg)
    pts = np.stack([w.bush_x, w.bush_z], axis=1)
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    clustered_nn = np.sqrt(d2.min(axis=1)).mean()

    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, pts.shape[0])
    rad = cfg.world.island_radius * np.sqrt(rng.uniform(0, 1, pts.shape[0]))
    upts = np.stack([rad * np.sin(theta), rad * np.cos(theta)], axis=1)
    ud2 = ((upts[:, None, :] - upts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(ud2, np.inf)
    uniform_nn = np.sqrt(ud2.min(axis=1)).mean()

    assert clustered_nn < uniform_nn


def test_movement_steps_the_configured_distance(cfg):
    w = make_world(cfg)
    w.pool.x[:] = 0.0
    w.pool.z[:] = 0.0
    before = np.stack([w.pool.x.copy(), w.pool.z.copy()], axis=1)
    w.step(np.full(cfg.world.num_agents, 2))  # east
    after = np.stack([w.pool.x, w.pool.z], axis=1)
    assert np.allclose(np.linalg.norm(after - before, axis=1), cfg.world.move_step)
    assert np.allclose(after[:, 1], before[:, 1])  # z unchanged when heading east


def test_agents_cannot_leave_the_island(cfg):
    w = make_world(cfg)
    w.pool.x[:] = cfg.world.island_radius - 0.1
    w.pool.z[:] = 0.0
    for _ in range(50):
        w.step(np.full(cfg.world.num_agents, 2))  # walk east into the sea
    r = np.hypot(w.pool.x, w.pool.z)
    assert (r <= cfg.world.island_radius + 1e-9).all()
    assert np.allclose(r, cfg.world.island_radius)


def test_idle_does_not_move(cfg):
    w = make_world(cfg)
    x0, z0 = w.pool.x.copy(), w.pool.z.copy()
    w.step(np.full(cfg.world.num_agents, IDLE))
    assert np.allclose(w.pool.x, x0)
    assert np.allclose(w.pool.z, z0)


# --- hunger, eating, death --------------------------------------------------


def test_hunger_drains_every_tick(cfg):
    w = make_world(cfg)
    start = w.pool.hunger.copy()
    w.step(np.full(cfg.world.num_agents, IDLE))
    assert np.allclose(w.pool.hunger, start - cfg.hunger.drain_per_tick)


def test_starvation_kills_and_the_body_stays_put(cfg):
    w = make_world(cfg, **{"world.max_ticks": 10_000})
    idle = np.full(cfg.world.num_agents, IDLE)
    ticks = 0
    while w.pool.alive.any() and ticks < 5000:
        res = w.step(idle)
        ticks += 1
    assert not w.pool.alive.any()
    expected = int(np.ceil(cfg.hunger.max / cfg.hunger.drain_per_tick))
    assert ticks == expected
    assert (w.pool.hunger == 0).all()
    # dead agents keep their last position so the viewer can mark the spot
    assert np.isfinite(w.pool.x).all()


def test_death_reward_is_paid_once(cfg):
    w = make_world(cfg, **{"world.max_ticks": 10_000})
    idle = np.full(cfg.world.num_agents, IDLE)
    total = np.zeros(cfg.world.num_agents)
    death_ticks = 0
    for _ in range(500):
        res = w.step(idle)
        total += res.rewards
        death_ticks += int(res.terminated.any())
        if not w.pool.alive.any():
            break
    assert death_ticks == 1  # all six starve on the same tick from a full start
    ticks_alive = int(np.ceil(cfg.hunger.max / cfg.hunger.drain_per_tick))
    expected = (ticks_alive - 1) * cfg.reward.alive_per_tick + cfg.reward.death
    assert np.allclose(total, expected)


def test_dead_agents_do_not_act_or_drain(cfg):
    w = make_world(cfg)
    w.pool.alive[0] = False
    w.pool.hunger[0] = 0.0
    x0 = w.pool.x[0]
    res = w.step(np.full(cfg.world.num_agents, 2))
    assert w.pool.x[0] == x0
    assert w.pool.hunger[0] == 0.0
    assert res.acted[0] == False  # noqa: E712
    assert res.rewards[0] == 0.0


def test_auto_eat_restores_hunger_and_consumes_food(cfg):
    w = make_world(cfg)
    threshold = cfg.hunger.eat_threshold
    w.pool.hunger[:] = threshold - 10.0
    w.pool.food[:] = 2
    before = w.pool.hunger.copy()
    res = w.step(np.full(cfg.world.num_agents, IDLE))
    drained = before - cfg.hunger.drain_per_tick
    assert np.allclose(w.pool.hunger, np.minimum(drained + cfg.hunger.eat_restore, cfg.hunger.max))
    assert (w.pool.food == 1).all()
    assert (res.ate == 1).all()


def test_no_eating_above_the_threshold(cfg):
    w = make_world(cfg)
    w.pool.hunger[:] = cfg.hunger.max
    w.pool.food[:] = 2
    res = w.step(np.full(cfg.world.num_agents, IDLE))
    assert (w.pool.food == 2).all()
    assert (res.ate == 0).all()


def test_eat_reward_scales_with_hunger_deficit(cfg):
    """Eating on the brink pays nearly the full bonus; eating at the threshold
    pays almost nothing."""
    starving = make_world(cfg)
    starving.pool.hunger[:] = 1.0
    starving.pool.food[:] = 1
    r_starving = starving.step(np.full(cfg.world.num_agents, IDLE)).rewards[0]

    peckish = make_world(cfg)
    peckish.pool.hunger[:] = cfg.hunger.eat_threshold - 0.1
    peckish.pool.food[:] = 1
    r_peckish = peckish.step(np.full(cfg.world.num_agents, IDLE)).rewards[0]

    assert r_starving > r_peckish
    assert r_starving <= cfg.reward.eat + cfg.reward.alive_per_tick + 1e-9


def test_hunger_never_exceeds_max(cfg):
    w = make_world(cfg)
    w.pool.hunger[:] = cfg.hunger.eat_threshold - 1.0
    w.pool.food[:] = cfg.food.capacity
    for _ in range(20):
        w.step(np.full(cfg.world.num_agents, IDLE))
    assert (w.pool.hunger <= cfg.hunger.max + 1e-9).all()


# --- gathering --------------------------------------------------------------


def test_gather_depletes_a_bush_and_fills_inventory(cfg):
    w = make_world(cfg)
    w.pool.x[0], w.pool.z[0] = w.bush_x[0], w.bush_z[0]
    berries_before = w.bush_berries[0]
    actions = np.full(cfg.world.num_agents, IDLE)
    actions[0] = GATHER
    res = w.step(actions)
    assert w.bush_berries[0] == berries_before - 1
    assert w.pool.food[0] == 1
    assert res.gathered[0] == 1
    assert res.rewards[0] == pytest.approx(cfg.reward.gather + cfg.reward.alive_per_tick)


def test_gather_out_of_range_does_nothing(cfg):
    w = make_world(cfg)
    # park the agent far from every bush
    far = np.argmax(np.hypot(w.bush_x, w.bush_z))
    w.pool.x[0] = -w.bush_x[far]
    w.pool.z[0] = -w.bush_z[far]
    d = np.hypot(w.bush_x - w.pool.x[0], w.bush_z - w.pool.z[0])
    if d.min() <= cfg.bushes.gather_radius:
        pytest.skip("no far-from-everything spot in this layout")
    actions = np.full(cfg.world.num_agents, IDLE)
    actions[0] = GATHER
    res = w.step(actions)
    assert w.pool.food[0] == 0
    assert res.gathered[0] == 0
    assert res.rewards[0] == pytest.approx(cfg.reward.alive_per_tick)


def test_gather_stops_at_carrying_capacity(cfg):
    w = make_world(cfg)
    w.pool.x[0], w.pool.z[0] = w.bush_x[0], w.bush_z[0]
    w.pool.hunger[0] = cfg.hunger.max  # keep it above the eat threshold
    actions = np.full(cfg.world.num_agents, IDLE)
    actions[0] = GATHER
    for _ in range(cfg.food.capacity + 3):
        w.pool.hunger[0] = cfg.hunger.max
        w.step(actions)
    assert w.pool.food[0] == cfg.food.capacity


def test_empty_bush_cannot_be_gathered(cfg):
    w = make_world(cfg, **{"bushes.regrow_ticks": 10_000})
    w.bush_berries[:] = 0
    w.pool.x[0], w.pool.z[0] = w.bush_x[0], w.bush_z[0]
    actions = np.full(cfg.world.num_agents, IDLE)
    actions[0] = GATHER
    w.step(actions)
    assert w.pool.food[0] == 0


def test_bush_regrows_on_cooldown(cfg):
    regrow = 5
    w = make_world(cfg, **{"bushes.regrow_ticks": regrow})
    w.bush_berries[:] = 0
    idle = np.full(cfg.world.num_agents, IDLE)
    for _ in range(regrow - 1):
        w.step(idle)
    assert (w.bush_berries == 0).all()
    w.step(idle)
    assert (w.bush_berries == 1).all()


def test_regrowth_stops_at_capacity(cfg):
    w = make_world(cfg, **{"bushes.regrow_ticks": 1})
    w.bush_berries[:] = 0
    idle = np.full(cfg.world.num_agents, IDLE)
    for _ in range(cfg.bushes.capacity + 10):
        w.step(idle)
    assert (w.bush_berries == cfg.bushes.capacity).all()


# --- episode lifecycle ------------------------------------------------------


def test_episode_truncates_at_max_ticks(cfg):
    w = make_world(cfg, **{"world.max_ticks": 25})
    idle = np.full(cfg.world.num_agents, IDLE)
    for t in range(24):
        assert not w.step(idle).episode_done
    res = w.step(idle)
    assert res.episode_done
    assert res.truncated


def test_episode_ends_when_everyone_dies(cfg):
    w = make_world(cfg, **{"world.max_ticks": 10_000})
    idle = np.full(cfg.world.num_agents, IDLE)
    res = w.step(idle)
    while not res.episode_done:
        res = w.step(idle)
    assert not w.pool.alive.any()
    assert not res.truncated  # extinction is termination, not truncation


def test_stats_track_lifespan_and_gathering(cfg):
    w = make_world(cfg, **{"world.max_ticks": 30})
    idle = np.full(cfg.world.num_agents, IDLE)
    for _ in range(30):
        w.step(idle)
    s = w.stats()
    assert s.ticks == 30
    assert s.mean_lifespan == pytest.approx(30.0)
    assert s.deaths == 0
    assert s.survivors == cfg.world.num_agents


# --- determinism ------------------------------------------------------------


def _rollout(cfg, seed: int, steps: int = 120) -> np.ndarray:
    w = World(cfg, seed=seed)
    rng = np.random.default_rng(1234)
    trace = []
    for _ in range(steps):
        actions = rng.integers(0, N_ACTIONS, size=cfg.world.num_agents)
        res = w.step(actions)
        trace.append(np.concatenate([w.pool.x, w.pool.z, w.pool.hunger, res.rewards]))
    return np.array(trace)


def test_same_seed_same_trajectory(cfg):
    assert np.array_equal(_rollout(cfg, 42), _rollout(cfg, 42))


def test_different_seed_different_trajectory(cfg):
    assert not np.array_equal(_rollout(cfg, 42), _rollout(cfg, 43))


def test_vec_world_envs_are_independent_and_reproducible(cfg):
    small = cfg.replace(**{"world.max_ticks": 20})
    actions = np.random.default_rng(0).integers(0, N_ACTIONS, size=(40, 4, cfg.world.num_agents))

    def run() -> list[np.ndarray]:
        vec = VecWorld(small, seed=3, num_envs=4)
        vec.reset()
        return [vec.step(actions[t])["rewards"].copy() for t in range(40)]

    a, b = run(), run()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    # different envs must not be running the same island
    vec = VecWorld(small, seed=3, num_envs=4)
    assert not np.allclose(vec.worlds[0].bush_x, vec.worlds[1].bush_x)


def test_vec_world_autoresets_and_reports_episodes(cfg):
    small = cfg.replace(**{"world.max_ticks": 10})
    vec = VecWorld(small, seed=5, num_envs=3)
    vec.reset()
    idle = np.full((3, cfg.world.num_agents), IDLE)
    for t in range(10):
        out = vec.step(idle)
    assert out["episode_done"].all()
    assert out["truncated"].all()  # nobody starved in 10 ticks
    assert len(vec.drain_episode_stats()) == 3
    assert all(w.tick == 0 for w in vec.worlds)  # auto-reset happened
    assert not np.allclose(out["final_obs"], 0.0)
