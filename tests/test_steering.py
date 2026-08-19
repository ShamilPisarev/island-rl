"""The steering override: it must change WHERE a policy walks and nothing else.

The number this module produced -- navigation being worth +89 to +116 ticks on m4h,
more than any other measured lever in the project -- is only meaningful if the
override leaves every non-move decision alone and does not quietly hand the policy
extra actions. These pin both.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.agents import N_MOVE_ACTIONS, action_names
from sim.config import load_config
from sim.steering import run_steered, steered_act
from sim.world import World


@pytest.fixture
def world_cfg():
    """m4h's world, short episodes: these are about the override, not about scores."""
    return load_config("config/m4h.yaml").replace(**{"world.max_ticks": 120})


def test_steering_only_rewrites_move_actions(world_cfg):
    """A decision is never replaced -- gather stays gather, build stays build.

    The claim rests on this: if steering also converted non-moves into moves it
    would be buying its gain by re-deciding, not by navigating.
    """
    names = action_names(world_cfg)
    rng = np.random.default_rng(0)
    chosen: list[np.ndarray] = []

    def base(obs, mask):
        a = np.array([rng.choice(np.flatnonzero(mask[i])) for i in range(mask.shape[0])],
                     dtype=np.int64)
        chosen.append(a.copy())
        return a

    act = steered_act(world_cfg, base, "food")
    w = World(world_cfg, seed=10000)
    obs = w.observations()
    for _ in range(60):
        mask = w.action_mask()
        out = act(w, obs, mask)
        before = chosen[-1]
        moved = before < N_MOVE_ACTIONS
        assert np.array_equal(out[~moved], before[~moved]), (
            f"a non-move decision was rewritten: {[names[k] for k in before[~moved]]}")
        assert (out[moved] < N_MOVE_ACTIONS).all(), "a move became a non-move"
        r = w.step(out)
        obs = r.obs
        if r.episode_done:
            break


def test_steering_actually_closes_on_food(world_cfg):
    """Sanity on the direction: steered agents end up nearer berries than unsteered.

    Without this the module could be measuring a bug that helps for some other
    reason -- the whole interpretation is "it walked to the target".
    """
    idle = action_names(world_cfg).index("idle")

    def always_north(obs, mask):
        return np.zeros(mask.shape[0], dtype=np.int64)      # action 0 is a move

    def distance_after(mode):
        act = steered_act(world_cfg, always_north, mode)
        w = World(world_cfg, seed=10000)
        obs = w.observations()
        for _ in range(80):
            r = w.step(act(w, obs, w.action_mask()))
            obs = r.obs
            if r.episode_done:
                break
        loaded = w.bush_berries > 0
        if not loaded.any():
            pytest.skip("no berry-bearing bush left to measure against")
        return float(np.mean([np.hypot(w.bush_x[loaded] - w.pool.x[i],
                                       w.bush_z[loaded] - w.pool.z[i]).min()
                              for i in np.flatnonzero(w.pool.alive)]))

    assert distance_after("food") < distance_after("off"), (
        "steering did not reduce the distance to food; the direction lookup is wrong")
    assert idle == action_names(world_cfg).index("idle")   # names unchanged by steering


def test_run_steered_is_deterministic_for_a_fixed_policy(world_cfg):
    """Common random numbers: two identical calls must give identical episodes.

    run_steered re-seeds torch precisely so variants can be compared at the ~5-tick
    scale, which sampling noise alone would swamp.
    """
    def always_idle(obs, mask):
        return np.full(mask.shape[0], action_names(world_cfg).index("idle"), dtype=np.int64)

    act = steered_act(world_cfg, always_idle, "food")
    a = [s.mean_lifespan for s in run_steered(world_cfg, act, 2, 10000)]
    b = [s.mean_lifespan for s in run_steered(world_cfg, act, 2, 10000)]
    assert a == b


def test_subset_steering_leaves_other_agents_alone(world_cfg):
    """agents={0} may rewrite only agent 0's moves; everyone else's actions pass
    through untouched, or the subset experiment is not measuring an individual."""
    from sim.steering import steered_act
    from sim.evaluate import load_checkpoint  # noqa: F401  (import parity with module)

    w = World(world_cfg, seed=3)

    def base(obs, mask):
        return np.zeros(world_cfg.world.num_agents, dtype=np.int64)  # everyone: north

    act = steered_act(world_cfg, base, "food", agents={0})
    actions = act(w, w.observations(), w.action_mask())
    assert (actions[1:] == 0).all()
    # agent 0's move was redirected at the nearest berry-bearing bush
    loaded = w.bush_berries > 0
    dx = w.bush_x[loaded] - w.pool.x[0]
    dz = w.bush_z[loaded] - w.pool.z[0]
    d = np.hypot(dx, dz)
    j = int(np.argmin(d))
    v = np.array([dx[j], dz[j]]) / max(float(d[j]), 1e-9)
    from sim.agents import MOVE_VECTORS
    assert actions[0] == int(np.argmax(MOVE_VECTORS[:N_MOVE_ACTIONS] @ v))
