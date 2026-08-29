"""Island 2.0 tech ladder, rung 2: the night predator.

Named after symptoms, as everything in this repo's test suite is. The two that
matter most are the bit-identity pin (a rung must not move a number in a world
that does not have it) and `test_a_sheltered_agent_is_never_hunted`, because if
that ever breaks the mechanic stops teaching anything about shelter and becomes
a flat tax on being alive.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from sim.agents import IDLE, observation_layout, predator_channels
from sim.config import load_config
from sim.economy import subsistence
from sim.society import make_runner, run_episodes
from sim.utility import ArbiterConfig
from sim.world import World

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "config" / "island2" / "society4_predator.yaml"
BASE = ROOT / "config" / "island2" / "society4.yaml"


@pytest.fixture
def pred_cfg():
    return load_config(PRED).replace(**{"world.num_agents": 20, "world.max_ticks": 240,
                                        "society.num_households": 4,
                                        "bushes.num_clusters": 4})


def _night_tick(cfg) -> int:
    """A tick index that is definitely night, so a hunt can be provoked."""
    from sim.agents import night_phase
    for t in range(cfg.construction.night_cycle * 2):
        if night_phase(t, cfg)[1]:
            return t
    raise AssertionError("this world has no night in it")


# --- the contract with every earlier world ----------------------------------

def test_a_world_without_predators_gains_no_channels(cfg):
    assert predator_channels(cfg) == 0
    assert "predator.dx" not in observation_layout(cfg)


def test_predators_add_three_channels_and_no_action():
    """No action and no goal, deliberately: there is nothing to DO about a
    predator that this world does not already offer, and adding a `flee` goal
    would score the response and then measure the score."""
    from sim.agents import num_actions
    pred, base = load_config(PRED), load_config(BASE)
    assert num_actions(pred) == num_actions(base)
    assert len(observation_layout(pred)) == len(observation_layout(base)) + 3
    assert set(observation_layout(pred)) - set(observation_layout(base)) == {
        "predator.dx", "predator.dz", "predator.hunting"}


def test_predators_do_not_move_a_world_that_has_none():
    """The pin. Same seed, same island, same trajectory as before rung 2 --
    taken from the code as it stood at the previous commit."""
    cfg = load_config(BASE).replace(
        **{"world.num_agents": 12, "world.max_ticks": 180,
           "society.num_households": 4, "bushes.num_clusters": 4})
    world = World(cfg, seed=10000)
    runner = make_runner(cfg, "utility", 10000, ArbiterConfig(), None)
    obs = world.observations()
    while True:
        res = world.step(runner.act(obs, world.action_mask()))
        obs = res.obs
        if res.episode_done:
            break
    h = hashlib.sha256()
    for arr in (world.pool.x, world.pool.z, world.pool.hunger, world.alive_ticks):
        h.update(np.asarray(arr).tobytes())
    assert h.hexdigest() == "e6de2dcb9052773a64eed2fb407f58d9f4f3cd0bc150d829cd16ea196a8b97cc"


# --- the mechanic ------------------------------------------------------------

def test_a_sheltered_agent_is_never_hunted(pred_cfg):
    """The shelter has to be the answer. A predator that besieged a hut would
    turn the rung into a tax on being alive rather than a reason to be indoors."""
    world = World(pred_cfg, seed=1)
    world.tick = _night_tick(pred_cfg)
    pool = world.pool
    # Everyone inside the first finished site, and the predators standing on them.
    world.site_wood_needed[0] = 0
    world.site_stone_needed[0] = 0
    pool.x[:] = world.site_x[0]
    pool.z[:] = world.site_z[0]
    world.predator_x[:] = world.site_x[0]
    world.predator_z[:] = world.site_z[0]
    world.step(np.full(pool.n, IDLE, dtype=np.int64))
    assert world.stats().attacks == 0


def test_an_exposed_agent_in_reach_loses_hunger(pred_cfg):
    world = World(pred_cfg, seed=1)
    world.tick = _night_tick(pred_cfg)
    pool = world.pool
    # Far from every site, with a predator on top of them.
    pool.x[:] = 0.0
    pool.z[:] = 0.0
    world.site_x[:] = 500.0
    world.site_z[:] = 500.0
    world.predator_x[:] = 0.0
    world.predator_z[:] = 0.0
    before = pool.hunger.copy()
    world.step(np.full(pool.n, IDLE, dtype=np.int64))
    assert world.stats().attacks == pool.n
    # The predator's damage is on TOP of the exposed night drain.
    exposed_drain = pred_cfg.hunger.drain_per_tick * pred_cfg.construction.night_drain_multiplier
    assert np.allclose(before - pool.hunger, exposed_drain + pred_cfg.predators.damage)


def test_predators_go_home_by_day(pred_cfg):
    """Somewhere to be, so the island is not permanently patrolled and a daytime
    forager is free."""
    world = World(pred_cfg, seed=1)
    world.tick = 0            # dawn
    world.predator_x[:] = 0.0
    world.predator_z[:] = 0.0
    start = np.hypot(world.predator_x - world.den_x, world.predator_z - world.den_z)
    for _ in range(5):
        world.step(np.full(world.pool.n, IDLE, dtype=np.int64))
    end = np.hypot(world.predator_x - world.den_x, world.predator_z - world.den_z)
    assert (end < start).all()


def test_a_predator_world_has_no_per_tick_randomness(pred_cfg):
    """Dens are drawn once off their own stream and every move afterwards is a
    function of positions, so two runs of one seed are byte-identical."""
    def run():
        world = World(pred_cfg, seed=7)
        runner = make_runner(pred_cfg, "utility", 7, ArbiterConfig(), None)
        obs = world.observations()
        while True:
            res = world.step(runner.act(obs, world.action_mask()))
            obs = res.obs
            if res.episode_done:
                break
        return world.predator_x.copy(), world.stats().attacks
    a, b = run(), run()
    assert np.array_equal(a[0], b[0])
    assert a[1] == b[1]


def test_adding_predators_does_not_shift_the_island(pred_cfg):
    """Their own rng stream, for the same reason shocks have one: a mechanic that
    reshuffled the bush layout would make every comparison against its own
    control a comparison of two different islands."""
    bare = pred_cfg.replace(**{"predators.enabled": False})
    a, b = World(pred_cfg, seed=11), World(bare, seed=11)
    assert np.array_equal(a.bush_x, b.bush_x)
    assert np.array_equal(a.pool.x, b.pool.x)


# --- the sizing --------------------------------------------------------------

def test_the_economy_prices_the_predator_into_exposed_demand():
    """A predator is a DEMAND-SIDE change, so the sizing has to know about it --
    rule 5, applied to the mechanic that most obviously invalidates a sizing."""
    with_p = subsistence(load_config(PRED))
    without = subsistence(load_config(BASE))
    assert with_p.demand_exposed > without.demand_exposed
    assert with_p.demand_sheltered == without.demand_sheltered, (
        "an agent indoors is never hunted, so its demand must not move")
    assert with_p.predator_damage > 0.0


def test_the_report_leads_with_whether_the_rung_was_tested_at_all(pred_cfg):
    """Pre-registered failure mode 3: if attacks are near zero nothing else in
    the section means anything, so the count comes first."""
    rep = run_episodes(pred_cfg, 1, 10000, ArbiterConfig(), policy="utility")
    assert rep.attacks and rep.caught_agents
    from sim.society import format_report
    text = format_report(pred_cfg, rep, "utility arbiter")
    assert "attacks / episode" in text
    assert text.index("attacks / episode") < text.index("hunger lost to them")


def test_a_predator_world_writes_v6_and_draws_the_predators(pred_cfg):
    """A predator that is not in the replay is the storm problem again: agents
    die at night to nothing visible and a watcher concludes the drain got
    harsher. Bumped rather than folded into v5, because a v5 reader shown this
    file would render a world with an invisible thing killing people in it."""
    from sim.replay import ReplayRecorder

    world = World(pred_cfg, seed=10000)
    runner = make_runner(pred_cfg, "utility", 10000, ArbiterConfig(), None)
    rec = ReplayRecorder(world, pred_cfg, label="t", source="test", seed=10000,
                         goal_source=runner)
    rec.snapshot()
    obs = world.observations()
    while True:
        res = world.step(runner.act(obs, world.action_mask()))
        obs = res.obs
        rec.snapshot()
        if res.episode_done:
            break
    d = rec.to_dict()
    assert d["schema_version"] == 6
    assert d["world"]["predator_count"] == pred_cfg.predators.count
    assert d["world"]["predator_attack_radius"] == pred_cfg.predators.attack_radius
    for tick in d["ticks"]:
        # Every tick, not only the hunting ones -- one that vanished by day would
        # look like a rendering bug rather than a predator going home.
        assert len(tick["d"]) == pred_cfg.predators.count
        assert all(len(p) == 2 for p in tick["d"])
    assert "predator_attacks" in d["summary"]


def test_a_world_without_predators_still_writes_v5(cfg):
    """Every replay ever written must still load, and a society world without
    predators must not claim a version that promises them."""
    from sim.replay import ReplayRecorder

    small = load_config(BASE).replace(**{"world.num_agents": 8, "world.max_ticks": 30,
                                         "society.num_households": 2,
                                         "bushes.num_clusters": 2})
    world = World(small, seed=1)
    rec = ReplayRecorder(world, small, label="t", source="test", seed=1)
    rec.snapshot()
    d = rec.to_dict()
    assert d["schema_version"] == 5
    assert "d" not in d["ticks"][0]
    assert "predator_count" not in d["world"]
