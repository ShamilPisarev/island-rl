"""Island 4.0: an array that grows, and a tribe that can take ground.

Every test here is named after the symptom it pins, which is this project's
convention for a correction rather than a preference -- a failing test should
say what went wrong, not which function it was in.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sim.agents import (TECH_CONQUERED, TECH_FARMING, observation_dim,
                        observation_layout)
from sim.config import load_config
from sim.replay import ReplayRecorder, island4_world
from sim.utility import CONQUER, N_GOALS, N_GOALS_ISLAND3, NEED_TERRITORY, utility_runner
from sim.world import World


def tiny(path: str, **over):
    cfg = load_config(path)
    base = {"world.max_ticks": 400, "bushes.resample_each_episode": False}
    base.update(over)
    return cfg.replace(**base)


def run(cfg, ticks: int, seed: int = 0, recorder: bool = False):
    w = World(cfg, seed=seed)
    w.reset()
    r = utility_runner(cfg, seed=seed, world=w)
    rec = ReplayRecorder(w, cfg, "t", "test", seed, goal_source=r) if recorder else None
    if rec:
        rec.snapshot()
    for _ in range(ticks):
        res = w.step(r.act(w.observations(), w.action_mask()))
        if rec:
            rec.snapshot()
        if res.episode_done:
            break
    return w, rec


# --- bit identity ----------------------------------------------------------

@pytest.mark.parametrize("path", [
    "config/island3/village.yaml",
    "config/island3/village_gen.yaml",
    "config/island3/village_frontier.yaml",
])
def test_island4_blocks_off_reproduce_island3_exactly(path):
    """Both new blocks default off, so every 3.0 world is bit-identical."""
    cfg = tiny(path)
    a, _ = run(cfg, 300)
    b, _ = run(cfg, 300)
    assert np.array_equal(a.pool.x, b.pool.x)
    assert a.pool.n == cfg.world.num_agents
    assert a.stats().conquests == 0 and a.stats().slots_grown == 0


def test_conquest_off_leaves_the_observation_width_alone():
    cfg = load_config("config/island3/village_frontier.yaml")
    assert observation_dim(cfg) == len(observation_layout(cfg))
    on = cfg.replace(**{"conquest.enabled": True})
    assert observation_dim(on) == observation_dim(cfg) + 6
    assert observation_layout(on)[-4:-3] == ("conquest.threat",)


# --- the growth registry ---------------------------------------------------

def test_every_per_agent_array_is_in_the_growth_registry():
    """A missing entry would surface 8,000 ticks later as a shape error.

    The world is built with an agent count that matches NOTHING else in it, so a
    per-bush or per-site array cannot masquerade as a per-agent one and get
    swept in (or, worse, be missed because the shapes coincided).
    """
    cfg = tiny("config/island4/empire.yaml", **{"world.num_agents": 137})
    w = World(cfg, seed=0)
    w.reset()
    n = 137
    # `grudge` (n, n) and the two (n, N_SKILLS) skill arrays are grown by hand
    # inside `_grow_slots` -- they are 2-D and the fill table is scalar.
    registry = set(World._GROWTH_FILL) | {"grudge", "skill", "_skill_carry"}
    missing = [k for k, v in vars(w).items()
               if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n
               and k not in registry]
    assert missing == [], f"per-agent arrays not in the growth registry: {missing}"


def test_growing_widens_every_array_to_one_width():
    cfg = tiny("config/island4/empire.yaml", **{"world.num_agents": 40})
    w = World(cfg, seed=0)
    w.reset()
    added = w._grow_slots(17)
    assert added == 17
    for name in World._GROWTH_FILL:
        assert getattr(w, name).shape[0] == 57, name
    for name in ("x", "z", "hunger", "food", "wood", "stone", "alive",
                 "last_action", "axe", "age", "born"):
        assert getattr(w.pool, name).shape[0] == 57, name
    assert w.grudge.shape == (57, 57)
    # A NEW ROW IS UNBORN, NOT DEAD. Both are `alive == False` and only the
    # second one ever lived; every per-agent statistic divides by the born.
    assert not w.pool.born[40:].any()
    assert not w.pool.alive[40:].any()
    assert (w._death_tick[40:] == -1).all()


def test_grown_rows_do_not_enter_the_per_agent_statistics():
    cfg = tiny("config/island4/empire.yaml", **{"world.num_agents": 40})
    w = World(cfg, seed=0)
    w.reset()
    before = w.stats().born
    w._grow_slots(60)
    assert w.stats().born == before


def test_max_slots_is_reported_not_silently_clamped():
    """A population bounded by memory must not read as a carrying capacity.

    This is read R6's mistake with the sign flipped: there, 200 rows looked like
    an island. Here the guard is allowed to bind, but the run has to SAY so.
    """
    cfg = tiny("config/island4/empire.yaml",
               **{"world.num_agents": 40, "world.max_slots": 45})
    w = World(cfg, seed=0)
    w.reset()
    assert w._grow_slots(3) == 3
    assert w._grow_slots(50) == 2      # clipped to the cap, and taken
    assert w.stats().slot_cap_hits == 0
    assert w._grow_slots(1) == 0       # nothing left
    assert w.stats().slot_cap_hits == 1


def test_step_result_arrays_match_the_new_width_after_growth():
    """A StepResult whose rewards and obs disagree is a shape error downstream."""
    cfg = tiny("config/island4/empire.yaml", **{"world.num_agents": 44})
    w = World(cfg, seed=0)
    w.reset()
    r = utility_runner(cfg, seed=0, world=w)
    grew = False
    for _ in range(400):
        n_before = w.pool.n
        res = w.step(r.act(w.observations(), w.action_mask()))
        if w.pool.n != n_before:
            grew = True
        assert res.obs.shape[0] == w.pool.n
        for arr in (res.rewards, res.acted, res.terminated, res.gathered, res.ate):
            assert arr.shape[0] == w.pool.n
        if res.episode_done:
            break
    assert grew, "the world never grew; this test is not testing anything"


def test_the_population_can_exceed_the_starting_array():
    """The whole point: alive-at-once was `num_agents` in every earlier world."""
    cfg = load_config("config/island4/empire.yaml").replace(**{
        "world.num_agents": 60, "world.max_ticks": 4000,
        "bushes.resample_each_episode": False})
    w, _ = run(cfg, 4000)
    assert int(w.pool.alive.sum()) > 60
    assert w.pool.n > 60


def test_capped_and_uncapped_differ_only_in_the_ceiling():
    a = load_config("config/island4/empire.yaml")
    b = load_config("config/island4/empire_capped.yaml")
    da, db = a.to_dict(), b.to_dict()
    diff = [k for k in da["world"] if da["world"][k] != db["world"][k]]
    assert diff == ["grow_slots"]


# --- conquest --------------------------------------------------------------

def test_a_siege_needs_superiority_and_time():
    cfg = tiny("config/island4/empire.yaml")
    w = World(cfg, seed=0)
    w.reset()
    qc = cfg.conquest
    # Park a foreign majority on household 0's doorstep by hand.
    victim = 0
    foreign = next(t for t in range(cfg.tribes.num_tribes)
                   if t != w.tribe_of_household[victim])
    mine = np.flatnonzero(w.household == victim)[:1]
    theirs = np.flatnonzero(w.tribe == foreign)[:4]
    for i in np.concatenate([mine, theirs]):
        w.pool.x[i] = w.stock_x[victim]
        w.pool.z[i] = w.stock_z[victim]
        w.pool.alive[i] = True
        w.pool.born[i] = True
        w.pool.age[i] = cfg.reproduction.maturity_ticks
    before = int(w.tribe_of_household[victim])
    for _ in range(qc.hold_ticks + 5):
        # Parked by hand every tick: no arbiter is driving, and `step` would
        # otherwise let them drift out of the ring on their own movement.
        for i in np.concatenate([mine, theirs]):
            w.pool.x[i] = w.stock_x[victim]
            w.pool.z[i] = w.stock_z[victim]
        w.step(np.zeros(w.pool.n, dtype=np.int64))
    assert int(w.tribe_of_household[victim]) == foreign, "the siege never resolved"
    assert before != foreign


def test_conquest_transfers_the_technology_and_labels_the_route():
    cfg = tiny("config/island4/empire.yaml")
    w = World(cfg, seed=0)
    w.reset()
    victim = 0
    foreign = next(t for t in range(cfg.tribes.num_tribes)
                   if t != w.tribe_of_household[victim])
    w.household_farming[victim] = True
    theirs = np.flatnonzero(w.tribe == foreign)[:4]
    for i in theirs:
        w.pool.alive[i] = True
        w.pool.born[i] = True
        w.pool.age[i] = cfg.reproduction.maturity_ticks
    for _ in range(cfg.conquest.hold_ticks + 5):
        for i in theirs:
            w.pool.x[i] = w.stock_x[victim]
            w.pool.z[i] = w.stock_z[victim]
        w.step(np.zeros(w.pool.n, dtype=np.int64))
    winners = np.unique(w.household[theirs])
    assert w.household_farming[winners].any(), "nobody learned anything"
    assert (w._tech_source[winners, TECH_FARMING] == TECH_CONQUERED).any()
    assert w.stats().tech_conquered[TECH_FARMING] > 0


def test_capture_tech_off_moves_the_ground_and_not_the_knowledge():
    cfg = tiny("config/island4/empire_notech.yaml")
    w = World(cfg, seed=0)
    w.reset()
    victim = 0
    foreign = next(t for t in range(cfg.tribes.num_tribes)
                   if t != w.tribe_of_household[victim])
    w.household_farming[victim] = True
    theirs = np.flatnonzero(w.tribe == foreign)[:4]
    for i in theirs:
        w.pool.alive[i] = True
        w.pool.born[i] = True
        w.pool.age[i] = cfg.reproduction.maturity_ticks
    for _ in range(cfg.conquest.hold_ticks + 5):
        for i in theirs:
            w.pool.x[i] = w.stock_x[victim]
            w.pool.z[i] = w.stock_z[victim]
        w.step(np.zeros(w.pool.n, dtype=np.int64))
    assert int(w.tribe_of_household[victim]) == foreign
    assert int(w.stats().tech_conquered[TECH_FARMING]) == 0


def test_a_tribe_wiped_off_the_map_keeps_its_column():
    """`max() + 1` was fine until a tribe could be annihilated.

    A per-tribe array that silently loses its last column reports the wiped
    tribe's zero as the NEXT tribe's total, which is a wrong answer wearing a
    right one's clothes.
    """
    cfg = tiny("config/island4/empire.yaml")
    w = World(cfg, seed=0)
    w.reset()
    w.tribe_of_household[:] = 0
    w.tribe[:] = 0
    st = w.stats()
    assert st.tribe_population.shape[0] == cfg.tribes.num_tribes
    assert st.tribe_households.shape[0] == cfg.tribes.num_tribes


def test_conquer_is_masked_for_children():
    """A siege is counted in grown heads, so a child wanting one is a doomed goal."""
    from sim.obsview import ObsView
    from sim.utility import goal_availability, compute_needs, ArbiterConfig
    cfg = tiny("config/island4/empire.yaml")
    w = World(cfg, seed=0)
    obs = w.reset()
    view = ObsView(obs, cfg)
    needs = compute_needs(view, cfg)
    avail, _, _ = goal_availability(view, cfg, needs, ArbiterConfig())
    assert not avail[~view.adult, CONQUER].any()


def test_the_goal_rung_is_frozen_so_island3_traits_do_not_move():
    """Appending a goal must not reshuffle an earlier world's trait draw.

    Rung 1 broke a golden checksum exactly this way, in a world with no tools in
    it, and `agent_traits` draws each rung in its own call so it cannot happen
    again. This is that guarantee, one rung further along.
    """
    from sim.arbiter import goal_width
    from sim.utility import agent_traits, ArbiterConfig
    acfg = ArbiterConfig()
    t3 = agent_traits(20, 7, acfg)[:, :N_GOALS_ISLAND3]
    assert t3.shape[1] == N_GOALS_ISLAND3 == N_GOALS - 1
    cfg3 = load_config("config/island3/village_frontier.yaml")
    cfg4 = load_config("config/island4/empire.yaml")
    assert goal_width(cfg3) == N_GOALS_ISLAND3
    assert goal_width(cfg4) == N_GOALS


def test_territory_need_is_zero_without_a_foreign_village():
    from sim.obsview import ObsView
    from sim.utility import compute_needs
    cfg = tiny("config/island4/empire.yaml")
    w = World(cfg, seed=0)
    w.tribe_of_household = None  # not used before reset
    obs = w.reset()
    # One tribe holding everything: nothing to take, so nothing to want.
    w.tribe_of_household[:] = 0
    w.tribe[:] = 0
    needs = compute_needs(ObsView(w.observations(), cfg), cfg)
    assert needs[:, NEED_TERRITORY].max() == 0.0


# --- the replay ------------------------------------------------------------

def test_a_frontier_replay_is_valid_json_with_no_infinity():
    """THE BUG THAT MADE A WHOLE MECHANIC UNWATCHABLE.

    Dormant sites are parked at `inf`; `json.dump` writes the literal `Infinity`
    and `JSON.parse` rejects it outright, so every frontier replay was
    unloadable in a browser and village fission had never once been seen.
    """
    cfg = tiny("config/island3/village_frontier.yaml", **{"world.max_ticks": 60})
    w, rec = run(cfg, 60, recorder=True)
    blob = rec.to_dict()
    text = json.dumps(blob)
    assert "Infinity" not in text and "NaN" not in text
    assert json.loads(text)["schema_version"] == 9
    assert all(np.isfinite([s["x"], s["z"]]).all() for s in blob["sites"])
    assert len(blob["households"]) == cfg.construction.num_sites
    assert sum(h["active"] for h in blob["households"]) == cfg.society.num_households


def test_the_replay_carries_the_history_and_the_household_stream():
    cfg = tiny("config/island4/empire.yaml", **{"world.max_ticks": 120})
    w, rec = run(cfg, 120, recorder=True)
    blob = rec.to_dict()
    assert island4_world(cfg)
    ticks = blob["ticks"]
    assert all("y" in t for t in ticks)
    assert len(ticks[0]["y"]) == 7
    assert all("yt" in t for t in ticks)
    # `hh` is SPARSE: the first tick states everything, later ticks only changes.
    assert len(ticks[0]["hh"]) == blob["world"]["num_household_slots"]
    assert sum("hh" in t for t in ticks) < len(ticks)


def test_a_grown_world_writes_an_agent_table_the_size_it_ENDED():
    cfg = load_config("config/island4/empire.yaml").replace(**{
        "world.num_agents": 50, "world.max_ticks": 1200,
        "bushes.resample_each_episode": False})
    w, rec = run(cfg, 1200, recorder=True)
    blob = rec.to_dict()
    assert w.pool.n > 50, "the world never grew"
    assert len(blob["agents"]) == w.pool.n
    assert blob["world"]["num_agents"] == w.pool.n
    # Early ticks are SHORTER than the header: those rows did not exist yet.
    assert len(blob["ticks"][0]["a"]) < len(blob["ticks"][-1]["a"])
    assert len(blob["ticks"][-1]["a"]) == w.pool.n


# --- terrain ---------------------------------------------------------------

def test_terrain_off_is_a_uniform_island():
    cfg = tiny("config/island4/empire_terrain_flat.yaml")
    w = World(cfg, seed=0)
    w.reset()
    wild = w.n_wild_bushes
    assert len(set(w.bush_cap[:wild].tolist())) == 1
    assert len(set(w.bush_regrow[:wild].tolist())) == 1
    assert np.allclose(w.fertility, 1.0)


def test_fertility_moves_capacity_and_regrow_in_opposite_directions():
    """Rich ground holds MORE and refills FASTER, or it is not rich ground."""
    cfg = tiny("config/island4/empire_terrain.yaml")
    w = World(cfg, seed=0)
    w.reset()
    per = cfg.bushes.bushes_per_cluster
    rich, poor = int(np.argmax(w.fertility)), int(np.argmin(w.fertility))
    assert w.bush_cap[rich * per] > w.bush_cap[poor * per]
    assert w.bush_regrow[rich * per] < w.bush_regrow[poor * per]
    # A field is a thing a household MADE; the ground under it must not scale it,
    # or agriculture becomes a geography mechanic too and the pair has two keys.
    assert len(set(w.bush_cap[w.n_wild_bushes:].tolist())) <= 1


def test_material_lands_on_ground_that_cannot_feed_you():
    """M5's own prescription: put the two economies in different places."""
    flat = tiny("config/island4/empire_terrain_flat.yaml")
    rough = tiny("config/island4/empire_terrain.yaml")
    a, b = World(flat, seed=0), World(rough, seed=0)
    a.reset(); b.reset()

    def fert_at(w, xs, zs):
        ccx, ccz = w._cluster_centres
        return np.mean([w.fertility[int(np.argmin((ccx - x) ** 2 + (ccz - z) ** 2))]
                        for x, z in zip(xs, zs)])

    assert fert_at(b, b.tree_x, b.tree_z) < b.fertility.mean() * 0.8
    assert fert_at(b, b.rock_x, b.rock_z) < b.fertility.mean() * 0.8
    # ORDERING ONLY: the island holds exactly as much material either way.
    assert a.tree_x.shape == b.tree_x.shape
    assert a.rock_x.shape == b.rock_x.shape


def test_terrain_does_not_change_the_observation_or_the_action_space():
    """No channel, on purpose: a rich bush already reads as a fuller bush."""
    flat = load_config("config/island4/empire_terrain_flat.yaml")
    rough = load_config("config/island4/empire_terrain.yaml")
    assert observation_layout(flat) == observation_layout(rough)


def test_terrain_and_flat_differ_in_exactly_one_section():
    a = load_config("config/island4/empire_terrain.yaml").to_dict()
    b = load_config("config/island4/empire_terrain_flat.yaml").to_dict()
    diff = [k for k in a if a[k] != b[k]]
    assert diff == ["terrain"]


# --- culture ---------------------------------------------------------------

def test_culture_weight_zero_is_pure_heredity():
    """The default must reproduce stage-2 heredity exactly, bit for bit."""
    from sim.utility import inherit_traits
    cfg = tiny("config/island4/empire.yaml",
               **{"reproduction.heritable_traits": True,
                  "reproduction.culture_weight": 0.0})
    w = World(cfg, seed=0)
    w.reset()
    traits = np.exp(np.random.default_rng(1).normal(0, 0.2, size=(w.pool.n, N_GOALS)))
    births = [(5, 0, 1)]
    a = traits.copy()
    inherit_traits(a, np.random.default_rng(3), births, cfg, w)
    b = traits.copy()
    inherit_traits(b, np.random.default_rng(3), births, cfg, None)   # no world at all
    assert np.allclose(a, b)


def test_culture_pulls_a_child_toward_its_village():
    from sim.utility import inherit_traits
    cfg = tiny("config/island4/empire.yaml",
               **{"reproduction.heritable_traits": True,
                  "reproduction.trait_mutation": 0.0,   # isolate the blend
                  "reproduction.culture_weight": 1.0})
    w = World(cfg, seed=0)
    w.reset()
    n = w.pool.n
    traits = np.ones((n, N_GOALS))
    # Membership assigned by hand: how many agents a household starts with is a
    # config detail (60 slots over 20 households is three each), and a test of
    # the BLEND should not depend on it.
    village = np.arange(6)
    w.household[village] = 0
    w.pool.alive[village] = True
    w.pool.born[village] = True
    traits[village] = 4.0            # the whole village is keen on everything
    traits[village[0]] = traits[village[1]] = 1.0   # ...except the two parents
    inherit_traits(traits, np.random.default_rng(0), [(int(village[3]), int(village[0]),
                                                       int(village[1]))], cfg, w)
    # Pure heredity would give the child its parents' 1.0. Pure culture gives it
    # the village's geometric mean, which is well above it.
    assert traits[village[3]].mean() > 1.5


# --- skills ----------------------------------------------------------------

def test_skill_off_changes_nothing():
    cfg = tiny("config/island4/empire.yaml", **{"world.max_ticks": 500})
    a, _ = run(cfg, 500)
    b, _ = run(cfg.replace(**{"skills.enabled": False}), 500)
    assert np.array_equal(a.pool.x, b.pool.x)
    assert not a.skill.any()


def test_skill_rises_only_on_a_SUCCESSFUL_harvest():
    """Swinging at an empty tree teaches nothing.

    Paying for the attempt would make standing at a stump a way to get good at
    chopping, which is the doomed-action failure the M3 mask deletes, wearing a
    learning curve's clothes.
    """
    from sim.agents import CHOP, SKILL_WOOD
    cfg = tiny("config/island4/empire.yaml", **{"skills.enabled": True})
    w = World(cfg, seed=0)
    w.reset()
    i = int(np.flatnonzero(w.pool.alive)[0])
    w.tree_wood[:] = 0                       # every tree stripped
    w.pool.x[i], w.pool.z[i] = w.tree_x[0], w.tree_z[0]
    for _ in range(20):
        a = np.zeros(w.pool.n, dtype=np.int64)
        a[i] = CHOP
        w.step(a)
    assert w.skill[i, SKILL_WOOD] == 0.0
    w.tree_wood[:] = 50
    for _ in range(20):
        a = np.zeros(w.pool.n, dtype=np.int64)
        a[i] = CHOP
        w.pool.wood[i] = w.pool.stone[i] = 0     # keep room in the pack
        w.pool.x[i], w.pool.z[i] = w.tree_x[0], w.tree_z[0]
        w.step(a)
    assert w.skill[i, SKILL_WOOD] > 0.0


def test_a_skill_bonus_can_never_overdraw_a_bush_or_a_pack():
    from sim.agents import SKILL_FOOD
    cfg = tiny("config/island4/empire.yaml", **{"skills.enabled": True})
    w = World(cfg, seed=0)
    w.reset()
    w.skill[:, SKILL_FOOD] = 1.0
    w._skill_carry[:, SKILL_FOOD] = 0.99      # every gather pays the bonus
    for _ in range(200):
        w.step(np.zeros(w.pool.n, dtype=np.int64) + 10)   # GATHER
        assert (w.bush_berries >= 0).all()
        assert (w.pool.food <= cfg.food.capacity).all()


def test_skill_is_deterministic():
    """No rng inside the harvest phase: the carry is what makes that possible."""
    cfg = tiny("config/island4/empire.yaml",
               **{"skills.enabled": True, "world.max_ticks": 900})
    a, _ = run(cfg, 900)
    b, _ = run(cfg, 900)
    assert np.allclose(a.skill, b.skill)
    assert np.array_equal(a.pool.x, b.pool.x)


def test_a_grown_row_starts_unskilled():
    cfg = tiny("config/island4/empire.yaml", **{"skills.enabled": True})
    w = World(cfg, seed=0)
    w.reset()
    w.skill[:] = 0.5
    old = w.pool.n
    w._grow_slots(9)
    assert w.skill.shape == (old + 9, w.skill.shape[1])
    assert not w.skill[old:].any()
    assert not w._skill_carry[old:].any()
