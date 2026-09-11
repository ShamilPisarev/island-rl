"""Island 3.0 stage 3: village fission.

Named after what each one protects, in this suite's style. The pins matter most:
no config written before this stage has `num_sites > num_households` in a society
world, so the dormant-slot rule cannot fire in any of them, and these tests are
what say so out loud.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from sim.agents import TECH_FARMING, TECH_INVENTED, TECH_SETTLED, observation_layout
from sim.config import load_config
from sim.society import make_runner, run_episodes
from sim.utility import ArbiterConfig
from sim.world import World

ROOT = Path(__file__).resolve().parent.parent
ISLAND2 = ROOT / "config" / "island2"
ISLAND3 = ROOT / "config" / "island3"


@pytest.fixture
def frontier():
    """Small enough to run in a test, big enough for a party to actually leave.

    The gates are all made cheap -- no full-house requirement, a short cooldown,
    a small party -- because what these tests check is the MECHANISM, and the
    gates have their own test.
    """
    return load_config(ISLAND3 / "village_frontier.yaml").replace(**{
        "world.num_agents": 90, "world.max_ticks": 4000,
        "reproduction.initial_agents": 40, "reproduction.max_age": 2000,
        "society.num_households": 20, "construction.num_sites": 40,
        "fission.require_full_house": False, "fission.cooldown": 300})


def _stack_household_zero(world, cfg, count=8):
    """Move `count` grown agents into household 0.

    The frontier world starts 40 founders in 20 households -- two each -- and a
    party of three that must leave two behind needs five. Over a full run a
    household grows into that; in a single-step unit test it has to be arranged.
    """
    ids = np.flatnonzero(world.pool.alive)[:count]
    world.household[ids] = 0
    world.tribe[ids] = world.tribe_of_household[0]
    world.pool.age[ids] = cfg.reproduction.maturity_ticks
    world.pool.x[ids] = world.stock_x[0]
    world.pool.z[ids] = world.stock_z[0]
    return ids


def _run(cfg, seed=10000):
    world = World(cfg, seed=seed)
    runner = make_runner(cfg, "utility", seed, ArbiterConfig(), None, world=world)
    obs = world.observations()
    while True:
        res = world.step(runner.act(obs, world.action_mask()))
        obs = res.obs
        if res.episode_done:
            break
    return world


# --- the pins ----------------------------------------------------------------

def test_no_world_before_stage3_has_a_dormant_slot():
    """The whole bit-identity argument in one assertion: the dormant-site rule
    fires only when `num_sites > num_households` in a society world, and nothing
    written before this stage does that."""
    import glob
    frontier = {"village_frontier.yaml", "village_frontier_nofission.yaml"}
    for path in sorted(glob.glob(str(ROOT / "config" / "**" / "*.yaml"), recursive=True)):
        if Path(path).name in frontier:
            continue          # the two worlds this stage exists for
        if Path(path).parent.name == "island4":
            continue          # a LATER stage, built on the frontier island. The
                              # claim under test is "nothing written BEFORE
                              # stage 3 has a dormant slot", and a stage-4 world
                              # is not evidence against it.
        cfg = load_config(path)
        if not (cfg.society.enabled and cfg.construction.enabled):
            continue
        assert not cfg.fission.enabled, path
        # Dormancy needs `reserve_sites` as well as spare sites, and nothing
        # outside the two frontier configs asks for either. Both halves are
        # asserted because relying on the site count alone is what broke two
        # stage-1 checksums: a TEST shrank their household count and left their
        # site count, and the rule fired in a world that had never heard of it.
        assert not cfg.fission.reserve_sites, path


def test_a_world_without_spare_sites_has_every_household_awake():
    world = World(load_config(ISLAND2 / "society4.yaml"), seed=1)
    assert world.household_active.all()
    assert world.household_active.shape[0] == world.cfg.society.num_households
    assert np.isfinite(world.site_x).all()


def test_fission_adds_no_action_and_no_observation_channel():
    before = observation_layout(load_config(ISLAND3 / "village_gen.yaml"))
    after = observation_layout(load_config(ISLAND3 / "village_frontier.yaml"))
    assert list(before) == list(after)


# --- dormancy ----------------------------------------------------------------

def test_a_dormant_site_is_invisible_to_everything(frontier):
    """Parked at infinity, like an unplanted field, so no branch anywhere has to
    know it exists: the k-nearest treats it as absent, every radius test fails,
    and its stockpile can never be raided."""
    from sim.agents import BUILD
    world = World(frontier, seed=1)
    n0 = frontier.society.num_households
    assert world.household_active[:n0].all()
    assert not world.household_active[n0:].any()
    assert np.isinf(world.site_x[n0:]).all()
    assert np.isinf(world.stock_x[n0:]).all()
    # Put an agent exactly where a dormant site WOULD be: it can build nothing.
    i = 0
    world.pool.x[i] = world.site_layout_x[n0]
    world.pool.z[i] = world.site_layout_z[n0]
    world.pool.wood[i] = 2
    world.pool.age[i] = frontier.reproduction.maturity_ticks
    d2 = ((world.site_x - world.pool.x[i]) ** 2 + (world.site_z - world.pool.z[i]) ** 2)
    assert not np.isfinite(d2[n0:]).any()


def test_a_field_slot_exists_for_every_household_slot(frontier):
    """The bug this catches crashed the first frontier run outright: field slots
    were sized to the households that exist at tick 0, and `plant` indexes them
    by household -- so the first daughter settlement to put a field in indexed
    off the end of the bush array."""
    world = World(frontier, seed=1)
    slots = world.bush_x.shape[0] - world.n_wild_bushes
    assert slots == frontier.construction.num_sites * frontier.agriculture.max_fields_per_household


# --- the mechanism -----------------------------------------------------------

def test_a_founding_party_claims_a_site_and_becomes_a_household(frontier):
    world = _run(frontier)
    st = world.stats()
    assert st.fissions > 0
    assert st.households_final == frontier.society.num_households + st.fissions
    new = np.flatnonzero(world.household_active)[frontier.society.num_households:]
    for h in new:
        assert np.isfinite(world.site_x[h])          # the site woke up
        assert np.isfinite(world.stock_x[h])
        assert world.site_x[h] == world.site_layout_x[h]


def test_settlers_carry_what_they_know(frontier):
    """MIGRATION, the third way a technology travels. Counted separately from
    invention and teaching, so 'it spread' stays a measurement."""
    world = World(frontier, seed=1)
    world.household_farming[0] = True
    world._tech_source[0, TECH_FARMING] = TECH_INVENTED
    h = 0
    _stack_household_zero(world, frontier)
    world.stock_food[h] = frontier.fission.food_cost
    before = int(world.household_active.sum())
    world.step(np.full(world.pool.n, 8, dtype=np.int64))
    assert int(world.household_active.sum()) == before + 1
    s_new = int(np.flatnonzero(world.household_active)[-1])
    assert world.household_farming[s_new]
    assert world._tech_source[s_new, TECH_FARMING] == TECH_SETTLED
    assert world.stats().tech_settled[TECH_FARMING] == 1


def test_a_daughter_settlement_keeps_its_parents_tribe(frontier):
    """The clause the whole stage turns on: a tribe that founds settlements
    spreads across the map, which is the only way one group can come to hold more
    ground than another."""
    world = World(frontier, seed=1)
    h = 0
    _stack_household_zero(world, frontier)
    world.stock_food[h] = frontier.fission.food_cost
    world.step(np.full(world.pool.n, 8, dtype=np.int64))
    s_new = int(np.flatnonzero(world.household_active)[-1])
    assert world.tribe_of_household[s_new] == world.tribe_of_household[h]
    settlers = np.flatnonzero(world.household == s_new)
    assert settlers.size == frontier.fission.party_size
    assert (world.tribe[settlers] == world.tribe_of_household[h]).all()


def test_a_party_never_leaves_a_household_that_cannot_carry_on(frontier):
    """`party_size + 2`: a birth needs two grown members at home, so a party may
    not take so many that the parent stops being able to reproduce."""
    world = World(frontier, seed=1)
    h = 0
    ids = _stack_household_zero(world, frontier)
    world.stock_food[h] = frontier.fission.food_cost
    # Leave exactly party_size + 1 grown: one short.
    world.pool.age[:] = 0
    world.pool.age[ids[:frontier.fission.party_size + 1]] = frontier.reproduction.maturity_ticks
    before = int(world.household_active.sum())
    world.step(np.full(world.pool.n, 8, dtype=np.int64))
    assert int(world.household_active.sum()) == before


def test_a_settlement_pays_for_itself_out_of_the_parents_larder(frontier):
    world = World(frontier, seed=1)
    h = 0
    cost = frontier.fission.food_cost
    _stack_household_zero(world, frontier)
    world.stock_food[h] = cost - 1
    before = int(world.household_active.sum())
    world.step(np.full(world.pool.n, 8, dtype=np.int64))
    assert int(world.household_active.sum()) == before      # cannot afford it
    world.stock_food[h] = cost
    world.step(np.full(world.pool.n, 8, dtype=np.int64))
    assert int(world.household_active.sum()) == before + 1
    s_new = int(np.flatnonzero(world.household_active)[-1])
    assert world.stock_food[h] == 0                 # the parent paid
    assert world.stock_food[s_new] == cost          # ...and the settlers carried it


def test_a_new_settlement_must_build_before_it_can_breed(frontier):
    """Not a rule anywhere: it falls out of a birth needing a bed and a bed
    needing a finished house. Worth a test because it is the one constraint that
    stops fission being free."""
    world = World(frontier, seed=1)
    h = 0
    _stack_household_zero(world, frontier)
    world.stock_food[h] = frontier.fission.food_cost
    world.step(np.full(world.pool.n, 8, dtype=np.int64))
    s_new = int(np.flatnonzero(world.household_active)[-1])
    assert int(world.site_capacity[s_new]) == 0     # no roof, so no beds


def test_the_full_house_gate_is_what_it_says(frontier):
    """With `require_full_house` on, a household that has room to grow stays put
    -- otherwise the mechanic is 'everybody spreads out', which is a different
    thing from an outgrown house."""
    strict = frontier.replace(**{"fission.require_full_house": True,
                                 "world.max_ticks": 1500})
    world = World(strict, seed=1)
    world.stock_food[:] = strict.fission.food_cost
    world.pool.age[world.pool.born] = strict.reproduction.maturity_ticks
    world.step(np.full(world.pool.n, 8, dtype=np.int64))
    # No house is finished at tick 0, so nobody qualifies.
    assert world.stats().fissions == 0


def test_the_report_counts_households_that_exist_not_the_ones_configured(frontier):
    rep = run_episodes(frontier.replace(**{"world.max_ticks": 2500}),
                       episodes=1, seed=10000)
    assert rep.fissions and rep.fissions[0] > 0
    assert rep.households_final[0] > frontier.society.num_households
    assert rep.tech_settled[0].shape == (2,)
