"""Island 3.0 stage 2: slot reuse, heredity, technology that spreads, a granary.

Same house style: every test is named after the thing that went wrong or the
claim it protects. The three that matter most are the bit-identity pins -- a
stage must not move a number in a world that does not have it -- and the two
about the life ledger, because a reused row is the one place a per-agent
statistic can silently sum two people.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from sim.agents import (N_TECHS, TECH_FARMING, TECH_GRANARY, TECH_INVENTED,
                        TECH_TAUGHT, observation_layout)
from sim.config import load_config
from sim.economy import subsistence
from sim.society import make_runner, run_episodes
from sim.utility import N_GOALS, ArbiterConfig, OptionRunner, UtilityArbiter
from sim.world import World

ROOT = Path(__file__).resolve().parent.parent
ISLAND2 = ROOT / "config" / "island2"
ISLAND3 = ROOT / "config" / "island3"


def _shrink(cfg):
    over = {"world.num_agents": 60, "world.max_ticks": 1200,
            "society.num_households": 6, "bushes.num_clusters": 6}
    if cfg.reproduction.enabled:
        over["reproduction.initial_agents"] = 12
    return cfg.replace(**over)


def _run(cfg, seed=10000):
    world = World(cfg, seed=seed)
    runner = make_runner(cfg, "utility", seed, ArbiterConfig(), None, world=world)
    obs = world.observations()
    while True:
        res = world.step(runner.act(obs, world.action_mask()))
        obs = res.obs
        if res.episode_done:
            break
    return world, runner


@pytest.fixture
def gen():
    """A village small enough to RUN OUT OF ROWS inside a test.

    30 slots against ~27 of carrying capacity and a 400-tick lifetime, so the
    array fills and reuse actually fires -- the first version used the standard
    shrink (60 slots, 12 founders) and never reused a single row, which made
    four tests pass vacuously.
    """
    return load_config(ISLAND3 / "village_gen.yaml").replace(**{
        # The GEOMETRY is left alone -- 20 households on 20 clusters -- because
        # shrinking the clusters starves the village before anything can be
        # tested. What is shrunk is the slot budget and the lifetime, which is
        # what makes rows run out inside a test.
        "world.num_agents": 70, "world.max_ticks": 4000,
        "reproduction.initial_agents": 40, "reproduction.max_age": 1500,
        "reproduction.reuse_delay": 20})


# --- the bit-identity pins ---------------------------------------------------

# Taken from the commit BEFORE stage 2 (`git archive` into a temp tree, same
# script, same seeds), so these are real pins rather than numbers read off the
# change they guard. The mortal row overrides `max_age` down to 500 because
# village_mortal's own 4000 is longer than any pin can afford, so at its shipped
# value it is bit-identical to `village` and pins nothing about mortality.
PRE_STAGE2 = {
    "village": ("7b99469697bdab1baa3df737737eda4def5b6d66a4b099752f734d2a404cedd6", {}),
    "village_static": ("0ccaa8612fbfed83d543bd62644e1a69cdc420805a45d8ed551c20ea6e708498", {}),
    "village_nohousing": ("15d8f415e17882c8e0c7093cf5be62bb6f70bd37a46bf43e3a0e50163723a5f4", {}),
    "village_notribes": ("ddd13b799d2b76a56dcbe22477884a39080ec68874295692d44a3f59d0b8c78e", {}),
    "village_mortal": ("0727e0995bf5c2ce18837699e5d7254cc432094823d551a38bb209d0445ca177",
                       {"reproduction.max_age": 500}),
    "society4_regrow": ("4f5f0249bc8d2d3d7590d077d60097478c5ddaa38367d7f15309798fbcf58766", {}),
}


@pytest.mark.parametrize("world_name", sorted(PRE_STAGE2))
def test_every_stage1_world_is_untouched_by_stage2(world_name):
    digest, over = PRE_STAGE2[world_name]
    cfg = _shrink(load_config(ISLAND3 / f"{world_name}.yaml"))
    if over:
        cfg = cfg.replace(**over)
    world, _ = _run(cfg)
    h = hashlib.sha256()
    for arr in (world.pool.x, world.pool.z, world.pool.hunger, world.alive_ticks,
                world.pool.age, world.pool.born, world.pool.food, world.stock_food,
                world.site_extra, world.bush_berries, world.household_farming,
                world.tree_wood):
        h.update(np.asarray(arr).tobytes())
    assert h.hexdigest() == digest


def test_a_stage1_world_gains_no_channel():
    stage1 = observation_layout(load_config(ISLAND3 / "village.yaml"))
    assert "own.granary" not in stage1
    gen = observation_layout(load_config(ISLAND3 / "village_gen.yaml"))
    assert set(gen) - set(stage1) == {"own.granary"}


# --- slot reuse and the life ledger ------------------------------------------

def test_a_row_is_used_once_unless_reuse_is_on(gen):
    """The contract stage 1 relied on, and the reason R6 measured an extinction
    that was an array bound: without reuse a village cannot hold more lives than
    it has rows."""
    off = gen.replace(**{"reproduction.reuse_slots": False})
    world, _ = _run(off)
    st = world.stats()
    assert st.slots_reused == 0
    assert st.born <= off.world.num_agents


def test_reuse_lets_a_village_hold_more_lives_than_it_has_rows(gen):
    world, _ = _run(gen)
    on = gen
    st = world.stats()
    assert st.slots_reused > 0
    assert st.born > on.world.num_agents      # the array is no longer the ceiling


def test_a_reused_row_does_not_sum_two_lifespans(gen):
    """The one arithmetic error that would have poisoned every stage-2 headline.
    `alive_ticks` is per ROW; a finished life goes to the ledger and the row is
    zeroed, so the reported mean is over LIVES."""
    cfg = gen
    world, _ = _run(cfg)
    st = world.stats()
    ledger = world._life_ticks
    assert len(ledger) == st.slots_reused
    lives = int(world.pool.born.sum()) + len(ledger)
    total = float(world.alive_ticks[world.pool.born].sum() + sum(ledger))
    assert st.born == lives
    assert st.mean_lifespan == pytest.approx(total / lives)
    # ...and no single life can exceed the episode.
    assert max(ledger) <= cfg.world.max_ticks


def test_a_newborn_owes_nothing_and_is_owed_nothing(gen):
    """A reused row's grudge ROW AND COLUMN are cleared. Clearing only the row
    would leave the village remembering a robbery by somebody who no longer
    exists, and taking it out on a child."""
    cfg = gen.replace(**{"society.reputation": True})
    world = World(cfg, seed=3)
    runner = make_runner(cfg, "utility", 3, ArbiterConfig(), None, world=world)
    obs = world.observations()
    seen = 0
    while True:
        res = world.step(runner.act(obs, world.action_mask()))
        obs = res.obs
        for slot, _pa, _pb in world.last_births:
            seen += 1
            assert world.grudge[slot, :].max() == 0.0
            assert world.grudge[:, slot].max() == 0.0
        if res.episode_done:
            break
    assert seen > 0


def test_a_corpse_is_not_replaced_the_instant_it_falls(gen):
    """`reuse_delay` is not cosmetic: the replay draws a corpse folding forward
    over several ticks, and a row that flips straight back to a walking newborn
    reads as a resurrection."""
    cfg = gen.replace(**{"reproduction.reuse_delay": 80})
    world = World(cfg, seed=5)
    runner = make_runner(cfg, "utility", 5, ArbiterConfig(), None, world=world)
    obs = world.observations()
    checked = 0
    while True:
        death_ticks = world._death_tick.copy()
        res = world.step(runner.act(obs, world.action_mask()))
        obs = res.obs
        for slot, _pa, _pb in world.last_births:
            if death_ticks[slot] >= 0:
                assert world.tick - death_ticks[slot] >= 80
                checked += 1
        if res.episode_done:
            break
    assert checked > 0


# --- heredity ----------------------------------------------------------------

def test_a_child_gets_its_parents_traits(gen):
    world = World(gen, seed=1)
    arb = UtilityArbiter(gen, ArbiterConfig(), seed=1)
    arb.traits[:] = 1.0
    arb.traits[3] = 4.0
    arb.traits[4] = 9.0
    arb.on_births([(7, 3, 4)], gen)
    # Geometric mean of 4 and 9 is 6, times a lognormal mutation.
    ratio = arb.traits[7] / 6.0
    assert np.all(ratio > 0.0)
    assert 0.3 < float(np.exp(np.log(ratio).mean())) < 3.0


def test_the_blend_is_geometric_because_arithmetic_would_drift_for_free(gen):
    """Traits are lognormal about 1.0. An arithmetic mean of two lognormals is
    biased upward, so a population would climb every generation with no
    selection at all -- and it would look exactly like evolution."""
    cfg = gen.replace(**{"reproduction.trait_mutation": 0.0})
    arb = UtilityArbiter(cfg, ArbiterConfig(), seed=1)
    arb.traits[0] = 0.5
    arb.traits[1] = 2.0
    arb.on_births([(2, 0, 1)], cfg)
    assert np.allclose(arb.traits[2], 1.0)          # sqrt(0.5 * 2), not 1.25


def test_a_runner_without_a_world_refuses_to_pretend_it_inherits(gen):
    """A driver that forgot to pass the world would silently give every newborn
    its ROW's founding traits, and the whole selection read would be a null with
    no way to see why."""
    with pytest.raises(ValueError, match="heritable_traits"):
        OptionRunner(UtilityArbiter(gen, ArbiterConfig(), seed=1), gen, seed=1)
    # ...and it is fine when heredity is off.
    off = gen.replace(**{"reproduction.heritable_traits": False})
    OptionRunner(UtilityArbiter(off, ArbiterConfig(), seed=1), off, seed=1)


def test_heredity_makes_a_childs_traits_correlate_with_its_parents(gen):
    """The whole mechanism in one number: with inheritance ON the children of a
    lineage resemble it, and with it OFF they do not."""
    def correlation(cfg):
        world = World(cfg, seed=2)
        runner = make_runner(cfg, "utility", 2, ArbiterConfig(), None, world=world)
        obs = world.observations()
        kid, parent, pending = [], [], []
        while True:
            # `act` is what APPLIES the previous step's births, so a newborn's
            # traits must be read after the next act, not the instant the world
            # creates it. Reading them a tick early is how this test first
            # measured a correlation of 0.05 and looked like a broken mechanic.
            actions = runner.act(obs, world.action_mask())
            for slot, pa, pb in pending:
                mid = np.sqrt(runner.arbiter.traits[pa] * runner.arbiter.traits[pb])
                kid.append(np.log(runner.arbiter.traits[slot]))
                parent.append(np.log(mid))
            res = world.step(actions)
            obs = res.obs
            pending = list(world.last_births)
            if res.episode_done:
                break
        assert len(kid) > 10
        k = np.concatenate(kid)
        p_ = np.concatenate(parent)
        return float(np.corrcoef(k, p_)[0, 1])

    cfg = gen
    assert correlation(cfg) > 0.5
    assert abs(correlation(cfg.replace(
        **{"reproduction.heritable_traits": False}))) < 0.3


# --- technology --------------------------------------------------------------

def test_a_household_can_be_taught_a_technology_it_never_invented(gen):
    world = World(gen, seed=1)
    # One household knows; put a member of another right next to one of theirs.
    world.household_farming[0] = True
    world._tech_source[0, TECH_FARMING] = TECH_INVENTED
    knower = int(np.flatnonzero((world.household == 0) & world.pool.alive)[0])
    learner = int(np.flatnonzero((world.household == 1) & world.pool.alive)[0])
    world.pool.x[learner] = world.pool.x[knower]
    world.pool.z[learner] = world.pool.z[knower]
    idle = np.full(world.pool.n, 8, dtype=np.int64)
    for _ in range(gen.agriculture.teach_ticks + 2):
        # Held together AND held alive: idling for 300 ticks starves everybody,
        # and a dead teacher teaches nothing (which is correct, and is why the
        # first version of this test failed).
        world.pool.hunger[:] = world.cfg.hunger.max
        world.pool.x[learner] = world.pool.x[knower]
        world.pool.z[learner] = world.pool.z[knower]
        world.step(idle)
    assert world.household_farming[1]
    assert world._tech_source[1, TECH_FARMING] == TECH_TAUGHT
    st = world.stats()
    assert st.tech_invented[TECH_FARMING] == 1
    assert st.tech_taught[TECH_FARMING] == 1


def test_nobody_is_taught_when_teaching_is_off(gen):
    off = gen.replace(**{"agriculture.teach_ticks": 0})
    world, _ = _run(off)
    assert int(world.stats().tech_taught.sum()) == 0


def test_a_granary_needs_farming_first(gen):
    """The prerequisite, which is the point of the second technology: it cannot
    be reached at all until the first one has been."""
    world = World(gen, seed=1)
    world.stock_food[:] = world.stock_food_capacity
    idle = np.full(world.pool.n, 8, dtype=np.int64)
    for _ in range(gen.tech.granary_full_ticks + 2):
        world.pool.hunger[:] = world.cfg.hunger.max
        world.stock_food[:] = world.stock_food_capacity
        world.step(idle)
    assert not world.household_granary.any()       # nobody farms yet
    world.household_farming[:] = True
    for _ in range(gen.tech.granary_full_ticks + 2):
        world.pool.hunger[:] = world.cfg.hunger.max
        world.stock_food[:] = world.stock_food_capacity
        world.step(idle)
    assert world.household_granary.all()


def test_a_granary_doubles_the_larder_everywhere_it_is_read(gen):
    """Three places cap or normalise a food store -- the engine's deposit, the
    action mask, and the observation. A granary that only one of them knew about
    is the doomed action masking exists to delete."""
    from sim.agents import DEPOSIT_FOOD
    world = World(gen, seed=1)
    base = gen.society.stockpile_food_capacity
    assert int(world.stock_food_capacity[0]) == base
    world.household_granary[0] = True
    assert int(world.stock_food_capacity[0]) == base * gen.tech.granary_multiplier
    # a full-by-old-standards larder still takes a deposit
    world.stock_food[0] = base
    i = int(np.flatnonzero(world.household == 0)[0])
    world.pool.x[i], world.pool.z[i] = world.stock_x[0], world.stock_z[0]
    world.pool.food[i] = 1
    world.pool.age[i] = gen.reproduction.maturity_ticks
    assert world.action_mask()[i, DEPOSIT_FOOD]
    act = np.full(world.pool.n, 8, dtype=np.int64)
    act[i] = DEPOSIT_FOOD
    world.step(act)
    assert world.stock_food[0] == base + 1


def test_the_economy_sizes_a_store_against_a_blight_not_against_a_founding_pair():
    """Rule 5 catching a mechanic before it ran. The first version divided the
    larder by a FOUNDING household (two agents) and reported 420 ticks of cover
    against a 60-tick blight, which would have made the granary read measure
    noise. A full house is the denominator that matters."""
    e = subsistence(load_config(ISLAND3 / "village_seasons.yaml"))
    assert e.store_ticks_granary == pytest.approx(2 * e.store_ticks)
    assert e.store_ticks < e.worst_blight        # a bare store does not bridge it
    assert e.store_ticks_granary < 200           # and neither does a granary, fully


# --- it all runs together ----------------------------------------------------

def test_the_generational_village_reports_every_stage2_number(gen):
    rep = run_episodes(gen, episodes=1, seed=10000)
    assert rep.slots_reused and rep.slots_reused[0] > 0
    assert rep.trait_final and rep.trait_founding
    assert rep.trait_final[0].shape == (N_GOALS,)
    assert rep.tech_invented[0].shape == (N_TECHS,)


# --- the replay (schema v8) --------------------------------------------------

def _record(cfg, seed=10000):
    from sim.replay import ReplayRecorder
    world = World(cfg, seed=seed)
    runner = make_runner(cfg, "utility", seed, ArbiterConfig(), None, world=world)
    rec = ReplayRecorder(world, cfg, label="t", source="test", seed=seed,
                         goal_source=runner)
    rec.snapshot()
    obs = world.observations()
    while True:
        res = world.step(runner.act(obs, world.action_mask()))
        obs = res.obs
        rec.snapshot()
        if res.episode_done:
            break
    return rec.to_dict(), world


def test_a_generational_replay_says_when_a_row_changed_occupant(gen):
    """Without `u` the viewer has no way to know the agent now walking is a
    different person from the one that died there -- and it would keep drawing
    the corpse on top of them."""
    from sim.replay import SCHEMA_VERSION_GENERATIONS
    blob, world = _record(gen.replace(**{"world.max_ticks": 3000}))
    assert blob["schema_version"] == SCHEMA_VERSION_GENERATIONS
    reborn = [row for t in blob["ticks"] for row in t.get("u", [])]
    assert len(reborn) == world.stats().slots_reused > 0
    # ...and every one of them names a real household and tribe.
    for slot, house, tribe in reborn:
        assert 0 <= house < gen.society.num_households
        assert 0 <= tribe < gen.tribes.num_tribes


def test_the_first_birth_into_a_fresh_row_is_not_a_reuse(gen):
    """`u` is an event key for a row that had an occupant BEFORE. A world with
    slots to spare must emit none of them, or the viewer would reset an identity
    that never changed."""
    roomy = gen.replace(**{"world.num_agents": 200, "world.max_ticks": 1500,
                           "reproduction.max_age": 0})
    blob, world = _record(roomy)
    assert world.stats().births > 0
    assert world.stats().slots_reused == 0
    assert not any("u" in t for t in blob["ticks"])


def test_a_generational_replay_is_strict_json_and_carries_the_granaries(gen):
    import json
    blob, _ = _record(gen.replace(**{"world.max_ticks": 1500}))
    raw = json.dumps(blob)
    assert "Infinity" not in raw and "NaN" not in raw
    json.loads(raw)
    assert all("q" in t for t in blob["ticks"])
    assert len(blob["ticks"][0]["q"]) == gen.society.num_households


def test_a_stage1_world_does_not_bump_to_v8():
    from sim.replay import SCHEMA_VERSION_ISLAND3
    cfg = _shrink(load_config(ISLAND3 / "village.yaml")).replace(
        **{"world.max_ticks": 300})
    blob, _ = _record(cfg)
    assert blob["schema_version"] == SCHEMA_VERSION_ISLAND3
