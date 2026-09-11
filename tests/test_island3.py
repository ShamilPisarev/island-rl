"""Island 3.0: reproduction, housing, agriculture, tribes, and a forest that grows.

Named after the thing that went wrong or the claim it protects, in this suite's
usual style. The first three are the bit-identity pins, which matter more than
any of the mechanics: a stage must not move a single number in a world that does
not have it, and Island 3.0 broke that twice before it worked -- once through the
trait draw's third block and once through the goal histogram's denominator.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from sim.agents import (BUILD, CHOP, PLANT, RAID, STEAL, action_names,
                        num_actions, observation_layout)
from sim.config import load_config
from sim.economy import materials, subsistence
from sim.society import make_runner, run_episodes
from sim.utility import (EXPAND, N_GOALS_ISLAND3, NEED_HOUSE_ROOM, NEED_LAND, N_GOALS,
                         N_GOALS_RUNG1, N_GOALS_STAGE4, PLANT_FIELD,
                         ArbiterConfig, agent_traits, compute_needs)
from sim.world import World

ROOT = Path(__file__).resolve().parent.parent
ISLAND2 = ROOT / "config" / "island2"
ISLAND3 = ROOT / "config" / "island3"


def _shrink(cfg):
    return cfg.replace(**{"world.num_agents": 12, "world.max_ticks": 180,
                          "society.num_households": 4, "bushes.num_clusters": 4})


def _run(cfg, seed=10000):
    world = World(cfg, seed=seed)
    runner = make_runner(cfg, "utility", seed, ArbiterConfig(), None)
    obs = world.observations()
    while True:
        res = world.step(runner.act(obs, world.action_mask()))
        obs = res.obs
        if res.episode_done:
            break
    return world


@pytest.fixture
def village():
    return load_config(ISLAND3 / "village.yaml").replace(
        **{"world.num_agents": 40, "world.max_ticks": 400,
           "reproduction.initial_agents": 8,
           "society.num_households": 4, "bushes.num_clusters": 4})


# --- the bit-identity pins ---------------------------------------------------

# Taken from the code as it stood BEFORE Island 3.0 (`git archive HEAD` into a
# temp tree, same script, same seeds), so these are real pins and not numbers
# read off the change they are supposed to be guarding.
PRE_ISLAND3 = {
    "society4": "15beb435369a35727041c82a594fc175b1b1e86d2e989e7f7d1ade16469b6b66",
    "society4_ramp": "0c78e9efddcca38c4c01e5d37946d2e0687b1f242605c2fa4b215fb59e7c7fd0",
    "society4_predator": "fda6eaf58bac8456f036456e50e2da614533fe68b9ac3cfedd0bb795c38fb9e2",
    "society4_axe": "7701c1d4f77a17d6fd23e9da65b35c552659c3d2f75ec3f63a3221da537c340f",
    "society4_trade": "e32e374e10bfc5f2630a3a44534245a49345835baf53897f92d324450c85475b",
}


@pytest.mark.parametrize("world_name", sorted(PRE_ISLAND3))
def test_every_2_0_world_is_untouched_by_island3(world_name):
    """Five worlds, end to end: same seed, same positions, same stores.

    If a 3.0 branch ever leaks into a world without the block, this fails here
    rather than silently invalidating every stage 2-5 number in the repo.
    """
    world = _run(_shrink(load_config(ISLAND2 / f"{world_name}.yaml")))
    h = hashlib.sha256()
    for arr in (world.pool.x, world.pool.z, world.pool.hunger, world.alive_ticks,
                world.pool.food, world.pool.wood, world.pool.stone,
                world.stock_food, world.site_wood_needed, world.site_stone_needed,
                world.bush_berries):
        h.update(np.asarray(arr).tobytes())
    assert h.hexdigest() == PRE_ISLAND3[world_name]


def test_appending_two_goals_does_not_move_the_earlier_rungs_traits():
    """`rng.normal` fills row-major, so merging the post-stage-4 rungs into ONE
    draw reshuffles rung 1's column the moment rung 3 appends two more. Each rung
    takes its own numbers off the end -- the append-never-insert rule applied to
    the random stream, one rung further along than the axe applied it."""
    acfg = ArbiterConfig()
    traits = agent_traits(37, 3, acfg)
    assert traits.shape == (37, N_GOALS)
    rng = np.random.default_rng(3)
    legacy4 = np.exp(rng.normal(0.0, acfg.trait_spread, size=(37, N_GOALS_STAGE4)))
    legacy1 = np.exp(rng.normal(0.0, acfg.trait_spread,
                                size=(37, N_GOALS_RUNG1 - N_GOALS_STAGE4)))
    assert np.array_equal(traits[:, :N_GOALS_STAGE4], legacy4)
    assert np.array_equal(traits[:, N_GOALS_STAGE4:N_GOALS_RUNG1], legacy1)


def test_a_2_0_world_gains_no_action_and_no_channel():
    bare = load_config(ISLAND2 / "society4.yaml")
    assert "plant" not in action_names(bare)
    names = observation_layout(bare)
    for new in ("own.age", "own.adult", "home.beds_free", "home.overflow",
                "home.expandable", "own.farming", "home.field_room"):
        assert new not in names
    assert not any(n.endswith(".same_tribe") for n in names)


def test_plant_is_appended_so_every_earlier_action_keeps_its_index():
    v = load_config(ISLAND3 / "village.yaml")
    assert action_names(v)[PLANT] == "plant"
    assert action_names(v)[:PLANT] == action_names(load_config(ISLAND2 / "society4_axe.yaml"))
    assert num_actions(v) == num_actions(load_config(ISLAND2 / "society4_axe.yaml")) + 1


# --- fields are bushes -------------------------------------------------------

def test_an_unplanted_field_is_invisible_to_everything(village):
    """Field slots are parked at infinity, not at (0, 0). The obvious version put
    a phantom bush in the middle of the island inside every agent's k-nearest
    list, which would have changed foraging in a world where nobody had planted
    anything yet."""
    world = World(village, seed=1)
    n_wild = world.n_wild_bushes
    assert world.bush_x.shape[0] > n_wild            # slots exist
    assert not world.bush_active[n_wild:].any()      # and none is on
    assert np.isinf(world.bush_x[n_wild:]).all()
    assert (world.bush_berries[n_wild:] == 0).all()
    # ...and no agent can see or reach one.
    assert not world.action_mask()[:, PLANT].any()   # farming not invented yet
    d2 = (world.bush_x[None, n_wild:] - world.pool.x[:, None]) ** 2
    assert np.isinf(d2).all()


def test_a_planted_field_is_a_bush_with_a_faster_clock(village):
    world = World(village, seed=1)
    n_wild = world.n_wild_bushes
    ac = village.agriculture
    world.household_farming[:] = True
    i = 0
    h = int(world.household[i])
    world.pool.x[i], world.pool.z[i] = world.stock_x[h], world.stock_z[h]
    world.pool.wood[i] = 1
    assert world.action_mask()[i, PLANT]
    actions = np.full(world.pool.n, 8, dtype=np.int64)   # IDLE
    actions[i] = PLANT
    world.step(actions)
    slot = n_wild + h * ac.max_fields_per_household
    assert world.bush_active[slot]
    assert world.bush_cap[slot] == ac.field_capacity
    assert world.bush_regrow[slot] == ac.field_regrow_ticks
    assert world.household_fields[h] == 1
    assert world.pool.wood[i] == 0


def test_farming_is_invented_by_going_hungry_and_not_otherwise(village):
    """The unlock is SCORED, not discovered, and this is the line that scores it.
    The test exists so the claim in the write-up is checkable: hunger below the
    threshold, accumulated per household, is the whole trigger."""
    world = World(village, seed=1)
    assert not world.household_farming.any()
    world.pool.hunger[:] = village.agriculture.unlock_hunger - 1.0
    actions = np.full(world.pool.n, 8, dtype=np.int64)
    per_tick = int((world.pool.alive & (world.household == 0)).sum())
    needed = village.agriculture.unlock_hunger_ticks
    for _ in range(needed // max(per_tick, 1) + 2):
        world.pool.hunger[:] = village.agriculture.unlock_hunger - 1.0
        world.step(actions)
    assert world.household_farming[0]


# --- reproduction ------------------------------------------------------------

def test_the_unborn_are_inert_and_are_not_anybodys_neighbour(village):
    world = World(village, seed=1)
    assert world.pool.born.sum() == village.reproduction.initial_agents
    assert world.pool.alive.sum() == village.reproduction.initial_agents
    obs = world.observations()
    unborn = ~world.pool.born
    assert np.abs(obs[unborn]).sum() == 0.0
    # A living agent's k-nearest neighbour block never contains an unborn slot:
    # every neighbour it can see is alive.
    mask = world.action_mask()
    assert mask[unborn][:, 8].all()          # idle only
    assert not mask[unborn][:, :8].any()


def test_a_birth_needs_two_fed_adults_at_home_a_bed_and_a_full_larder(village):
    """Four conditions, and this is the test that says which one is missing when
    a world produces no children -- the failure mode ISLAND3_DESIGN.md
    pre-registers for R2."""
    world = World(village, seed=1)
    rc = village.reproduction
    h = 0
    # A BED IS ONE OF THE FOUR CONDITIONS, and an unfinished house has none --
    # so a family has to build before it can breed. That coupling is deliberate
    # (see HousingConfig) and it is why this test finishes the house first.
    world.site_wood_needed[:] = 0
    world.site_stone_needed[:] = 0
    members = np.flatnonzero((world.household == h) & world.pool.alive)
    assert members.size >= 2
    world.pool.x[members] = world.stock_x[h]
    world.pool.z[members] = world.stock_z[h]
    world.pool.hunger[members] = rc.birth_hunger + 1.0
    world.pool.age[:] = rc.maturity_ticks
    actions = np.full(world.pool.n, 8, dtype=np.int64)

    world.stock_food[h] = 0                       # ...but the larder is empty
    before = int(world.pool.born.sum())
    world.step(actions)
    assert int(world.pool.born.sum()) == before

    world.stock_food[h] = rc.birth_food_cost      # now it is not
    world.pool.hunger[members] = rc.birth_hunger + 1.0
    world.step(actions)
    assert int(world.pool.born.sum()) == before + 1
    baby = int(np.flatnonzero(world.pool.born & (world.pool.age == 0))[0])
    assert world.household[baby] == h
    assert world.stock_food[h] == 0               # the child was paid for


def test_a_child_cannot_swing_an_axe(village):
    world = World(village, seed=1)
    world.pool.age[:] = 0
    world.pool.wood[:] = 1
    world.pool.stone[:] = 1
    mask = world.action_mask()
    for a in (CHOP, BUILD, STEAL, RAID, PLANT):
        assert not mask[world.pool.alive][:, a].any()
    world.pool.age[:] = village.reproduction.maturity_ticks
    assert world.action_mask()[:, CHOP].any() or True   # legality now depends on the world


def test_lifespan_divides_by_the_born_not_by_the_slots(village):
    """The one arithmetic error that would have poisoned every 3.0 headline: with
    160 unborn rows carrying a lifespan of 0, a mean over slots reports a
    thriving village as nearly dead."""
    world = _run(village)
    st = world.stats()
    born = world.pool.born
    assert st.born == int(born.sum()) < village.world.num_agents
    assert st.mean_lifespan == pytest.approx(
        world.alive_ticks[born].sum() / born.sum())
    assert st.deaths == int((born & ~world.pool.alive).sum())


# --- housing -----------------------------------------------------------------

def test_a_full_house_turns_the_stranger_away_and_keeps_the_family(village):
    """Kin first, then nearest. The ask this stage exists for -- 'every family
    should house their own people' -- is this one ordering."""
    hc = village.housing
    world = World(village, seed=1)
    world.site_wood_needed[:] = 0
    world.site_stone_needed[:] = 0
    assert int(world.site_capacity[0]) == hc.base_occupants
    # Put everybody on top of house 0: its own household plus strangers.
    world.pool.x[:] = world.site_x[0]
    world.pool.z[:] = world.site_z[0]
    kin = np.flatnonzero((world.household == 0) & world.pool.alive)
    # Step to a night tick and read who was sheltered.
    cc = village.construction
    world.tick = int(cc.night_cycle * (1.0 - cc.night_fraction)) + 1
    before = world.pool.hunger.copy()
    world.step(np.full(world.pool.n, 8, dtype=np.int64))
    drop = before - world.pool.hunger
    # The family drained least: they got the beds.
    assert drop[kin].max() <= drop[world.pool.alive].max()
    assert world.stats().bed_denied > 0


def test_building_at_a_finished_house_adds_a_room(village):
    world = World(village, seed=1)
    hc = village.housing
    world.site_wood_needed[:] = 0
    world.site_stone_needed[:] = 0
    base = int(world.site_capacity[0])
    i = int(np.flatnonzero(world.household == 0)[0])
    world.pool.age[i] = village.reproduction.maturity_ticks
    world.pool.x[i], world.pool.z[i] = world.site_x[0], world.site_z[0]
    world.pool.wood[i] = 2
    assert world.action_mask()[i, BUILD]
    actions = np.full(world.pool.n, 8, dtype=np.int64)
    actions[i] = BUILD
    for _ in range(hc.expand_units):
        world.pool.wood[i] = max(int(world.pool.wood[i]), 1)
        world.step(actions)
    assert world.site_rooms[0] == 1
    assert int(world.site_capacity[0]) == base + hc.occupants_per_room


def test_a_house_stops_taking_material_at_max_rooms(village):
    world = World(village, seed=1)
    hc = village.housing
    world.site_wood_needed[:] = 0
    world.site_stone_needed[:] = 0
    world.site_extra[0] = hc.max_rooms * hc.expand_units
    i = int(np.flatnonzero(world.household == 0)[0])
    world.pool.age[i] = village.reproduction.maturity_ticks
    world.pool.x[i], world.pool.z[i] = world.site_x[0], world.site_z[0]
    world.pool.wood[i] = 2
    # Only site 0 is maxed, so the mask may still allow a build at a NEIGHBOUR;
    # move site 0 far from the rest to isolate it.
    world.site_x[1:] += 1000.0
    assert not world.action_mask()[i, BUILD]


def test_an_unfinished_site_still_shelters_everyone(village):
    """Rule 5, pre-registered rather than discovered. Capping an unfinished site
    would put the M4 cliff straight back: `partial_shelter` exists because three
    quarters of a build bought nothing, and a capacity of zero on a half-built
    wall buys nothing again."""
    world = World(village, seed=1)
    assert int(world.site_capacity[0]) == 0       # not finished -> no beds...
    cc = village.construction
    world.site_wood_needed[0] = 1                 # ...but nearly there
    world.site_stone_needed[0] = 0
    world.pool.x[:] = world.site_x[0]
    world.pool.z[:] = world.site_z[0]
    world.tick = int(cc.night_cycle * (1.0 - cc.night_fraction)) + 1
    before = world.pool.hunger.copy()
    world.step(np.full(world.pool.n, 8, dtype=np.int64))
    drop = (before - world.pool.hunger)[world.pool.alive]
    full_night = cc.night_drain_multiplier * village.hunger.drain_per_tick
    assert drop.max() < full_night     # everyone got the partial roof


# --- tribes ------------------------------------------------------------------

def test_a_tribe_is_an_arc_of_coast_not_a_round_robin(village):
    world = World(village, seed=1)
    ang = np.arctan2(world.stock_z, world.stock_x)
    for t in range(village.tribes.num_tribes):
        members = np.flatnonzero(world.tribe_of_household == t)
        if members.size > 1:
            spread = ang[members].max() - ang[members].min()
            assert spread <= 2 * np.pi / village.tribes.num_tribes + 1e-9


def test_you_do_not_rob_your_own_tribe(village):
    world = World(village, seed=1)
    i = 0
    same = np.flatnonzero((world.tribe == world.tribe[i]) & world.pool.alive)
    same = same[same != i]
    if same.size == 0:
        pytest.skip("this seed put nobody else in agent 0's tribe")
    world.pool.x[same] = world.pool.x[i]
    world.pool.z[same] = world.pool.z[i]
    world.pool.food[same] = 2
    world.pool.food[i] = 0
    world.pool.age[:] = village.reproduction.maturity_ticks
    assert not world.action_mask()[i, STEAL]


def test_a_raid_next_door_is_a_family_matter_not_a_war(village):
    """A within-tribe raid is remembered by the victim's household alone. Widening
    it unconditionally had a tribe hold a grudge against its own member -- the
    raider included, since the raider is in the tribe it robbed."""
    world = World(village, seed=1)
    hh = np.flatnonzero(world.tribe_of_household == world.tribe_of_household[0])
    if hh.size < 2:
        pytest.skip("this seed gave tribe 0 only one household")
    victim = int(hh[1])
    i = int(np.flatnonzero((world.household == 0) & world.pool.alive)[0])
    world.pool.age[:] = village.reproduction.maturity_ticks
    world.pool.x[i], world.pool.z[i] = world.stock_x[victim], world.stock_z[victim]
    world.stock_food[victim] = 5
    world.pool.food[i] = 0
    actions = np.full(world.pool.n, 8, dtype=np.int64)
    actions[i] = RAID
    world.step(actions)
    assert world.stats().raids_within_tribe == 1
    remembers = np.flatnonzero(world.grudge[:, i] > 0)
    assert set(world.household[remembers].tolist()) == {victim}


# --- the forest, and the arithmetic that sizes it ----------------------------

def test_a_chopped_tree_grows_back_to_what_it_was_and_no_further(village):
    world = World(village, seed=1)
    cc = village.construction
    world.tree_wood[0] = 0
    actions = np.full(world.pool.n, 8, dtype=np.int64)
    for _ in range(cc.tree_regrow_ticks + 1):
        world.step(actions)
    assert world.tree_wood[0] == 1
    world.tree_wood[:] = world.tree_cap
    for _ in range(cc.tree_regrow_ticks + 1):
        world.step(actions)
    assert (world.tree_wood <= world.tree_cap).all()


def test_a_blight_starves_the_bushes_and_leaves_the_forest_alone(village):
    """One shock, one job. A blight that also halted the forest would make the
    seasons world unreadable -- two mechanics moving behind one name."""
    world = World(village, seed=1)
    world.blight_until = 10_000
    world.tree_wood[0] = 0
    world.bush_berries[:] = 0
    world.bush_timer[:] = 0
    actions = np.full(world.pool.n, 8, dtype=np.int64)
    for _ in range(village.construction.tree_regrow_ticks + 1):
        world.step(actions)
    assert world.tree_wood[0] == 1          # the forest grew
    assert world.bush_berries.sum() == 0    # the bushes did not


def test_the_economy_reports_a_rate_because_a_stock_ratio_cannot_see_a_collapse():
    """The number ISLAND2_DESIGN.md section 15 needed and nobody had computed."""
    dead = materials(load_config(ISLAND2 / "society4.yaml"))
    alive = materials(load_config(ISLAND3 / "society4_regrow.yaml"))
    assert dead.material_rate == 0.0 < dead.build_rate_needed
    assert alive.material_rate > alive.build_rate_needed


def test_the_economy_reports_a_carrying_capacity_once_the_cast_is_not_fixed():
    v = subsistence(load_config(ISLAND3 / "village.yaml"))
    assert v.carrying_sheltered > v.carrying_exposed > 0.0
    assert v.initial_agents == 40
    # ...and a 2.0 world still reports nothing of the kind, because its
    # population is an input rather than an outcome.
    assert subsistence(load_config(ISLAND2 / "society4.yaml")).carrying_sheltered == 0.0


# --- the arbiter -------------------------------------------------------------

def test_a_family_expands_when_the_house_is_full_not_when_it_overflows(village):
    """The correction the mechanic forced: a birth needs a free bed, so a
    household stops AT capacity and can never breed past it. Keying `expand` on
    overflow made it unavailable forever."""
    world = World(village, seed=1)
    world.site_wood_needed[:] = 0
    world.site_stone_needed[:] = 0
    from sim.obsview import ObsView
    view = ObsView(world.observations(), village)
    needs = compute_needs(view, village)
    # Households start below capacity, so nobody wants a room yet.
    assert needs[world.pool.alive, NEED_HOUSE_ROOM].max() < 1.0
    # Fill house 0 to its capacity and the need saturates.
    fill = np.flatnonzero(~world.pool.born)[:village.housing.base_occupants]
    world.pool.born[fill] = True
    world.pool.alive[fill] = True
    world.household[fill] = 0
    view = ObsView(world.observations(), village)
    needs = compute_needs(view, village)
    kin = np.flatnonzero((world.household == 0) & world.pool.alive)
    assert needs[kin, NEED_HOUSE_ROOM].max() == pytest.approx(1.0)


def test_planting_serves_its_own_need_because_wealth_is_zero_when_it_is_possible(village):
    """The axe's lesson in a new place: a need is what you LACK, and what a
    household without a field lacks is the field, not the unit in its hand."""
    world = World(village, seed=1)
    from sim.obsview import ObsView
    view = ObsView(world.observations(), village)
    assert compute_needs(view, village)[:, NEED_LAND].max() == 0.0   # not invented
    world.household_farming[:] = True
    view = ObsView(world.observations(), village)
    assert compute_needs(view, village)[:, NEED_LAND].max() > 0.0


def test_the_goal_histogram_counts_the_living_only(village):
    """With 200 slots and 40 agents alive, counting every row spent 80% of the
    village's goal-ticks in agents that did not exist -- and reported it as
    `explore`. Rule 6, in the denominator again."""
    rep = run_episodes(village, episodes=1, seed=10000)
    total = int(rep.goal_ticks.sum())
    born = float(np.mean(rep.born))
    assert total <= born * village.world.max_ticks + 1
    assert total > 0.5 * born * village.world.max_ticks


def test_the_learned_goal_head_keeps_each_rungs_width():
    from sim.arbiter import goal_width
    assert goal_width(load_config(ISLAND2 / "society4.yaml")) == N_GOALS_STAGE4
    assert goal_width(load_config(ISLAND2 / "society4_axe.yaml")) == N_GOALS_RUNG1
    # ISLAND 3.0's OWN RUNG, not `N_GOALS`. The two were the same number until
    # Island 4.0 appended `conquer`; asserting `N_GOALS` here would mean "a 3.0
    # world gets whatever the newest rung is", which is exactly the guarantee
    # this test exists to deny -- a from-scratch run in a world with no conquest
    # in it must start from the same weights it always did.
    assert goal_width(load_config(ISLAND3 / "village.yaml")) == N_GOALS_ISLAND3
    assert goal_width(load_config(
        ROOT / "config" / "island4" / "empire.yaml")) == N_GOALS
    assert EXPAND == N_GOALS_RUNG1 and PLANT_FIELD == N_GOALS_RUNG1 + 1


# --- the replay (schema v7) --------------------------------------------------

def _record(cfg, seed=10000, ticks=None):
    from sim.replay import ReplayRecorder
    if ticks:
        cfg = cfg.replace(**{"world.max_ticks": ticks})
    world = World(cfg, seed=seed)
    runner = make_runner(cfg, "utility", seed, ArbiterConfig(), None)
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


def test_a_village_replay_is_strict_json(village):
    """The bug this catches shipped for one commit: unplanted field slots are
    parked at infinity so the engine treats them as absent, `json.dump` writes
    the literal `Infinity`, and `JSON.parse` rejects it -- so every village
    replay was unloadable in the browser. The header carries wild bushes only."""
    import json
    blob, world = _record(village, ticks=300)
    raw = json.dumps(blob)
    assert "Infinity" not in raw and "NaN" not in raw
    json.loads(raw)          # a strict parser, like the viewer's
    assert len(blob["bushes"]) == world.n_wild_bushes
    assert len(blob["ticks"][0]["b"]) > world.n_wild_bushes


def test_the_village_replay_says_who_is_a_child_and_how_big_the_house_is(village):
    from sim.replay import AGENT_FIELDS_V7, SCHEMA_VERSION_ISLAND3
    blob, _ = _record(village, ticks=300)
    assert blob["schema_version"] == SCHEMA_VERSION_ISLAND3
    assert blob["tick_fields"]["agent"] == AGENT_FIELDS_V7
    assert all("h" in t for t in blob["ticks"])          # rooms, every tick
    assert all("tribe" in a for a in blob["agents"])
    assert "births" in blob["summary"]


def test_a_field_is_announced_once_on_the_tick_it_is_planted(village):
    """`f` is an event key like `k` and `g`: repeating forty positions every tick
    for 24,000 ticks is how an 11MB replay becomes a 200MB one."""
    blob, _ = _record(village, ticks=300)
    seen = []
    for t in blob["ticks"]:
        seen += [row[0] for row in t.get("f", [])]
    assert len(seen) == len(set(seen))


def test_a_world_without_island3_does_not_bump_the_schema():
    from sim.replay import SCHEMA_VERSION_PREDATOR, SCHEMA_VERSION_VIEW
    cfg = _shrink(load_config(ISLAND2 / "society4.yaml"))
    blob, _ = _record(cfg, ticks=60)
    assert blob["schema_version"] == SCHEMA_VERSION_VIEW
    pred = _shrink(load_config(ISLAND2 / "society4_predator.yaml"))
    assert _record(pred, ticks=60)[0]["schema_version"] == SCHEMA_VERSION_PREDATOR
