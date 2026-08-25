"""Island 2.0 stage 4: regions, households, stockpiles, reputation, shocks.

As with `test_utility.py`, the tests that earn their keep here are the ones named
after a symptom that was actually measured, because each of those looked correct
in code and only showed up as a population-level number:

  * `test_deposit_and_withdraw_cannot_both_be_available` -- 5858 deposits and
    5475 withdrawals an episode with the pile never above 1.14 of 12: the M5 gift
    farm, rebuilt out of a stockpile.
  * `test_housemates_cannot_be_robbed` -- spawning a household together packs
    five agents permanently inside steal_radius, and an opportunistic steal goal
    then fires every tick (8105 steals an episode, tripping the pre-registered
    permanent-war check).
  * `test_explore_ends_when_food_comes_into_view` -- a 25-tick commitment served
    out long after the search succeeded, 28.3% of all goal-ticks.
  * `test_shelter_prefers_own_household` -- without it, 55.7% of agents slept
    nearer a foreign home than their own and "household" stopped being a place.
  * `test_economy_accounts_for_blights` -- the blight cut supply to 0.98x
    subsistence and the sizing tool did not know (CLAUDE.md rule 5).

The first block is the invariants: enabling `society` must not disturb any 1.0
world, and the append-never-insert rule has to hold for both the action space and
the observation layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim.agents import (DEPOSIT_FOOD, DEPOSIT_MATERIAL, GATHER, IDLE, RAID, STEAL,
                        WITHDRAW_FOOD, WITHDRAW_MATERIAL, action_names,
                        num_actions, observation_dim, observation_layout)
from sim.config import load_config
from sim.economy import subsistence
from sim.obsview import ObsView
from sim.society import household_dispersion, run_episodes
from sim.utility import (DRAW_FOOD, DRAW_MATERIAL, EXPLORE, FORAGE, GOAL_RAID,
                         GOAL_TIER, N_GOALS, NEED_HUNGER, NEED_TIER, RESTORE,
                         SHELTER, STORE_FOOD, STORE_MATERIAL, ArbiterConfig,
                         compute_needs, goal_viable, score_goals, utility_runner)
from sim.world import World

ROOT = Path(__file__).resolve().parent.parent
STAGE2 = ROOT / "config" / "island2" / "society100.yaml"
STAGE4 = ROOT / "config" / "island2" / "society4.yaml"


@pytest.fixture(scope="module")
def cfg4():
    return load_config(STAGE4)


@pytest.fixture(scope="module")
def cfg2():
    return load_config(STAGE2)


def small(cfg, **over):
    """A 12-agent, 4-household version of a config, for tests that need to count."""
    base = {"world.num_agents": 12, "world.max_ticks": 120,
            "society.num_households": 4, "bushes.num_clusters": 4}
    base.update(over)
    return cfg.replace(**base)


# --- invariants -------------------------------------------------------------

def test_society_off_leaves_every_earlier_world_untouched():
    """Every 1.0 config must be bit-identical with the stage-4 code in place.

    The same guarantee `construction` and `exchange` carry, and the reason both of
    those are opt-in flags. Checked on the observation width, the action count and
    an actual episode's state, because a new default that widened any of them
    would silently invalidate every checkpoint in `checkpoints/`.
    """
    for name in ("default.yaml", "m3_masked.yaml", "m4h.yaml", "m5b.yaml"):
        cfg = load_config(ROOT / "config" / name)
        assert not cfg.society.enabled
        layout = observation_layout(cfg)
        assert observation_dim(cfg) == len(layout)
        assert not [c for c in layout
                    if c.startswith(("home.", "raid.", "shock."))
                    or c.endswith((".same_household", ".grudge"))
                    or c in ("own.stock_food", "own.stock_material")]
        assert "deposit_food" not in action_names(cfg)


def test_new_actions_are_appended_never_inserted(cfg4):
    """A checkpoint from any earlier milestone must mean the same thing here."""
    m5b = load_config(ROOT / "config" / "m5b.yaml")
    old = action_names(m5b)
    new = action_names(cfg4)
    assert new[:len(old)] == old
    assert new[DEPOSIT_FOOD] == "deposit_food"
    assert new[RAID] == "raid"
    assert num_actions(cfg4) == len(old) + 5


def test_observation_layout_matches_its_width(cfg4):
    assert observation_dim(cfg4) == len(observation_layout(cfg4))
    world = World(cfg4, seed=3)
    assert world.observations().shape[1] == observation_dim(cfg4)
    # Every column still has to land in [-1, 1], the invariant the whole
    # observation design rests on -- a stockpile count that overflowed it would
    # quietly dominate a trained first layer.
    obs = world.observations()
    assert np.isfinite(obs).all()
    assert obs.min() >= -1.0 - 1e-6 and obs.max() <= 1.0 + 1e-6


def test_more_households_than_sites_is_refused(cfg4):
    """A household's stockpile sits at its own site, so the counts must allow it."""
    with pytest.raises(ValueError, match="num_households"):
        World(cfg4.replace(**{"society.num_households": 999}), seed=0)


def test_determinism_at_a_hundred_agents(cfg4):
    """Same seed, same config, identical trajectory -- a project-wide invariant."""
    def run():
        world = World(cfg4, seed=7)
        runner = utility_runner(cfg4, seed=7)
        obs = world.observations()
        for _ in range(120):
            res = world.step(runner.act(obs, world.action_mask()))
            obs = res.obs
        return (world.pool.x.copy(), world.stock_food.copy(),
                world.grudge.copy(), list(world.raid_ledger))
    a, b = run(), run()
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x, dtype=object) if isinstance(x, list) else x,
                              np.asarray(y, dtype=object) if isinstance(y, list) else y)


# --- regions ----------------------------------------------------------------

def test_region_split_puts_wood_and_stone_on_opposite_sides(cfg4):
    """The whole point of the split: one material is a journey, so a relay pays.

    Checked as a separation of means rather than a hard boundary, because trees
    and rocks are jittered onto cluster centres and a cluster can straddle the
    axis.
    """
    world = World(cfg4, seed=11)
    assert world.tree_z.mean() > 10.0
    assert world.rock_z.mean() < -10.0
    plain = World(cfg4.replace(**{"society.region_split": False}), seed=11)
    # Without the split, `materials_at_clusters` deals both onto every cluster, so
    # neither mean is pushed to a side.
    assert abs(plain.tree_z.mean() - plain.rock_z.mean()) < abs(
        world.tree_z.mean() - world.rock_z.mean())


# --- stockpiles -------------------------------------------------------------

def test_deposit_moves_a_unit_from_pocket_to_pile(cfg4):
    cfg = small(cfg4)
    world = World(cfg, seed=5)
    world.pool.x[:] = world.stock_x[world.household]
    world.pool.z[:] = world.stock_z[world.household]
    world.pool.food[:] = 2
    actions = np.full(cfg.world.num_agents, DEPOSIT_FOOD, dtype=np.int64)
    res = world.step(actions)
    assert res.deposited.sum() == cfg.world.num_agents
    assert world.stock_food.sum() == cfg.world.num_agents
    assert (world.pool.food == 1).all()


def test_a_full_pile_refuses_a_deposit(cfg4):
    cfg = small(cfg4, **{"society.stockpile_food_capacity": 1})
    world = World(cfg, seed=5)
    world.pool.x[:] = world.stock_x[world.household]
    world.pool.z[:] = world.stock_z[world.household]
    world.pool.food[:] = 2
    world.step(np.full(cfg.world.num_agents, DEPOSIT_FOOD, dtype=np.int64))
    assert (world.stock_food <= 1).all()


def test_withdraw_needs_the_agent_to_be_at_its_own_pile(cfg4):
    cfg = small(cfg4)
    world = World(cfg, seed=5)
    world.stock_food[:] = 4
    # Standing on somebody else's home does not open your own larder.
    other = (world.household + 1) % cfg.society.num_households
    world.pool.x[:] = world.stock_x[other]
    world.pool.z[:] = world.stock_z[other]
    res = world.step(np.full(cfg.world.num_agents, WITHDRAW_FOOD, dtype=np.int64))
    assert res.withdrew.sum() == 0
    assert (world.stock_food == 4).all()


def test_raid_takes_from_a_foreign_pile_and_never_your_own(cfg4):
    cfg = small(cfg4)
    world = World(cfg, seed=5)
    world.stock_food[:] = 4
    # Put every agent on household 0's doorstep. Household 0's own members must
    # not be able to raid it; everyone else must.
    world.pool.x[:] = world.stock_x[0]
    world.pool.z[:] = world.stock_z[0]
    res = world.step(np.full(cfg.world.num_agents, RAID, dtype=np.int64))
    own = world.household == 0
    assert res.raided[own].sum() == 0
    assert res.raided[~own].sum() > 0
    assert all(victim == 0 for _r, victim, _i in res.raids)


def test_a_raid_is_remembered_by_the_whole_victim_household(cfg4):
    """Collective memory is what lets retaliation happen without scripted war."""
    cfg = small(cfg4)
    world = World(cfg, seed=5)
    world.stock_food[:] = 4
    world.pool.x[:] = world.stock_x[0]
    world.pool.z[:] = world.stock_z[0]
    actions = np.full(cfg.world.num_agents, IDLE, dtype=np.int64)
    raider = int(np.flatnonzero(world.household != 0)[0])
    actions[raider] = RAID
    world.step(actions)
    victims = np.flatnonzero(world.household == 0)
    assert (world.grudge[victims, raider] > 0).all()
    # ...and by nobody else.
    outsiders = np.flatnonzero((world.household != 0) & (np.arange(cfg.world.num_agents) != raider))
    assert (world.grudge[outsiders, raider] == 0).all()


def test_grudges_decay_so_a_feud_can_end(cfg4):
    cfg = small(cfg4)
    world = World(cfg, seed=5)
    world.grudge[1, 0] = 1.0
    before = world.grudge[1, 0]
    for _ in range(50):
        world.step(np.full(cfg.world.num_agents, IDLE, dtype=np.int64))
    assert world.grudge[1, 0] < before * 0.9


def test_housemates_cannot_be_robbed(cfg4):
    """Immunity within the household. Symptom: 8105 steals an episode.

    Spawning a household together keeps five agents permanently inside
    `steal_radius`, so an opportunistic steal fires every tick and the
    pre-registered permanent-war check trips. All three places that encode the
    rule -- the world, the mask and the arbiter's target choice -- are checked,
    because a mask that promises a steal the world refuses is the doomed action
    masking exists to delete.
    """
    cfg = small(cfg4)
    world = World(cfg, seed=5)
    housemates = np.flatnonzero(world.household == 0)
    i, j = int(housemates[0]), int(housemates[1])
    world.pool.x[:] = 300.0        # everyone else far away
    world.pool.z[:] = 300.0
    world.pool.x[[i, j]] = 0.0
    world.pool.z[[i, j]] = 0.0
    world.pool.food[:] = 0
    world.pool.food[j] = 2
    assert not world.action_mask()[i, STEAL]
    actions = np.full(cfg.world.num_agents, IDLE, dtype=np.int64)
    actions[i] = STEAL
    res = world.step(actions)
    assert res.stole[i] == 0
    assert world.pool.food[j] == 2


# --- shocks -----------------------------------------------------------------

def test_a_blight_stops_regrowth_and_does_not_repay_it_afterwards(cfg4):
    """A shock the world quietly makes up afterwards is not a shock.

    The regrowth TIMER stops too, so a blight costs the island its full duration
    of income rather than delivering it in a burst the moment it lifts.
    """
    cfg = small(cfg4, **{"society.shock_interval": 0})
    world = World(cfg, seed=5)
    world.bush_berries[:] = 0
    world.blight_until = 10_000
    for _ in range(cfg.bushes.regrow_ticks * 2):
        world.step(np.full(cfg.world.num_agents, IDLE, dtype=np.int64))
    assert world.bush_berries.sum() == 0
    assert world.bush_timer.max() == 0          # the timer never accumulated
    world.blight_until = -1
    for _ in range(cfg.bushes.regrow_ticks + 1):
        world.step(np.full(cfg.world.num_agents, IDLE, dtype=np.int64))
    assert world.bush_berries.sum() > 0         # exactly one regrowth, not two


def test_a_storm_damages_finished_shelters_and_keeps_the_cost_invariant(cfg4):
    """Damage lands on the wood counter so need_wood + need_stone still sums to
    the units outstanding -- every protection and progress calculation reads that
    sum, and a storm must not be the one place the invariant breaks."""
    cfg = small(cfg4, **{"society.shock_interval": 4, "society.blight_ticks": 0})
    world = World(cfg, seed=5)
    world.site_wood_needed[:] = 0
    world.site_stone_needed[:] = 0
    for _ in range(40):
        world.step(np.full(cfg.world.num_agents, IDLE, dtype=np.int64))
    stats = world.stats()
    assert stats.storms > 0
    assert stats.shelters_damaged > 0
    assert world.site_wood_needed.sum() > 0
    assert (world.site_stone_needed >= 0).all()


def test_shocks_do_not_perturb_the_rest_of_the_world(cfg4):
    """Shocks come off their own RNG stream, so adding them cannot move the bush
    layout or the spawn positions of an otherwise identical world."""
    a = World(cfg4.replace(**{"society.shock_interval": 0}), seed=13)
    b = World(cfg4.replace(**{"society.shock_interval": 25}), seed=13)
    assert np.array_equal(a.bush_x, b.bush_x)
    assert np.array_equal(a.pool.x, b.pool.x)


def test_economy_accounts_for_blights(cfg4, cfg2):
    """CLAUDE.md rule 5: re-derive the arithmetic when you change a mechanic.

    Left un-taught, `sim.economy` reported stage 2's 1.31x for a world whose
    blights actually put it at 0.98x -- below subsistence, where nothing
    behavioural can be read off it.
    """
    with_shocks = subsistence(cfg4)
    without = subsistence(cfg4.replace(**{"society.shock_interval": 0}))
    assert with_shocks.blight_loss > 0
    assert with_shocks.supply < without.supply
    assert subsistence(cfg2).blight_loss == 0.0
    # And the shipped stage-4 world must still sit in the doc's watchable band,
    # between sheltered and exposed subsistence.
    assert 1.1 <= with_shocks.ratio_sheltered <= 1.5
    assert with_shocks.ratio_exposed < 1.0


# --- the arbiter ------------------------------------------------------------

def _view_at_home(cfg, **state):
    """A world with everybody standing on their own stockpile, plus an ObsView."""
    world = World(cfg, seed=5)
    world.pool.x[:] = world.stock_x[world.household]
    world.pool.z[:] = world.stock_z[world.household]
    for key, value in state.items():
        target = getattr(world.pool, key, None)
        if target is not None:
            target[:] = value
        else:
            getattr(world, key)[:] = value
    return world, ObsView(world.observations(), cfg)


def test_deposit_and_withdraw_cannot_both_be_available(cfg4):
    """The M5 gift farm, rebuilt out of a stockpile. Measured: 5858 deposits and
    5475 withdrawals an episode with the pile never rising above 1.14 of 12.

    A deposit needs a real surplus and a draw needs a real shortage, so the two
    are mutually exclusive by construction rather than by weight -- rule 2's
    "fix the shape of the problem, not the size of the number".
    """
    cfg = small(cfg4)
    acfg = ArbiterConfig()
    traits = np.ones((cfg.world.num_agents, N_GOALS))
    for food, stock in ((3, 6), (0, 6), (1, 6), (3, 0)):
        world, view = _view_at_home(cfg, food=food, stock_food=stock)
        needs = compute_needs(view, cfg)
        scores = score_goals(view, cfg, needs, traits, acfg)
        both = (scores[:, STORE_FOOD] > 0) & (scores[:, DRAW_FOOD] > 0)
        assert not both.any(), f"food={food} stock={stock} offers both directions"


def test_a_full_larder_is_not_worth_walking_home_for(cfg4):
    cfg = small(cfg4)
    traits = np.ones((cfg.world.num_agents, N_GOALS))
    world, view = _view_at_home(cfg, food=3,
                                stock_food=cfg.society.stockpile_food_capacity)
    needs = compute_needs(view, cfg)
    scores = score_goals(view, cfg, needs, traits, ArbiterConfig())
    assert (scores[:, STORE_FOOD] == 0).all()


def test_raiding_is_gated_on_motive_not_on_proximity(cfg4):
    """Stage 2's correction 1, applied to a fatter target.

    Theft scored purely on distance handed itself every contest once 100 agents
    were packed together, and a stockpile is a bigger prize than a pocket. So a
    raid needs either desperation or a grudge; being next to a full pantry is not
    a reason.
    """
    cfg = small(cfg4)
    traits = np.ones((cfg.world.num_agents, N_GOALS))
    acfg = ArbiterConfig()
    world = World(cfg, seed=5)
    world.stock_food[:] = 6
    # Right next to a foreign pile, well fed, no grudge: not available.
    other = (world.household + 1) % cfg.society.num_households
    world.pool.x[:] = world.stock_x[other]
    world.pool.z[:] = world.stock_z[other]
    view = ObsView(world.observations(), cfg)
    needs = compute_needs(view, cfg)
    assert (score_goals(view, cfg, needs, traits, acfg)[:, GOAL_RAID] == 0).all()

    # Same position, starving, own larder empty: available.
    world.stock_food[:] = 0
    world.stock_food[other[0]] = 6
    world.pool.hunger[:] = cfg.hunger.max * 0.15
    view = ObsView(world.observations(), cfg)
    needs = compute_needs(view, cfg)
    scores = score_goals(view, cfg, needs, traits, acfg)
    assert scores[0, GOAL_RAID] > 0


def test_goals_that_serve_hunger_are_never_gated_above_it(cfg4):
    """Stage 2's correction 2, restated as an invariant over the whole table.

    A goal that RESTORES a tier-0 need must itself sit at tier 0, or the Maslow
    gate zeroes the only thing that could fix the emergency. That is the bug that
    put 39.6% of stage 2's intentions into `rest`.
    """
    for goal in range(N_GOALS):
        serves_tier0 = (RESTORE[goal][NEED_TIER == 0] > 0).any()
        if serves_tier0:
            assert GOAL_TIER[goal] == 0, f"{goal} serves hunger from tier {GOAL_TIER[goal]}"


def test_explore_ends_when_food_comes_into_view(cfg4):
    """Symptom: 28.3% of all goal-ticks spent exploring while the mean forage
    score among the explorers was 0.40 against explore's 0.17 -- a commitment
    served out long after its reason expired."""
    cfg = small(cfg4)
    world = World(cfg, seed=5)
    view = ObsView(world.observations(), cfg)
    goals = np.full(cfg.world.num_agents, EXPLORE, dtype=np.int64)
    visible = view.loaded_bushes.any(axis=1) & (view.food < 1.0 - 1e-6)
    ok = goal_viable(view, cfg, goals)
    assert np.array_equal(ok, ~visible)


def test_shelter_prefers_own_household(cfg4):
    """Symptom: 55.7% of agents slept nearer a foreign home than their own, so
    "my group is who sleeps where I sleep" was quietly false and every household
    statistic described a round-robin index rather than a group."""
    from sim.utility import shelter_target
    cfg = small(cfg4)
    world = World(cfg, seed=5)
    world.site_wood_needed[:] = 0
    world.site_stone_needed[:] = 0
    # Park everyone on household 0's doorstep -- but not EXACTLY on the site, which
    # ObsView's own docstring warns about: a site's presence is read off a non-zero
    # offset, so an agent standing dead centre makes it look like padding.
    world.pool.x[:] = world.stock_x[0] + 0.5
    world.pool.z[:] = world.stock_z[0] + 0.5
    view = ObsView(world.observations(), cfg)
    d, dx, dz = shelter_target(view, cfg)
    # Compared against the CLIPPED offset, because that is what an agent can see:
    # a home past `distance_scale` reads as exactly `distance_scale` away, and the
    # arbiter's job is to walk in the right direction, not to know the metric.
    scale = cfg.observation.distance_scale
    home_x = np.clip(world.stock_x[world.household] - world.pool.x, -scale, scale)
    home_z = np.clip(world.stock_z[world.household] - world.pool.z, -scale, scale)
    assert np.allclose(dx, home_x, atol=1e-2)
    assert np.allclose(dz, home_z, atol=1e-2)
    # ...and with the roof off, anything finished will do.
    world.site_wood_needed[0] = 0
    world.site_wood_needed[1:] = 3
    view = ObsView(world.observations(), cfg)
    d2, _, _ = shelter_target(view, cfg)
    outsiders = world.household != 0
    assert (d2[outsiders] < d[outsiders]).all()


# --- the population ---------------------------------------------------------

def test_household_dispersion_reports_membership_not_crowding(cfg4):
    cfg = small(cfg4)
    world = World(cfg, seed=5)
    world.pool.x[:] = world.stock_x[world.household]
    world.pool.z[:] = world.stock_z[world.household]
    own, away = household_dispersion(world)
    assert own == pytest.approx(0.0, abs=1e-6)
    assert away == 0.0
    # Swap everyone onto the wrong doorstep and it must read fully displaced.
    other = (world.household + 1) % cfg.society.num_households
    world.pool.x[:] = world.stock_x[other]
    world.pool.z[:] = world.stock_z[other]
    _own, away = household_dispersion(world)
    assert away == 1.0


def test_a_short_run_stays_alive_and_fills_its_stores(cfg4):
    """The stage-4 smoke test: the population must beat the random floor and the
    stores must actually accumulate rather than churn."""
    cfg = cfg4.replace(**{"world.max_ticks": 240})
    rep = run_episodes(cfg, 1, 10_000)
    floor = run_episodes(cfg, 1, 10_000, policy="random")
    assert np.concatenate(rep.lifespans).mean() > np.concatenate(floor.lifespans).mean()
    assert np.mean(rep.stock_trace) > 1.0
    assert np.mean(rep.deposits) > np.mean(rep.raids)
