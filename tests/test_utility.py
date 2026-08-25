"""Island 2.0 stage 2: the observation view, the needs arbiter, and options.

The tests that matter most here are the ones pinning the three modelling errors
found while building this, because every one of them looked correct in code and
only showed up as a population-level symptom:

  * a Maslow gate that suppressed `explore`, the goal that SERVES an unmet need
  * a safety need scaled DOWN by being near a shelter
  * a safety need that zeroed on arrival, emptying the shelter mid-night

Each has a test named after the symptom, so a future tidy-up that reintroduces
one fails here instead of six hundred ticks later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim.agents import GATHER, IDLE, action_mask, num_actions
from sim.config import load_config
from sim.obsview import ObsView
from sim.society import expected_nearest_neighbour, gini, run_episodes
from sim.utility import (EXPLORE, FORAGE, GOAL_GIVE_FOOD, GOAL_NAMES, GOAL_STEAL,
                         GOAL_TIER, N_GOALS, NEED_HUNGER, NEED_SAFETY, REST,
                         RESTORE, SHELTER, ArbiterConfig, OptionRunner,
                         UtilityArbiter, compute_needs, execute_goals,
                         goal_viable, score_goals, utility_runner)
from sim.world import World

CONFIG = Path(__file__).resolve().parent.parent / "config" / "island2" / "society100.yaml"


@pytest.fixture(scope="module")
def cfg2():
    return load_config(CONFIG)


@pytest.fixture
def world(cfg2):
    return World(cfg2, seed=4242)


@pytest.fixture
def view(world, cfg2):
    return ObsView(world.observations(), cfg2)


# --- ObsView ----------------------------------------------------------------

def test_obsview_distances_are_world_units(world, cfg2, view):
    """A bush 3 units away must read as 3, not as 3/distance_scale."""
    pool = world.pool
    d2 = ((world.bush_x[None, :] - pool.x[:, None]) ** 2
          + (world.bush_z[None, :] - pool.z[:, None]) ** 2)
    truth = np.sqrt(d2.min(axis=1))
    observed = view.bushes.distance.min(axis=1)
    # Only compare where the true nearest is inside the clip, since beyond
    # distance_scale the observation genuinely cannot say how far it is.
    inside = truth < cfg2.observation.distance_scale * 0.9
    assert inside.any()
    np.testing.assert_allclose(observed[inside], truth[inside], atol=0.05)


def test_obsview_padding_is_not_mistaken_for_an_entity(cfg2):
    """A zero-padded slot reads as "nothing, at zero offset" and must be absent.

    This is the trap the scripted trader documents: a padded neighbour looks like
    a starving empty-handed agent standing on top of you, i.e. the perfect
    recipient.
    """
    cfg = cfg2.replace(**{"world.num_agents": 2})
    w = World(cfg, seed=1)
    v = ObsView(w.observations(), cfg)
    # 2 agents, k_agents slots: every slot past the first is padding.
    assert v.neighbours.present[:, 0].all()
    assert not v.neighbours.present[:, 1:].any()
    assert np.isinf(v.neighbours.distance[:, 1:]).all()


def test_obsview_nearest_returns_zero_offsets_when_nothing_qualifies(view):
    none = np.zeros_like(view.bushes.present)
    d, dx, dz, _ = view.bushes.nearest(none)
    assert np.isinf(d).all()
    assert (dx == 0).all() and (dz == 0).all()


# --- the three modelling corrections ---------------------------------------

def test_explore_is_never_gated_by_the_needs_it_serves(cfg2, view):
    """The bug: an empty-handed agent's tier-2 urgency zeroed every higher tier,
    including the search that was the only way to fix the shortage. Measured, it
    put 39.6% of all intentions into `rest`."""
    assert GOAL_TIER[EXPLORE] == 0, "explore must be ungated: searching is not a luxury"
    assert RESTORE[EXPLORE].sum() > 0, "explore must serve a real need to scale with it"
    # An agent with everything unsatisfied must still score explore above zero.
    needs = np.ones((view.n, RESTORE.shape[1]))
    traits = np.ones((view.n, N_GOALS))
    scores = score_goals(view, cfg2, needs, traits, ArbiterConfig())
    assert (scores[:, EXPLORE] > 0).all()


def test_safety_need_does_not_shrink_when_shelter_is_near(cfg2):
    """The bug: deficit scaled by distance-to-shelter made an agent 9.7 units
    from cover at night only 19% unsafe. Distance is the cost of fixing a need,
    never its size."""
    cfg = cfg2
    w = World(cfg, seed=7)
    # Finish every site, then place one agent on top of a shelter and one far off.
    w.site_wood_needed[:] = 0
    w.site_stone_needed[:] = 0
    w.tick = int(cfg.construction.night_cycle * 0.9)   # deep in the night
    w.pool.x[0], w.pool.z[0] = w.site_x[0], w.site_z[0]
    w.pool.x[1], w.pool.z[1] = 0.0, 0.0
    v = ObsView(w.observations(), cfg)
    needs = compute_needs(v, cfg)
    near, far = needs[0, NEED_SAFETY], needs[1, NEED_SAFETY]
    assert near == pytest.approx(far), (
        f"safety deficit varied with distance to shelter ({near:.3f} vs {far:.3f})")
    assert near > 0.9, "night exposure must read as near-total, whatever the distance"


def test_safety_need_stays_up_while_sheltering(cfg2):
    """The bug: zeroing the need on arrival emptied the shelter mid-night --
    satisfied need, shelter scores 0, agent wanders back out. Being under cover
    is how the need goes on being met, not a state that discharges it."""
    cfg = cfg2
    w = World(cfg, seed=7)
    w.site_wood_needed[:] = 0
    w.site_stone_needed[:] = 0
    w.tick = int(cfg.construction.night_cycle * 0.9)
    w.pool.x[:] = w.site_x[0]
    w.pool.z[:] = w.site_z[0]
    v = ObsView(w.observations(), cfg)
    needs = compute_needs(v, cfg)
    assert needs[:, NEED_SAFETY].min() > 0.9
    traits = np.ones((v.n, N_GOALS))
    scores = score_goals(v, cfg, needs, traits, ArbiterConfig())
    # Sitting inside, shelter must still be the winning intention.
    assert (scores.argmax(axis=1) == SHELTER).mean() > 0.9


def test_safety_is_zero_in_broad_daylight(cfg2):
    cfg = cfg2
    w = World(cfg, seed=7)
    w.site_wood_needed[:] = 0
    w.site_stone_needed[:] = 0
    w.tick = 0                      # dawn
    v = ObsView(w.observations(), cfg)
    needs = compute_needs(v, cfg)
    assert needs[:, NEED_SAFETY].max() == pytest.approx(0.0)


def test_dusk_lead_is_long_enough_to_walk_home(cfg2):
    """The lead time is derived from move_step, not picked. A hardcoded 0.2 had
    agents setting off after dark on the radius-100 island."""
    cfg = cfg2
    cross_ticks = cfg.observation.distance_scale / cfg.world.move_step
    lead_cycles = cross_ticks / cfg.construction.night_cycle
    w = World(cfg, seed=7)
    w.site_wood_needed[:] = 0
    w.site_stone_needed[:] = 0
    dusk = 1.0 - cfg.construction.night_fraction
    # Just inside the derived lead: safety must already be non-zero.
    w.tick = int(cfg.construction.night_cycle * (dusk - lead_cycles * 0.5))
    v = ObsView(w.observations(), cfg)
    assert compute_needs(v, cfg)[:, NEED_SAFETY].max() > 0.0
    # Well before it: still zero, so the population is not sheltering all day.
    w.tick = int(cfg.construction.night_cycle * max(dusk - lead_cycles * 1.5, 0.0))
    v = ObsView(w.observations(), cfg)
    assert compute_needs(v, cfg)[:, NEED_SAFETY].max() == pytest.approx(0.0)


# --- theft stays opportunistic ---------------------------------------------

def test_steal_is_only_available_within_reach(cfg2):
    """Pursued as a travelling goal, theft produced the pre-registered PERMANENT
    WAR: 6932 steals an episode, foraging down to 10.9% of intentions, and the
    population harvesting 685 berries against a demand of 857."""
    cfg = cfg2
    w = World(cfg, seed=11)
    # One loaded victim, and a would-be thief far outside steal_radius.
    w.pool.food[:] = 0
    w.pool.food[1] = 2
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1], w.pool.z[1] = cfg.competition.steal_radius * 6, 0.0
    v = ObsView(w.observations(), cfg)
    needs = compute_needs(v, cfg)
    traits = np.ones((v.n, N_GOALS))
    scores = score_goals(v, cfg, needs, traits, ArbiterConfig())
    assert scores[0, GOAL_STEAL] == 0.0, "theft must not be a destination"
    # Now stand next to them: it becomes available.
    w.pool.x[1] = cfg.competition.steal_radius * 0.5
    v = ObsView(w.observations(), cfg)
    scores = score_goals(v, cfg, needs, traits, ArbiterConfig())
    assert scores[0, GOAL_STEAL] > 0.0


def test_a_steal_goal_terminates_when_the_victim_walks_away(cfg2):
    cfg = cfg2
    w = World(cfg, seed=11)
    w.pool.food[:] = 0
    w.pool.food[1] = 2
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1], w.pool.z[1] = cfg.competition.steal_radius * 8, 0.0
    v = ObsView(w.observations(), cfg)
    goals = np.full(v.n, GOAL_STEAL, dtype=np.int64)
    assert not goal_viable(v, cfg, goals)[0]


# --- executors --------------------------------------------------------------

def test_execute_never_emits_an_illegal_action(cfg2, world):
    """Whatever the arbiter intends, the primitive handed to the world must be
    one the mask allows -- otherwise the tick is spent on something already
    known to be doomed, which is the failure action masking exists to remove."""
    cfg = cfg2
    runner = utility_runner(cfg, seed=3)
    obs = world.observations()
    for _ in range(120):
        mask = world.action_mask()
        actions = runner.act(obs, mask)
        legal = np.take_along_axis(mask, actions[:, None], axis=1).ravel()
        assert legal.all(), "arbiter emitted an action the mask forbids"
        obs = world.step(actions).obs


def test_forage_walks_when_far_and_gathers_when_close(cfg2):
    cfg = cfg2
    w = World(cfg, seed=5)
    v = ObsView(w.observations(), cfg)
    mask = w.action_mask()
    goals = np.full(v.n, FORAGE, dtype=np.int64)
    heading = np.zeros(v.n, dtype=np.int64)
    actions = execute_goals(goals, v, cfg, mask, heading)
    d_bush, _, _, _ = v.bushes.nearest(v.loaded_bushes)
    close = mask[:, GATHER]
    assert (actions[close] == GATHER).all()
    walking = np.isfinite(d_bush) & ~close
    assert (actions[walking] < 8).all(), "a visible but distant bush must be walked at"


def test_dead_agents_are_left_idle(cfg2):
    cfg = cfg2
    w = World(cfg, seed=5)
    w.pool.alive[:10] = False
    runner = utility_runner(cfg, seed=1)
    actions = runner.act(w.observations(), w.action_mask())
    assert (actions[:10] == IDLE).all()


# --- options ----------------------------------------------------------------

def test_options_commit_rather_than_re_deciding_every_tick(cfg2, world):
    """The whole point of the goal level: a 20-tick crossing is ONE decision."""
    cfg = cfg2
    runner = utility_runner(cfg, seed=8, acfg=ArbiterConfig(commit_ticks=25))
    obs = world.observations()
    ticks = 200
    for _ in range(ticks):
        obs = world.step(runner.act(obs, world.action_mask())).obs
    per_agent = runner.decisions.mean()
    assert per_agent < ticks / 3, (
        f"{per_agent:.1f} decisions in {ticks} ticks -- commitment is not holding")
    assert per_agent > 1, "options never re-decided at all"


def test_a_hunger_emergency_interrupts_a_running_option(cfg2):
    cfg = cfg2
    w = World(cfg, seed=9)
    acfg = ArbiterConfig(commit_ticks=999)
    runner = OptionRunner(UtilityArbiter(cfg, acfg, seed=0), cfg, seed=0)
    runner.goals[:] = REST
    runner.ticks_left[:] = 999
    # Starve everyone past the critical threshold.
    w.pool.hunger[:] = cfg.hunger.max * 0.05
    runner.act(w.observations(), w.action_mask())
    assert runner.last_decided.all(), "a starving agent kept resting"
    assert (runner.goals != REST).any()


def test_explore_holds_one_heading_for_the_whole_option(cfg2):
    """Ballistic travel, not a fresh random step per tick -- the one thing
    `nav-commit` showed is worth real ticks (+50 zero-shot) with no direction
    learned."""
    cfg = cfg2
    w = World(cfg, seed=2)
    runner = utility_runner(cfg, seed=2, acfg=ArbiterConfig(commit_ticks=30))
    obs = w.observations()
    headings = []
    for _ in range(10):
        runner.act(obs, w.action_mask())
        headings.append(runner.explore_heading.copy())
        obs = w.step(np.full(cfg.world.num_agents, IDLE)).obs
    assert all(np.array_equal(headings[0], h) for h in headings[1:])


def test_runner_is_deterministic(cfg2):
    cfg = cfg2

    def rollout():
        w = World(cfg, seed=77)
        r = utility_runner(cfg, seed=77)
        obs = w.observations()
        out = []
        for _ in range(60):
            a = r.act(obs, w.action_mask())
            out.append(a.copy())
            obs = w.step(a).obs
        return np.stack(out)

    np.testing.assert_array_equal(rollout(), rollout())


def test_traits_differentiate_agents(cfg2):
    cfg = cfg2
    arb = UtilityArbiter(cfg, ArbiterConfig(), seed=5)
    assert arb.traits.shape == (cfg.world.num_agents, N_GOALS)
    # Every agent has its own preference vector, and none is degenerate.
    assert arb.traits.std(axis=0).min() > 0.0
    assert (arb.traits > 0).all()


# --- the population-level result -------------------------------------------

def test_utility_population_beats_the_random_floor(cfg2):
    """Stage 2's exit condition, as a test rather than a claim in a note.

    One episode each, so this stays a cheap regression rather than the real
    measurement (`python -m sim.society --episodes 5 --random`).
    """
    cfg = cfg2
    good = run_episodes(cfg, 1, 10000)
    floor = run_episodes(cfg, 1, 10000, policy="random")
    a = np.concatenate(good.lifespans).mean()
    b = np.concatenate(floor.lifespans).mean()
    assert a > 2.0 * b, f"utility {a:.0f} vs random {b:.0f} -- barely above chance"
    assert good.shelters[0] > 0, "nobody built anything"


def test_failure_mode_two_stays_clear(cfg2):
    """Theft must not crowd out foraging. The pre-registered collapse."""
    cfg = cfg2
    rep = run_episodes(cfg, 1, 10000)
    assert rep.goal_ticks[GOAL_STEAL] < rep.goal_ticks[FORAGE], "permanent war"


def test_gini_and_crowding_controls():
    assert gini(np.array([5.0, 5.0, 5.0])) == pytest.approx(0.0)
    assert gini(np.array([0.0, 0.0, 9.0])) > 0.6
    # Denser islands put neighbours closer, which is the control for failure 3.
    assert (expected_nearest_neighbour(40.0, 100)
            < expected_nearest_neighbour(100.0, 100))


def test_goal_names_and_tables_agree():
    assert len(GOAL_NAMES) == N_GOALS
    assert RESTORE.shape[0] == N_GOALS
    assert GOAL_TIER.shape[0] == N_GOALS


# --- persist-until-goal options (stage 5, lever 3) ---------------------------
#
# The flag turns a 25-tick budget into "run until the goal state", so a whole
# build programme is one semi-MDP decision. Two things have to be pinned hard:
# that it changes NOTHING when off (every result in this project was measured
# under commit_ticks), and that the pieces the design doc warns about -- the
# tier-0 interruption and the goal-state termination -- still hold when it is on.

PERSIST_OFF_CHECKSUM = "60d41e718002d8f854175d65cb454fad09951e54b83a90991d87be550d587497"


def _persist_cfg(cfg2):
    return cfg2.replace(**{"world.num_agents": 12, "world.max_ticks": 240,
                           "society.num_households": 4, "bushes.num_clusters": 4})


def _trajectory_checksum(cfg, acfg=None, ticks=240):
    import hashlib
    w = World(cfg, seed=5)
    runner = utility_runner(cfg, seed=5, acfg=acfg)
    obs = w.observations()
    h = hashlib.sha256()
    for _ in range(ticks):
        actions = runner.act(obs, w.action_mask())
        res = w.step(actions)
        obs = res.obs
        h.update(actions.tobytes())
        h.update(w.pool.x.tobytes())
        h.update(w.pool.hunger.tobytes())
        if res.episode_done:
            break
    return h.hexdigest()


def test_persist_off_is_bit_identical(cfg2):
    """The golden constant was taken from the code as it stood BEFORE this lever
    existed (checksummed side by side against the previous commit), so a future
    edit to the persist paths that leaks into the default world fails here rather
    than silently invalidating every stage 2-5 number."""
    from sim.config import load_config as _load
    cfg = _persist_cfg(cfg2)
    ROOT = Path(__file__).resolve().parent.parent
    cfg4 = _load(ROOT / "config" / "island2" / "society4.yaml").replace(
        **{"world.num_agents": 12, "world.max_ticks": 240,
           "society.num_households": 4, "bushes.num_clusters": 4})
    assert _trajectory_checksum(cfg4) == PERSIST_OFF_CHECKSUM
    # ...and the persist knobs are inert while the flag is off
    assert _trajectory_checksum(
        cfg4, ArbiterConfig(persist_timeout=999)) == PERSIST_OFF_CHECKSUM
    del cfg


def test_persist_extends_only_the_goals_with_a_reachable_goal_state():
    """Five goals are excluded and each exclusion is a measured symptom, not a
    taste call: `rest` never fails its viability test (paralysis, not
    persistence), the gifts are single-tick, `steal`/`raid` are opportunistic by
    construction (stage 2's permanent-war correction -- persisting raid took it
    from 8.3% to 13.4% of intentions), and `explore`'s goal state is a
    PERCEPTION a full-handed agent can never reach, so persisting it produced a
    150-tick wander (27% -> 36% of intentions)."""
    from sim.utility import (DELIVER, GOAL_GIVE_MATERIAL, GOAL_RAID,
                             commit_budget)
    acfg = ArbiterConfig(persist_until_goal=True, persist_timeout=150)
    goals = np.array([DELIVER, FORAGE, SHELTER, REST, GOAL_STEAL,
                      GOAL_GIVE_FOOD, GOAL_GIVE_MATERIAL, EXPLORE, GOAL_RAID])
    got = commit_budget(acfg, goals)
    want = np.array([150, 150, 150, 25, 25, 25, 25, 25, 25])
    assert np.array_equal(got, want)
    # off: everything is commit_ticks, whatever the timeout says
    assert (commit_budget(ArbiterConfig(persist_timeout=999), goals) == 25).all()


def test_the_curfew_fires_once_and_only_under_persistence(cfg2):
    """A 150-tick option decided at noon must not run through the night -- the
    scripted population lost 60 ticks of life and a quarter of its sheltered
    nights to exactly that. But it fires ONCE per option: re-testing every tick
    would hand every agent a decision per tick for a third of the day, and a
    night owl that looks at the sky and forages on keeps its commitment."""
    from sim.utility import NEED_SAFETY, option_interrupted
    n = 3
    needs = np.zeros((n, 7))
    needs[:, NEED_SAFETY] = 0.4                       # the dusk ramp has begun
    goals = np.array([FORAGE, SHELTER, FORAGE])
    decided_safe = np.array([True, True, False])      # the third looked already
    off = option_interrupted(needs, goals, ArbiterConfig(), decided_safe)
    on = option_interrupted(needs, goals,
                            ArbiterConfig(persist_until_goal=True), decided_safe)
    assert not off.any(), "the curfew must not touch the 25-tick world"
    assert list(on) == [True, False, False]


def test_persisted_deliver_survives_running_out_of_material(cfg2):
    """The goal state is 'the site is fed', not 'my hands are empty'. Cutting the
    option when the carried unit is spent is precisely the compound-prize wall
    this lever exists to remove -- six units at a capacity of two is three
    separate decisions otherwise."""
    from sim.utility import DELIVER
    cfg = _persist_cfg(cfg2)
    w = World(cfg, seed=11)
    w.pool.wood[:] = 0.0
    w.pool.stone[:] = 0.0
    view = ObsView(w.observations(), cfg)
    goals = np.full(cfg.world.num_agents, DELIVER, dtype=np.int64)
    plain = goal_viable(view, cfg, goals, ArbiterConfig())
    persist = goal_viable(view, cfg, goals,
                          ArbiterConfig(persist_until_goal=True))
    assert not plain.any(), "empty-handed deliver should die without the flag"
    assert persist.any(), "the programme must outlive the carried unit"


def test_persisted_deliver_goes_and_fetches_instead_of_idling(cfg2):
    """The supply leg: an agent that chose 'build that shelter' and is holding
    nothing walks at a node and harvests. This is the honest cost of the lever --
    the programme is scripted muscle, so what can emerge is when-to-build."""
    from sim.utility import CHOP, DELIVER, MINE
    cfg = _persist_cfg(cfg2)
    w = World(cfg, seed=11)
    w.pool.wood[:] = 0.0
    w.pool.stone[:] = 0.0
    view = ObsView(w.observations(), cfg)
    mask = w.action_mask()
    goals = np.full(cfg.world.num_agents, DELIVER, dtype=np.int64)
    off = execute_goals(goals, view, cfg, mask, np.zeros(cfg.world.num_agents,
                                                         dtype=np.int64))
    on = execute_goals(goals, view, cfg, mask,
                       np.zeros(cfg.world.num_agents, dtype=np.int64),
                       ArbiterConfig(persist_until_goal=True))
    # Without the flag an empty-handed deliverer can only trudge to the site and
    # stand there -- it is not a state the arbiter can reach (deliver is masked
    # out when you carry nothing), which is exactly why the programme needed the
    # widening.
    assert not np.isin(off, (CHOP, MINE)).any()
    fetching = np.isin(on, (CHOP, MINE)) | (on != off)
    assert fetching.any(), "the supply leg never fired"
    assert set(np.unique(on)).issubset(set(range(8)) | {CHOP, MINE, IDLE})


def test_tier0_interruption_still_ends_a_persisted_option(cfg2):
    """utility.py's own warning: the semi-MDP bookkeeping is the one place this
    can silently rot. A 150-tick commitment must not let a committed builder walk
    past its own death."""
    from sim.utility import DELIVER
    cfg = _persist_cfg(cfg2)
    w = World(cfg, seed=9)
    acfg = ArbiterConfig(persist_until_goal=True)
    runner = OptionRunner(UtilityArbiter(cfg, acfg, seed=0), cfg, seed=0)
    runner.goals[:] = DELIVER
    runner.ticks_left[:] = acfg.persist_timeout
    w.pool.hunger[:] = cfg.hunger.max * 0.05
    runner.act(w.observations(), w.action_mask())
    assert runner.last_decided.all(), "a starving agent kept building"


def test_persist_actually_lengthens_options_in_a_real_episode(cfg2):
    """The population-level read: fewer decisions per agent over the same ticks,
    with the commitment holding rather than the agents dying earlier."""
    cfg = _persist_cfg(cfg2)
    def decisions(acfg):
        w = World(cfg, seed=7)
        runner = utility_runner(cfg, seed=7, acfg=acfg)
        obs = w.observations()
        for _ in range(240):
            res = w.step(runner.act(obs, w.action_mask()))
            obs = res.obs
            if res.episode_done:
                break
        return float(runner.decisions.mean())
    assert decisions(ArbiterConfig(persist_until_goal=True)) < decisions(ArbiterConfig())
