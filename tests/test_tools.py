"""Island 2.0 tech ladder, rung 1: the craftable axe.

Every test here is named after the thing that went wrong or the claim it
protects, in this file's usual style. The two that matter most are the
bit-identity pins: appending a rung must not move a single number in any world
that does not have it, and rung 1 broke that twice before it worked -- once
through the trait draw and once through the arbiter's net width.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from sim.agents import CRAFT, action_names, num_actions, observation_layout
from sim.config import load_config
from sim.economy import materials
from sim.society import make_runner, run_episodes
from sim.utility import (CRAFT_AXE, NEED_TOOL, N_GOALS, N_GOALS_STAGE4,
                         ArbiterConfig, agent_traits, compute_needs)
from sim.world import World

ROOT = Path(__file__).resolve().parent.parent
AXE = ROOT / "config" / "island2" / "society4_axe.yaml"
CONTROL = ROOT / "config" / "island2" / "society4_axe_control.yaml"


@pytest.fixture
def axe_cfg():
    return load_config(AXE).replace(**{"world.num_agents": 20, "world.max_ticks": 240,
                                       "society.num_households": 4,
                                       "bushes.num_clusters": 4})


def _run(cfg, seed=10000, arm=False):
    world = World(cfg, seed=seed)
    runner = make_runner(cfg, "utility", seed, ArbiterConfig(), None)
    if arm:
        world.pool.axe[:] = 1
    obs = world.observations()
    while True:
        res = world.step(runner.act(obs, world.action_mask()))
        obs = res.obs
        if res.episode_done:
            break
    return world


# --- the append-never-insert contract ---------------------------------------

def test_craft_is_appended_so_every_earlier_action_keeps_its_index():
    cfg = load_config(AXE)
    names = action_names(cfg)
    assert names[CRAFT] == "craft"
    assert names[:CRAFT] == action_names(load_config(CONTROL))
    assert num_actions(cfg) == num_actions(load_config(CONTROL)) + 1


def test_own_axe_is_the_only_new_observation_channel():
    axe = observation_layout(load_config(AXE))
    bare = observation_layout(load_config(CONTROL))
    assert set(axe) - set(bare) == {"own.axe"}
    assert len(axe) == len(bare) + 1


def test_a_toolless_world_is_untouched(cfg):
    """The default 1.0 world must not gain an action or a channel."""
    assert "craft" not in action_names(cfg)
    assert "own.axe" not in observation_layout(cfg)


# --- the two bit-identity pins ----------------------------------------------

def test_appending_a_goal_does_not_move_the_stage4_trait_draw():
    """`rng.normal` fills row-major, so ONE draw of (agents, N_GOALS) reshuffles
    every agent's every trait the moment a goal is appended -- which is how this
    rung first broke the persist-off golden checksum in a world with no tools in
    it. The stage-4 block keeps the numbers it always had."""
    acfg = ArbiterConfig()
    traits = agent_traits(37, 3, acfg)
    assert traits.shape == (37, N_GOALS)
    rng = np.random.default_rng(3)
    legacy = np.exp(rng.normal(0.0, acfg.trait_spread, size=(37, N_GOALS_STAGE4)))
    assert np.array_equal(traits[:, :N_GOALS_STAGE4], legacy)


def test_a_toolless_arbiter_keeps_the_stage4_goal_width(cfg):
    """A net's random init is shaped by its widths, so widening the goal head
    would change every from-scratch stage-5 number in a world with no tools."""
    from sim.arbiter import goal_width
    assert goal_width(cfg) == N_GOALS_STAGE4
    assert goal_width(load_config(AXE)) == N_GOALS


# Taken from the code as it stood BEFORE tools existed (`git archive HEAD` into a
# temp tree, same script, same seed), so this is a real pin and not a number read
# off the change it is supposed to be guarding.
TOOLLESS_TRAJECTORY = "e6de2dcb9052773a64eed2fb407f58d9f4f3cd0bc150d829cd16ea196a8b97cc"


def test_toolless_trajectory_is_unchanged_by_the_rung():
    """The scripted side, end to end: same seed, same world, same positions.

    Companion to test_utility.py's persist-off checksum. If a tools branch ever
    leaks into a world without tools, this fails here rather than silently
    invalidating every stage 2-5 number.
    """
    cfg = load_config(ROOT / "config" / "island2" / "society4.yaml").replace(
        **{"world.num_agents": 12, "world.max_ticks": 180,
           "society.num_households": 4, "bushes.num_clusters": 4})
    world = _run(cfg)
    h = hashlib.sha256()
    for arr in (world.pool.x, world.pool.z, world.pool.hunger, world.alive_ticks):
        h.update(np.asarray(arr).tobytes())
    assert h.hexdigest() == TOOLLESS_TRAJECTORY


# --- the mechanic ------------------------------------------------------------

def test_an_axe_needs_a_haft_and_a_head_even_in_a_fungible_world(axe_cfg):
    """A site takes whatever arrives because a wall is a wall. An axe does not:
    keeping the composition is the only thing that stops `mine` becoming a dead
    action once the village is built."""
    assert axe_cfg.construction.fungible_materials
    world = World(axe_cfg, seed=1)
    pool = world.pool
    pool.x[:] = world.site_x[0]
    pool.z[:] = world.site_z[0]
    pool.wood[:] = 2
    pool.stone[:] = 0
    assert not world.action_mask()[:, CRAFT].any()
    pool.wood[:] = 1
    pool.stone[:] = 1
    assert world.action_mask()[:, CRAFT].all()


def test_a_second_axe_is_never_offered(axe_cfg):
    """A doomed action is exactly what masking exists to delete."""
    world = World(axe_cfg, seed=1)
    pool = world.pool
    pool.x[:] = world.site_x[0]
    pool.z[:] = world.site_z[0]
    pool.wood[:] = 1
    pool.stone[:] = 1
    pool.axe[:] = 1
    assert not world.action_mask()[:, CRAFT].any()


def test_an_axe_doubles_the_chop_and_never_overfills(axe_cfg):
    """Bounded by carrying room and by what the tree still holds -- an
    unbounded multiplier would let one swing empty a tree."""
    from sim.agents import CHOP
    world = World(axe_cfg, seed=2)
    pool = world.pool
    pool.x[:] = world.tree_x[0]
    pool.z[:] = world.tree_z[0]
    pool.wood[:] = 0
    pool.stone[:] = 0
    pool.axe[0] = 1
    world.step(np.full(pool.n, CHOP, dtype=np.int64))
    assert pool.wood[0] == axe_cfg.tools.chop_multiplier
    assert pool.wood[1] == 1
    cap = axe_cfg.construction.material_capacity
    assert (pool.wood + pool.stone <= cap).all()


def test_crafting_spends_the_materials_and_pays_nothing(axe_cfg):
    """Rule 1: nothing in the world pays for a craft. Its whole return is the
    wood every later chop brings in."""
    world = World(axe_cfg, seed=3)
    pool = world.pool
    pool.x[:] = world.site_x[0]
    pool.z[:] = world.site_z[0]
    pool.wood[:] = 1
    pool.stone[:] = 1
    res = world.step(np.full(pool.n, CRAFT, dtype=np.int64))
    assert (pool.axe == 1).all()
    assert (pool.wood == 0).all() and (pool.stone == 0).all()
    # The only reward on a craft tick is the per-tick survival term every agent
    # gets for being alive -- identical to an idle tick.
    idle = World(axe_cfg, seed=3)
    idle.pool.x[:] = world.site_x[0]
    idle.pool.z[:] = world.site_z[0]
    idle.pool.wood[:] = 1
    idle.pool.stone[:] = 1
    from sim.agents import IDLE
    res_idle = idle.step(np.full(pool.n, IDLE, dtype=np.int64))
    assert np.allclose(res.rewards, res_idle.rewards)


# --- the arbiter -------------------------------------------------------------

def test_the_tool_need_is_what_makes_crafting_choosable_at_all(axe_cfg):
    """Restoring `wealth` alone can never fire: `craft` is available exactly when
    the agent is carrying a wood and a stone, and on a 2-unit inventory that
    means FULL -- so the wealth deficit is zero at the moment the goal becomes
    possible. A need is what you LACK, and an unarmed agent lacks the tool."""
    from sim.obsview import ObsView
    world = World(axe_cfg, seed=4)
    view = ObsView(world.observations(), axe_cfg)
    needs = compute_needs(view, axe_cfg)
    assert (needs[:, NEED_TOOL] == 1.0).all()
    world.pool.axe[:] = 1
    view = ObsView(world.observations(), axe_cfg)
    assert (compute_needs(view, axe_cfg)[:, NEED_TOOL] == 0.0).all()


def test_craft_is_off_the_menu_in_a_world_without_tools(cfg):
    from sim.arbiter import goal_mask
    from sim.obsview import ObsView
    from sim.utility import goal_availability
    world = World(cfg, seed=5)
    view = ObsView(world.observations(), cfg)
    needs = compute_needs(view, cfg)
    available, _, _ = goal_availability(view, cfg, needs, ArbiterConfig())
    assert not available[:, CRAFT_AXE].any()
    # ...and the learned chooser never even sees the column.
    assert goal_mask(view, cfg, ArbiterConfig()).shape[1] == N_GOALS_STAGE4


def test_axes_get_made_and_the_report_says_how_many():
    """The FULL world, not the shrunk fixture, and that is the finding rather
    than a test convenience: crafting needs an unarmed agent standing at a site
    holding one wood and one stone, which is 1.6% of the ticks it stands there
    loaded at all. A 20-agent island simply never rolls it."""
    rep = run_episodes(load_config(AXE), 1, 10000, ArbiterConfig(), policy="utility")
    assert rep.axes_crafted and rep.axe_holders
    assert sum(rep.axes_crafted) > 0
    assert sum(rep.axe_holders) < load_config(AXE).world.num_agents, (
        "universal adoption would mean the composition rule stopped biting")


# --- the sizing --------------------------------------------------------------

def test_the_economy_says_which_regime_the_axe_world_is_in():
    """An axe never CREATES wood -- trees hold a finite stock -- so it only helps
    when labour, not stock, is the constraint. Sizing a tool world without
    knowing which is rule 5's mistake with a new mechanic."""
    m = materials(load_config(AXE))
    assert m.ratio > 1.3, "the axe world must have slack stock, or it measures scarcity"
    bill = m.axes_if_everyone * (m.axe_wood_cost + m.axe_stone_cost)
    assert m.supply > m.demand + bill, "arming everyone must not starve construction"
    assert m.chop_trips_axed * 2 == pytest.approx(m.chop_trips_bare)


def test_a_stage4_checkpoint_grows_into_a_tools_world(cfg, tmp_path):
    """Appending a goal widens the arbiter's net at BOTH ends -- the trait vector
    on the input and the goal head on the output -- so every arb4/arb5 checkpoint
    would simply stop loading. The project's rule since M3 is that a milestone
    GROWS a policy rather than orphaning it, and the mapping is by NAME because
    `own.axe` is inserted mid-observation: by position, the grown net would feed
    its shoreline weights the axe flag.
    """
    import numpy as np
    from sim.arbiter import ArbiterTrainer, TrainConfig, load_arbiter, save_checkpoint

    trainer = ArbiterTrainer(cfg, TrainConfig(num_envs=1, rollout_ticks=10), seed=1,
                             learn_agents=np.arange(4))
    path = tmp_path / "stage4.pt"
    save_checkpoint(path, trainer, update=1)

    axe_cfg = load_config(AXE)
    grown = load_arbiter(path, axe_cfg)
    assert grown.policy.n_actions == N_GOALS
    assert grown.traits.shape[1] == N_GOALS
    # The appended logit's weights are zero, so the grown net scores the goals it
    # was trained on exactly as it did.
    head = grown.policy.policy_head
    assert float(head.weight[CRAFT_AXE].detach().abs().sum()) == 0.0
    assert float(head.bias[CRAFT_AXE].detach()) == 0.0
    # ...and loading into its OWN world needs no growth at all.
    same = load_arbiter(path, cfg)
    assert same.policy.n_actions == N_GOALS_STAGE4


def test_arm_all_is_the_counterfactual_that_separates_value_from_adoption():
    """A low adoption rate and a worthless tool look identical from the outside.
    Arming everyone at spawn takes adoption off the table -- the movement
    counterpart of `sim.opportunity --force`, and the measurement that turned
    rung 1 from a shrug into a result.
    """
    cfg = load_config(AXE).replace(**{"world.max_ticks": 120})
    rep = run_episodes(cfg, 1, 10000, ArbiterConfig(), policy="utility", arm_all=True)
    assert rep.axe_holders == [cfg.world.num_agents]
    assert sum(rep.axes_crafted) == 0, "nobody needs to craft what they already hold"
    assert int(rep.goal_ticks_unarmed.sum()) == 0
