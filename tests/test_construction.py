"""Milestone 4: wood, stone, shelter construction, and the night hazard."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sim.agents import (
    BUILD,
    CHOP,
    GATHER,
    IDLE,
    MINE,
    STEAL,
    action_names,
    night_phase,
    num_actions,
    observation_dim,
    observation_layout,
)
from sim.config import load_config
from sim.policy import ActorCritic, build_policy, column_map, grow_actor_critic
from sim.ppo import PPOTrainer
from sim.world import VecWorld, World


@pytest.fixture
def m4(m3=None):
    return load_config("config/m3.yaml").replace(**{"construction.enabled": True})


def park(w, i, x, z, hunger=None):
    w.pool.x[i], w.pool.z[i] = x, z
    w.pool.hunger[i] = hunger if hunger is not None else w.cfg.hunger.max


def idle_all(cfg):
    return np.full(cfg.world.num_agents, IDLE)


# --- space and layout ---------------------------------------------------------


def test_construction_is_off_everywhere_before_m4(cfg):
    for name in ("config/default.yaml", "config/scarce.yaml", "config/m3.yaml"):
        c = load_config(name)
        assert c.construction.enabled is False
        assert "chop" not in action_names(c)


def test_construction_actions_are_appended_after_steal(m4):
    names = action_names(m4)
    assert names[STEAL] == "steal"
    assert names[CHOP] == "chop"
    assert names[MINE] == "mine"
    assert names[BUILD] == "build"
    assert num_actions(m4) == 14


def test_steal_slot_is_kept_even_if_stealing_is_disabled(m4):
    """Construction always includes the (inert) steal slot, so CHOP/MINE/BUILD
    are stable indices -- the appended-not-inserted rule, held under composition."""
    no_steal = m4.replace(**{"competition.enable_steal": False})
    names = action_names(no_steal)
    assert names[CHOP] == "chop" and num_actions(no_steal) == 14


def test_observation_layout_matches_dim_and_is_unique(m4):
    layout = observation_layout(m4)
    assert len(layout) == observation_dim(m4) == 55
    assert len(set(layout)) == len(layout)
    assert "own.wood" in layout and "night.is_night" in layout


def test_m3_checkpoint_grows_into_m4_by_feature(m4):
    m3 = load_config("config/m3.yaml")
    torch.manual_seed(0)
    old = ActorCritic(observation_dim(m3), num_actions(m3), (24,))
    obs_map = column_map(observation_layout(m3), observation_layout(m4))
    act_map = column_map(action_names(m3), action_names(m4))
    new = grow_actor_critic(old, observation_dim(m4), num_actions(m4), obs_map, act_map)

    obs_old = torch.randn(6, observation_dim(m3))
    obs_new = torch.zeros(6, observation_dim(m4))
    obs_new[:, obs_map] = obs_old
    assert torch.allclose(old(obs_old)[0], new(obs_new)[0][:, act_map], atol=1e-6)
    # the three new actions start at logit zero: reachable, unpreferred
    for a in (CHOP, MINE, BUILD):
        assert torch.allclose(new(obs_new)[0][:, a], torch.zeros(6), atol=1e-6)


# --- harvesting ----------------------------------------------------------------


def test_chop_takes_wood_and_depletes_the_tree(m4):
    w = World(m4, seed=1)
    park(w, 0, w.tree_x[0], w.tree_z[0])
    before = int(w.tree_wood[0])
    a = idle_all(m4); a[0] = CHOP
    res = w.step(a)
    assert int(w.pool.wood[0]) == 1
    assert int(w.tree_wood[0]) == before - 1
    assert res.rewards[0] == pytest.approx(m4.reward.alive_per_tick)  # unshaped: no pay


def test_mine_takes_stone(m4):
    w = World(m4, seed=2)
    park(w, 0, w.rock_x[0], w.rock_z[0])
    a = idle_all(m4); a[0] = MINE
    w.step(a)
    assert int(w.pool.stone[0]) == 1


def test_material_capacity_is_shared_between_wood_and_stone(m4):
    w = World(m4, seed=3)
    park(w, 0, w.tree_x[0], w.tree_z[0])
    w.pool.stone[0] = m4.construction.material_capacity
    a = idle_all(m4); a[0] = CHOP
    w.step(a)
    assert int(w.pool.wood[0]) == 0, "carrying capacity is wood+stone combined"


def test_empty_tree_yields_nothing(m4):
    w = World(m4, seed=4)
    park(w, 0, w.tree_x[0], w.tree_z[0])
    w.tree_wood[:] = 0
    a = idle_all(m4); a[0] = CHOP
    w.step(a)
    assert int(w.pool.wood[0]) == 0


def test_harvest_out_of_range_does_nothing(m4):
    w = World(m4, seed=5)
    far = 35.0
    park(w, 0, far, far / 2)
    d = np.hypot(w.tree_x - far, w.tree_z - far / 2)
    if d.min() <= m4.construction.harvest_radius:
        pytest.skip("a tree happens to be at the parking spot")
    a = idle_all(m4); a[0] = CHOP
    w.step(a)
    assert int(w.pool.wood[0]) == 0


# --- building -------------------------------------------------------------------


def test_build_delivers_carried_material(m4):
    w = World(m4, seed=6)
    park(w, 0, w.site_x[0], w.site_z[0])
    w.pool.wood[0] = 2
    need_before = int(w.site_wood_needed[0])
    a = idle_all(m4); a[0] = BUILD
    res = w.step(a)
    assert int(w.site_wood_needed[0]) == need_before - 1
    assert int(w.pool.wood[0]) == 1
    assert res.rewards[0] == pytest.approx(m4.reward.alive_per_tick)  # unshaped


def test_build_delivers_stone_when_wood_is_covered(m4):
    w = World(m4, seed=7)
    park(w, 0, w.site_x[0], w.site_z[0])
    w.site_wood_needed[0] = 0
    w.pool.stone[0] = 1
    a = idle_all(m4); a[0] = BUILD
    w.step(a)
    assert int(w.site_stone_needed[0]) == m4.construction.site_stone_cost - 1
    assert int(w.pool.stone[0]) == 0


def test_build_with_nothing_needed_or_nothing_carried_is_inert(m4):
    w = World(m4, seed=8)
    park(w, 0, w.site_x[0], w.site_z[0])
    a = idle_all(m4); a[0] = BUILD
    w.step(a)   # carries nothing
    assert int(w.site_wood_needed[0]) == m4.construction.site_wood_cost

    w.site_wood_needed[0] = 0
    w.site_stone_needed[0] = 0
    w.pool.wood[0] = 1
    w.step(a)   # site complete
    assert int(w.pool.wood[0]) == 1, "complete sites take no more material"


def test_completion_is_counted_and_shelter_is_communal(m4):
    """Two agents finish one site between them; the count is per site, not per
    contributor."""
    w = World(m4, seed=9)
    w.site_wood_needed[0] = 1
    w.site_stone_needed[0] = 1
    park(w, 0, w.site_x[0], w.site_z[0])
    park(w, 1, w.site_x[0], w.site_z[0])
    w.pool.wood[0] = 1
    w.pool.stone[1] = 1
    a = idle_all(m4); a[0] = BUILD; a[1] = BUILD
    w.step(a)
    assert w.stats().shelters_completed == 1
    assert w.stats().builds == 2


# --- night ----------------------------------------------------------------------


def ticks_to_night(cfg) -> int:
    cc = cfg.construction
    day_len = int(cc.night_cycle * (1.0 - cc.night_fraction))
    return day_len


def test_night_phase_arithmetic(m4):
    cc = m4.construction
    assert night_phase(0, m4) == (0.0, False)
    assert night_phase(ticks_to_night(m4), m4)[1] is True
    assert night_phase(cc.night_cycle - 1, m4)[1] is True
    assert night_phase(cc.night_cycle, m4)[1] is False  # dawn wraps


def test_night_multiplies_drain_for_the_exposed(m4):
    w = World(m4, seed=10)
    w.tick = ticks_to_night(m4)          # jump straight to nightfall
    w.pool.x[:], w.pool.z[:] = 30.0, 0.0  # nowhere near any shelter
    h0 = w.pool.hunger.copy()
    w.step(idle_all(m4))
    expected = m4.hunger.drain_per_tick * m4.construction.night_drain_multiplier
    assert np.allclose(h0 - w.pool.hunger, expected)
    assert w.stats().night_ticks_exposed == m4.world.num_agents


def test_completed_shelter_protects_at_night(m4):
    w = World(m4, seed=11)
    w.site_wood_needed[0] = 0
    w.site_stone_needed[0] = 0           # completed shelter
    w.tick = ticks_to_night(m4)
    for i in range(m4.world.num_agents):
        park(w, i, w.site_x[0], w.site_z[0])
    h0 = w.pool.hunger.copy()
    w.step(idle_all(m4))
    assert np.allclose(h0 - w.pool.hunger, m4.hunger.drain_per_tick)
    assert w.stats().night_ticks_sheltered == m4.world.num_agents


def test_incomplete_shelter_protects_nobody(m4):
    w = World(m4, seed=12)
    w.site_wood_needed[0] = 1            # one unit short
    w.site_stone_needed[0] = 0
    w.tick = ticks_to_night(m4)
    park(w, 0, w.site_x[0], w.site_z[0])
    h0 = float(w.pool.hunger[0])
    w.step(idle_all(m4))
    expected = m4.hunger.drain_per_tick * m4.construction.night_drain_multiplier
    assert h0 - float(w.pool.hunger[0]) == pytest.approx(expected)


def test_daytime_drain_is_normal(m4):
    w = World(m4, seed=13)
    assert night_phase(w.tick, m4)[1] is False
    h0 = w.pool.hunger.copy()
    w.step(idle_all(m4))
    assert np.allclose(h0 - w.pool.hunger, m4.hunger.drain_per_tick)


# --- masking --------------------------------------------------------------------


def masked(m4):
    return m4.replace(**{"competition.mask_invalid_actions": True})


def test_mask_offers_chop_only_at_a_stocked_tree(m4):
    cfg = masked(m4)
    w = World(cfg, seed=14)
    w.pool.x[:], w.pool.z[:] = 0.0, 0.0
    d = np.hypot(w.tree_x, w.tree_z)
    if d.min() <= cfg.construction.harvest_radius:
        w.tree_wood[d <= cfg.construction.harvest_radius] = 0
    assert not w.action_mask()[:, CHOP].any()

    park(w, 0, w.tree_x[1], w.tree_z[1])
    w.tree_wood[1] = 3
    assert w.action_mask()[0, CHOP]

    w.pool.wood[0] = cfg.construction.material_capacity
    assert not w.action_mask()[0, CHOP], "no room, no chop"


def test_mask_offers_build_only_when_agent_can_contribute(m4):
    cfg = masked(m4)
    w = World(cfg, seed=15)
    park(w, 0, w.site_x[0], w.site_z[0])
    assert not w.action_mask()[0, BUILD], "empty-handed"

    w.pool.wood[0] = 1
    assert w.action_mask()[0, BUILD]

    w.site_wood_needed[0] = 0
    assert not w.action_mask()[0, BUILD], "site no longer needs wood"

    w.pool.stone[0] = 1
    assert w.action_mask()[0, BUILD], "but it still needs stone"


# --- observations ----------------------------------------------------------------


def test_observation_reports_own_materials_and_night(m4):
    w = World(m4, seed=16)
    w.pool.wood[0] = m4.construction.material_capacity
    layout = observation_layout(m4)
    obs = w.observations()
    assert obs[0, layout.index("own.wood")] == pytest.approx(1.0)
    assert obs[0, layout.index("night.is_night")] == pytest.approx(0.0)

    w.tick = ticks_to_night(m4)
    obs = w.observations()
    assert obs[0, layout.index("night.is_night")] == pytest.approx(1.0)


def test_site_need_channels_track_delivery_per_material(m4):
    """Per-material needs, not blended progress: whether to bring wood or stone
    is a decision, and a policy that cannot see which is missing can only guess."""
    w = World(m4, seed=17)
    layout = observation_layout(m4)
    park(w, 0, w.site_x[0], w.site_z[0])

    obs = w.observations()
    assert obs[0, layout.index("site0.need_wood")] == pytest.approx(1.0)
    assert obs[0, layout.index("site0.need_stone")] == pytest.approx(1.0)
    assert obs[0, layout.index("site0.complete")] == pytest.approx(0.0)

    w.site_wood_needed[0] = 0
    obs = w.observations()
    assert obs[0, layout.index("site0.need_wood")] == pytest.approx(0.0)
    assert obs[0, layout.index("site0.need_stone")] == pytest.approx(1.0)
    assert obs[0, layout.index("site0.complete")] == pytest.approx(0.0)

    w.site_stone_needed[0] = 0
    obs = w.observations()
    assert obs[0, layout.index("site0.complete")] == pytest.approx(1.0)


def test_observation_bounds_hold_with_construction_on(m4):
    w = World(m4.replace(**{"world.max_ticks": 10_000}), seed=18)
    rng = np.random.default_rng(0)
    for _ in range(400):
        res = w.step(rng.integers(0, num_actions(m4), size=m4.world.num_agents))
        assert res.obs.min() >= -1.0 and res.obs.max() <= 1.0
        if not w.pool.alive.any():
            break


# --- shaping and training ---------------------------------------------------------


def test_shaping_rewards_flow_only_when_configured(m4):
    shaped = m4.replace(**{"reward.wood": 0.3, "reward.build": 0.5, "reward.complete": 2.0})
    w = World(shaped, seed=19)
    park(w, 0, w.tree_x[0], w.tree_z[0])
    a = idle_all(shaped); a[0] = CHOP
    res = w.step(a)
    assert res.rewards[0] == pytest.approx(0.3 + shaped.reward.alive_per_tick)

    w2 = World(shaped, seed=20)
    park(w2, 0, w2.site_x[0], w2.site_z[0])
    w2.site_wood_needed[0] = 1
    w2.site_stone_needed[0] = 0
    w2.pool.wood[0] = 1
    a = idle_all(shaped); a[0] = BUILD
    res = w2.step(a)
    assert res.rewards[0] == pytest.approx(0.5 + 2.0 + shaped.reward.alive_per_tick)


def test_ppo_runs_in_a_construction_world(m4):
    tiny = m4.replace(**{
        "competition.mask_invalid_actions": True,
        "ppo.num_envs": 4, "ppo.rollout_ticks": 24, "ppo.num_minibatches": 2,
        "ppo.epochs": 2, "world.max_ticks": 60, "policy.mode": "individual",
    })
    torch.manual_seed(0)
    envs = VecWorld(tiny, seed=0, num_envs=tiny.ppo.num_envs)
    assert envs.obs_dim == 55
    trainer = PPOTrainer(tiny, build_policy(tiny, envs.obs_dim), envs, device="cpu")
    for i in range(2):
        m = trainer.train_update(i)
        assert np.isfinite(m.policy_loss) and np.isfinite(m.value_loss)
    assert trainer.policy.n_actions == 14


def test_construction_world_is_deterministic(m4):
    def run():
        w = World(m4, seed=33)
        rng = np.random.default_rng(7)
        out = []
        for _ in range(80):
            r = w.step(rng.integers(0, num_actions(m4), size=m4.world.num_agents))
            out.append((r.rewards.copy(), w.pool.wood.copy(), w.site_wood_needed.copy()))
        return out
    a, b = run(), run()
    assert all(np.array_equal(x[i], y[i]) for x, y in zip(a, b) for i in range(3))
