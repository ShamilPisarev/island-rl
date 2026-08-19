"""Milestone 3: bush contention, stealing, and the config layering behind them."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sim.agents import (
    BASE_ACTION_NAMES,
    GATHER,
    IDLE,
    N_MOVE_ACTIONS,
    STEAL,
    action_mask,
    action_names,
    num_actions,
    observation_dim,
    observation_layout,
)
from sim.config import load_config
from sim.policy import (
    ActorCritic,
    build_policy,
    column_map,
    greedy_thief_actions,
    grow_actor_critic,
    grow_policy,
    PolicyGroup,
)
from sim.ppo import PPOTrainer
from sim.world import VecWorld, World


@pytest.fixture
def m3() -> object:
    return load_config("config/m3.yaml")


@pytest.fixture
def scarce() -> object:
    return load_config("config/scarce.yaml")


# --- config layering --------------------------------------------------------


def test_extends_inherits_and_overrides(cfg, scarce):
    assert scarce.world.island_radius == cfg.world.island_radius     # inherited
    assert scarce.hunger.drain_per_tick == cfg.hunger.drain_per_tick  # inherited
    assert scarce.bushes.capacity == 2 and cfg.bushes.capacity == 6   # overridden
    assert scarce.bushes.count == 6 and cfg.bushes.count == 20


def test_extends_chains_two_deep(m3, cfg):
    """m3 -> scarce -> default: values must come from the nearest one that sets them."""
    assert m3.bushes.capacity == 2                     # from scarce
    assert m3.world.island_radius == cfg.world.island_radius  # from default
    assert m3.competition.enable_steal is True         # from m3


def test_default_config_has_competition_off(cfg):
    """M1 and M2 worlds must be untouched by anything added for M3."""
    assert cfg.competition.contest_bushes is False
    assert cfg.competition.enable_steal is False
    assert cfg.competition.observe_neighbour_food is False
    assert num_actions(cfg) == len(BASE_ACTION_NAMES) == 10
    assert observation_dim(cfg) == 26


def test_circular_extends_is_caught(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="circular config extends"):
        load_config(tmp_path / "a.yaml")


# --- action and observation space -------------------------------------------


def test_steal_is_appended_not_inserted(m3):
    """Every pre-M3 action must keep its index, or old checkpoints change meaning."""
    names = action_names(m3)
    assert names[:10] == BASE_ACTION_NAMES
    assert names[STEAL] == "steal"
    assert num_actions(m3) == 11


def test_neighbour_food_widens_the_observation(m3, cfg):
    assert observation_dim(m3) == observation_dim(cfg) + m3.observation.k_agents
    assert observation_dim(m3) == 29


def test_neighbour_food_channel_reports_carried_food(m3):
    w = World(m3, seed=1)
    w.pool.alive[:] = False
    w.pool.alive[0] = w.pool.alive[1] = True
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1], w.pool.z[1] = 3.0, 0.0
    w.pool.food[1] = m3.food.capacity

    start = 2 + 3 * m3.observation.k_bushes
    neighbours = w.observations()[0, start:start + 4 * m3.observation.k_agents]
    assert neighbours.reshape(m3.observation.k_agents, 4)[0, 3] == pytest.approx(1.0)

    w.pool.food[1] = 0
    neighbours = w.observations()[0, start:start + 4 * m3.observation.k_agents]
    assert neighbours.reshape(m3.observation.k_agents, 4)[0, 3] == pytest.approx(0.0)


def test_contested_flag_marks_the_agent_that_is_not_closest(m3):
    """With exclusive bushes, whether you may harvest depends on whether a rival
    stands nearer. That has to be perceivable or the policy can only gather and
    hope -- the same argument that put neighbours' food in the observation."""
    cfg = m3.replace(**{"competition.observe_bush_contested": True})
    w = World(cfg, seed=40)
    w.pool.x[0], w.pool.z[0] = w.bush_x[0], w.bush_z[0]          # on the bush
    w.pool.x[1], w.pool.z[1] = w.bush_x[0] + 1.0, w.bush_z[0]    # one unit out
    w.pool.x[2:], w.pool.z[2:] = 35.0, 0.0

    kb = cfg.observation.k_bushes
    bushes = w.observations()[:, 2:2 + 4 * kb].reshape(cfg.world.num_agents, kb, 4)
    assert bushes[0, 0, 3] == pytest.approx(0.0)   # closest: free
    assert bushes[1, 0, 3] == pytest.approx(1.0)   # further: blocked


def test_contested_flag_ignores_dead_rivals(m3):
    cfg = m3.replace(**{"competition.observe_bush_contested": True})
    w = World(cfg, seed=41)
    w.pool.x[0], w.pool.z[0] = w.bush_x[0] + 1.0, w.bush_z[0]
    w.pool.x[1], w.pool.z[1] = w.bush_x[0], w.bush_z[0]   # nearer, but dead
    w.pool.alive[1] = False
    w.pool.x[2:], w.pool.z[2:] = 35.0, 0.0

    kb = cfg.observation.k_bushes
    bushes = w.observations()[:, 2:2 + 4 * kb].reshape(cfg.world.num_agents, kb, 4)
    assert bushes[0, 0, 3] == pytest.approx(0.0)


def test_contested_flag_widens_the_observation_only_when_enabled(m3):
    assert observation_dim(m3) == 29
    assert observation_dim(m3.replace(**{"competition.observe_bush_contested": True})) == 33


def test_observation_bounds_still_hold_with_competition_on(m3):
    w = World(m3.replace(**{"world.max_ticks": 10_000}), seed=2)
    rng = np.random.default_rng(0)
    for _ in range(400):
        res = w.step(rng.integers(0, num_actions(m3), size=m3.world.num_agents))
        assert res.obs.min() >= -1.0 and res.obs.max() <= 1.0
        if not w.pool.alive.any():
            break


# --- bush contention --------------------------------------------------------


def _park_on_bush(world, agents, bush=0):
    for i in agents:
        world.pool.x[i], world.pool.z[i] = world.bush_x[bush], world.bush_z[bush]
        world.pool.hunger[i] = world.cfg.hunger.max


def test_contested_bush_yields_one_berry_per_tick(m3):
    """Two agents on one bush: exactly one berry moves, and the loser is recorded."""
    w = World(m3, seed=3)
    w.bush_berries[:] = m3.bushes.capacity
    _park_on_bush(w, [0, 1])
    actions = np.full(m3.world.num_agents, IDLE)
    actions[0] = actions[1] = GATHER

    before = int(w.bush_berries[0])
    res = w.step(actions)

    assert int(w.bush_berries[0]) == before - 1
    assert int(w.pool.food[0]) + int(w.pool.food[1]) == 1
    assert res.gathered.sum() == 1
    assert res.contested.sum() == 1


def test_without_contention_both_agents_take_a_berry(scarce):
    """The control: on the same setup with contest_bushes off, both succeed."""
    w = World(scarce, seed=3)
    w.bush_berries[:] = scarce.bushes.capacity
    _park_on_bush(w, [0, 1])
    actions = np.full(scarce.world.num_agents, IDLE)
    actions[0] = actions[1] = GATHER

    before = int(w.bush_berries[0])
    res = w.step(actions)
    assert int(w.bush_berries[0]) == before - 2
    assert res.gathered.sum() == 2
    assert res.contested.sum() == 0


def test_closest_agent_blocks_the_others(m3):
    """Exclusive bushes: standing nearest denies the bush to everyone else, which
    is what makes a bush worth holding rather than merely visiting."""
    w = World(m3, seed=20)
    w.bush_berries[:] = m3.bushes.capacity
    w.pool.hunger[:] = m3.hunger.max
    # agent 0 right on the bush, agent 1 nearby but further out (still in range)
    w.pool.x[0], w.pool.z[0] = w.bush_x[0], w.bush_z[0]
    w.pool.x[1] = w.bush_x[0] + m3.bushes.gather_radius * 0.8
    w.pool.z[1] = w.bush_z[0]
    w.pool.x[2:], w.pool.z[2:] = 35.0, 0.0   # everyone else far away

    actions = np.full(m3.world.num_agents, IDLE)
    actions[0] = actions[1] = GATHER
    res = w.step(actions)

    assert res.gathered[0] == 1 and int(w.pool.food[0]) == 1
    assert res.gathered[1] == 0 and int(w.pool.food[1]) == 0
    assert res.contested[1] == 1


def test_exclusion_does_not_block_a_lone_gatherer(m3):
    w = World(m3, seed=21)
    w.bush_berries[:] = m3.bushes.capacity
    w.pool.hunger[:] = m3.hunger.max
    w.pool.x[0], w.pool.z[0] = w.bush_x[0], w.bush_z[0]
    w.pool.x[1:], w.pool.z[1:] = 35.0, 0.0
    actions = np.full(m3.world.num_agents, IDLE)
    actions[0] = GATHER
    res = w.step(actions)
    assert res.gathered[0] == 1
    assert res.contested.sum() == 0


def test_a_dead_agent_cannot_block(m3):
    """A corpse next to a bush must not hold it forever."""
    w = World(m3, seed=22)
    w.bush_berries[:] = m3.bushes.capacity
    w.pool.hunger[:] = m3.hunger.max
    w.pool.x[1], w.pool.z[1] = w.bush_x[0], w.bush_z[0]   # closest, but dead
    w.pool.alive[1] = False
    w.pool.x[0] = w.bush_x[0] + m3.bushes.gather_radius * 0.5
    w.pool.z[0] = w.bush_z[0]
    w.pool.x[2:], w.pool.z[2:] = 35.0, 0.0

    actions = np.full(m3.world.num_agents, IDLE)
    actions[0] = GATHER
    res = w.step(actions)
    assert res.gathered[0] == 1


def test_exclusion_is_off_without_the_flag(m3):
    """The control: same geometry, exclusion disabled, and the further agent gets
    its berry (subject only to the same-tick rule, which targets a different bush
    here because the nearest one is claimed)."""
    lenient = m3.replace(**{"competition.exclusive_bushes": False,
                            "competition.contest_bushes": False})
    w = World(lenient, seed=20)
    w.bush_berries[:] = lenient.bushes.capacity
    w.pool.hunger[:] = lenient.hunger.max
    w.pool.x[0], w.pool.z[0] = w.bush_x[0], w.bush_z[0]
    w.pool.x[1] = w.bush_x[0] + lenient.bushes.gather_radius * 0.8
    w.pool.z[1] = w.bush_z[0]
    w.pool.x[2:], w.pool.z[2:] = 35.0, 0.0

    actions = np.full(lenient.world.num_agents, IDLE)
    actions[0] = actions[1] = GATHER
    res = w.step(actions)
    assert res.gathered[0] == 1 and res.gathered[1] == 1
    assert res.contested.sum() == 0


def test_contention_never_overdraws_a_bush(m3):
    """Three agents, one berry: the bush must not go negative."""
    w = World(m3, seed=4)
    w.bush_berries[:] = 0
    w.bush_berries[0] = 1
    _park_on_bush(w, [0, 1, 2])
    actions = np.full(m3.world.num_agents, IDLE)
    actions[:3] = GATHER
    res = w.step(actions)
    assert int(w.bush_berries[0]) == 0
    assert res.gathered.sum() == 1
    assert (w.bush_berries >= 0).all()


# --- stealing ---------------------------------------------------------------


def test_steal_moves_one_berry_between_agents(m3):
    w = World(m3, seed=5)
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1], w.pool.z[1] = 1.0, 0.0
    w.pool.food[1] = 2
    w.pool.hunger[:] = m3.hunger.max

    actions = np.full(m3.world.num_agents, IDLE)
    actions[0] = STEAL
    res = w.step(actions)

    assert int(w.pool.food[0]) == 1
    assert int(w.pool.food[1]) == 1
    assert res.stole[0] == 1 and res.robbed[1] == 1


def test_steal_pays_no_reward(m3):
    """The brief allows no reward terms beyond survival, so theft must be worth
    only the food. If this starts paying out, we are rewarding aggression."""
    w = World(m3, seed=6)
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1], w.pool.z[1] = 1.0, 0.0
    w.pool.food[1] = 1
    w.pool.hunger[:] = m3.hunger.max

    actions = np.full(m3.world.num_agents, IDLE)
    actions[0] = STEAL
    res = w.step(actions)
    assert res.stole[0] == 1
    assert res.rewards[0] == pytest.approx(m3.reward.alive_per_tick)


def test_shaped_config_pays_for_theft_and_is_clearly_an_ablation(m3):
    """The shaped config exists to answer a question, not to be the M3 result.
    If its reward ever leaks into m3.yaml, the headline number stops being
    faithful to the brief -- so both halves are pinned here."""
    shaped = load_config("config/m3_shaped.yaml")
    assert m3.reward.steal == 0.0
    assert shaped.reward.steal == m3.reward.gather
    assert shaped.competition.enable_steal is True   # inherited from m3.yaml

    w = World(shaped, seed=30)
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1], w.pool.z[1] = 1.0, 0.0
    w.pool.food[1] = 1
    w.pool.hunger[:] = shaped.hunger.max
    actions = np.full(shaped.world.num_agents, IDLE)
    actions[0] = STEAL
    res = w.step(actions)
    assert res.stole[0] == 1
    assert res.rewards[0] == pytest.approx(shaped.reward.gather + shaped.reward.alive_per_tick)


def test_steal_is_conserving(m3):
    """Total food in the world must not change: theft moves berries, never mints
    or destroys them."""
    w = World(m3.replace(**{"hunger.eat_threshold": 0.0}), seed=7)  # no eating
    w.pool.food[:] = 1
    w.pool.x[:] = 0.0
    w.pool.z[:] = 0.0
    total = int(w.pool.food.sum())
    for _ in range(20):
        w.step(np.full(m3.world.num_agents, STEAL))
        assert int(w.pool.food.sum()) == total
        assert (w.pool.food >= 0).all()


def test_steal_out_of_range_does_nothing(m3):
    w = World(m3, seed=8)
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1:], w.pool.z[1:] = 30.0, 0.0
    w.pool.food[1:] = 2
    actions = np.full(m3.world.num_agents, IDLE)
    actions[0] = STEAL
    res = w.step(actions)
    assert int(w.pool.food[0]) == 0
    assert res.stole.sum() == 0


def test_cannot_steal_from_an_empty_neighbour(m3):
    w = World(m3, seed=9)
    w.pool.x[:] = 0.0
    w.pool.z[:] = 0.0
    w.pool.food[:] = 0
    res = w.step(np.full(m3.world.num_agents, STEAL))
    assert res.stole.sum() == 0
    assert int(w.pool.food.sum()) == 0


def test_cannot_steal_from_the_dead(m3):
    w = World(m3, seed=10)
    w.pool.x[:] = 0.0
    w.pool.z[:] = 0.0
    w.pool.food[1] = 2
    w.pool.alive[1] = False
    actions = np.full(m3.world.num_agents, IDLE)
    actions[0] = STEAL
    res = w.step(actions)
    assert res.stole[0] == 0
    assert int(w.pool.food[1]) == 2


def test_full_agent_does_not_steal(m3):
    w = World(m3, seed=11)
    w.pool.x[:] = 0.0
    w.pool.z[:] = 0.0
    w.pool.hunger[:] = m3.hunger.max
    w.pool.food[0] = m3.food.capacity
    w.pool.food[1] = 1
    actions = np.full(m3.world.num_agents, IDLE)
    actions[0] = STEAL
    w.step(actions)
    assert int(w.pool.food[1]) == 1


def test_steal_action_is_inert_when_disabled(scarce):
    """Action 10 does not exist in a non-competition world; passing it must not
    steal, and must not crash either."""
    w = World(scarce, seed=12)
    w.pool.x[:] = 0.0
    w.pool.z[:] = 0.0
    w.pool.food[1] = 2
    actions = np.full(scarce.world.num_agents, IDLE)
    actions[0] = STEAL
    res = w.step(actions)
    assert int(w.pool.food[0]) == 0
    assert int(w.pool.food[1]) == 2
    assert res.stole.sum() == 0


def test_episode_stats_tally_theft_and_contests(m3):
    w = World(m3, seed=13)
    w.pool.x[:] = 0.0
    w.pool.z[:] = 0.0
    w.pool.food[:] = 1
    w.pool.hunger[:] = m3.hunger.max
    w.step(np.full(m3.world.num_agents, STEAL))
    assert w.stats().steals >= 1


# --- scripted thief ---------------------------------------------------------


def test_scripted_thief_robs_a_loaded_neighbour_in_reach(m3):
    w = World(m3, seed=14)
    w.pool.alive[:] = False
    w.pool.alive[0] = w.pool.alive[1] = True
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1], w.pool.z[1] = 1.0, 0.0
    w.pool.food[1] = 2
    assert greedy_thief_actions(w.observations(), m3)[0] == STEAL


def test_scripted_thief_ignores_an_empty_neighbour(m3):
    w = World(m3, seed=15)
    w.pool.alive[:] = False
    w.pool.alive[0] = w.pool.alive[1] = True
    w.pool.x[0], w.pool.z[0] = 0.0, 0.0
    w.pool.x[1], w.pool.z[1] = 1.0, 0.0
    w.pool.food[1] = 0
    assert greedy_thief_actions(w.observations(), m3)[0] != STEAL


def test_scripted_thief_needs_the_food_channel(scarce):
    steal_only = scarce.replace(**{"competition.enable_steal": True})
    with pytest.raises(ValueError, match="observe_neighbour_food"):
        greedy_thief_actions(np.zeros((6, observation_dim(steal_only)), dtype=np.float32),
                             steal_only)


# --- growing a policy across a milestone boundary ---------------------------


def maps_between(source_cfg, target_cfg):
    return (column_map(observation_layout(source_cfg), observation_layout(target_cfg)),
            column_map(action_names(source_cfg), action_names(target_cfg)))


def scatter(source_cfg, target_cfg, obs_old: torch.Tensor) -> torch.Tensor:
    """Place source observations into target columns *by feature name*."""
    obs_map, _ = maps_between(source_cfg, target_cfg)
    obs_new = torch.zeros(obs_old.shape[0], observation_dim(target_cfg))
    obs_new[:, obs_map] = obs_old
    return obs_new


def test_observation_layout_matches_the_real_observation(cfg, m3):
    """The layout names are what growing a policy relies on, so they must not be
    allowed to drift from build_observations."""
    for c in (cfg, m3, m3.replace(**{"competition.observe_bush_contested": True})):
        assert len(observation_layout(c)) == observation_dim(c)
        assert len(set(observation_layout(c))) == observation_dim(c)   # no duplicates


def test_milestone_boundary_actually_shifts_columns(cfg, m3):
    """Guards the premise of the next test. The optional neighbour-food channel is
    inserted *inside* the neighbour block, so widening the observation renumbers
    every column after it -- a positional copy is silently wrong."""
    obs_map, _ = maps_between(cfg, m3)
    assert obs_map != list(range(observation_dim(cfg)))
    moved = [i for i, j in enumerate(obs_map) if i != j]
    assert len(moved) >= 9
    # the shoreline features in particular must not stay where they were
    layout_old = observation_layout(cfg)
    assert obs_map[layout_old.index("edge.room")] != layout_old.index("edge.room")


def test_grown_policy_behaves_like_its_source_feature_for_feature(cfg, m3):
    """An M2 brain dropped into an M3 world must start out doing exactly what it
    used to when shown the same *features* -- not the same column indices.

    The earlier version of this test fed the old observation into the first N
    columns of the new one, which is what the buggy positional copy did, so it
    passed while the grown policy was reading its shoreline weights off a
    neighbour's carried food.
    """
    torch.manual_seed(0)
    old = ActorCritic(observation_dim(cfg), num_actions(cfg), (32,))
    obs_map, act_map = maps_between(cfg, m3)
    new = grow_actor_critic(old, observation_dim(m3), num_actions(m3), obs_map, act_map)

    obs_old = torch.randn(8, observation_dim(cfg))
    old_logits, old_value = old(obs_old)
    new_logits, new_value = new(scatter(cfg, m3, obs_old))

    assert torch.allclose(old_logits, new_logits[:, act_map], atol=1e-6)
    assert torch.allclose(old_value, new_value, atol=1e-6)


def test_grown_policy_starts_the_new_action_at_logit_zero(cfg, m3):
    """Reachable but unpreferred: PPO gets to find out if stealing is worth it."""
    old = ActorCritic(observation_dim(cfg), num_actions(cfg), (32,))
    obs_map, act_map = maps_between(cfg, m3)
    new = grow_actor_critic(old, observation_dim(m3), num_actions(m3), obs_map, act_map)
    logits, _ = new(torch.randn(4, observation_dim(m3)))
    assert torch.allclose(logits[:, STEAL], torch.zeros(4), atol=1e-6)


def test_growing_ignores_the_new_observation_channels_at_first(cfg, m3):
    old = ActorCritic(observation_dim(cfg), num_actions(cfg), (32,))
    obs_map, act_map = maps_between(cfg, m3)
    new = grow_actor_critic(old, observation_dim(m3), num_actions(m3), obs_map, act_map)

    base = torch.randn(6, observation_dim(m3))
    perturbed = base.clone()
    added = [i for i in range(observation_dim(m3)) if i not in set(obs_map)]
    assert added, "no new columns to perturb"
    perturbed[:, added] = 1.0
    assert torch.allclose(new(base)[0], new(perturbed)[0], atol=1e-6)


def test_growing_two_channels_at_once_stays_aligned(cfg):
    """M2 -> a world with both extra channels: two separate insertions, so the
    column shift compounds."""
    both = load_config("config/m3.yaml").replace(
        **{"competition.observe_bush_contested": True})
    torch.manual_seed(1)
    old = ActorCritic(observation_dim(cfg), num_actions(cfg), (24,))
    obs_map, act_map = maps_between(cfg, both)
    new = grow_actor_critic(old, observation_dim(both), num_actions(both), obs_map, act_map)

    obs_old = torch.randn(5, observation_dim(cfg))
    assert torch.allclose(old(obs_old)[1], new(scatter(cfg, both, obs_old))[1], atol=1e-6)


def test_grow_refuses_to_shrink(cfg, m3):
    big = ActorCritic(observation_dim(m3), num_actions(m3), (32,))
    with pytest.raises(ValueError, match="cannot shrink"):
        grow_actor_critic(big, observation_dim(cfg), num_actions(cfg))


def test_column_map_refuses_to_drop_a_trained_feature(cfg, m3):
    with pytest.raises(ValueError, match="missing source features"):
        column_map(observation_layout(m3), observation_layout(cfg))


def test_grow_preserves_policy_group_mode(cfg, m3):
    group = PolicyGroup([ActorCritic(observation_dim(cfg), num_actions(cfg), (16,))
                         for _ in range(3)])
    obs_map, act_map = maps_between(cfg, m3)
    grown = grow_policy(group, observation_dim(m3), num_actions(m3), obs_map, act_map)
    assert isinstance(grown, PolicyGroup)
    assert grown.num_agents == 3
    assert grown.obs_dim == observation_dim(m3)
    assert grown.n_actions == num_actions(m3)


# --- training ---------------------------------------------------------------


def test_ppo_runs_in_a_competition_world(m3):
    tiny = m3.replace(**{
        "ppo.num_envs": 4, "ppo.rollout_ticks": 24, "ppo.num_minibatches": 2,
        "ppo.epochs": 2, "ppo.total_updates": 3, "world.max_ticks": 60,
        "policy.mode": "individual",
    })
    torch.manual_seed(0)
    envs = VecWorld(tiny, seed=0, num_envs=tiny.ppo.num_envs)
    assert envs.obs_dim == observation_dim(tiny)
    trainer = PPOTrainer(tiny, build_policy(tiny, envs.obs_dim), envs, device="cpu")
    for i in range(2):
        m = trainer.train_update(i)
        assert np.isfinite(m.policy_loss) and np.isfinite(m.value_loss)
    assert trainer.policy.n_actions == 11


def test_competition_world_is_deterministic(m3):
    tiny = m3.replace(**{"world.max_ticks": 60})

    def run() -> list:
        w = World(tiny, seed=21)
        rng = np.random.default_rng(5)
        out = []
        for _ in range(60):
            res = w.step(rng.integers(0, num_actions(tiny), size=tiny.world.num_agents))
            out.append((res.rewards.copy(), res.stole.copy(), res.contested.copy()))
        return out

    a, b = run(), run()
    assert all(np.array_equal(x[i], y[i]) for x, y in zip(a, b) for i in range(3))


# --- action masking ---------------------------------------------------------


def test_mask_is_all_true_when_disabled(m3):
    """Callers never branch on the flag, so the mask must be inert by default."""
    assert m3.competition.mask_invalid_actions is False
    w = World(m3, seed=50)
    assert w.action_mask().all()
    assert w.action_mask().shape == (m3.world.num_agents, num_actions(m3))


def test_mask_hides_gather_with_no_berry_in_reach(m3):
    cfg = m3.replace(**{"competition.mask_invalid_actions": True})
    w = World(cfg, seed=51)
    w.pool.x[:], w.pool.z[:] = 35.0, 0.0        # far from every bush
    mask = w.action_mask()
    assert not mask[:, GATHER].any()
    assert mask[:, :N_MOVE_ACTIONS + 1].all()   # moving and idling always allowed

    w.pool.x[0], w.pool.z[0] = w.bush_x[0], w.bush_z[0]
    w.bush_berries[0] = cfg.bushes.capacity
    assert w.action_mask()[0, GATHER]


def test_mask_hides_gather_at_an_empty_bush_and_when_full(m3):
    cfg = m3.replace(**{"competition.mask_invalid_actions": True})
    w = World(cfg, seed=52)
    w.pool.x[0], w.pool.z[0] = w.bush_x[0], w.bush_z[0]

    w.bush_berries[:] = 0
    assert not w.action_mask()[0, GATHER], "empty bush should not offer gather"

    w.bush_berries[0] = cfg.bushes.capacity
    w.pool.food[0] = cfg.food.capacity
    assert not w.action_mask()[0, GATHER], "full inventory should not offer gather"


def test_mask_hides_steal_with_no_loaded_neighbour(m3):
    cfg = m3.replace(**{"competition.mask_invalid_actions": True})
    w = World(cfg, seed=53)
    w.pool.x[:], w.pool.z[:] = 0.0, 0.0     # all together
    w.pool.food[:] = 0
    assert not w.action_mask()[:, STEAL].any(), "nobody is carrying anything"

    w.pool.food[1] = 1
    mask = w.action_mask()
    assert mask[0, STEAL], "agent 0 can rob the loaded neighbour"
    assert not mask[1, STEAL], "agent 1 cannot rob itself"


def test_mask_never_leaves_a_dead_agent_without_an_action(m3):
    """A fully masked row makes the action distribution undefined and NaNs
    propagate silently, so every row must keep at least one action."""
    cfg = m3.replace(**{"competition.mask_invalid_actions": True})
    w = World(cfg, seed=54)
    w.pool.alive[:] = False
    mask = w.action_mask()
    assert mask.any(axis=1).all()
    assert mask[:, IDLE].all()
    assert mask.sum() == cfg.world.num_agents   # idle and nothing else


def test_masked_policy_never_emits_a_masked_action(m3):
    """End to end through the trainer: the world must never be handed an action
    its own mask forbade."""
    cfg = m3.replace(**{
        "competition.mask_invalid_actions": True,
        "ppo.num_envs": 3, "ppo.rollout_ticks": 40, "ppo.num_minibatches": 1,
        "ppo.epochs": 1, "world.max_ticks": 80, "policy.mode": "individual",
    })
    torch.manual_seed(0)
    envs = VecWorld(cfg, seed=0, num_envs=cfg.ppo.num_envs)
    trainer = PPOTrainer(cfg, build_policy(cfg, envs.obs_dim), envs, device="cpu")

    sent: list = []
    original = envs.step
    envs.step = lambda a: (sent.append(a.copy()), original(a))[1]
    rollout = trainer.collect()

    masks = rollout.masks.numpy()
    active = rollout.active.numpy()
    violations = 0
    for t, actions in enumerate(sent):
        for e in range(cfg.ppo.num_envs):
            for i in range(cfg.world.num_agents):
                if active[t, e, i] and not masks[t, e, i, actions[e, i]]:
                    violations += 1
    assert violations == 0
    assert not masks.all(), "masking never actually bit; the test proves nothing"


def test_masking_removes_doomed_attempts_that_go_unmasked(m3):
    """The point of the mask, measured: under it a uniform-random policy cannot
    spend a tick on gather/steal that could not succeed, and without it a third
    of such attempts are doomed. This is the 30% the learned policy was wasting."""
    def doomed_share(mask_on: bool) -> float:
        cfg = m3.replace(**{"competition.mask_invalid_actions": mask_on,
                            "world.max_ticks": 250})
        rng = np.random.default_rng(0)
        attempts = doomed = 0
        for s in range(4):
            w = World(cfg, seed=s)
            for _ in range(cfg.world.max_ticks):
                offered = w.action_mask()
                truth = action_mask(w.pool, w.bush_x, w.bush_z, w.bush_berries, cfg)
                a = np.array([rng.choice(np.flatnonzero(offered[i])) for i in range(w.pool.n)])
                for i in np.flatnonzero(w.pool.alive):
                    if a[i] in (GATHER, STEAL):
                        attempts += 1
                        doomed += int(not truth[i, a[i]])
                if w.step(a).episode_done:
                    break
        return doomed / max(attempts, 1)

    assert doomed_share(True) == pytest.approx(0.0, abs=1e-9)
    assert doomed_share(False) > 0.3


def test_equal_width_layout_change_fails_loudly_instead_of_misaligning(cfg):
    """The silent case a width check misses, and what should happen instead.

    Two configs can carry the same NUMBER of observation columns holding different
    features: 3 bushes with a `blocked` channel is 26 dims, and so is 4 bushes
    without it. A grow path that triggers on width would pass a checkpoint straight
    through and read every trained weight off the wrong feature, printing nothing.

    Note what the right answer is here, because it is not "remap it": with the layout
    built in a fixed order (own, bushes, neighbours, construction, night, edge), equal
    width plus a different layout means a feature was *swapped*, so some trained
    input has no home in the target. `column_map` refuses, and refusing is correct --
    the point of keying the check off the layout rather than the width is to turn a
    silent misalignment into an error.
    """
    from sim.agents import observation_dim, observation_layout
    from sim.policy import column_map

    a = cfg.replace(**{"competition.observe_bush_contested": True,
                       "observation.k_bushes": 3})
    b = cfg.replace(**{"observation.k_bushes": 4})
    assert observation_dim(a) == observation_dim(b) == 26, (
        "fixture no longer exercises the equal-width case; pick other channels")
    assert observation_layout(a) != observation_layout(b)

    for src, dst in ((a, b), (b, a)):
        with pytest.raises(ValueError, match="missing source features"):
            column_map(observation_layout(src), observation_layout(dst))
