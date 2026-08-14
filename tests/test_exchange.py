"""Milestone 5: transfers between agents, and the ledger they leave behind."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from sim.agents import (
    BUILD,
    GATHER,
    GIVE_FOOD,
    GIVE_MATERIAL,
    IDLE,
    ITEM_FOOD,
    ITEM_STONE,
    ITEM_WOOD,
    STEAL,
    action_names,
    num_actions,
    observation_dim,
    observation_layout,
)
from sim.config import load_config
from sim.exchange import analyse
from sim.evaluate import make_act_fn
from sim.policy import ActorCritic, column_map, grow_actor_critic
from sim.ppo import PPOTrainer
from sim.replay import load_replay, record_episode
from sim.world import VecWorld, World


@pytest.fixture
def m5():
    return load_config("config/m5.yaml")


def idle_all(cfg):
    return np.full(cfg.world.num_agents, IDLE)


def park(w, i, x, z, hunger=None):
    w.pool.x[i], w.pool.z[i] = x, z
    w.pool.hunger[i] = hunger if hunger is not None else w.cfg.hunger.max


def line_up(w, spacing=1.0):
    """Put every agent on a row, one unit apart, all fed."""
    for i in range(w.pool.n):
        park(w, i, 0.0, i * spacing)


# --- space and layout ---------------------------------------------------------


def test_exchange_is_off_everywhere_before_m5():
    for name in ("config/default.yaml", "config/scarce.yaml", "config/m3.yaml",
                 "config/m4.yaml", "config/m4c.yaml", "config/m4c_anneal.yaml"):
        c = load_config(name)
        assert c.exchange.enabled is False
        assert "give_food" not in action_names(c)


def test_give_actions_are_appended_after_build(m5):
    names = action_names(m5)
    assert names[STEAL] == "steal" and names[BUILD] == "build"
    assert names[GIVE_FOOD] == "give_food"
    assert names[GIVE_MATERIAL] == "give_material"
    assert num_actions(m5) == 16


def test_construction_slots_are_kept_even_without_construction(m5):
    """Exchange implies the whole construction action block, inert or not, so the
    give indices cannot shift out from under an earlier checkpoint."""
    no_build = m5.replace(**{"construction.enabled": False})
    names = action_names(no_build)
    assert names[GIVE_FOOD] == "give_food" and num_actions(no_build) == 16


def test_observation_layout_matches_dim_and_is_unique(m5):
    layout = observation_layout(m5)
    assert len(layout) == observation_dim(m5) == 61
    assert len(set(layout)) == len(layout)
    assert "neighbour0.wood" in layout and "neighbour0.stone" in layout


def test_neighbour_material_channels_report_what_the_neighbour_carries(m5):
    w = World(m5, seed=1)
    line_up(w)
    w.pool.wood[1] = 2
    w.pool.stone[1] = 0
    obs = w.observations()
    col = {name: i for i, name in enumerate(observation_layout(m5))}
    cap = m5.construction.material_capacity
    # agent 0's nearest neighbour is agent 1
    assert obs[0, col["neighbour0.wood"]] == pytest.approx(2 / cap)
    assert obs[0, col["neighbour0.stone"]] == pytest.approx(0.0)


def test_observation_stays_in_bounds_with_the_new_channels(m5):
    w = World(m5, seed=2)
    for _ in range(40):
        w.step(np.random.default_rng(0).integers(0, num_actions(m5), size=m5.world.num_agents))
    obs = w.observations()
    assert obs.shape == (m5.world.num_agents, observation_dim(m5))
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)


def test_m4_checkpoint_grows_into_m5_by_feature(m5):
    m4 = load_config("config/m4c_anneal.yaml")
    torch.manual_seed(0)
    old = ActorCritic(observation_dim(m4), num_actions(m4), (24,))
    obs_map = column_map(observation_layout(m4), observation_layout(m5))
    act_map = column_map(action_names(m4), action_names(m5))
    new = grow_actor_critic(old, observation_dim(m5), num_actions(m5), obs_map, act_map)

    obs_old = torch.randn(6, observation_dim(m4))
    obs_new = torch.zeros(6, observation_dim(m5))
    obs_new[:, obs_map] = obs_old
    assert torch.allclose(old(obs_old)[0], new(obs_new)[0][:, act_map], atol=1e-6)
    for a in (GIVE_FOOD, GIVE_MATERIAL):
        assert torch.allclose(new(obs_new)[0][:, a], torch.zeros(6), atol=1e-6)


# --- the transfer itself -------------------------------------------------------


def test_give_food_moves_one_unit_to_the_nearest_neighbour(m5):
    w = World(m5, seed=3)
    line_up(w)
    w.pool.food[0] = 2
    a = idle_all(m5); a[0] = GIVE_FOOD
    res = w.step(a)
    assert int(w.pool.food[0]) == 1
    assert int(w.pool.food[1]) == 1
    assert res.transfers == [(0, 1, ITEM_FOOD)]
    assert res.gave[0] == 1 and res.received[1] == 1


def test_give_material_prefers_wood_then_falls_back_to_stone(m5):
    w = World(m5, seed=4)
    line_up(w)
    w.pool.wood[0], w.pool.stone[0] = 1, 1
    a = idle_all(m5); a[0] = GIVE_MATERIAL
    assert w.step(a).transfers == [(0, 1, ITEM_WOOD)]
    assert w.step(a).transfers == [(0, 1, ITEM_STONE)]
    assert int(w.pool.wood[1]) == 1 and int(w.pool.stone[1]) == 1


def test_giving_nothing_is_a_wasted_tick_not_a_free_unit(m5):
    w = World(m5, seed=5)
    line_up(w)
    a = idle_all(m5); a[0] = GIVE_FOOD
    res = w.step(a)
    assert res.transfers == []
    assert int(w.pool.food.sum()) == 0


def test_a_gift_needs_someone_in_range(m5):
    w = World(m5, seed=6)
    line_up(w, spacing=m5.exchange.give_radius * 4)
    w.pool.food[0] = 1
    a = idle_all(m5); a[0] = GIVE_FOOD
    assert w.step(a).transfers == []
    assert int(w.pool.food[0]) == 1


def test_a_full_neighbour_is_skipped_for_one_with_room(m5):
    w = World(m5, seed=7)
    line_up(w)
    w.pool.food[0] = 1
    w.pool.food[1] = m5.food.capacity        # nearest, but full
    a = idle_all(m5); a[0] = GIVE_FOOD
    assert w.step(a).transfers == [(0, 2, ITEM_FOOD)]


def test_the_dead_cannot_receive(m5):
    w = World(m5, seed=8)
    line_up(w)
    w.pool.food[0] = 1
    w.pool.alive[1] = False
    a = idle_all(m5); a[0] = GIVE_FOOD
    assert w.step(a).transfers == [(0, 2, ITEM_FOOD)]
    assert int(w.pool.food[1]) == 0


def test_nobody_gives_to_themselves(m5):
    w = World(m5, seed=9)
    for i in range(w.pool.n):
        park(w, i, 0.0, 0.0)          # everyone on the same spot
    w.pool.food[0] = 1
    res = w.step(np.array([GIVE_FOOD] + [IDLE] * (m5.world.num_agents - 1)))
    assert res.transfers and res.transfers[0][1] != 0


def test_a_gift_arrives_before_the_drain_so_it_can_save_a_life(m5):
    """Same rule as gathering: the transfer resolves in phase 1, so a berry
    handed over on the receiver's last tick is still eaten in time."""
    w = World(m5, seed=10)
    line_up(w)
    w.pool.food[0] = 1
    w.pool.hunger[1] = m5.hunger.drain_per_tick / 2   # dies this tick without help
    a = idle_all(m5); a[0] = GIVE_FOOD
    res = w.step(a)
    assert not res.terminated[1]
    assert bool(w.pool.alive[1])
    assert res.ate[1] == 1


def test_giving_pays_nothing_by_default(m5):
    """The brief-faithful default, and the same call as reward.steal. If this
    ever starts paying out, that is a deliberate ablation, not a tweak."""
    assert m5.reward.give == 0.0
    w = World(m5, seed=11)
    line_up(w)
    w.pool.food[0] = 1
    a = idle_all(m5); a[0] = GIVE_FOOD
    res = w.step(a)
    assert res.transfers
    assert res.rewards[0] == pytest.approx(m5.reward.alive_per_tick)


def test_the_shaped_ablation_pays_a_gather_for_a_gift():
    shaped = load_config("config/m5_shaped.yaml")
    assert shaped.reward.give == shaped.reward.gather
    assert shaped.exchange.enabled is True     # inherited from m5.yaml
    w = World(shaped, seed=12)
    line_up(w)
    w.pool.food[0] = 1
    a = idle_all(shaped); a[0] = GIVE_FOOD
    res = w.step(a)
    assert res.rewards[0] == pytest.approx(shaped.reward.gather + shaped.reward.alive_per_tick)


def test_transfers_conserve_units(m5):
    """Exchange moves things; it must never create or destroy one."""
    rng = np.random.default_rng(0)
    w = World(m5, seed=13)
    line_up(w)
    w.pool.food[:] = 1
    w.pool.wood[:] = 1
    before = int(w.pool.food.sum() + w.pool.wood.sum() + w.pool.stone.sum())
    eaten = 0
    for _ in range(30):
        a = rng.choice([GIVE_FOOD, GIVE_MATERIAL, IDLE], size=w.pool.n)
        res = w.step(a)
        eaten += int(res.ate.sum())
    after = int(w.pool.food.sum() + w.pool.wood.sum() + w.pool.stone.sum())
    assert after == before - eaten


# --- masking -------------------------------------------------------------------


def test_give_is_masked_out_when_it_cannot_do_anything(m5):
    assert m5.competition.mask_invalid_actions is True   # inherited from m3_masked
    w = World(m5, seed=14)
    line_up(w)
    mask = w.action_mask()
    assert not mask[:, GIVE_FOOD].any()          # nobody is carrying anything
    assert not mask[:, GIVE_MATERIAL].any()

    w.pool.food[0] = 1
    w.pool.stone[0] = 1
    mask = w.action_mask()
    assert mask[0, GIVE_FOOD] and mask[0, GIVE_MATERIAL]
    assert not mask[1, GIVE_FOOD]


def test_a_masked_in_give_succeeds(m5):
    """The mask promises reachability, so anything it allows must resolve. A mask
    that says yes to a give that then does nothing is the doomed-action bug M3
    spent seven interventions on.

    Checked on ticks with a single giver: the mask is computed before anyone
    acts, so two givers can still race for one free slot (below)."""
    rng = np.random.default_rng(1)
    w = World(m5, seed=15)
    checked = 0
    for _ in range(300):
        mask = w.action_mask()
        actions = np.array([rng.choice(np.flatnonzero(row)) for row in mask])
        givers = np.flatnonzero((actions == GIVE_FOOD) | (actions == GIVE_MATERIAL))
        res = w.step(actions)
        if givers.size == 1:
            checked += 1
            assert res.gave[givers[0]] == 1, "a give the mask allowed did nothing"
        if res.episode_done:
            break
    assert checked > 0, "the rollout never produced a lone giver to check"


def test_a_thief_can_take_the_berry_you_were_about_to_give(m5):
    """Theft resolves in phase 1c and gifts in 1e, so a robbery lands first and
    the gift simply does not happen. Not a mask bug -- it is 75% of every failed
    give the learned policy makes, and it is the correct reading of the tick
    order: you cannot hand over what was just taken from you."""
    w = World(m5, seed=20)
    line_up(w)
    w.pool.food[0] = 1
    w.pool.food[1] = 0
    a = idle_all(m5); a[0] = GIVE_FOOD; a[1] = STEAL
    res = w.step(a)
    assert res.stole[1] == 1 and res.robbed[0] == 1
    assert res.gave[0] == 0 and res.transfers == []
    assert int(w.pool.food[0]) == 0 and int(w.pool.food[1]) == 1


def test_two_givers_can_race_for_the_last_free_slot(m5):
    """A known and accepted limit, shared with gather under contest_bushes: the
    mask is a start-of-tick promise, and transfers resolve in agent order. The
    second giver of the tick can find the slot it was promised already taken."""
    w = World(m5, seed=15)
    for i in range(w.pool.n):
        park(w, i, 0.0, 100.0 + i)                     # everyone else out of reach
    park(w, 0, 0.0, 0.0)
    park(w, 1, 0.0, 1.5)
    park(w, 2, 0.0, 3.0)                               # in reach of 1, not of 0
    w.pool.food[0] = w.pool.food[2] = 1
    w.pool.food[1] = m5.food.capacity - 1              # room for exactly one
    mask = w.action_mask()
    assert mask[0, GIVE_FOOD] and mask[2, GIVE_FOOD]   # both are promised a taker

    a = idle_all(m5); a[0] = GIVE_FOOD; a[2] = GIVE_FOOD
    res = w.step(a)
    assert res.transfers == [(0, 1, ITEM_FOOD)]
    assert res.gave[2] == 0 and int(w.pool.food[2]) == 1


def test_a_dead_agent_keeps_only_idle(m5):
    w = World(m5, seed=16)
    w.pool.alive[2] = False
    mask = w.action_mask()
    assert mask[2, IDLE] and mask[2].sum() == 1


# --- the ledger ----------------------------------------------------------------


def test_the_ledger_is_off_by_default_and_on_when_asked(m5):
    assert m5.exchange.log_transfers is False
    quiet = World(m5, seed=17)
    line_up(quiet)
    quiet.pool.food[0] = 1
    a = idle_all(m5); a[0] = GIVE_FOOD
    quiet.step(a)
    assert quiet.stats().gifts == 1
    assert quiet.stats().transfers == []       # counted, not recorded

    loud = World(m5.replace(**{"exchange.log_transfers": True}), seed=17)
    line_up(loud)
    loud.pool.food[0] = 1
    loud.step(a)
    assert loud.stats().transfers == [(0, 0, 1, ITEM_FOOD)]   # tick 0, 0 -> 1, food


def test_the_ledger_is_cleared_between_episodes(m5):
    w = World(m5.replace(**{"exchange.log_transfers": True}), seed=18)
    line_up(w)
    w.pool.food[0] = 1
    a = idle_all(m5); a[0] = GIVE_FOOD
    w.step(a)
    assert w.stats().transfers
    w.reset()
    assert w.stats().transfers == []
    assert w.stats().gifts == 0


def test_stats_split_food_from_materials(m5):
    w = World(m5, seed=19)
    line_up(w)
    w.pool.food[0] = 1
    w.pool.wood[0] = 1
    a = idle_all(m5); a[0] = GIVE_FOOD
    w.step(a)
    a = idle_all(m5); a[0] = GIVE_MATERIAL
    w.step(a)
    stats = w.stats()
    assert (stats.gifts, stats.food_given, stats.materials_given) == (2, 1, 1)


def test_the_ledger_is_deterministic(m5):
    cfg = m5.replace(**{"exchange.log_transfers": True})
    act = make_act_fn("trader", cfg, None, 10_000)
    ledgers = []
    for _ in range(2):
        w = World(cfg, seed=99)
        obs = w.observations()
        for _ in range(200):
            res = w.step(act(obs, w.action_mask()))
            obs = res.obs
            if res.episode_done:
                break
        ledgers.append(w.stats().transfers)
    assert ledgers[0] == ledgers[1]
    assert ledgers[0], "the scripted trader should transfer something in 200 ticks"


# --- replay schema v3 ----------------------------------------------------------


def test_replay_records_transfers_and_bumps_the_schema(tmp_path, m5):
    act = make_act_fn("trader", m5, None, 10_000)
    rec = record_episode(m5, seed=10_000, act_fn=act, label="t", source="scripted")
    path = rec.save(tmp_path / "m5.json", update_manifest_file=False)
    data = load_replay(path)

    assert data["schema_version"] == 3
    assert data["world"]["give_radius"] == m5.exchange.give_radius
    assert data["item_names"] == ["food", "wood", "stone"]
    assert data["summary"]["gifts"] == data["summary"]["food_given"] + data["summary"]["materials_given"]

    listed = sum(len(tick["g"]) for tick in data["ticks"] if "g" in tick)
    assert listed == data["summary"]["gifts"] > 0
    assert "g" not in data["ticks"][0], "the pre-action snapshot cannot contain a transfer"
    for tick in data["ticks"]:
        for giver, receiver, item in tick.get("g", []):
            assert giver != receiver
            assert 0 <= item < 3


def test_a_world_without_exchange_still_writes_the_old_schema(tmp_path):
    m4 = load_config("config/m4c.yaml")
    act = make_act_fn("builder", m4, None, 10_000)
    rec = record_episode(m4, seed=10_000, act_fn=act, label="t", source="scripted",
                         max_ticks=20)
    data = load_replay(rec.save(tmp_path / "m4.json", update_manifest_file=False))
    assert data["schema_version"] == 2
    assert all("g" not in tick for tick in data["ticks"])
    assert "gifts" not in data["summary"]


# --- the analysis tool ---------------------------------------------------------


def test_analyse_accounts_for_every_transfer(m5):
    act = make_act_fn("trader", m5, None, 10_000)
    report, records = analyse(m5, act, episodes=2, seed=10_000)

    flow = np.array(report["flow"])
    assert report["totals"]["transfers"] == flow.sum() == len(records)
    assert flow.trace() == 0, "a self-transfer would be a bug in the world"
    given = sum(a["given"] for a in report["agents"])
    received = sum(a["received"] for a in report["agents"])
    assert given == received == flow.sum()
    assert report["totals"]["food"] + report["totals"]["material"] == flow.sum()
    assert 0.0 <= report["reciprocity"] <= 1.0
    assert set(records[0]) == {"episode", "tick", "giver", "receiver", "item"}
    json.dumps(report)   # must survive the trip to the viewer


def test_analyse_refuses_a_world_with_no_exchange():
    with pytest.raises(ValueError, match="exchange disabled"):
        analyse(load_config("config/m4c.yaml"), lambda o, m=None: None, 1, 0)


def test_the_scripted_trader_never_wastes_a_give(m5):
    """It reads the same mask the learned policy does, so a give it chooses must
    land -- and the recipient is the world's choice, not its own."""
    act = make_act_fn("trader", m5, None, 10_000)
    w = World(m5, seed=10_000)
    obs = w.observations()
    gifts = 0
    for _ in range(300):
        actions = act(obs, w.action_mask())
        res = w.step(actions)
        for i in np.flatnonzero((actions == GIVE_FOOD) | (actions == GIVE_MATERIAL)):
            assert res.gave[i] == 1
            gifts += 1
        obs = res.obs
        if res.episode_done:
            break
    assert gifts > 0


# --- training ------------------------------------------------------------------


def test_ppo_runs_an_update_in_an_exchange_world(m5):
    cfg = m5.replace(**{"ppo.num_envs": 2, "ppo.rollout_ticks": 16,
                        "ppo.num_minibatches": 2, "ppo.epochs": 1})
    envs = VecWorld(cfg, seed=0, num_envs=2)
    from sim.policy import build_policy
    policy = build_policy(cfg, envs.obs_dim, mode="shared")
    assert policy.n_actions == 16
    trainer = PPOTrainer(cfg, policy, envs, device="cpu")
    metrics = trainer.train_update(0)
    assert np.isfinite(metrics.policy_loss) and np.isfinite(metrics.value_loss)
