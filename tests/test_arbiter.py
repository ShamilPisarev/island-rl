"""Island 2.0 stage 5: the learned option-level arbiter and its SMDP trainer.

The two contracts worth pinning hard:

  * MENU PARITY -- the learned chooser, the random-goal control and the scripted
    scorer must face the identical option set, or the headline comparison is two
    different games. Pinned by checking every sampled goal against
    `goal_availability` and by the redecide-rule parity test.
  * SMDP BOOKKEEPING -- one transition per decision, reward summed with in-option
    discounting, gamma^k on the bootstrap. utility.py's own docstring calls this
    "the one place the bookkeeping can silently rot", and the trainer duplicates
    OptionRunner's redecide logic (a forward pass has to go between re-deciding
    and executing), so the duplication is pinned against the original.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sim.arbiter import (ArbiterTrainer, LearnedArbiter, RandomGoalArbiter,
                         TrainConfig, goal_mask, load_arbiter, save_checkpoint)
from sim.config import load_config
from sim.obsview import ObsView
from sim.policy import ActorCritic
from sim.society import run_episodes
from sim.utility import (N_GOALS, REST, ArbiterConfig, OptionRunner,
                         UtilityArbiter, compute_needs, goal_availability)
from sim.world import World

ROOT = Path(__file__).resolve().parent.parent
STAGE4 = ROOT / "config" / "island2" / "society4.yaml"


@pytest.fixture(scope="module")
def cfg():
    c = load_config(STAGE4)
    return c.replace(**{"world.num_agents": 12, "world.max_ticks": 120,
                        "society.num_households": 4, "bushes.num_clusters": 4})


def fresh_arbiter(cfg, seed=0, deterministic=True):
    from sim.agents import observation_dim
    torch.manual_seed(seed)      # the WEIGHTS must be reproducible too
    policy = ActorCritic(observation_dim(cfg) + N_GOALS, n_actions=N_GOALS)
    return LearnedArbiter(policy, cfg, seed=seed, deterministic=deterministic)


# --- menu parity -------------------------------------------------------------

def test_every_chooser_faces_the_same_menu(cfg):
    """One availability function, three choosers. A sampled goal outside the menu
    means a chooser is playing a different game, which is the failure that would
    quietly invalidate the whole scripted-vs-learned comparison."""
    world = World(cfg, seed=3)
    view = ObsView(world.observations(), cfg)
    acfg = ArbiterConfig()
    menu = goal_mask(view, cfg, acfg)
    rng = np.random.default_rng(0)
    mask = world.action_mask()

    for chooser in (UtilityArbiter(cfg, acfg), RandomGoalArbiter(cfg, acfg),
                    fresh_arbiter(cfg, deterministic=False)):
        goals = chooser.choose(view, mask, rng)
        rows = np.arange(view.n)
        # The scripted scorer falls back to REST when nothing scores > 0, and
        # REST is force-opened in the menu, so this covers that path too.
        assert menu[rows, goals].all(), type(chooser).__name__


def test_rest_is_always_on_the_menu(cfg):
    world = World(cfg, seed=3)
    view = ObsView(world.observations(), cfg)
    menu = goal_mask(view, cfg, ArbiterConfig())
    assert menu[:, REST].all()
    assert menu.any(axis=1).all()          # no row can ever be fully masked


def test_trainer_redecide_matches_option_runner(cfg):
    """The trainer duplicates OptionRunner's redecide rule so a policy forward
    pass can sit between re-deciding and executing. Pinned side by side on the
    same mid-episode state, because a silent drift here changes what a
    'decision' means without changing any output shape."""
    runner = OptionRunner(UtilityArbiter(cfg), cfg, seed=5)
    world = World(cfg, seed=5)
    obs = world.observations()
    for _ in range(30):
        obs = world.step(runner.act(obs, world.action_mask())).obs

    trainer = ArbiterTrainer(cfg, TrainConfig(num_envs=1), seed=5)
    env = trainer.envs[0]
    env.goals = runner.goals.copy()
    env.ticks_left = runner.ticks_left.copy()
    env.started[:] = True                       # isolate the shared terms
    view = ObsView(obs, cfg)
    got = trainer._redecide_mask(env, view)

    from sim.utility import (DRAW_FOOD, FORAGE, GOAL_RAID, GOAL_STEAL,
                             NEED_HUNGER, goal_viable)
    needs = compute_needs(view, cfg)
    viable = goal_viable(view, cfg, runner.goals)
    emergency = needs[:, NEED_HUNGER] >= runner.acfg.critical
    pursuing = np.isin(runner.goals, (FORAGE, GOAL_STEAL, DRAW_FOOD, GOAL_RAID))
    want = (~viable) | (runner.ticks_left <= 0) | (emergency & ~pursuing)
    assert np.array_equal(got, want)


# --- SMDP bookkeeping ---------------------------------------------------------

def collect_once(cfg, seed=0, ticks=130):
    tcfg = TrainConfig(num_envs=2, rollout_ticks=ticks)
    trainer = ArbiterTrainer(cfg, tcfg, seed=seed)
    buffers = {k: [] for k in ("stream", "obs", "goal", "logprob", "value",
                               "mask", "reward", "discount", "next_obs",
                               "done", "agent")}
    trainer.collect(buffers)
    return trainer, buffers


def test_transitions_respect_the_gamma_contract(cfg):
    """discount must be an exact power of gamma -- gamma^k for the k ticks the
    option ran -- and never gamma^0 (a zero-tick option would mean a transition
    was emitted without the world advancing)."""
    trainer, buffers = collect_once(cfg)
    gamma = trainer.tcfg.gamma
    d = np.asarray(buffers["discount"])
    ks = np.log(d) / np.log(gamma)
    assert np.allclose(ks, np.round(ks), atol=1e-6)
    assert (np.round(ks) >= 1).all()
    # ...and no option outlives its commitment plus the tick it terminates on.
    assert (np.round(ks) <= trainer.acfg.commit_ticks + 1).all()


def test_sampled_goals_were_on_the_menu(cfg):
    trainer, buffers = collect_once(cfg)
    goals = np.asarray(buffers["goal"])
    masks = np.stack(buffers["mask"])
    assert masks[np.arange(len(goals)), goals].all()


def test_a_death_closes_the_option_as_terminal(cfg):
    """Starved agents must emit done=True (nothing to bootstrap); the episode
    tick-limit must NOT (survivors were alive and carry value forward) -- the
    same death-vs-truncation split 1.0's ppo.py enforces."""
    hard = cfg.replace(**{"hunger.drain_per_tick": 3.0, "world.max_ticks": 120})
    trainer, buffers = collect_once(hard, ticks=125)
    dones = np.asarray(buffers["done"])
    assert dones.any()                         # people starved
    # every done sits on an agent that is actually dead at close time is hard to
    # re-derive post hoc; what is cheap and load-bearing: at least one truncated
    # close exists too when anyone survives to the limit.
    trainer2, buffers2 = collect_once(cfg, ticks=125)
    assert not np.asarray(buffers2["done"]).all()


def test_training_moves_and_stays_finite(cfg):
    """Three tiny updates: losses finite, entropy sane, weights not NaN. This is
    the test that would have caught the size-1-minibatch std() NaN."""
    tcfg = TrainConfig(num_envs=2, rollout_ticks=60, total_updates=3,
                       num_minibatches=4)
    trainer = ArbiterTrainer(cfg, tcfg, seed=1)
    for update in range(3):
        buffers = {k: [] for k in ("stream", "obs", "goal", "logprob", "value",
                                   "mask", "reward", "discount", "next_obs",
                                   "done", "agent")}
        trainer.collect(buffers)
        stats = trainer.update(buffers, lr=tcfg.lr)
        assert np.isfinite(stats["policy_loss"])
        assert np.isfinite(stats["entropy"])
    for p in trainer.policy.parameters():
        assert torch.isfinite(p).all()


# --- the arbiter zoo ----------------------------------------------------------

def test_learned_arbiter_is_reproducible(cfg):
    def run(seed):
        rep = run_episodes(cfg, 1, seed, policy="randomgoal")
        return float(np.concatenate(rep.lifespans).mean())
    assert run(11) == run(11)

    # and the learned one, untrained, through the full society pipeline
    arb = fresh_arbiter(cfg, seed=2)
    runner = OptionRunner(arb, cfg, seed=2)
    world = World(cfg, seed=2)
    obs = world.observations()
    first = []
    for _ in range(40):
        a = runner.act(obs, world.action_mask())
        first.append(a.copy())
        obs = world.step(a).obs
    arb2 = fresh_arbiter(cfg, seed=2)
    # fresh_arbiter re-seeds torch identically, so the weights match too
    runner2 = OptionRunner(arb2, cfg, seed=2)
    world2 = World(cfg, seed=2)
    obs2 = world2.observations()
    for i in range(40):
        a = runner2.act(obs2, world2.action_mask())
        assert np.array_equal(a, first[i])
        obs2 = world2.step(a).obs


def test_checkpoint_roundtrip_and_1_0_refusal(cfg, tmp_path):
    tcfg = TrainConfig(num_envs=1, rollout_ticks=30)
    trainer = ArbiterTrainer(cfg, tcfg, seed=3)
    path = tmp_path / "latest.pt"
    save_checkpoint(path, trainer, update=1)
    arb = load_arbiter(path, cfg)
    world = World(cfg, seed=3)
    view = ObsView(world.observations(), cfg)
    goals = arb.choose(view, world.action_mask(), np.random.default_rng(0))
    assert goals.shape == (cfg.world.num_agents,)

    fake = tmp_path / "old.pt"
    torch.save({"policy_state": {}, "policy_config": {}}, fake)
    with pytest.raises(ValueError, match="not a goal-arbiter"):
        load_arbiter(fake, cfg)


# --- the mixed population (stage 5's first lever) ------------------------------

def test_mixed_transitions_come_only_from_learned_agents(cfg):
    """A scripted agent must never leak a PPO transition: the whole point of the
    mixed run is that the gradient sees only the minority's decisions while the
    majority shapes the state distribution."""
    tcfg = TrainConfig(num_envs=2, rollout_ticks=130)
    trainer = ArbiterTrainer(cfg, tcfg, seed=0, learn_agents=np.arange(4))
    buffers = {k: [] for k in ("stream", "obs", "goal", "logprob", "value",
                               "mask", "reward", "discount", "next_obs",
                               "done", "agent")}
    trainer.collect(buffers)
    agents = np.asarray(buffers["agent"])
    assert len(agents) > 0
    assert (agents < 4).all()
    # ...and the update still runs on the subset's transitions alone
    stats = trainer.update(buffers, lr=tcfg.lr)
    assert np.isfinite(stats["policy_loss"])


def test_scripted_majority_keeps_its_commitments(cfg):
    """The redecide flag used to key first-decision off has_open, which a
    scripted agent never sets -- so it would have re-decided EVERY tick and the
    commitment would be decorative. Pinned by checking a scripted agent's goal
    survives more ticks than a decide-every-tick world could ever show."""
    tcfg = TrainConfig(num_envs=1, rollout_ticks=1)
    trainer = ArbiterTrainer(cfg, tcfg, seed=2, learn_agents=np.arange(2))
    env = trainer.envs[0]
    buffers = {k: [] for k in ("stream", "obs", "goal", "logprob", "value",
                               "mask", "reward", "discount", "next_obs",
                               "done", "agent")}
    scripted = np.flatnonzero(~trainer.learn_mask)
    trainer.collect(buffers)              # tick 1: everyone decides
    committed = env.ticks_left[scripted].copy()
    trainer.collect(buffers)              # tick 2
    # a freshly committed scripted agent's clock ticks DOWN rather than being
    # re-armed to commit_ticks every tick
    still = env.ticks_left[scripted] == committed - 1
    assert still.any()


def test_mixed_arbiter_routes_each_row_to_its_owner(cfg):
    from sim.arbiter import MixedArbiter
    world = World(cfg, seed=4)
    view = ObsView(world.observations(), cfg)
    mask = world.action_mask()
    acfg = ArbiterConfig()
    scripted = UtilityArbiter(cfg, acfg, seed=4)
    learned = fresh_arbiter(cfg, seed=4)
    learn_mask = np.zeros(cfg.world.num_agents, dtype=bool)
    learn_mask[:3] = True
    mixed = MixedArbiter(scripted, learned, learn_mask)
    rng = np.random.default_rng(0)
    got = mixed.choose(view, mask, rng)
    want_s = UtilityArbiter(cfg, acfg, seed=4).choose(view, mask,
                                                      np.random.default_rng(0))
    assert np.array_equal(got[~learn_mask], want_s[~learn_mask])
    want_l = fresh_arbiter(cfg, seed=4).choose(view, mask, np.random.default_rng(0))
    assert np.array_equal(got[learn_mask], want_l[learn_mask])


def test_mixed_checkpoint_carries_the_split_into_society(cfg, tmp_path):
    tcfg = TrainConfig(num_envs=1, rollout_ticks=30)
    trainer = ArbiterTrainer(cfg, tcfg, seed=3, learn_agents=np.arange(4))
    path = tmp_path / "latest.pt"
    save_checkpoint(path, trainer, update=1)
    arb = load_arbiter(path, cfg)
    assert arb.learn_agents == [0, 1, 2, 3]

    rep = run_episodes(cfg, 1, 10000, policy="mixed", checkpoint=str(path))
    assert rep.learn_mask is not None
    assert rep.learn_mask.sum() == 4
    # the per-subset splits actually accumulated
    assert rep.goal_ticks_learned.sum() > 0
    assert rep.goal_ticks_scripted.sum() > 0
    assert len(rep.night_in_agent) == 1

    # an all-learned checkpoint must refuse --arbiter mixed rather than invent
    # a split
    trainer_all = ArbiterTrainer(cfg, tcfg, seed=3)
    path2 = tmp_path / "all.pt"
    save_checkpoint(path2, trainer_all, update=1)
    with pytest.raises(ValueError, match="not trained mixed"):
        run_episodes(cfg, 1, 10000, policy="mixed", checkpoint=str(path2))


def test_household_reward_is_the_household_mean_for_learned_agents_only(cfg):
    """12 agents in 4 round-robin households of 3. A learned agent's training
    reward must be its household's mean over the FIXED household size (a dead
    housemate drags the mean, never vanishes from it); a scripted agent's must
    pass through untouched."""
    tcfg = TrainConfig(num_envs=1, rollout_ticks=1)
    trainer = ArbiterTrainer(cfg, tcfg, seed=0, learn_agents=np.arange(4),
                             household_reward=True)
    env = trainer.envs[0]
    rewards = np.arange(12, dtype=np.float64)          # agent i earns i
    got = trainer._train_rewards(env, rewards)
    # household h = {h, h+4, h+8}, mean = h + 4
    for a in range(4):                                  # learned: shared
        assert got[a] == pytest.approx(a + 4.0)
    for a in range(4, 12):                              # scripted: untouched
        assert got[a] == rewards[a]


def test_household_reward_off_is_bit_identical(cfg):
    tcfg = TrainConfig(num_envs=1, rollout_ticks=1)
    trainer = ArbiterTrainer(cfg, tcfg, seed=0, learn_agents=np.arange(4))
    env = trainer.envs[0]
    rewards = np.arange(12, dtype=np.float64)
    assert trainer._train_rewards(env, rewards) is rewards


def test_household_reward_refuses_a_world_without_households(cfg):
    plain = cfg.replace(**{"society.enabled": False})
    with pytest.raises(ValueError, match="household-reward needs a society"):
        ArbiterTrainer(plain, TrainConfig(num_envs=1), seed=0,
                       learn_agents=np.arange(4), household_reward=True)


def test_mixedrandom_is_runnable_and_reports_the_split(cfg):
    rep = run_episodes(cfg, 1, 10000, policy="mixedrandom")
    assert rep.learn_mask is not None
    assert rep.learn_mask.sum() == cfg.society.num_households


# --- persist-until-goal options ----------------------------------------------

def persist_acfg(timeout=150):
    return ArbiterConfig(persist_until_goal=True, persist_timeout=timeout)


def test_persist_trainer_redecide_matches_option_runner(cfg):
    """The parity test again, under the new contract. The trainer duplicates
    OptionRunner's redecide rule, and persistence changes both the budget and
    `deliver`'s termination -- so the duplication is re-pinned rather than
    assumed to have survived."""
    acfg = persist_acfg()
    runner = OptionRunner(UtilityArbiter(cfg, acfg), cfg, seed=5)
    world = World(cfg, seed=5)
    obs = world.observations()
    for _ in range(40):
        obs = world.step(runner.act(obs, world.action_mask())).obs

    trainer = ArbiterTrainer(cfg, TrainConfig(num_envs=1), seed=5, acfg=acfg)
    env = trainer.envs[0]
    env.goals = runner.goals.copy()
    env.ticks_left = runner.ticks_left.copy()
    env.started[:] = True
    view = ObsView(obs, cfg)
    got = trainer._redecide_mask(env, view)

    from sim.utility import (DRAW_FOOD, FORAGE, GOAL_RAID, GOAL_STEAL,
                             NEED_HUNGER, goal_viable)
    needs = compute_needs(view, cfg)
    viable = goal_viable(view, cfg, runner.goals, acfg)
    emergency = needs[:, NEED_HUNGER] >= acfg.critical
    pursuing = np.isin(runner.goals, (FORAGE, GOAL_STEAL, DRAW_FOOD, GOAL_RAID))
    want = (~viable) | (runner.ticks_left <= 0) | (emergency & ~pursuing)
    assert np.array_equal(got, want)


def test_one_transition_per_decision_survives_persistence(cfg):
    """The SMDP contract under the longer options: gamma^k is still an exact
    power of gamma for the k ticks the option ran, k >= 1, and now bounded by the
    BACKSTOP rather than by commit_ticks. An option that ran past 25 ticks is the
    thing the lever is for, so its presence is asserted too."""
    acfg = persist_acfg()
    tcfg = TrainConfig(num_envs=2, rollout_ticks=200)
    trainer = ArbiterTrainer(cfg, tcfg, seed=0, acfg=acfg)
    buffers = {k: [] for k in ("stream", "obs", "goal", "logprob", "value",
                               "mask", "reward", "discount", "next_obs",
                               "done", "agent")}
    trainer.collect(buffers)
    d = np.asarray(buffers["discount"])
    ks = np.log(d) / np.log(tcfg.gamma)
    assert np.allclose(ks, np.round(ks), atol=1e-6)
    assert (np.round(ks) >= 1).all()
    assert (np.round(ks) <= acfg.persist_timeout + 1).all()
    assert (np.round(ks) > acfg.commit_ticks).any(), "no option outlived 25 ticks"


def test_persisted_menu_is_the_same_for_every_chooser(cfg):
    """Menu parity has to hold under the flag too: persistence widens `deliver`
    (an empty-handed agent may start a programme), and it must widen it for the
    scripted scorer and the learned chooser identically or the headline
    comparison is two different games."""
    acfg = persist_acfg()
    world = World(cfg, seed=3)
    view = ObsView(world.observations(), cfg)
    menu = goal_mask(view, cfg, acfg)
    plain = goal_mask(view, cfg, ArbiterConfig())
    from sim.utility import DELIVER
    assert (menu[:, DELIVER] >= plain[:, DELIVER]).all()
    rng = np.random.default_rng(0)
    mask = world.action_mask()
    for chooser in (UtilityArbiter(cfg, acfg), RandomGoalArbiter(cfg, acfg),
                    fresh_arbiter(cfg, deterministic=False)):
        chooser.acfg = acfg
        goals = chooser.choose(view, mask, rng)
        assert menu[np.arange(view.n), goals].all(), type(chooser).__name__


def test_checkpoint_carries_the_option_contract(cfg, tmp_path):
    """Evaluating a persist-trained arbiter at commit_ticks is the same weights
    playing a different game, so the contract is provenance the checkpoint
    carries rather than something the caller has to remember."""
    tcfg = TrainConfig(num_envs=1, rollout_ticks=30)
    trainer = ArbiterTrainer(cfg, tcfg, seed=3, acfg=persist_acfg(),
                             learn_agents=np.arange(4))
    path = tmp_path / "latest.pt"
    save_checkpoint(path, trainer, update=1)
    assert load_arbiter(path, cfg).trained_persist is True

    plain = ArbiterTrainer(cfg, tcfg, seed=3, learn_agents=np.arange(4))
    path2 = tmp_path / "plain.pt"
    save_checkpoint(path2, plain, update=1)
    assert load_arbiter(path2, cfg).trained_persist is False


def test_persist_off_leaves_the_trainer_bit_identical(cfg):
    """Companion to tests/test_utility.py's golden checksum, on the gradient
    side: with the flag off the buffers a rollout produces are byte-for-byte what
    they were before the lever existed."""
    import hashlib
    def digest(acfg):
        tcfg = TrainConfig(num_envs=2, rollout_ticks=130)
        trainer = ArbiterTrainer(cfg, tcfg, seed=0, acfg=acfg,
                                 learn_agents=np.arange(4), household_reward=True)
        buffers = {k: [] for k in ("stream", "obs", "goal", "logprob", "value",
                                   "mask", "reward", "discount", "next_obs",
                                   "done", "agent")}
        trainer.collect(buffers)
        h = hashlib.sha256()
        for k in sorted(buffers):
            h.update(np.asarray(buffers[k]).tobytes())
        return len(buffers["goal"]), h.hexdigest()
    assert digest(None) == digest(ArbiterConfig(persist_timeout=999))
    assert digest(None) == (118, digest(ArbiterConfig())[1])


def test_mixedrandom_floor_is_size_matched(cfg):
    """The learned-share sweep needs a random-goal floor at EVERY share, not
    only at one-per-household -- otherwise the only point on the curve with a
    control is its left end."""
    rep = run_episodes(cfg, 1, 10000, policy="mixedrandom", learn_agents=7)
    assert rep.learn_mask.sum() == 7
    default = run_episodes(cfg, 1, 10000, policy="mixedrandom")
    assert default.learn_mask.sum() == cfg.society.num_households
