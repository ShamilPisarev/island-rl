"""Per-agent brains (Milestone 2): dispatch, independence, and forking."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sim.agents import N_ACTIONS, observation_dim
from sim.policy import (
    ActorCritic,
    PolicyGroup,
    build_policy,
    policy_from_config_dict,
)
from sim.ppo import PPOTrainer
from sim.world import VecWorld


@pytest.fixture
def obs_dim(cfg) -> int:
    return observation_dim(cfg)


def distinct_group(obs_dim: int, n: int = 3) -> PolicyGroup:
    """A group whose brains are trivially distinguishable by their output.

    Each brain's policy head is zeroed except for a constant bias of `i` on
    action `i`, so the logits alone say which brain produced a row.
    """
    group = PolicyGroup([ActorCritic(obs_dim, N_ACTIONS, (8,)) for _ in range(n)])
    with torch.no_grad():
        for i, policy in enumerate(group.policies):
            policy.policy_head.weight.zero_()
            policy.policy_head.bias.zero_()
            policy.policy_head.bias[i] = 10.0
            policy.value_head.weight.zero_()
            policy.value_head.bias.fill_(float(i))
    return group


# --- construction -----------------------------------------------------------


def test_build_policy_respects_mode(cfg, obs_dim):
    shared = build_policy(cfg.replace(**{"policy.mode": "shared"}), obs_dim)
    group = build_policy(cfg.replace(**{"policy.mode": "individual"}), obs_dim)
    assert isinstance(shared, ActorCritic)
    assert isinstance(group, PolicyGroup)
    assert group.num_agents == cfg.world.num_agents


def test_unknown_mode_is_rejected(cfg, obs_dim):
    with pytest.raises(ValueError, match="unknown policy mode"):
        build_policy(cfg, obs_dim, mode="hivemind")


def test_group_has_one_brains_worth_of_parameters_per_agent(cfg, obs_dim):
    shared = build_policy(cfg.replace(**{"policy.mode": "shared"}), obs_dim)
    group = build_policy(cfg.replace(**{"policy.mode": "individual"}), obs_dim)
    per_brain = sum(p.numel() for p in shared.parameters())
    assert sum(p.numel() for p in group.parameters()) == per_brain * cfg.world.num_agents


# --- dispatch ---------------------------------------------------------------


def test_rows_are_scored_by_their_own_brain(obs_dim):
    """The core invariant: row i must be evaluated by brain agent_ids[i], and the
    output must land back in row i. Getting this wrong trains each agent on
    another agent's experience and is invisible in the loss curves."""
    group = distinct_group(obs_dim)
    obs = torch.randn(12, obs_dim)
    agent_ids = torch.tensor([0, 1, 2, 2, 1, 0, 0, 0, 1, 2, 1, 0])

    logits, values = group(obs, agent_ids)

    assert torch.equal(logits.argmax(dim=-1), agent_ids)
    assert torch.allclose(values, agent_ids.float())


def test_dispatch_is_order_independent(obs_dim):
    """Shuffling the batch must permute the outputs, not change them."""
    group = distinct_group(obs_dim)
    obs = torch.randn(9, obs_dim)
    agent_ids = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2])
    logits, values = group(obs, agent_ids)

    perm = torch.randperm(9)
    p_logits, p_values = group(obs[perm], agent_ids[perm])
    assert torch.allclose(p_logits, logits[perm])
    assert torch.allclose(p_values, values[perm])


def test_group_handles_a_batch_missing_some_agents(obs_dim):
    """A minibatch may contain no rows at all for some agent; that brain simply
    gets no gradient that step rather than erroring."""
    group = distinct_group(obs_dim)
    obs = torch.randn(4, obs_dim)
    agent_ids = torch.tensor([1, 1, 1, 1])
    logits, values = group(obs, agent_ids)
    assert torch.equal(logits.argmax(dim=-1), agent_ids)
    assert torch.allclose(values, torch.ones(4))


def test_shared_policy_ignores_agent_ids(cfg, obs_dim):
    """ActorCritic accepts agent_ids so it stays swappable with a group; it must
    genuinely make no difference to the output."""
    shared = build_policy(cfg, obs_dim, mode="shared")
    obs = torch.randn(8, obs_dim)
    a, _ = shared(obs, torch.zeros(8, dtype=torch.long))
    b, _ = shared(obs, torch.arange(8))
    c, _ = shared(obs)
    assert torch.allclose(a, b) and torch.allclose(a, c)


# --- independence -----------------------------------------------------------


def test_gradients_reach_only_the_brain_that_produced_the_row(obs_dim):
    group = distinct_group(obs_dim)
    obs = torch.randn(6, obs_dim)
    agent_ids = torch.full((6,), 1, dtype=torch.long)

    logits, values = group(obs, agent_ids)
    (logits.sum() + values.sum()).backward()

    for i, policy in enumerate(group.policies):
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        has_grad = any(g.abs().sum() > 0 for g in grads)
        assert has_grad == (i == 1), f"brain {i}: unexpected gradient state"


def test_grad_clipping_is_per_brain_not_global(obs_dim):
    """Clipping the union would let one agent's bad update shrink everyone else's
    gradient. Brain 1's small gradient must survive brain 0's enormous one."""
    group = distinct_group(obs_dim)
    for i, policy in enumerate(group.policies):
        for p in policy.parameters():
            p.grad = torch.full_like(p, 100.0 if i == 0 else 1e-4)

    before = [p.grad.clone() for p in group.policies[1].parameters()]
    group.clip_grad_norm(0.5)
    after = [p.grad for p in group.policies[1].parameters()]

    # brain 0 was way over the threshold and got clipped
    norm0 = torch.cat([p.grad.flatten() for p in group.policies[0].parameters()]).norm()
    assert norm0 == pytest.approx(0.5, rel=1e-4)
    # brain 1 was under it and is untouched
    assert all(torch.allclose(b, a) for b, a in zip(before, after))


# --- forking ----------------------------------------------------------------


def test_fork_from_shared_reproduces_the_shared_policy_exactly(cfg, obs_dim):
    """Immediately after forking, every agent must behave exactly as the shared
    brain did -- otherwise Milestone 2 starts by throwing away Milestone 1."""
    torch.manual_seed(0)
    shared = build_policy(cfg, obs_dim, mode="shared")
    group = PolicyGroup.from_shared(shared, cfg.world.num_agents)
    obs = torch.randn(cfg.world.num_agents, obs_dim)
    agent_ids = torch.arange(cfg.world.num_agents)

    shared_logits, shared_values = shared(obs)
    group_logits, group_values = group(obs, agent_ids)
    assert torch.allclose(shared_logits, group_logits, atol=1e-6)
    assert torch.allclose(shared_values, group_values, atol=1e-6)


def test_forked_brains_are_copies_not_aliases(cfg, obs_dim):
    """They must not share storage, or "independent training" trains one brain
    six times."""
    shared = build_policy(cfg, obs_dim, mode="shared")
    group = PolicyGroup.from_shared(shared, 3)
    with torch.no_grad():
        group.policies[0].policy_head.bias.fill_(5.0)
    assert not torch.allclose(group.policies[1].policy_head.bias,
                              group.policies[0].policy_head.bias)
    assert not torch.allclose(shared.policy_head.bias, group.policies[0].policy_head.bias)


# --- checkpoint round-trip --------------------------------------------------


@pytest.mark.parametrize("mode", ["shared", "individual"])
def test_checkpoint_config_rebuilds_the_right_brain(cfg, obs_dim, mode):
    original = build_policy(cfg, obs_dim, mode=mode)
    rebuilt = policy_from_config_dict(cfg, original.config_dict())
    rebuilt.load_state_dict(original.state_dict())

    obs = torch.randn(cfg.world.num_agents, obs_dim)
    ids = torch.arange(cfg.world.num_agents)
    assert type(rebuilt) is type(original)
    assert torch.allclose(original(obs, ids)[0], rebuilt(obs, ids)[0])


def test_pre_milestone2_checkpoints_load_as_shared(cfg, obs_dim):
    """Checkpoints written before the mode key existed must still load."""
    legacy = {"obs_dim": obs_dim, "n_actions": N_ACTIONS, "hidden_sizes": [128, 128]}
    assert isinstance(policy_from_config_dict(cfg, legacy), ActorCritic)


# --- training ---------------------------------------------------------------


@pytest.fixture
def tiny_individual(cfg):
    return cfg.replace(**{
        "policy.mode": "individual",
        "ppo.num_envs": 4, "ppo.rollout_ticks": 24, "ppo.num_minibatches": 2,
        "ppo.epochs": 2, "ppo.total_updates": 4, "world.max_ticks": 40,
    })


def test_individual_training_runs_end_to_end(tiny_individual):
    torch.manual_seed(0)
    envs = VecWorld(tiny_individual, seed=0, num_envs=tiny_individual.ppo.num_envs)
    policy = build_policy(tiny_individual, envs.obs_dim)
    trainer = PPOTrainer(tiny_individual, policy, envs, device="cpu")

    for i in range(2):
        m = trainer.train_update(i)
        assert np.isfinite(m.policy_loss) and np.isfinite(m.value_loss)
        assert m.entropy > 0.0


def test_independent_training_actually_diverges(tiny_individual):
    """Forked brains start identical; after training they must not be.

    This is the substance of Milestone 2 -- if the weights stayed tied, the
    per-agent split would be decoration.
    """
    torch.manual_seed(0)
    envs = VecWorld(tiny_individual, seed=0, num_envs=tiny_individual.ppo.num_envs)
    shared = build_policy(tiny_individual, envs.obs_dim, mode="shared")
    group = PolicyGroup.from_shared(shared, tiny_individual.world.num_agents)

    def head(i: int) -> torch.Tensor:
        return group.policies[i].policy_head.weight.detach().clone()

    before = [head(i) for i in range(group.num_agents)]
    assert all(torch.allclose(before[0], b) for b in before[1:]), "fork was not identical"

    trainer = PPOTrainer(tiny_individual, group, envs, device="cpu")
    for i in range(3):
        trainer.train_update(i)

    after = [head(i) for i in range(group.num_agents)]
    assert all(not torch.allclose(before[i], after[i]) for i in range(group.num_agents)), \
        "some brain never updated"
    assert any(not torch.allclose(after[0], after[i], atol=1e-6)
               for i in range(1, group.num_agents)), "brains stayed identical"


def test_individual_training_is_deterministic(tiny_individual):
    def run() -> list[float]:
        torch.manual_seed(3)
        envs = VecWorld(tiny_individual, seed=3, num_envs=tiny_individual.ppo.num_envs)
        trainer = PPOTrainer(tiny_individual, build_policy(tiny_individual, envs.obs_dim),
                             envs, device="cpu")
        return [trainer.train_update(i).mean_reward for i in range(3)]

    assert run() == run()
