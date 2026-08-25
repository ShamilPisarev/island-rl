"""Island 2.0 stage 5: a learned option-level arbiter (design doc option B).

THE EXPERIMENT THIS MODULE EXISTS TO RUN. Same world, same controllers, same
option menu: does a PPO policy choosing GOALS beat the scripted needs scorer
choosing them? Island 1.0's one remaining open problem is that PPO cannot take a
~20-step trip whose every one-step prefix its own accurate critic prices <= 0.
At the option level a trip IS one decision, so the wall this policy trains
against is the one 1.0 could never reach: not "can it walk", but "does it know
when walking, building, raiding or giving is worth it".

WHAT IS SHARED AND WHAT IS LEARNED, exactly:

  shared with the scripted arbiter (sim/utility.py):
    * `goal_availability` -- the option menu, including its mechanic-level
      corrections (raid motive gate, store/draw surplus rules, opportunistic
      theft). Both choosers pick from the same list.
    * `execute_goals`, `goal_viable`, `OptionRunner` -- the muscles and the
      semi-MDP bookkeeping (commitment, termination, interruption).
    * the per-agent TRAIT VECTOR, appended to the observation. The scripted
      scorer multiplies by it; the learned one is free to use or ignore it.
      Option D's individual character, one shared brain either way.
  learned:
    * the choice. `LearnedArbiter.choose(view, mask, rng)` is a policy forward
      pass where `UtilityArbiter.choose` is a scored argmax -- the one seam the
      whole stage-2 build kept clean for exactly this swap.

THE SEMI-MDP TRAINING CONTRACT, because this is the place the bookkeeping can
silently rot (utility.py's own warning): one PPO transition per DECISION, not
per tick. Between an agent's decisions the world runs k ticks under the
committed goal; the transition stores R = sum_j gamma^j r_j and bootstraps with
gamma^k V(s'), so gamma discounts real time, not decision count -- an option
that wastes 25 ticks is charged for 25 ticks. Decisions are PER-AGENT
ASYNCHRONOUS (one agent re-decides while its neighbour is mid-commitment),
which is why this trainer keeps per-agent open transitions instead of reusing
`world.decision_interval`'s synchronous machinery. GAE runs per agent-stream
with each transition's own gamma^k; lambda stays per-decision (a modelling
choice, noted, not a theorem).

Rewards are the world's own (alive/gather/eat/death), unshaped -- no goal is
paid for. Rule 1 applies at this level too: the question is whether choosing
well emerges from survival pressure, not whether we can pay for choices.

Train:
    python -m sim.arbiter --config config/island2/society4.yaml --run-name arb4
Compare (in sim.society):
    python -m sim.society --config config/island2/society4.yaml \\
        --arbiter learned --checkpoint checkpoints/arb4/latest.pt
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .agents import IDLE, num_actions, observation_dim
from .config import Config, load_config
from .obsview import ObsView
from .policy import ActorCritic
from .utility import (GOAL_NAMES, N_GOALS, REST, ArbiterConfig, agent_traits,
                      compute_needs, goal_availability)
from .world import World


def goal_mask(view: ObsView, cfg: Config, acfg: ArbiterConfig) -> np.ndarray:
    """The option menu as a boolean mask, (agents, goals). REST is always open.

    REST is forced on so no row is ever fully masked -- the same belt-and-braces
    the action mask gives `idle` -- and because doing nothing is genuinely always
    an option, which is not true of any other goal.
    """
    needs = compute_needs(view, cfg)
    available, _, _ = goal_availability(view, cfg, needs, acfg)
    available[:, REST] = True
    return available


class LearnedArbiter:
    """The stage-5 chooser. Same interface as `UtilityArbiter`, so `OptionRunner`
    and everything built on it (sim.society, the replay recorder) run unchanged.

    `deterministic=True` plays the argmax, which is how evaluation runs; training
    samples. The rng argument is accepted for interface parity and unused --
    torch's own generator does the sampling, seeded once at construction, so a
    learned replay is exactly as reproducible as a scripted one.
    """

    def __init__(self, policy: ActorCritic, cfg: Config,
                 acfg: ArbiterConfig | None = None, seed: int = 0,
                 deterministic: bool = True) -> None:
        self.policy = policy
        self.cfg = cfg
        self.acfg = acfg or ArbiterConfig()
        self.traits = agent_traits(cfg.world.num_agents, seed, self.acfg)
        self.deterministic = deterministic
        torch.manual_seed(seed)

    def _inputs(self, view: ObsView) -> torch.Tensor:
        return torch.as_tensor(
            np.concatenate([view.obs, self.traits], axis=1), dtype=torch.float32)

    def choose(self, view: ObsView, mask: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
        available = goal_mask(view, self.cfg, self.acfg)
        with torch.no_grad():
            goals, _, _ = self.policy.act(
                self._inputs(view), deterministic=self.deterministic,
                mask=torch.as_tensor(available))
        return goals.numpy()


class MixedArbiter:
    """A scripted majority and a learned minority behind one `choose` interface.

    The evaluation half of the mixed-population experiment: `OptionRunner` and
    everything on top of it (sim.society, the replay recorder) run unchanged,
    and the split is a boolean mask over agent rows. Both sub-choosers see the
    same view; each agent's goal comes from whichever chooser owns it.
    """

    def __init__(self, scripted, learned, learn_mask: np.ndarray) -> None:
        self.scripted = scripted
        self.learned = learned
        self.learn_mask = np.asarray(learn_mask, dtype=bool)
        self.acfg = scripted.acfg

    def choose(self, view: ObsView, mask: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
        g = self.scripted.choose(view, mask, rng)
        gl = self.learned.choose(view, mask, rng)
        return np.where(self.learn_mask, gl, g)


class RandomGoalArbiter:
    """Uniform over the available goals: the floor the learned arbiter must beat.

    "Learned beats scripted" means nothing on its own -- the menu plus the
    scripted muscles might carry any chooser. This control prices the menu
    itself, the way the random-action floor prices the world.
    """

    def __init__(self, cfg: Config, acfg: ArbiterConfig | None = None,
                 seed: int = 0) -> None:
        self.cfg = cfg
        self.acfg = acfg or ArbiterConfig()
        self.traits = agent_traits(cfg.world.num_agents, seed, self.acfg)

    def choose(self, view: ObsView, mask: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
        available = goal_mask(view, self.cfg, self.acfg)
        # Inverse-CDF over the uniform distribution on available goals, one draw
        # per agent, vectorised -- same trick as the utility arbiter's softmax.
        p = available / available.sum(axis=1, keepdims=True)
        u = rng.random((view.n, 1))
        return (p.cumsum(axis=1) < u).sum(axis=1).clip(0, N_GOALS - 1)


# --------------------------------------------------------------------------
# The SMDP trainer
# --------------------------------------------------------------------------

@dataclass
class TrainConfig:
    num_envs: int = 8
    rollout_ticks: int = 200        # one day/night cycle per rollout
    total_updates: int = 150
    epochs: int = 4
    num_minibatches: int = 4
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (128, 128)


class _EnvState:
    """One world plus the per-agent semi-MDP bookkeeping the trainer owns.

    Mirrors `OptionRunner`'s redecide logic exactly (viability, timeout,
    tier-0 interruption) -- duplicated rather than subclassed because the runner
    re-decides and executes in one call, and the trainer has to put a policy
    forward pass between those two steps. `tests/test_arbiter.py` pins the two
    redecide rules against each other so they cannot drift.
    """

    def __init__(self, cfg: Config, acfg: ArbiterConfig, seed: int) -> None:
        from .utility import N_MOVE_ACTIONS  # local: avoid a circular top import
        self.cfg = cfg
        self.acfg = acfg
        self.world = World(cfg, seed=seed)
        self.rng = np.random.default_rng(seed + 7_777)
        n = cfg.world.num_agents
        self.obs = self.world.observations()
        self.goals = np.full(n, REST, dtype=np.int64)
        self.ticks_left = np.zeros(n, dtype=np.int64)
        self.explore_heading = self.rng.integers(0, 8, size=n)
        # the open transition, per agent
        self.dec_obs = np.zeros((n, 0), dtype=np.float32)   # set on first decide
        self.dec_goal = np.full(n, -1, dtype=np.int64)
        self.dec_logprob = np.zeros(n, dtype=np.float32)
        self.dec_value = np.zeros(n, dtype=np.float32)
        self.dec_mask = np.zeros((n, N_GOALS), dtype=bool)
        self.reward_acc = np.zeros(n, dtype=np.float64)
        self.disc = np.ones(n, dtype=np.float64)            # gamma^j inside the option
        self.has_open = np.zeros(n, dtype=bool)
        # `started` is the "this agent has decided at least once this episode"
        # flag. It used to be `has_open`, which is the same thing when every
        # agent is learned -- but a SCRIPTED agent in a mixed population never
        # opens a transition, so keying first-decision off has_open would make
        # it redecide every tick and turn its commitment decorative.
        self.started = np.zeros(n, dtype=bool)


class ArbiterTrainer:
    """PPO over goals, one transition per option decision."""

    def __init__(self, cfg: Config, tcfg: TrainConfig, seed: int = 0,
                 acfg: ArbiterConfig | None = None,
                 learn_agents: np.ndarray | None = None) -> None:
        self.cfg = cfg
        self.tcfg = tcfg
        self.acfg = acfg or ArbiterConfig()
        self.seed = seed
        n = cfg.world.num_agents
        self.traits = agent_traits(n, seed, self.acfg)
        # Mixed population (design doc section 10, the first lever): only the
        # agents in `learn_agents` are driven by -- and train -- the policy; the
        # rest run the scripted arbiter in the SAME worlds. The point is the
        # state distribution: a minority among scripted builders EXPERIENCES
        # sheltered nights from tick 0, so the critic can price one without the
        # chicken-and-egg that sank the from-scratch runs. With households
        # assigned round-robin (agent i -> household i % H), the first H agents
        # are one per household, each with scripted housemates.
        self.learn_mask = np.zeros(n, dtype=bool)
        if learn_agents is None:
            self.learn_mask[:] = True
            self.learn_agents = np.arange(n)
            self.teacher = None
        else:
            self.learn_agents = np.asarray(learn_agents, dtype=np.int64)
            self.learn_mask[self.learn_agents] = True
            from .utility import UtilityArbiter
            # Same seed and construction as self.traits, so the scripted agents
            # behave exactly as an all-scripted population's would.
            self.teacher = UtilityArbiter(cfg, self.acfg, seed=seed)
        self.in_dim = observation_dim(cfg) + N_GOALS
        torch.manual_seed(seed)
        self.policy = ActorCritic(self.in_dim, n_actions=N_GOALS,
                                  hidden_sizes=tcfg.hidden_sizes)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=tcfg.lr, eps=1e-5)
        self.mb_rng = np.random.default_rng(seed + 12_345)   # never global numpy state
        seeds = np.random.SeedSequence(seed).generate_state(tcfg.num_envs, dtype=np.uint32)
        self.envs = [_EnvState(cfg, self.acfg, int(s)) for s in seeds]
        self.finished: list = []
        self.finished_learned: list[float] = []   # learned subset's mean lifespan

    # --- collection --------------------------------------------------------

    def _inputs(self, obs: np.ndarray, rows: np.ndarray | None = None) -> torch.Tensor:
        traits = self.traits if rows is None else self.traits[rows]
        return torch.as_tensor(np.concatenate([obs, traits], axis=1),
                               dtype=torch.float32)

    def _redecide_mask(self, env: _EnvState, view: ObsView) -> np.ndarray:
        from .utility import (FORAGE, GOAL_STEAL, DRAW_FOOD, GOAL_RAID,
                              NEED_HUNGER, goal_viable)
        needs = compute_needs(view, self.cfg)
        viable = goal_viable(view, self.cfg, env.goals)
        emergency = needs[:, NEED_HUNGER] >= self.acfg.critical
        pursuing_food = np.isin(env.goals, (FORAGE, GOAL_STEAL, DRAW_FOOD, GOAL_RAID))
        return (~viable) | (env.ticks_left <= 0) | (emergency & ~pursuing_food) \
            | (~env.started)

    def collect(self, buffers: dict) -> None:
        """Advance every env `rollout_ticks`, appending completed transitions."""
        from .utility import EXPLORE, execute_goals, N_MOVE_ACTIONS
        cfg, acfg, gamma = self.cfg, self.acfg, self.tcfg.gamma
        for _ in range(self.tcfg.rollout_ticks):
            for e, env in enumerate(self.envs):
                view = ObsView(env.obs, cfg)
                alive = env.world.pool.alive
                redecide = self._redecide_mask(env, view) & alive
                if redecide.any():
                    rows = np.flatnonzero(redecide)
                    lrows = rows[self.learn_mask[rows]]
                    srows = rows[~self.learn_mask[rows]]
                    # close the open transition: the state it lands in is the
                    # state the NEXT decision is made from, so V(s') bootstraps it
                    self._close(buffers, e, env, lrows, env.obs, done=False)
                    if srows.size:
                        choice = self.teacher.choose(view, None, env.rng)
                        env.goals[srows] = choice[srows]
                    if lrows.size:
                        available = goal_mask(view, cfg, acfg)
                        with torch.no_grad():
                            g, lp, v = self.policy.act(
                                self._inputs(env.obs[lrows], lrows),
                                mask=torch.as_tensor(available[lrows]))
                        if env.dec_obs.shape[1] == 0:
                            env.dec_obs = np.zeros(
                                (cfg.world.num_agents, env.obs.shape[1]), dtype=np.float32)
                        env.dec_obs[lrows] = env.obs[lrows]
                        env.dec_goal[lrows] = g.numpy()
                        env.dec_logprob[lrows] = lp.numpy()
                        env.dec_value[lrows] = v.numpy()
                        env.dec_mask[lrows] = available[lrows]
                        env.goals[lrows] = g.numpy()
                        env.has_open[lrows] = True
                    env.ticks_left[rows] = acfg.commit_ticks
                    env.reward_acc[rows] = 0.0
                    env.disc[rows] = 1.0
                    env.started[rows] = True
                    fresh_explore = redecide & (env.goals == EXPLORE)
                    if fresh_explore.any():
                        roll = env.rng.integers(0, N_MOVE_ACTIONS, size=view.n)
                        env.explore_heading = np.where(fresh_explore, roll,
                                                       env.explore_heading)
                env.ticks_left -= 1

                mask = env.world.action_mask()
                actions = execute_goals(env.goals, ObsView(env.obs, cfg), cfg,
                                        mask, env.explore_heading)
                res = env.world.step(actions)
                env.reward_acc += env.disc * res.rewards
                env.disc *= gamma
                env.obs = res.obs

                died = res.terminated & env.has_open
                if died.any():
                    # a starved agent's option ends with nothing after it
                    self._close(buffers, e, env, np.flatnonzero(died), res.obs,
                                done=True)
                if res.episode_done:
                    # truncation: survivors' open options bootstrap V(final_obs);
                    # done=False is exactly 1.0's death-vs-timeout distinction.
                    rows = np.flatnonzero(env.has_open)
                    self._close(buffers, e, env, rows, res.obs,
                                done=not res.truncated)
                    self.finished.append(env.world.stats())
                    self.finished_learned.append(
                        float(env.world.alive_ticks[self.learn_agents].mean()))
                    env.obs = env.world.reset()
                    env.goals[:] = REST
                    env.ticks_left[:] = 0
                    env.has_open[:] = False
                    env.started[:] = False

    def _close(self, buffers: dict, e: int, env: _EnvState, rows: np.ndarray,
               next_obs: np.ndarray, done: bool) -> None:
        rows = rows[env.has_open[rows]]
        if rows.size == 0:
            return
        for i in rows:
            buffers["stream"].append(e * self.cfg.world.num_agents + int(i))
            buffers["obs"].append(env.dec_obs[i].copy())
            buffers["goal"].append(int(env.dec_goal[i]))
            buffers["logprob"].append(float(env.dec_logprob[i]))
            buffers["value"].append(float(env.dec_value[i]))
            buffers["mask"].append(env.dec_mask[i].copy())
            buffers["reward"].append(float(env.reward_acc[i]))
            buffers["discount"].append(float(env.disc[i]))   # gamma^k for GAE
            buffers["next_obs"].append(next_obs[i].copy())
            buffers["done"].append(bool(done))
            buffers["agent"].append(int(i))
        env.has_open[rows] = False

    # --- imitation warm start -------------------------------------------------

    def imitate(self, updates: int, lr: float = 1e-3) -> float:
        """Behaviour-clone the scripted arbiter's choices, then hand over to PPO.

        Rule 3, translated to the option level. From scratch, option-level PPO
        converged to a hyper-competent PURE FORAGER -- 93% of the island
        harvested, food stores at 9.8 of 12, and 0% of nights indoors, because a
        population where nobody builds never experiences a sheltered night (so
        the critic cannot price one) and never finishes a shelter (so the
        `shelter` goal is masked out of the menu all episode -- the option-level
        chicken-and-egg). -122.5 +- 8.3 against the scripted arbiter, worse on
        10 of 10 islands.

        This is the M1 -> M2 move -- fork from competence, then diverge -- with
        the teacher being a program instead of a checkpoint: collect states with
        the SCRIPTED arbiter driving (so shelters exist and the state
        distribution is the one competent play visits) and train the goal head
        by cross-entropy on its choices at decision points, masked by the shared
        menu. The critic trains on nothing here; PPO fits it afterwards on the
        semi-MDP returns. What the headline comparison then asks is sharper, not
        weaker: given the scripted arbiter's own behaviour as a starting point,
        does PPO find improvements the needs scorer cannot express?
        """
        from .utility import UtilityArbiter
        teacher = UtilityArbiter(self.cfg, self.acfg, seed=self.seed)
        # The teacher's traits ARE this trainer's traits (same seed, same
        # construction), so the student sees the inputs that explain the
        # teacher's per-agent quirks rather than having to average over them.
        opt = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        last_acc = 0.0
        for _ in range(updates):
            batch_obs, batch_goal, batch_mask, batch_agent = [], [], [], []
            for env in self.envs:
                from .utility import EXPLORE, N_MOVE_ACTIONS, execute_goals
                for _ in range(self.tcfg.rollout_ticks):
                    view = ObsView(env.obs, self.cfg)
                    alive = env.world.pool.alive
                    redecide = self._redecide_mask(env, view) & alive
                    if redecide.any():
                        rows = np.flatnonzero(redecide)
                        available = goal_mask(view, self.cfg, self.acfg)
                        choice = teacher.choose(view, None, env.rng)
                        batch_obs.append(env.obs[rows].copy())
                        batch_goal.append(choice[rows])
                        batch_mask.append(available[rows])
                        batch_agent.append(rows)
                        env.goals[rows] = choice[rows]
                        env.ticks_left[rows] = self.acfg.commit_ticks
                        env.started[rows] = True   # bookkeeping only; no PPO buffer
                        fresh = redecide & (env.goals == EXPLORE)
                        if fresh.any():
                            roll = env.rng.integers(0, N_MOVE_ACTIONS, size=view.n)
                            env.explore_heading = np.where(fresh, roll,
                                                           env.explore_heading)
                    env.ticks_left -= 1
                    mask = env.world.action_mask()
                    actions = execute_goals(env.goals, ObsView(env.obs, self.cfg),
                                            self.cfg, mask, env.explore_heading)
                    res = env.world.step(actions)
                    env.obs = res.obs
                    if res.episode_done:
                        env.obs = env.world.reset()
                        env.goals[:] = REST
                        env.ticks_left[:] = 0
                        env.has_open[:] = False
                        env.started[:] = False
            obs = np.concatenate(batch_obs)
            goals = torch.as_tensor(np.concatenate(batch_goal))
            masks = torch.as_tensor(np.concatenate(batch_mask))
            agents = np.concatenate(batch_agent)
            inputs = self._inputs(obs, agents)
            logits, _ = self.policy(inputs)
            logits = self.policy._masked(logits, masks)
            loss = torch.nn.functional.cross_entropy(logits, goals)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_acc = float((logits.argmax(dim=1) == goals).float().mean())
        # PPO must not start with open transitions half-recorded under the
        # teacher: reset every env so the first collect() is clean.
        seeds = np.random.SeedSequence(self.seed).generate_state(
            self.tcfg.num_envs, dtype=np.uint32)
        self.envs = [_EnvState(self.cfg, self.acfg, int(s)) for s in seeds]
        return last_acc

    # --- update -------------------------------------------------------------

    def update(self, buffers: dict, lr: float, value_only: bool = False) -> dict:
        t = self.tcfg
        obs = np.stack(buffers["obs"])
        goals = np.asarray(buffers["goal"])
        logprobs = np.asarray(buffers["logprob"], dtype=np.float32)
        values = np.asarray(buffers["value"], dtype=np.float32)
        masks = np.stack(buffers["mask"])
        rewards = np.asarray(buffers["reward"], dtype=np.float32)
        discounts = np.asarray(buffers["discount"], dtype=np.float32)
        dones = np.asarray(buffers["done"])
        streams = np.asarray(buffers["stream"])
        agents = np.asarray(buffers["agent"])
        next_obs = np.stack(buffers["next_obs"])

        # Bootstrap values for every transition's landing state, in one batch.
        with torch.no_grad():
            traits = self.traits[agents]
            nv = self.policy.value(torch.as_tensor(
                np.concatenate([next_obs, traits], axis=1),
                dtype=torch.float32)).numpy()

        # GAE per agent-stream, walking each stream backward in collection
        # order. gamma^k is per transition; a done transition bootstraps nothing.
        advantages = np.zeros_like(rewards)
        last_adv: dict[int, float] = {}
        for idx in range(len(rewards) - 1, -1, -1):
            s = int(streams[idx])
            nonterminal = 0.0 if dones[idx] else 1.0
            delta = (rewards[idx] + discounts[idx] * nv[idx] * nonterminal
                     - values[idx])
            carry = last_adv.get(s, 0.0) * nonterminal
            advantages[idx] = delta + discounts[idx] * t.gae_lambda * carry
            last_adv[s] = advantages[idx]
        returns = advantages + values

        n = len(rewards)
        device_obs = torch.as_tensor(
            np.concatenate([obs, self.traits[agents]], axis=1), dtype=torch.float32)
        b_goals = torch.as_tensor(goals)
        b_logprobs = torch.as_tensor(logprobs)
        b_masks = torch.as_tensor(masks)
        b_adv = torch.as_tensor(advantages)
        b_ret = torch.as_tensor(returns)

        for group in self.optim.param_groups:
            group["lr"] = lr
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "count": 0}
        mb_size = max(n // t.num_minibatches, 1)
        for _ in range(t.epochs):
            order = self.mb_rng.permutation(n)
            for start in range(0, n, mb_size):
                mb = torch.as_tensor(order[start:start + mb_size])
                new_lp, entropy, value = self.policy.evaluate_actions(
                    device_obs[mb], b_goals[mb], mask=b_masks[mb])
                ratio = (new_lp - b_logprobs[mb]).exp()
                adv = b_adv[mb]
                # unbiased=False: the last minibatch can be a single leftover
                # transition, and Bessel's correction on n=1 is NaN -- which then
                # NaNs every weight in the network one backward pass later.
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
                pg = torch.max(-adv * ratio,
                               -adv * ratio.clamp(1 - t.clip_coef, 1 + t.clip_coef)).mean()
                v_loss = 0.5 * ((value - b_ret[mb]) ** 2).mean()
                if value_only:
                    # Critic warm-up: the imitation phase trains the goal head on
                    # nothing but cross-entropy, so PPO would otherwise begin with
                    # a RANDOM critic -- and its first advantage estimates wreck
                    # the warm-started policy before the critic can price the
                    # sheltered nights that policy produces (1.0's own learning
                    # curves put the critic ~60 updates ahead of the policy).
                    # Value loss only; the trunk is shared, so the policy head is
                    # perturbed only through features, not through a gradient of
                    # its own.
                    loss = t.vf_coef * v_loss
                else:
                    loss = pg - t.ent_coef * entropy.mean() + t.vf_coef * v_loss
                self.optim.zero_grad()
                loss.backward()
                self.policy.clip_grad_norm(t.max_grad_norm)
                self.optim.step()
                stats["policy_loss"] += float(pg.detach())
                stats["value_loss"] += float(v_loss.detach())
                stats["entropy"] += float(entropy.mean().detach())
                stats["count"] += 1
        c = max(stats.pop("count"), 1)
        out = {k: v / c for k, v in stats.items()}
        out["transitions"] = n
        var_y = float(np.var(returns))
        out["explained_variance"] = (float("nan") if var_y == 0 else
                                     1.0 - float(np.var(returns - values)) / var_y)
        return out


def save_checkpoint(path: Path, trainer: ArbiterTrainer, update: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "arbiter",
        "update": update,
        "policy_state": trainer.policy.state_dict(),
        "policy_config": trainer.policy.config_dict(),
        "config": trainer.cfg.to_dict(),
        "seed": trainer.seed,
        # None means "all agents were learned"; a list is the mixed-population
        # subset, so evaluation can rebuild the same split without being told.
        "learn_agents": (None if trainer.learn_mask.all()
                         else trainer.learn_agents.tolist()),
    }, path)


def load_arbiter(path: str | Path, cfg: Config | None = None,
                 deterministic: bool = True) -> LearnedArbiter:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("kind") != "arbiter":
        raise ValueError(f"{path} is not a goal-arbiter checkpoint -- 1.0 "
                         f"checkpoints act on primitive actions, not goals")
    from .config import config_from_dict
    cfg = cfg or config_from_dict(blob["config"])
    pc = blob["policy_config"]
    policy = ActorCritic(pc["obs_dim"], n_actions=pc["n_actions"],
                         hidden_sizes=pc["hidden_sizes"])
    policy.load_state_dict(blob["policy_state"])
    policy.eval()
    arb = LearnedArbiter(policy, cfg, seed=blob.get("seed", 0),
                         deterministic=deterministic)
    arb.learn_agents = blob.get("learn_agents")   # None unless mixed-trained
    return arb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/island2/society4.yaml")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--envs", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=None,
                    help="override the semi-MDP discount. 1.0's 0.99 is myopic "
                         "at this level: measured, the pure-forager policy's "
                         "DISCOUNTED return beats the scripted arbiter's (4.47 "
                         "vs 4.41) while dying 127 ticks sooner -- the night "
                         "bill lands 100-300 ticks after the build decision and "
                         "0.99^300 is 0.05. Undiscounted, sheltering wins.")
    ap.add_argument("--imitate", type=int, default=0,
                    help="behaviour-clone the scripted arbiter for this many "
                         "updates before PPO (the option-level fork)")
    ap.add_argument("--value-warmup", type=int, default=0,
                    help="after imitation, fit the critic alone for this many "
                         "updates before any policy gradient flows")
    ap.add_argument("--learn-agents", type=int, default=0,
                    help="mixed population: only the FIRST N agents train under "
                         "PPO; the rest run the scripted arbiter in the same "
                         "worlds. Households are round-robin (agent i -> "
                         "household i%%H), so N=num_households puts one learned "
                         "agent in every household. 0 = all learned.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    kw = {}
    if args.updates is not None:
        kw["total_updates"] = args.updates
    if args.envs is not None:
        kw["num_envs"] = args.envs
    if args.gamma is not None:
        kw["gamma"] = args.gamma
    tcfg = TrainConfig(**kw)
    torch.set_num_threads(cfg.ppo.threads or 4)

    learn = np.arange(args.learn_agents) if args.learn_agents else None
    trainer = ArbiterTrainer(cfg, tcfg, seed=args.seed, learn_agents=learn)
    if learn is not None:
        print(f"mixed population: agents 0..{args.learn_agents - 1} learn, "
              f"{cfg.world.num_agents - args.learn_agents} run the scripted arbiter")
    if args.imitate:
        acc = trainer.imitate(args.imitate)
        print(f"imitation warm start: {args.imitate} updates, "
              f"final agreement with the teacher {100 * acc:.1f}%")
    run_dir = Path(cfg.logging.run_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(cfg.logging.checkpoint_dir) / args.run_name / "latest.pt"

    fields = ["update", "lifespan", "lifespan_learned", "deaths", "transitions",
              "entropy", "explained_variance", "policy_loss", "value_loss", "sps"]
    log = open(run_dir / "metrics.csv", "w", newline="")
    writer = csv.DictWriter(log, fieldnames=fields)
    writer.writeheader()

    start = time.time()
    for w in range(args.value_warmup):
        buffers = {k: [] for k in ("stream", "obs", "goal", "logprob", "value",
                                   "mask", "reward", "discount", "next_obs",
                                   "done", "agent")}
        trainer.collect(buffers)
        stats = trainer.update(buffers, lr=tcfg.lr, value_only=True)
        if (w + 1) % 5 == 0:
            print(f"[warmup {w + 1}/{args.value_warmup}] "
                  f"ev {stats['explained_variance']:.3f} "
                  f"v_loss {stats['value_loss']:.3f}")
    trainer.finished = []
    trainer.finished_learned = []

    for update in range(1, tcfg.total_updates + 1):
        buffers: dict = {k: [] for k in ("stream", "obs", "goal", "logprob",
                                         "value", "mask", "reward", "discount",
                                         "next_obs", "done", "agent")}
        trainer.collect(buffers)
        frac = 1.0 - (update - 1) / tcfg.total_updates if tcfg.anneal_lr else 1.0
        stats = trainer.update(buffers, lr=tcfg.lr * frac)

        eps = trainer.finished
        trainer.finished = []
        eps_l = trainer.finished_learned
        trainer.finished_learned = []
        lifespan = float(np.mean([e.mean_lifespan for e in eps])) if eps else float("nan")
        life_l = float(np.mean(eps_l)) if eps_l else float("nan")
        deaths = float(np.mean([e.deaths for e in eps])) if eps else float("nan")
        ticks_done = update * tcfg.rollout_ticks * tcfg.num_envs * cfg.world.num_agents
        row = {"update": update, "lifespan": round(lifespan, 1),
               "lifespan_learned": round(life_l, 1),
               "deaths": round(deaths, 2), "transitions": stats["transitions"],
               "entropy": round(stats["entropy"], 4),
               "explained_variance": round(stats["explained_variance"], 4),
               "policy_loss": round(stats["policy_loss"], 5),
               "value_loss": round(stats["value_loss"], 4),
               "sps": int(ticks_done / (time.time() - start))}
        writer.writerow(row)
        log.flush()
        if update % 5 == 0 or update == 1:
            print(f"[{update:>4}/{tcfg.total_updates}] lifespan {row['lifespan']} "
                  f"(learned {row['lifespan_learned']}) "
                  f"deaths {row['deaths']} ent {row['entropy']} "
                  f"ev {row['explained_variance']} sps {row['sps']}")
        if update % 25 == 0 or update == tcfg.total_updates:
            save_checkpoint(ckpt, trainer, update)
    log.close()
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
