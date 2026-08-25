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


class ArbiterTrainer:
    """PPO over goals, one transition per option decision."""

    def __init__(self, cfg: Config, tcfg: TrainConfig, seed: int = 0,
                 acfg: ArbiterConfig | None = None) -> None:
        self.cfg = cfg
        self.tcfg = tcfg
        self.acfg = acfg or ArbiterConfig()
        self.seed = seed
        n = cfg.world.num_agents
        self.traits = agent_traits(n, seed, self.acfg)
        self.in_dim = observation_dim(cfg) + N_GOALS
        torch.manual_seed(seed)
        self.policy = ActorCritic(self.in_dim, n_actions=N_GOALS,
                                  hidden_sizes=tcfg.hidden_sizes)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=tcfg.lr, eps=1e-5)
        self.mb_rng = np.random.default_rng(seed + 12_345)   # never global numpy state
        seeds = np.random.SeedSequence(seed).generate_state(tcfg.num_envs, dtype=np.uint32)
        self.envs = [_EnvState(cfg, self.acfg, int(s)) for s in seeds]
        self.finished: list = []

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
            | (~env.has_open)

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
                    # close the open transition: the state it lands in is the
                    # state the NEXT decision is made from, so V(s') bootstraps it
                    self._close(buffers, e, env, rows, env.obs, done=False)
                    available = goal_mask(view, cfg, acfg)
                    with torch.no_grad():
                        g, lp, v = self.policy.act(
                            self._inputs(env.obs[rows], rows),
                            mask=torch.as_tensor(available[rows]))
                    if env.dec_obs.shape[1] == 0:
                        env.dec_obs = np.zeros(
                            (cfg.world.num_agents, env.obs.shape[1]), dtype=np.float32)
                    env.dec_obs[rows] = env.obs[rows]
                    env.dec_goal[rows] = g.numpy()
                    env.dec_logprob[rows] = lp.numpy()
                    env.dec_value[rows] = v.numpy()
                    env.dec_mask[rows] = available[rows]
                    env.goals[rows] = g.numpy()
                    env.ticks_left[rows] = acfg.commit_ticks
                    env.reward_acc[rows] = 0.0
                    env.disc[rows] = 1.0
                    env.has_open[rows] = True
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
                    env.obs = env.world.reset()
                    env.goals[:] = REST
                    env.ticks_left[:] = 0
                    env.has_open[:] = False

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

    # --- update -------------------------------------------------------------

    def update(self, buffers: dict, lr: float) -> dict:
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
    return LearnedArbiter(policy, cfg, seed=blob.get("seed", 0),
                          deterministic=deterministic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/island2/society4.yaml")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--envs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = TrainConfig()
    if args.updates is not None:
        tcfg = TrainConfig(total_updates=args.updates,
                           num_envs=args.envs or tcfg.num_envs)
    elif args.envs is not None:
        tcfg = TrainConfig(num_envs=args.envs)
    torch.set_num_threads(cfg.ppo.threads or 4)

    trainer = ArbiterTrainer(cfg, tcfg, seed=args.seed)
    run_dir = Path(cfg.logging.run_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(cfg.logging.checkpoint_dir) / args.run_name / "latest.pt"

    fields = ["update", "lifespan", "deaths", "transitions", "entropy",
              "explained_variance", "policy_loss", "value_loss", "sps"]
    log = open(run_dir / "metrics.csv", "w", newline="")
    writer = csv.DictWriter(log, fieldnames=fields)
    writer.writeheader()

    start = time.time()
    for update in range(1, tcfg.total_updates + 1):
        buffers: dict = {k: [] for k in ("stream", "obs", "goal", "logprob",
                                         "value", "mask", "reward", "discount",
                                         "next_obs", "done", "agent")}
        trainer.collect(buffers)
        frac = 1.0 - (update - 1) / tcfg.total_updates if tcfg.anneal_lr else 1.0
        stats = trainer.update(buffers, lr=tcfg.lr * frac)

        eps = trainer.finished
        trainer.finished = []
        lifespan = float(np.mean([e.mean_lifespan for e in eps])) if eps else float("nan")
        deaths = float(np.mean([e.deaths for e in eps])) if eps else float("nan")
        ticks_done = update * tcfg.rollout_ticks * tcfg.num_envs * cfg.world.num_agents
        row = {"update": update, "lifespan": round(lifespan, 1),
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
                  f"deaths {row['deaths']} ent {row['entropy']} "
                  f"ev {row['explained_variance']} sps {row['sps']}")
        if update % 25 == 0 or update == tcfg.total_updates:
            save_checkpoint(ckpt, trainer, update)
    log.close()
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
