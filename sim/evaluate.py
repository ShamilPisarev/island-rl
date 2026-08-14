"""Run a policy, report statistics, and optionally record a replay.

Also supplies the baselines the Milestone 1 definition of done is measured
against: uniform-random actions (the floor) and the hand-written greedy forager
(a rough ceiling for memoryless reactive foraging).

    python -m sim.evaluate --checkpoint checkpoints/latest.pt --episodes 20
    python -m sim.evaluate --policy random --episodes 20
    python -m sim.evaluate --checkpoint checkpoints/latest.pt --replay viewer/replays/run.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .agents import observation_dim
from .config import Config, config_from_dict, load_config
from .metrics import EvalResult
from .policy import ActorCritic, build_policy, greedy_forager_actions, random_actions
from .replay import record_episode
from .world import EpisodeStats, World

ActFn = Callable[[np.ndarray], np.ndarray]


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[ActorCritic, Config, dict]:
    """Rebuild a policy from a checkpoint, using the config stored inside it.

    The config travels with the weights so an evaluation cannot silently use a
    different world than the one the policy was trained on -- a mismatch that
    produces plausible-looking numbers and no error at all.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(blob["config"])
    policy = build_policy(cfg, blob["policy_config"]["obs_dim"])
    policy.load_state_dict(blob["policy_state"])
    policy.eval()
    return policy, cfg, blob


def policy_act_fn(policy: ActorCritic, deterministic: bool = False,
                  device: str = "cpu") -> ActFn:
    """Wrap a network as an ``act_fn(obs) -> actions``.

    Sampling (not argmax) is the default: an argmax policy in a world with ties
    and symmetric compass directions can lock into a corner and stand still,
    which flatters or maligns the policy depending on the seed.
    """
    torch_device = torch.device(device)

    def act(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=torch_device)
            action, _, _ = policy.act(tensor, deterministic=deterministic)
        return action.cpu().numpy()

    return act


def make_act_fn(kind: str, cfg: Config, policy: ActorCritic | None,
                seed: int, deterministic: bool = False, device: str = "cpu") -> ActFn:
    if kind == "random":
        rng = np.random.default_rng(seed)
        return lambda obs: random_actions(obs, rng)
    if kind == "greedy":
        return lambda obs: greedy_forager_actions(obs, cfg)
    if policy is None:
        raise ValueError("a checkpoint is required to evaluate a learned policy")
    return policy_act_fn(policy, deterministic=deterministic, device=device)


def run_episodes(cfg: Config, act_fn: ActFn, episodes: int, seed: int) -> list[EpisodeStats]:
    """Evaluate over a fixed seed block, so every policy sees the same islands."""
    stats = []
    for i in range(episodes):
        world = World(cfg, seed=seed + i)
        obs = world.observations()
        for _ in range(cfg.world.max_ticks):
            result = world.step(act_fn(obs))
            obs = result.obs
            if result.episode_done:
                break
        stats.append(world.stats())
    return stats


def evaluate(cfg: Config, act_fn: ActFn, episodes: int, seed: int, label: str) -> EvalResult:
    return EvalResult.from_episodes(label, run_episodes(cfg, act_fn, episodes, seed),
                                    cfg.world.max_ticks)


def baselines(cfg: Config, episodes: int, seed: int) -> list[EvalResult]:
    """Random floor and scripted ceiling, on the same islands as everything else."""
    return [
        evaluate(cfg, make_act_fn("random", cfg, None, seed), episodes, seed, "random actions"),
        evaluate(cfg, make_act_fn("greedy", cfg, None, seed), episodes, seed, "scripted forager"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None,
                        help="ignored when --checkpoint is given; the checkpoint carries its own")
    parser.add_argument("--policy", choices=["learned", "random", "greedy"], default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000,
                        help="evaluation seeds are offset from training seeds by default")
    parser.add_argument("--deterministic", action="store_true",
                        help="take the argmax action instead of sampling")
    parser.add_argument("--replay", default=None, help="also record one episode to this path")
    parser.add_argument("--baselines", action="store_true", help="also evaluate random and scripted")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    policy = None
    if args.checkpoint:
        policy, cfg, blob = load_checkpoint(args.checkpoint, device=args.device)
        print(f"loaded {args.checkpoint} (update {blob.get('update', '?')}, "
              f"{blob.get('agent_steps', 0):,} agent-steps)")
    else:
        cfg = load_config(args.config)

    kind = args.policy or ("learned" if policy is not None else "random")
    if kind == "learned" and policy is None:
        parser.error("--policy learned needs --checkpoint")

    results = []
    if args.baselines or kind != "learned":
        results += baselines(cfg, args.episodes, args.seed)
    if kind == "learned":
        act = make_act_fn(kind, cfg, policy, args.seed, args.deterministic, args.device)
        suffix = " (argmax)" if args.deterministic else ""
        results.append(evaluate(cfg, act, args.episodes, args.seed, f"learned policy{suffix}"))

    print(f"\n{args.episodes} episodes, {cfg.world.max_ticks} ticks max, "
          f"{cfg.world.num_agents} agents, seeds {args.seed}..{args.seed + args.episodes - 1}")
    print("-" * 118)
    for r in results:
        print(r.line())
    print("-" * 118)

    if len(results) > 1 and kind == "learned":
        floor = next(r for r in results if r.label == "random actions")
        learned = results[-1]
        lift = learned.mean_lifespan / max(floor.mean_lifespan, 1e-9)
        print(f"learned policy survives {lift:.2f}x the random baseline "
              f"({learned.mean_lifespan:.1f} vs {floor.mean_lifespan:.1f} ticks)")

    if args.replay:
        act = make_act_fn(kind, cfg, policy, args.seed, args.deterministic, args.device)
        rec = record_episode(cfg, seed=args.seed, act_fn=act,
                             label=f"{kind} policy", source="evaluate")
        path = rec.save(args.replay)
        print(f"\nwrote replay {path}")


if __name__ == "__main__":
    main()
