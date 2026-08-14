"""Training entrypoint.

    python -m sim.train                              # full run from config/default.yaml
    python -m sim.train --updates 40 --run-name quick
    python -m sim.train --resume checkpoints/latest.pt

Reports the random-action baseline before training starts, because "lifespan
went up" means nothing without the floor it went up from, and the floor moves
whenever the world config is touched.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config, load_config
from .evaluate import baselines, evaluate, load_checkpoint, make_act_fn, policy_act_fn
from .metrics import MetricsLogger
from .policy import Brain, PolicyGroup, build_policy
from .ppo import PPOTrainer
from .replay import record_episode
from .world import VecWorld


def seed_everything(seed: int) -> None:
    """Seed torch and numpy. World RNGs are seeded separately and explicitly."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def fork_from_checkpoint(policy: Brain, cfg: Config, obs_dim: int) -> Brain:
    """Initialise from ``cfg.policy.init_from``.

    The Milestone 2 move: take the trained shared brain and hand every agent its
    own copy. They start identical and competent, then diverge under independent
    gradients -- which is the point, since a divergence measured against six
    randomly-initialised networks would mostly be measuring initialisation noise.

    Forking a shared checkpoint into an individual policy is the interesting
    case, but shared->shared (warm start) and individual->individual (resume with
    a fresh optimiser) both work.
    """
    source, _, blob = load_checkpoint(cfg.policy.init_from, device=cfg.ppo.device)
    if source.config_dict()["obs_dim"] != obs_dim:
        raise ValueError(
            f"{cfg.policy.init_from} was trained with obs_dim "
            f"{source.config_dict()['obs_dim']}, this config gives {obs_dim}"
        )

    source_mode = source.config_dict()["mode"]
    target_mode = cfg.policy.mode
    if source_mode == "shared" and target_mode == "individual":
        forked = PolicyGroup.from_shared(source, cfg.world.num_agents)
        print(f"forked {cfg.policy.init_from} (shared, update {blob.get('update', '?')}) "
              f"into {cfg.world.num_agents} individual brains")
        return forked
    if source_mode == target_mode:
        print(f"warm-started from {cfg.policy.init_from} "
              f"({source_mode}, update {blob.get('update', '?')})")
        return source
    raise ValueError(
        f"cannot initialise a '{target_mode}' policy from an '{source_mode}' checkpoint"
    )


def save_checkpoint(path: Path, trainer: PPOTrainer, cfg: Config, update: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state": trainer.policy.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "policy_config": trainer.policy.config_dict(),
            "config": cfg.to_dict(),
            "update": update,
            "agent_steps": trainer.global_step,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--policy-mode", choices=["shared", "individual"], default=None,
                        help="shared = M1 parameter sharing; individual = M2, one brain per agent")
    parser.add_argument("--init-from", default=None,
                        help="checkpoint to fork individual brains from (M2)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides: dict = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.updates is not None:
        overrides["ppo.total_updates"] = args.updates
    if args.num_envs is not None:
        overrides["ppo.num_envs"] = args.num_envs
    if args.device is not None:
        overrides["ppo.device"] = args.device
    if args.policy_mode is not None:
        overrides["policy.mode"] = args.policy_mode
    if args.init_from is not None:
        overrides["policy.init_from"] = args.init_from
    if overrides:
        cfg = cfg.replace(**overrides)

    run_name = args.run_name or time.strftime("run-%Y%m%d-%H%M%S")
    run_dir = Path(cfg.logging.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)

    seed_everything(cfg.seed)
    envs = VecWorld(cfg, seed=cfg.seed, num_envs=cfg.ppo.num_envs)
    policy = build_policy(cfg, envs.obs_dim)

    if cfg.policy.init_from:
        policy = fork_from_checkpoint(policy, cfg, envs.obs_dim)

    trainer = PPOTrainer(cfg, policy, envs, device=cfg.ppo.device)

    start_update = 0
    if args.resume:
        blob = torch.load(args.resume, map_location=cfg.ppo.device, weights_only=False)
        trainer.policy.load_state_dict(blob["policy_state"])
        trainer.optimizer.load_state_dict(blob["optimizer_state"])
        trainer.global_step = blob.get("agent_steps", 0)
        start_update = blob.get("update", 0) + 1
        print(f"resumed {args.resume} at update {start_update}")

    print(f"run        : {run_name}")
    print(f"device     : {cfg.ppo.device}")
    print(f"obs dim    : {envs.obs_dim}   agents: {cfg.world.num_agents}   "
          f"envs: {cfg.ppo.num_envs}   rollout: {cfg.ppo.rollout_ticks}")
    print(f"batch      : {cfg.ppo.rollout_ticks * cfg.ppo.num_envs * cfg.world.num_agents:,} "
          f"agent-steps per update")
    brains = getattr(policy, "num_agents", 1)
    print(f"policy     : {cfg.policy.mode}, {brains} brain(s), "
          f"{sum(p.numel() for p in policy.parameters()):,} parameters total\n")

    baseline_results = []
    if not args.no_baseline:
        print(f"baselines ({cfg.logging.baseline_episodes} episodes, before training):")
        baseline_results = baselines(cfg, cfg.logging.baseline_episodes, seed=10_000)
        for r in baseline_results:
            print("  " + r.line())
        with open(run_dir / "baselines.json", "w") as fh:
            json.dump([r.__dict__ for r in baseline_results], fh, indent=2)
        print()

    logger = MetricsLogger(run_dir / "metrics.csv")
    # Scoped by run: a Milestone 2 run forks from a Milestone 1 checkpoint, and a
    # flat directory would have the fork overwrite the very file it started from.
    checkpoint_dir = Path(cfg.logging.checkpoint_dir) / run_name
    replay_dir = Path(cfg.logging.replay_dir)
    started = time.perf_counter()

    try:
        for update in range(start_update, cfg.ppo.total_updates):
            metrics = trainer.train_update(update)
            if update % cfg.logging.log_every == 0 or update == cfg.ppo.total_updates - 1:
                logger.log(metrics)

            if cfg.logging.checkpoint_every and (update + 1) % cfg.logging.checkpoint_every == 0:
                save_checkpoint(checkpoint_dir / "latest.pt", trainer, cfg, update)
                save_checkpoint(checkpoint_dir / f"update_{update + 1:05d}.pt", trainer, cfg, update)

            if cfg.logging.replay_every and (update + 1) % cfg.logging.replay_every == 0:
                act = policy_act_fn(trainer.policy, device=cfg.ppo.device)
                rec = record_episode(cfg, seed=10_000, act_fn=act,
                                     label=f"{run_name} @ update {update + 1}", source="train")
                rec.save(replay_dir / f"{run_name}_u{update + 1:05d}.json")
    except KeyboardInterrupt:
        print("\ninterrupted; saving a checkpoint before exiting")
    finally:
        logger.close()
        save_checkpoint(checkpoint_dir / "latest.pt", trainer, cfg, cfg.ppo.total_updates - 1)

    elapsed = time.perf_counter() - started
    took = f"{elapsed:.0f}s" if elapsed < 90 else f"{elapsed / 60:.1f} min"
    print(f"\ntrained {trainer.global_step:,} agent-steps in {took}")

    print(f"\nfinal evaluation ({cfg.logging.baseline_episodes} episodes):")
    for r in baseline_results:
        print("  " + r.line())
    act = policy_act_fn(trainer.policy, device=cfg.ppo.device)
    final = evaluate(cfg, act, cfg.logging.baseline_episodes, seed=10_000, label="learned policy")
    print("  " + final.line())

    if baseline_results:
        floor = baseline_results[0]
        print(f"\n  learned / random lifespan ratio: "
              f"{final.mean_lifespan / max(floor.mean_lifespan, 1e-9):.2f}x")

    rec = record_episode(cfg, seed=10_000, act_fn=act,
                         label=f"{run_name} final", source="train")
    path = rec.save(replay_dir / f"{run_name}_final.json")
    print(f"\ncheckpoint : {checkpoint_dir / 'latest.pt'}")
    print(f"metrics    : {run_dir / 'metrics.csv'}")
    print(f"replay     : {path}")


if __name__ == "__main__":
    main()
