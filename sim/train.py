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
from .evaluate import baselines, evaluate, make_act_fn, policy_act_fn
from .metrics import MetricsLogger
from .policy import build_policy
from .ppo import PPOTrainer
from .replay import record_episode
from .world import VecWorld


def seed_everything(seed: int) -> None:
    """Seed torch and numpy. World RNGs are seeded separately and explicitly."""
    torch.manual_seed(seed)
    np.random.seed(seed)


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
    print(f"policy     : {sum(p.numel() for p in policy.parameters()):,} parameters\n")

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
    checkpoint_dir = Path(cfg.logging.checkpoint_dir)
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
