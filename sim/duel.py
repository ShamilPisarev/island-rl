"""Two saved goal networks compete in one island; no training is performed."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from .arbiter import goal_mask, load_arbiter
from .config import load_config
from .replay import ReplayRecorder
from .utility import GOAL_NAMES, OptionRunner, agent_traits
from .world import World


class TribeArbiter:
    """Route decisions by current allegiance, including births and captures.

    Appended goals have no trained weights. Mask them rather than presenting a
    zero-initialized conquest output as a strategy the network learned.
    """

    def __init__(self, world, checkpoints, seed):
        if world.cfg.world.grow_slots or world.cfg.tribes.num_tribes != 2:
            raise ValueError("Duel requires two tribes and fixed population slots")
        self.world = world
        self.models = [load_arbiter(p, world.cfg) for p in checkpoints]
        self.widths = [torch.load(p, map_location="cpu", weights_only=False)
                       ["policy_config"]["n_actions"] for p in checkpoints]
        self.acfg = self.models[0].acfg
        # Both networks see the same individual traits, independent of side.
        for model in self.models:
            model.traits = agent_traits(world.num_agents, seed, self.acfg)[
                :, :model.traits.shape[1]]
        self.previous_tribes = world.tribe.copy()

    def choose(self, view, mask, rng):
        choices = []
        for model, width in zip(self.models, self.widths):
            available = goal_mask(view, model.cfg, model.acfg).copy()
            available[:, width:] = False
            with torch.inference_mode():
                goals, _, _ = model.policy.act(model._inputs(view),
                    deterministic=True, mask=torch.as_tensor(available))
            choices.append(goals.numpy())
        return np.where(self.world.tribe == 0, choices[0], choices[1])


def play(cfg, checkpoints, seed, path):
    world = World(cfg, seed=seed)
    arb = TribeArbiter(world, checkpoints, seed)
    runner = OptionRunner(arb, cfg, seed=seed, world=world)
    names = [Path(p).parent.name for p in checkpoints]
    rec = ReplayRecorder(world, cfg,
        label=f"Two networks: tribe 0 {names[0]} / tribe 1 {names[1]}",
        source="sim.duel", seed=seed, goal_source=runner,
        learn_mask=np.ones(cfg.world.num_agents, dtype=bool))
    rec.snapshot()
    obs = world.observations()
    initial = [int((world.pool.alive & (world.tribe == t)).sum()) for t in range(2)]
    counts = np.zeros((2, len(GOAL_NAMES)), dtype=int)
    captures = 0
    peak = int(world.pool.alive.sum())
    start = time.perf_counter()
    while True:
        # An assimilated agent must consult its new tribe's network immediately.
        changed = world.tribe != arb.previous_tribes
        runner.ticks_left[changed] = 0
        arb.previous_tribes = world.tribe.copy()
        actions = runner.act(obs, world.action_mask())
        for tribe in range(2):
            alive = world.pool.alive & (world.tribe == tribe)
            np.add.at(counts[tribe], runner.goals[alive], 1)
        result = world.step(actions)
        obs = result.obs
        captures += len(world.last_conquests)
        peak = max(peak, int(world.pool.alive.sum()))
        rec.snapshot()
        if result.episode_done:
            break
    population = [int((world.pool.alive & (world.tribe == t)).sum())
                  for t in range(2)]
    report = {"seed": seed, "networks_by_tribe": names,
        "initial_population": initial,
        "population": population, "captures": captures, "peak_population": peak,
        "population_cap": cfg.world.num_agents,
        "cap_reached": peak >= cfg.world.num_agents,
        "simulation_seconds": round(time.perf_counter() - start, 2),
        "goals_by_tribe": [{name: int(n) for name, n in zip(GOAL_NAMES, row) if n}
                           for row in counts],
        "interpretation": "Exploratory frozen-policy match; population is not proof of military superiority. New untrained goals are masked."}
    # The generic recorder rebuilds its manifest by reading every old replay.
    # Avoid loading hundreds of recordings (possibly offloaded by macOS).
    rec.save(path, update_manifest_file=False)
    manifest_path = Path(path).parent / "index.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"replays": []}
    blob = rec.to_dict()
    manifest["replays"].insert(0, {"file": Path(path).name,
        "label": blob["label"], "source": "sim.duel",
        "schema_version": blob["schema_version"], "ticks": len(blob["ticks"]),
        "generated_at": blob.get("generated_at", ""), "summary": blob.get("summary", {})})
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/duel.yaml")
    ap.add_argument("--a", default="checkpoints/arb7-village/latest.pt")
    ap.add_argument("--b", default="checkpoints/arb5-hh/latest.pt")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--ticks", type=int, default=1200)
    ap.add_argument("--swap", action="store_true", help="also swap networks on the same map")
    args = ap.parse_args()
    if args.ticks < 1:
        ap.error("ticks must be positive")
    torch.set_num_threads(2)
    cfg = load_config(args.config).replace(**{"world.max_ticks": args.ticks})
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    reports = []
    pairs = [(args.a, args.b)] + ([(args.b, args.a)] if args.swap else [])
    for i, pair in enumerate(pairs):
        path = Path(cfg.logging.replay_dir) / f"duel-{stamp}-{i + 1}.json"
        report = play(cfg, pair, args.seed, path)
        report["replay"] = str(path)
        reports.append(report)
        print(json.dumps(report), flush=True)
    # macOS reports bytes; Linux reports KiB.
    import sys
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {"matches": reports, "peak_process_memory_mb": round(
        peak / (1024 ** 2 if sys.platform == "darwin" else 1024), 1)}
    target = Path("viewer/reports") / f"duel-{stamp}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Report: {target}; peak process memory: {out['peak_process_memory_mb']} MB")


if __name__ == "__main__":
    main()
