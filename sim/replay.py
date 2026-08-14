"""Replay recording, schema, and the manifest the viewer reads.

REPLAY SCHEMA v1
================

A replay is a single JSON object. Training and evaluation write them; the viewer
reads them and renders them back. The viewer MUST refuse to render anything whose
``schema_version`` it does not know -- a silently mis-rendered replay is worse
than no replay, because you will believe what you are looking at.

Top level::

    schema_version : int      -- bump on ANY change to the tick encoding
    generated_at   : str      -- ISO-8601 UTC
    label          : str      -- human name shown in the viewer's dropdown
    source         : str      -- "train" | "evaluate" | "random" | "scripted" | "fake"
    seed           : int
    config         : object   -- the full Config used, for reproduction
    world          : object   -- the handful of constants the viewer needs:
                                 island_radius, num_agents, max_ticks, max_hunger,
                                 food_capacity, bush_capacity, gather_radius
    action_names   : [str]    -- index -> label, so the viewer never hardcodes them
    agents         : [{id, color}]
    bushes         : [{x, z}] -- static positions, index-aligned with the tick's `b`
    tick_fields    : object   -- self-describing column names for `a` and `b`
    ticks          : [Tick]
    summary        : object   -- end-of-episode stats

Tick::

    t : int
    a : [[x, z, hunger, food, alive, action], ...]   -- one row per agent, in id order
    b : [berries, ...]                               -- one entry per bush

Positions are rounded to 2dp and hunger to 1dp. On a 40-unit island that is far
below what the eye can resolve, and it roughly halves the file size.

The first entry is the world state at t=0 before any action is taken (its
``action`` column is ``idle`` for every agent), so ``len(ticks) == steps + 1``.
"""

from __future__ import annotations

import colorsys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .agents import ACTION_NAMES, IDLE
from .config import Config
from .world import World

SCHEMA_VERSION = 1
SUPPORTED_VERSIONS = frozenset({1})

AGENT_FIELDS = ["x", "z", "hunger", "food", "alive", "action"]
BUSH_FIELDS = ["berries"]

MANIFEST_NAME = "index.json"


class ReplaySchemaError(ValueError):
    """Raised when a replay's schema version is not one we can read."""


def agent_color(index: int, total: int) -> str:
    """Evenly spaced hues so six agents stay distinguishable at a glance."""
    h = (index / max(total, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.62, 0.98)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


class ReplayRecorder:
    """Accumulates one episode of one world into the v1 schema.

    Call :meth:`snapshot` once before stepping and once after every step. The
    recorder reads the world directly rather than being handed diffs, which keeps
    it impossible for the replay to disagree with the simulation.
    """

    def __init__(self, world: World, cfg: Config, label: str, source: str, seed: int) -> None:
        self.world = world
        self.cfg = cfg
        self.label = label
        self.source = source
        self.seed = seed
        self.ticks: list[dict[str, Any]] = []
        self.bushes = [
            {"x": round(float(x), 2), "z": round(float(z), 2)}
            for x, z in zip(world.bush_x, world.bush_z)
        ]

    def snapshot(self) -> None:
        pool = self.world.pool
        rows = [
            [
                round(float(pool.x[i]), 2),
                round(float(pool.z[i]), 2),
                round(float(pool.hunger[i]), 1),
                int(pool.food[i]),
                int(pool.alive[i]),
                int(pool.last_action[i]) if self.ticks else IDLE,
            ]
            for i in range(pool.n)
        ]
        self.ticks.append(
            {
                "t": int(self.world.tick),
                "a": rows,
                "b": [int(v) for v in self.world.bush_berries],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        cfg = self.cfg
        stats = self.world.stats()
        n = cfg.world.num_agents
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label": self.label,
            "source": self.source,
            "seed": self.seed,
            "config": cfg.to_dict(),
            "world": {
                "island_radius": cfg.world.island_radius,
                "num_agents": n,
                "max_ticks": cfg.world.max_ticks,
                "max_hunger": cfg.hunger.max,
                "food_capacity": cfg.food.capacity,
                "bush_capacity": cfg.bushes.capacity,
                "gather_radius": cfg.bushes.gather_radius,
            },
            "action_names": list(ACTION_NAMES),
            "agents": [{"id": i, "color": agent_color(i, n)} for i in range(n)],
            "bushes": self.bushes,
            "tick_fields": {"agent": AGENT_FIELDS, "bush": BUSH_FIELDS},
            "ticks": self.ticks,
            "summary": {
                "ticks": stats.ticks,
                "deaths": stats.deaths,
                "survivors": stats.survivors,
                "berries_gathered": stats.berries_gathered,
                "meals": stats.meals,
                "mean_lifespan": round(stats.mean_lifespan, 2),
                "mean_final_hunger": round(stats.mean_final_hunger, 2),
            },
        }

    def save(self, path: str | Path, update_manifest_file: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, separators=(",", ":"))
        if update_manifest_file:
            update_manifest(path.parent)
        return path


def load_replay(path: str | Path) -> dict[str, Any]:
    """Load and version-check a replay."""
    with open(path, "r") as fh:
        data = json.load(fh)
    version = data.get("schema_version")
    if version not in SUPPORTED_VERSIONS:
        raise ReplaySchemaError(
            f"{path}: replay schema v{version!r} is not supported "
            f"(this build reads {sorted(SUPPORTED_VERSIONS)})"
        )
    return data


def replay_to_arrays(data: dict[str, Any]) -> dict[str, np.ndarray]:
    """Tick list -> arrays, for tests and post-hoc analysis.

    Returns ``agents`` of shape (T, A, 6) and ``bushes`` of shape (T, B).
    """
    agents = np.array([tick["a"] for tick in data["ticks"]], dtype=np.float64)
    bushes = np.array([tick["b"] for tick in data["ticks"]], dtype=np.int64)
    return {"agents": agents, "bushes": bushes}


def update_manifest(replay_dir: str | Path) -> Path:
    """Rewrite ``replays/index.json``.

    The viewer is a static page: it cannot list a directory over http, and
    ``fetch`` on ``file://`` is blocked outright. So the writer maintains an
    index and the viewer reads that. Drag-and-drop remains the fallback.
    """
    replay_dir = Path(replay_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted(replay_dir.glob("*.json")):
        if path.name == MANIFEST_NAME:
            continue
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        entries.append(
            {
                "file": path.name,
                "label": data.get("label", path.stem),
                "source": data.get("source", "?"),
                "schema_version": data.get("schema_version"),
                "ticks": len(data.get("ticks", [])),
                "generated_at": data.get("generated_at", ""),
                "summary": data.get("summary", {}),
            }
        )
    entries.sort(key=lambda e: e["generated_at"], reverse=True)
    manifest = replay_dir / MANIFEST_NAME
    with open(manifest, "w") as fh:
        json.dump({"schema_version": SCHEMA_VERSION, "replays": entries}, fh, indent=1)
    return manifest


def record_episode(
    cfg: Config,
    seed: int,
    act_fn,
    label: str,
    source: str,
    max_ticks: int | None = None,
) -> ReplayRecorder:
    """Run one episode under ``act_fn(obs) -> actions`` and record it.

    ``act_fn`` takes an ``(A, obs_dim)`` array and returns ``(A,)`` int actions.
    Every replay producer in the project (random baseline, scripted forager,
    trained policy) goes through here, so they cannot drift apart.
    """
    world = World(cfg, seed=seed)
    recorder = ReplayRecorder(world, cfg, label=label, source=source, seed=seed)
    recorder.snapshot()
    limit = max_ticks if max_ticks is not None else cfg.world.max_ticks
    obs = world.observations()
    for _ in range(limit):
        actions = act_fn(obs)
        result = world.step(actions)
        recorder.snapshot()
        obs = result.obs
        if result.episode_done:
            break
    return recorder
