"""Replay recording, schema, and the manifest the viewer reads.

REPLAY SCHEMA v1 / v2 / v3
==========================

v3 (Milestone 5) extends v2 and is emitted only for exchange worlds. It adds one
optional per-tick field:

    per tick    : g = [[giver, receiver, item], ...] -- every transfer that
                  resolved during this tick, item being 0 food / 1 wood / 2 stone
    world block : give_radius

A transfer is the only event that leaves no trace in the post-step state: the
inventories move, but nothing says who handed what to whom, and "agent 3 gained
a berry" is indistinguishable from a gather. Hence an explicit list. Ticks with
no transfers omit the key entirely, so a replay from a world where nobody ever
gives is the same size as a v2 one.


v2 (Milestone 4) extends v1 and is emitted only for construction worlds:
non-construction worlds keep the v1 encoding unchanged, so every replay written
before M4 stays valid. What v2 adds:

    agent rows  : two extra columns, wood and stone carried
    trees       : [{x, z}]  static, index-aligned with per-tick `w`
    rocks       : [{x, z}]  static, index-aligned with per-tick `r`
    sites       : [{x, z}]  static, index-aligned with per-tick `s`
    per tick    : w = wood left per tree, r = stone left per rock,
                  s = [wood_still_needed, stone_still_needed] per site
    world block : night_cycle, night_fraction, shelter_radius, site costs --
                  the viewer derives day/night from the tick index and these,
                  rather than trusting a redundant per-tick flag

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

v4 (Island 2.0 stage 4) additionally carries, per tick::

    p : [[stock_food, stock_material], ...]   -- one pair per household
    k : [[raider, victim household, item], ...]  -- raids resolved this tick

and, at top level, a ``households`` array (id, x, z, colour -- static for the
episode, index-aligned with ``p``) plus a ``household`` key on every agent.

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

from .agents import IDLE, ITEM_NAMES, action_names
from .config import Config
from .world import World

SCHEMA_VERSION = 1
SCHEMA_VERSION_CONSTRUCTION = 2
SCHEMA_VERSION_EXCHANGE = 3
# Island 2.0 stage 4. Bumped because the tick encoding genuinely changed: a tick
# now carries the household stockpiles (`p`) and the raids that happened (`k`).
# Stage 3 deliberately did NOT bump -- the design doc asked whether animation
# state needed a schema change and the action column already carried it -- and the
# rule that made that the right call is the same one that makes this the wrong
# place to economise: bump on any change to the tick encoding.
SCHEMA_VERSION_SOCIETY = 4
SUPPORTED_VERSIONS = frozenset({1, 2, 3, 4})

AGENT_FIELDS = ["x", "z", "hunger", "food", "alive", "action"]
AGENT_FIELDS_V2 = AGENT_FIELDS + ["wood", "stone"]
BUSH_FIELDS = ["berries"]

MANIFEST_NAME = "index.json"


class ReplaySchemaError(ValueError):
    """Raised when a replay's schema version is not one we can read."""


def agent_color(index: int, total: int) -> str:
    """A distinguishable colour per agent, at six agents and at a hundred.

    Evenly spaced hues were fine for six and fail for a hundred: adjacent ids
    land 3.6 degrees apart, which is the same colour. Two changes fix it.

    The golden-angle step (0.381966..., i.e. 1 - 1/phi) walks the hue wheel so
    that *consecutive* indices are always far apart -- the standard trick for
    "I do not know how many I will need". And saturation/value cycle through a
    few levels on a different period from the hue, so two agents that do land on
    a similar hue differ in weight instead. Six-agent replays stay well spread
    (hues ~137 degrees apart) and their colours simply differ from those written
    before this change, which is cosmetic: nothing keys off an agent's colour.
    """
    h = (index * 0.381966011250105) % 1.0
    sat = (0.72, 0.52, 0.88)[index % 3]
    val = (0.98, 0.80, 0.90, 0.68)[index % 4]
    r, g, b = colorsys.hsv_to_rgb(h, sat, val)
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
        construction = self.cfg.construction.enabled
        rows = []
        for i in range(pool.n):
            row = [
                round(float(pool.x[i]), 2),
                round(float(pool.z[i]), 2),
                round(float(pool.hunger[i]), 1),
                int(pool.food[i]),
                int(pool.alive[i]),
                int(pool.last_action[i]) if self.ticks else IDLE,
            ]
            if construction:
                row += [int(pool.wood[i]), int(pool.stone[i])]
            rows.append(row)
        tick: dict[str, Any] = {
            "t": int(self.world.tick),
            "a": rows,
            "b": [int(v) for v in self.world.bush_berries],
        }
        if construction:
            tick["w"] = [int(v) for v in self.world.tree_wood]
            tick["r"] = [int(v) for v in self.world.rock_stone]
            tick["s"] = [[int(a), int(b)] for a, b in
                         zip(self.world.site_wood_needed, self.world.site_stone_needed)]
        # The first snapshot is the pre-action state, so it can carry no
        # transfers even if the world object still holds some from a previous
        # episode -- `self.ticks` being empty is the same "before anything
        # happened" test the action column uses.
        if self.cfg.society.enabled:
            # Stockpiles as a pair per household, so one array covers both
            # economies and the viewer needs no second key. Raids are events, like
            # transfers: nothing in the post-step state records that a pile was
            # robbed rather than drawn down by its owners.
            tick["p"] = [[int(f), int(m)] for f, m in
                         zip(self.world.stock_food, self.world.stock_material)]
            if self.ticks and self.world.last_raids:
                tick["k"] = [[int(a), int(h), int(item)]
                             for a, h, item in self.world.last_raids]
        if self.cfg.exchange.enabled and self.ticks and self.world.last_transfers:
            tick["g"] = [[int(g), int(r), int(item)] for g, r, item in self.world.last_transfers]
        self.ticks.append(tick)

    def to_dict(self) -> dict[str, Any]:
        cfg = self.cfg
        stats = self.world.stats()
        n = cfg.world.num_agents
        construction = cfg.construction.enabled
        exchange = cfg.exchange.enabled
        society = cfg.society.enabled
        version = SCHEMA_VERSION
        if society:
            version = SCHEMA_VERSION_SOCIETY
        elif exchange:
            version = SCHEMA_VERSION_EXCHANGE
        elif construction:
            version = SCHEMA_VERSION_CONSTRUCTION
        blob = {
            "schema_version": version,
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
            "action_names": list(action_names(cfg)),
            "agents": [{"id": i, "color": agent_color(i, n)} for i in range(n)],
            "bushes": self.bushes,
            "tick_fields": {"agent": AGENT_FIELDS_V2 if construction else AGENT_FIELDS,
                            "bush": BUSH_FIELDS},
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
        if construction:
            cc = cfg.construction
            world = self.world
            blob["world"].update({
                "night_cycle": cc.night_cycle,
                "night_fraction": cc.night_fraction,
                "shelter_radius": cc.shelter_radius,
                "partial_shelter": cc.partial_shelter,
                "site_wood_cost": cc.site_wood_cost,
                "site_stone_cost": cc.site_stone_cost,
                "material_capacity": cc.material_capacity,
                "tree_wood": cc.tree_wood,
                "rock_stone": cc.rock_stone,
            })
            blob["trees"] = [{"x": round(float(x), 2), "z": round(float(z), 2)}
                             for x, z in zip(world.tree_x, world.tree_z)]
            blob["rocks"] = [{"x": round(float(x), 2), "z": round(float(z), 2)}
                             for x, z in zip(world.rock_x, world.rock_z)]
            blob["sites"] = [{"x": round(float(x), 2), "z": round(float(z), 2)}
                             for x, z in zip(world.site_x, world.site_z)]
            blob["summary"].update({
                "wood_gathered": stats.wood_gathered,
                "stone_gathered": stats.stone_gathered,
                "shelters_completed": stats.shelters_completed,
                "night_ticks_sheltered": stats.night_ticks_sheltered,
                "night_ticks_exposed": stats.night_ticks_exposed,
            })
        if exchange:
            blob["world"]["give_radius"] = cfg.exchange.give_radius
            blob["item_names"] = list(ITEM_NAMES)
            blob["summary"].update({
                "gifts": stats.gifts,
                "food_given": stats.food_given,
                "materials_given": stats.materials_given,
            })
        if society:
            sc = cfg.society
            blob["world"].update({
                "num_households": sc.num_households,
                "stockpile_radius": sc.stockpile_radius,
                "stockpile_food_capacity": sc.stockpile_food_capacity,
                "stockpile_material_capacity": sc.stockpile_material_capacity,
            })
            # Household membership and home position are static for an episode, so
            # they belong here and not on every tick. `home` is index-aligned with
            # the tick's `p`.
            blob["households"] = [
                {"id": h,
                 "x": round(float(self.world.stock_x[h]), 2),
                 "z": round(float(self.world.stock_z[h]), 2),
                 "color": agent_color(h * 7 + 3, sc.num_households)}
                for h in range(sc.num_households)
            ]
            for agent, h in zip(blob["agents"], self.world.household):
                agent["household"] = int(h)
            blob["summary"].update({
                "deposits": stats.deposits,
                "withdrawals": stats.withdrawals,
                "raids": stats.raids,
                "blight_ticks": stats.blight_ticks,
                "storms": stats.storms,
                "shelters_damaged": stats.shelters_damaged,
            })
        return blob

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

    ``act_fn`` takes an ``(A, obs_dim)`` array plus the world's action mask and
    returns ``(A,)`` int actions.
    Every replay producer in the project (random baseline, scripted forager,
    trained policy) goes through here, so they cannot drift apart.
    """
    world = World(cfg, seed=seed)
    recorder = ReplayRecorder(world, cfg, label=label, source=source, seed=seed)
    recorder.snapshot()
    limit = max_ticks if max_ticks is not None else cfg.world.max_ticks
    obs = world.observations()
    for _ in range(limit):
        actions = act_fn(obs, world.action_mask())
        result = world.step(actions)
        recorder.snapshot()
        obs = result.obs
        if result.episode_done:
            break
    return recorder
