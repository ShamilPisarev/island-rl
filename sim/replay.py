"""Replay recording, schema, and the manifest the viewer reads.

REPLAY SCHEMA v1 ... v7
=======================

v7 (Island 3.0) extends v6 and is emitted for any world with a 3.0 block on.
Three optional per-tick fields and one new agent column, all of them things a v6
replay could not show at all:

    agent row   : `adult` -- 0 while an agent is a child. A village whose
                  children are drawn the same size as its parents is a village
                  where the whole reproduction result is invisible.
    per tick    : f = [[slot, x, z], ...] -- fields PLANTED on this tick. A field
                  is a bush that did not exist when the header was written, so
                  the static bush list cannot carry it; the viewer accumulates.
                  `b` already covers every slot, wild bushes first, so a field's
                  berry count needs nothing new.
    per tick    : h = [rooms, ...] -- rooms per site. Without it a house that
                  has been extended four times looks exactly like one that never
                  was, and R3 is the difference between those two pictures.
    agent block : `tribe`, plus a `tribes` array at top level

Same rule as every bump before it: the tick encoding changed, so the version
changes. A v6 reader shown a v7 file would draw children as adults and fields as
nothing at all -- and the second of those is a berry patch appearing from
nowhere, which is precisely what a version check exists to prevent.

REPLAY SCHEMA v1 / v2 / v3 / v4 / v5 / v6
=========================================

v6 (tech ladder rung 2) extends v5 and is emitted only for worlds with
predators. One optional per-tick field, and a bump rather than a quiet addition
for the reason the whole ladder exists:

    per tick    : d = [[x, z], ...] -- where each predator stands, in index
                  order, rounded to 2dp like every other position
    world block : predator_count, predator_attack_radius, predator_speed

A predator that is not drawn is the storm problem again -- agents die at night
to nothing visible, and a watcher concludes the night drain got harsher. The
rung is also the most watchable thing in the project so far, which is a poor
reason to build a mechanic and an excellent reason to render one.

REPLAY SCHEMA v1 / v2 / v3 / v4 / v5
====================================

v5 (Island 2.0 stage 5 viewing) extends v4 and is emitted only for society
worlds. It adds two optional per-tick fields, both of them things a watcher
could previously only INFER:

    per tick    : o = [goal, ...]  -- one arbiter goal id per agent, in id
                  order, index-aligned with `goal_names` at top level
    per tick    : n = [blight, storm_sites]  -- the shock state, omitted
                  entirely on ticks where neither is happening
    agent block : `learn` -- true for an agent whose goals came from a learned
                  arbiter, in a mixed population

Every one of these exists because watching a stage-4 replay could not answer a
question the run was about. The action column says "NE"; it does not say the
agent is walking to a site to deliver, which is the difference between a builder
and a free-rider and the whole subject of stage 5. A storm at tick 500 destroyed
76 units of shelter and was pixel-for-pixel invisible, because the damage is only
readable as a jump in `s` between two frames nobody compares. And in a mixed run
the twenty learned agents were indistinguishable from the eighty scripted ones,
which is the one distinction the run exists to make.

`o` is a separate array rather than two more agent columns because a goal exists
only when an ARBITER drives the world: a replay written by `sim.train` has
actions and no goals, and an optional key is how `g`, `k` and `p` already handle
"this world has no such thing".

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
# v5 adds the goal channel and the shock channel (see the module docstring). Same
# rule again, and the same reason it is a separate constant: a society world with
# no arbiter driving it still carries the shock channel, so the bump is about the
# WORLD, not about who is playing it.
SCHEMA_VERSION_VIEW = 5
# v6 adds the predators' positions. Bumped rather than folded into v5 because a
# v5 reader shown a v6 file would draw a world with an invisible thing killing
# people in it -- which is exactly the failure the version check exists for.
SCHEMA_VERSION_PREDATOR = 6
# v7 adds a per-agent `adult` column, planted fields and per-site rooms. Bumped
# for the usual reason and one specific to this stage: a v6 reader would draw a
# field as nothing, i.e. a berry patch that appears out of thin air.
SCHEMA_VERSION_ISLAND3 = 7
SUPPORTED_VERSIONS = frozenset({1, 2, 3, 4, 5, 6, 7})

AGENT_FIELDS = ["x", "z", "hunger", "food", "alive", "action"]
AGENT_FIELDS_V2 = AGENT_FIELDS + ["wood", "stone"]
AGENT_FIELDS_V7 = AGENT_FIELDS_V2 + ["adult"]
BUSH_FIELDS = ["berries"]

MANIFEST_NAME = "index.json"


def island3_world(cfg: Config) -> bool:
    """Does this config have any Island 3.0 block on?

    One predicate, used by the version choice, the tick encoder and the header,
    so the three cannot drift on what a v7 file is -- the same reason
    `goal_viable` is one function.
    """
    return bool(cfg.reproduction.enabled or cfg.housing.enabled
                or cfg.agriculture.enabled or cfg.tribes.enabled)


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

    def __init__(self, world: World, cfg: Config, label: str, source: str, seed: int,
                 goal_source: Any = None, learn_mask: np.ndarray | None = None) -> None:
        self.world = world
        self.cfg = cfg
        self.label = label
        self.source = source
        self.seed = seed
        # `goal_source` is anything with a `.goals` array of one goal id per
        # agent -- in practice the OptionRunner driving the episode. Duck-typed
        # rather than imported, because replay.py must not depend on utility.py:
        # a recorder that can only be fed by one arbiter is a recorder that has
        # to be edited every time a new one is written.
        self.goal_source = goal_source
        self.learn_mask = None if learn_mask is None else np.asarray(learn_mask, dtype=bool)
        self.ticks: list[dict[str, Any]] = []
        # Which field slots have already been announced. See `snapshot`.
        self._seen_fields: set[int] = set()
        # WILD BUSHES ONLY. Island 3.0 parks its unplanted field slots at
        # infinity so the engine treats them as absent, and `float('inf')` is not
        # representable in JSON -- python writes the literal `Infinity`, which
        # `JSON.parse` rejects outright. So a village replay was unloadable, and
        # would have been even if it had parsed: 40 phantom bushes at the edge of
        # the universe. Fields arrive through the per-tick `f` key instead.
        n_wild = getattr(world, "n_wild_bushes", world.bush_x.shape[0])
        self.bushes = [
            {"x": round(float(x), 2), "z": round(float(z), 2)}
            for x, z in zip(world.bush_x[:n_wild], world.bush_z[:n_wild])
        ]

    def snapshot(self) -> None:
        pool = self.world.pool
        construction = self.cfg.construction.enabled
        island3 = island3_world(self.cfg)
        adult = pool.adult(self.cfg) if island3 else None
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
            if island3:
                row.append(int(adult[i]))
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
            # v5. The goal is the INTENTION behind the action column, and it is
            # the only one of the two that says whether an agent walking north is
            # fetching wood or running home. Recorded on the first snapshot too,
            # where every goal is still the arbiter's initial `rest` -- honest,
            # and the same convention the action column uses.
            if self.goal_source is not None:
                tick["o"] = [int(g) for g in self.goal_source.goals]
            # Omitted entirely on a calm tick, so a world with `shock_interval: 0`
            # writes a file byte-for-byte the size of a v4 one.
            blight = int(bool(self.world.blight_active))
            storm = int(getattr(self.world, "last_storm", 0)) if self.ticks else 0
            if blight or storm:
                tick["n"] = [blight, storm]
        # v6. Every tick, not only the hunting ones: a predator walking home at
        # dawn is part of what makes the island read as inhabited, and a thing
        # that vanishes by day looks like a rendering bug.
        if self.cfg.predators.enabled and self.world.predator_x.size:
            tick["d"] = [[round(float(x), 2), round(float(z), 2)]
                         for x, z in zip(self.world.predator_x, self.world.predator_z)]
        if self.cfg.exchange.enabled and self.ticks and self.world.last_transfers:
            tick["g"] = [[int(g), int(r), int(item)] for g, r, item in self.world.last_transfers]
        # v7. Fields are emitted ONLY on the tick they are planted, because a
        # field never moves afterwards and repeating forty positions every tick
        # for 24,000 ticks is how an 11MB replay becomes a 200MB one. The viewer
        # accumulates, which is the same contract `k` and `g` already have for
        # events.
        if island3:
            n_wild = self.world.n_wild_bushes
            active = self.world.bush_active[n_wild:]
            new_fields = [j for j in range(active.shape[0])
                          if active[j] and j not in self._seen_fields]
            if new_fields:
                for j in new_fields:
                    self._seen_fields.add(j)
                tick["f"] = [[int(j), round(float(self.world.bush_x[n_wild + j]), 2),
                              round(float(self.world.bush_z[n_wild + j]), 2)]
                             for j in new_fields]
            if self.cfg.housing.enabled and self.world.site_x.size:
                tick["h"] = [int(v) for v in self.world.site_rooms]
        self.ticks.append(tick)

    def to_dict(self) -> dict[str, Any]:
        cfg = self.cfg
        stats = self.world.stats()
        n = cfg.world.num_agents
        construction = cfg.construction.enabled
        exchange = cfg.exchange.enabled
        society = cfg.society.enabled
        version = SCHEMA_VERSION
        if island3_world(cfg):
            version = SCHEMA_VERSION_ISLAND3
        elif society:
            # A society world always carries the shock channel, whether or not a
            # shock ever fires, so it is always v5 -- the version says what the
            # reader must be able to parse, not what this particular episode
            # happened to contain.
            version = (SCHEMA_VERSION_PREDATOR if cfg.predators.enabled
                       else SCHEMA_VERSION_VIEW)
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
            "tick_fields": {"agent": (AGENT_FIELDS_V7 if island3_world(cfg)
                                      else AGENT_FIELDS_V2 if construction
                                      else AGENT_FIELDS),
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
            if self.goal_source is not None:
                # Index -> label, exactly as `action_names` is carried, so the
                # viewer never hardcodes a goal list that can drift from the
                # arbiter's.
                from .utility import GOAL_NAMES
                blob["goal_names"] = list(GOAL_NAMES)
            if self.learn_mask is not None:
                # WHICH agents a learned arbiter is driving. Without it a mixed
                # replay is 100 identical-looking agents and the twenty the run is
                # about cannot be picked out of them.
                for agent, flag in zip(blob["agents"], self.learn_mask):
                    agent["learn"] = bool(flag)
            blob["world"]["shock_ramp"] = sc.shock_ramp
            if island3_world(cfg):
                # Static for the episode, so it belongs here rather than on
                # every tick -- the same call `households` already makes. A
                # newborn's tribe is its household's, and a household never
                # changes tribe, so an agent born on tick 4000 is covered by a
                # header written at the end.
                blob["tribes"] = [
                    {"id": t, "color": agent_color(t * 5 + 1,
                                                   max(cfg.tribes.num_tribes, 1))}
                    for t in range(max(cfg.tribes.num_tribes, 1)
                                   if cfg.tribes.enabled else 1)]
                for agent, t, h in zip(blob["agents"], self.world.tribe,
                                       self.world.household):
                    agent["tribe"] = int(t)
                    agent["household"] = int(h)
                blob["world"].update({
                    "num_tribes": cfg.tribes.num_tribes if cfg.tribes.enabled else 1,
                    "housing": cfg.housing.enabled,
                    "base_occupants": cfg.housing.base_occupants,
                    "occupants_per_room": cfg.housing.occupants_per_room,
                    "max_rooms": cfg.housing.max_rooms,
                    "wild_bushes": int(self.world.n_wild_bushes),
                    "field_capacity": cfg.agriculture.field_capacity,
                    "maturity_ticks": cfg.reproduction.maturity_ticks,
                })
                blob["summary"].update({
                    "births": stats.births,
                    "born": stats.born,
                    "population_final": stats.population_final,
                    "house_expansions": stats.house_expansions,
                    "fields_planted": stats.fields_planted,
                    "farming_households": stats.farming_households,
                    "tribe_population": [int(v) for v in stats.tribe_population],
                })
            if cfg.predators.enabled:
                pc = cfg.predators
                blob["world"].update({
                    "predator_count": pc.count,
                    "predator_attack_radius": pc.attack_radius,
                    "predator_speed": pc.speed,
                })
                blob["summary"]["predator_attacks"] = stats.attacks
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
