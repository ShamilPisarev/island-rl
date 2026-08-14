"""Typed configuration loaded from YAML.

Everything tunable lives in ``config/default.yaml``; this module turns it into
frozen dataclasses so typos surface at load time rather than as an AttributeError
2000 updates into a training run. ``Config.to_dict`` round-trips back to plain
data so a run's exact configuration can be embedded in its replay file.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


@dataclass(frozen=True)
class WorldConfig:
    island_radius: float = 40.0
    num_agents: int = 6
    max_ticks: int = 600
    move_step: float = 0.8
    spawn_radius_frac: float = 0.6


@dataclass(frozen=True)
class HungerConfig:
    max: float = 100.0
    drain_per_tick: float = 0.5
    eat_threshold: float = 60.0
    eat_restore: float = 35.0


@dataclass(frozen=True)
class FoodConfig:
    capacity: int = 3


@dataclass(frozen=True)
class BushConfig:
    num_clusters: int = 5
    bushes_per_cluster: int = 4
    cluster_std: float = 5.0
    cluster_radius_frac: float = 0.75
    capacity: int = 6
    initial_berries: int = 6
    regrow_ticks: int = 50
    gather_radius: float = 2.0
    resample_each_episode: bool = True

    @property
    def count(self) -> int:
        return self.num_clusters * self.bushes_per_cluster


@dataclass(frozen=True)
class CompetitionConfig:
    """Milestone 3. All off by default, so M1/M2 worlds are bit-identical."""

    contest_bushes: bool = False   # only one agent may take from a bush per tick
    exclusive_bushes: bool = False  # ...and only the closest agent may take at all
    enable_steal: bool = False     # adds an 11th action: take a berry from a neighbour
    steal_radius: float = 2.5
    observe_neighbour_food: bool = False  # neighbours' carried food enters the observation
    observe_bush_contested: bool = False  # per bush: is a living rival closer than me?
    mask_invalid_actions: bool = False    # hide gather/steal when they cannot succeed


@dataclass(frozen=True)
class ConstructionConfig:
    """Milestone 4. All off by default, so M1-M3 worlds are bit-identical."""

    enabled: bool = False
    # material nodes, placed like bushes (clustered)
    num_trees: int = 8
    tree_wood: int = 6              # units per tree; no regrowth within an episode
    num_rocks: int = 5
    rock_stone: int = 4
    harvest_radius: float = 2.0
    material_capacity: int = 2      # carried wood+stone combined
    # shelter sites
    num_sites: int = 3
    sites_at_clusters: bool = False  # place shelter sites on the berry clusters
    site_wood_cost: int = 4         # delivered units to complete a site
    site_stone_cost: int = 2
    build_radius: float = 2.5
    shelter_radius: float = 6.0     # protection range of a COMPLETED shelter
    # the hazard shelter protects from
    night_cycle: int = 200          # ticks per full day
    night_fraction: float = 0.25    # last quarter of each cycle is night
    night_drain_multiplier: float = 3.0
    # observation channels
    k_trees: int = 2
    k_rocks: int = 2
    k_sites: int = 2


@dataclass(frozen=True)
class ObservationConfig:
    k_bushes: int = 4
    k_agents: int = 3
    distance_scale: float = 20.0


@dataclass(frozen=True)
class RewardConfig:
    alive_per_tick: float = 0.01
    gather: float = 1.0
    eat: float = 2.0
    death: float = -10.0
    steal: float = 0.0   # 0.0 is the brief-faithful default: theft earns nothing
                         # directly and must pay for itself through the food. Only
                         # config/m3_shaped.yaml raises it, as a labelled ablation.
    # Milestone 4 shaping. The brief calls M4 the milestone that NEEDS shaping --
    # the terminal chain (chop -> carry -> build -> survive the night) is far too
    # long for the survival signal alone. These are documented bootstraps: m4.yaml
    # sets them non-zero, m4_unshaped.yaml is the control, and the annealing test
    # retrains with them returned to zero.
    wood: float = 0.0
    stone: float = 0.0
    build: float = 0.0
    complete: float = 0.0  # split among contributors when a shelter completes


@dataclass(frozen=True)
class PolicyConfig:
    hidden_sizes: tuple[int, ...] = (128, 128)
    mode: str = "shared"          # "shared" (M1) or "individual" (M2)
    init_from: str | None = None  # checkpoint to fork individual brains from


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 32
    rollout_ticks: int = 128
    total_updates: int = 300
    epochs: int = 4
    num_minibatches: int = 4
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    ent_coef_final: float | None = None  # anneal ent_coef to this by the last update
    max_grad_norm: float = 0.5
    device: str = "cpu"


@dataclass(frozen=True)
class LoggingConfig:
    run_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    replay_dir: str = "viewer/replays"
    log_every: int = 1
    checkpoint_every: int = 25
    replay_every: int = 50
    baseline_episodes: int = 20


@dataclass(frozen=True)
class Config:
    seed: int = 0
    world: WorldConfig = field(default_factory=WorldConfig)
    hunger: HungerConfig = field(default_factory=HungerConfig)
    food: FoodConfig = field(default_factory=FoodConfig)
    bushes: BushConfig = field(default_factory=BushConfig)
    competition: CompetitionConfig = field(default_factory=CompetitionConfig)
    construction: ConstructionConfig = field(default_factory=ConstructionConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **overrides: Any) -> "Config":
        """Return a copy with dotted-path overrides applied, e.g. ``ppo.num_envs=4``.

        Used by tests and CLI flags to shrink a run without maintaining a second
        YAML file that will inevitably drift from the real one.
        """
        nested: dict[str, dict[str, Any]] = {}
        top: dict[str, Any] = {}
        for key, value in overrides.items():
            if "." in key:
                section, _, leaf = key.partition(".")
                nested.setdefault(section, {})[leaf] = value
            else:
                top[key] = value
        for section, leaves in nested.items():
            current = getattr(self, section)
            top[section] = dataclasses.replace(current, **leaves)
        return dataclasses.replace(self, **top)


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    """Instantiate a (flat) config dataclass from a mapping, rejecting unknown keys.

    Silently ignoring a typo'd key is the classic way to spend an afternoon
    wondering why a config change did nothing, so unknown keys are fatal.
    """
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    if "hidden_sizes" in data:
        data["hidden_sizes"] = tuple(int(v) for v in data["hidden_sizes"])
    return cls(**data)


_SECTIONS: dict[str, type] = {
    "world": WorldConfig,
    "hunger": HungerConfig,
    "food": FoodConfig,
    "bushes": BushConfig,
    "competition": CompetitionConfig,
    "construction": ConstructionConfig,
    "observation": ObservationConfig,
    "reward": RewardConfig,
    "policy": PolicyConfig,
    "ppo": PPOConfig,
    "logging": LoggingConfig,
}


def config_from_dict(data: dict[str, Any]) -> Config:
    data = dict(data)
    data.pop("extends", None)  # resolved by load_config before we get here
    sections = {name: _build(cls, data.pop(name, None)) for name, cls in _SECTIONS.items()}
    unknown = set(data) - {"seed"}
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    return Config(seed=int(data.get("seed", 0)), **sections)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, leaving both untouched."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a YAML config, resolving a chain of ``extends`` parents.

    A config may name a parent to inherit from::

        extends: default.yaml
        bushes:
          capacity: 3

    Only the keys it restates are overridden, so a milestone that changes four
    numbers says exactly those four and cannot silently drift from the base
    config. Parent paths are relative to the child's own directory.
    """
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in (*seen, path))
        raise ValueError(f"circular config extends: {chain}")
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.get("extends")
    if not parent:
        return raw
    return _deep_merge(_load_raw(path.parent / parent, (*seen, path)), raw)


def load_config(path: str | Path | None = None) -> Config:
    return config_from_dict(_load_raw(Path(path) if path is not None else DEFAULT_CONFIG_PATH))
