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
    # An action, once chosen, persists for this many ticks; agents decide only on
    # ticks where tick % decision_interval == 0. Enforced by World.step itself, so
    # every driver (trainer, evaluation, replay, scripted baselines) lives under
    # the same commitment and none of them can drift.
    #
    # The temporal-resolution lever from the nav-spread postmortem: the per-step
    # advantage of walking toward food is ~0.01-0.05 against ~0.9 of advantage
    # noise (sim.advantage), so a single-tick direction choice is invisible to
    # PPO. Committing for k ticks multiplies the per-decision slope by k without
    # touching the reward. 1 reproduces every earlier world bit-identically.
    decision_interval: int = 1


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
    # Place trees and rocks on the berry clusters too.
    #
    # m4b moved the sites onto the clusters and stopped there, which relocated
    # the uncreditable walk to the harvest leg instead of deleting it. Measured
    # on the m4c-anneal policy: mean distance to the nearest tree 10.2 and to the
    # nearest rock 15.5, against 2.4 to the nearest bush, and agents stand within
    # harvest_radius on 2.7% of ticks -- so `chop` is *reachable* on 0.2% of
    # ticks, and it is taken on 82-93% of those. The policy is not declining to
    # harvest; it is almost never in a position to.
    materials_at_clusters: bool = False
    # Let either material count toward a site's remaining cost.
    #
    # With separate wood and stone costs, sites deadlock on COMPOSITION rather
    # than on volume. Measured on m4g: 22.5% of sites end an episode with all
    # their wood in and one stone missing, 10% the reverse, and agents finish
    # holding 3.7 stone -- the material exists, it is simply in the wrong hands,
    # because each agent feeds its nearest site whatever it happens to carry and
    # nothing routes the last stone to the site that wants it. The scripted
    # builder dodges this by targeting one focal site globally; six independent
    # brains have no such coordination.
    #
    # Fungible deliveries remove the mismatch without removing either economy:
    # both wood and stone still have to be harvested and carried, and `mine` is
    # still worth taking, but a site takes whatever arrives.
    fungible_materials: bool = False
    site_wood_cost: int = 4         # delivered units to complete a site
    site_stone_cost: int = 2
    build_radius: float = 2.5
    shelter_radius: float = 6.0     # protection range of a COMPLETED shelter
    # the hazard shelter protects from
    night_cycle: int = 200          # ticks per full day
    night_fraction: float = 0.25    # last quarter of each cycle is night
    night_drain_multiplier: float = 3.0
    partial_shelter: bool = False    # half-built walls give half the protection
    # How much of a shelter's protection is withheld until it is FINISHED.
    #
    # partial_shelter alone makes protection linear in build progress, which
    # fixed the credit-assignment cliff by removing the reason to complete: the
    # last unit buys exactly what the first one did. Measured on m4c, every unit
    # of a 4-unit site is worth 0.25 of the night drain, final unit included.
    # That is why nobody finishes -- not a perception failure, correct play.
    #
    # With a premium p, a site at progress q protects q*(1-p) and a finished one
    # protects 1.0. Every unit still buys something (the gradient partial_shelter
    # was introduced for survives) and the last one additionally buys p. 0.0
    # reproduces every earlier M4 result exactly.
    completion_premium: float = 0.0
    # observation channels
    k_trees: int = 2
    k_rocks: int = 2
    k_sites: int = 2
    # One extra site channel: "one more unit, of the kind I am already carrying,
    # finishes this". Derivable in principle from own.wood/own.stone and the
    # site's need channels, but only as a conjunction across distant parts of a
    # 55-dim vector, which is exactly the shape of thing a small MLP does badly.
    # Same call as observe_neighbour_food before theft and the M3 action mask:
    # a mechanic the policy cannot see is one it cannot respond to.
    observe_final_unit: bool = False


@dataclass(frozen=True)
class ExchangeConfig:
    """Milestone 5. Off by default, so M1-M4 worlds are bit-identical.

    Enabling exchange appends two actions, ``give_food`` and ``give_material``.
    Food and materials are split because they are two different economies -- one
    keeps an agent alive, the other builds shelter -- and an agent carrying both
    would otherwise be unable to choose which it is participating in. Wood and
    stone share one action: both are construction inputs, the receiver's ``build``
    already resolves which the site needs, and a third give action would be a
    third thing PPO has to discover the value of.
    """

    enabled: bool = False
    give_radius: float = 2.5          # same reach as a steal, so giving is not cheaper
    observe_neighbour_materials: bool = False  # neighbours' carried wood/stone
    log_transfers: bool = False       # accumulate a per-episode transfer ledger


@dataclass(frozen=True)
class SocietyConfig:
    """Island 2.0 stage 4: regions, households, stockpiles, reputation, shocks.

    All off by default, so every 1.0 world (and stages 1-3) stays bit-identical.
    Enabling this appends five actions -- deposit/withdraw food and material, and
    raid -- and widens the observation; the append-never-insert rule that carried
    an M1 checkpoint into M3 applies here too.

    WHY THESE FIVE MECHANICS AND NOT MORE. Stage 2 measured what the 100-agent
    utility population was missing, and every item here answers one of those
    findings rather than being a feature someone fancied:

      * `region_split` -- stage 2's world had wood and stone on every cluster
        (the m4f lineage's `materials_at_clusters`), so nobody ever needed
        anything from anybody. M5's own postmortem says a relay needs its chain
        shortened by GEOGRAPHY, not its deliveries made fungible; splitting the
        island is that.
      * households + stockpiles -- "construction is over by the first nightfall"
        (20 sites x 4 units against 100 agents carrying one each), and lifespan
        Gini was 0.058 because there was nothing for inequality to accumulate
        IN. A stockpile is a store, a sustained demand, and a raid target.
      * reputation -- retaliation and guarding without any scripted war logic.
      * shocks -- populations that never get stressed never visibly cooperate.
        Deterministic from the seed, like everything else here.
    """

    enabled: bool = False

    # --- regions. Trees to one side of the island, rocks to the other, berries
    # everywhere. `region_axis` is the compass angle (radians) of the wood half's
    # outward normal; 0 puts wood at +z ("north") and stone at -z.
    region_split: bool = False
    region_axis: float = 0.0

    # --- households. Agents are dealt round-robin, so a household is a stable
    # group from tick 0 and "my group" needs no learning to identify. Each
    # household owns the site of the same index (num_sites must be >= this), and
    # that site's position is also its stockpile's.
    num_households: int = 1
    observe_household: bool = False   # per-neighbour same-household flag

    # --- stockpiles. Capacity is per household, not per agent, which is the
    # whole point: it is the first thing in this project bigger than a pocket.
    stockpile_food_capacity: int = 12
    stockpile_material_capacity: int = 12
    stockpile_radius: float = 2.5
    # May an agent rob a member of its OWN household?
    #
    # This is a correction, not a taste call, and it is stage 2's correction 1
    # arriving in a new world exactly as rule 5 says to expect. Stage 4 spawns a
    # household together at its own site, which packs five agents inside
    # `steal_radius` of each other permanently -- and an opportunistic steal goal
    # that only needs a loaded victim in reach then fires every tick. Measured:
    # 8105 steals an episode against stage 2's 5.8% of goal-ticks becoming 13.8%,
    # tripping the pre-registered PERMANENT WAR check, and cascading into raids
    # because every housemate held a saturated grudge against every other.
    #
    # Immunity is also the coherent reading of the mechanic: a household shares a
    # stockpile, so taking from a housemate's pocket is not how you get food out
    # of your own group -- `withdraw` is. Theft stays available against everyone
    # else, which is what keeps competition between households real.
    household_theft_immunity: bool = False

    # --- reputation. grudge[i, j] in [0, 1] is how much i remembers j taking
    # from it (a steal from i, or a raid on i's household stockpile). Decays
    # geometrically so an old robbery stops mattering, which is what lets a
    # feud end.
    reputation: bool = False
    grudge_per_theft: float = 0.34    # three thefts saturate the memory
    grudge_decay: float = 0.995       # half-life ~138 ticks
    observe_grudge: bool = False      # per-neighbour grudge channel

    # --- shocks. Every `shock_interval` ticks one fires, alternating
    # deterministically between a blight (berry regrowth stops) and a storm
    # (every finished shelter loses units and needs rebuilding). 0 disables.
    shock_interval: int = 0
    blight_ticks: int = 60            # how long a blight suspends regrowth
    storm_damage: int = 2             # units knocked out of each finished site
    observe_shock: bool = False       # one channel: is a blight running
    # Escalation: shock SEVERITY grows linearly with episode progress. A shock
    # at tick t is scaled by 1 + shock_ramp * t/max_ticks, so at ramp 2.0 the
    # last shock of an episode is ~3x the first (blights ~3x longer, storms
    # ~3x the damage, clamped so a site never needs more than it costs). The
    # cadence is untouched -- same clock, same seed stream -- so 0.0 reproduces
    # every earlier world bit-identically. sim.economy models the ramp; re-size
    # before reading anything behavioural off a ramped world (rule 5).
    shock_ramp: float = 0.0


@dataclass(frozen=True)
class ToolsConfig:
    """Island 2.0 tech ladder, rung 1: a craftable axe.

    Off by default, so every 1.0 world and every island2 stage stays
    bit-identical. Enabling it appends ONE action (`craft`, index 21) and ONE
    observation channel (`own.axe`), by the append-never-insert rule that carried
    an M1 checkpoint into M3 -- and, like `society`, enabling it implies the whole
    block below it is present, so `craft` is a fixed index in every tool world.

    WHY AN AXE AND WHY THIS SHAPE. It is the cheapest thing in this project that
    is honestly a *technology*: a durable object an agent makes out of two things
    it already gathers, which then changes the rate at which it can gather one of
    them. The chain is short enough to be creditable (craft once, benefit every
    chop afterwards) -- which matters, because every long chain this project has
    tried has failed, and the point of rung 1 is to establish the ladder, not to
    re-run the credit-assignment wall.

    IT DOES NOT CREATE WOOD. Trees hold a finite stock, so an axe buys TICKS, not
    supply: the same wood arrives in half the trips. `sim.economy` says which of
    those two the world is actually short of, and it says so before a config is
    sized (rule 5) -- if material stock is the binding constraint, the axe cannot
    help and the run would measure nothing.

    An axe is permanent once made. Durability was left out deliberately: it is a
    second mechanic (a decay rate to tune) wearing the same name, and rung 1 has
    to answer "does a tool get adopted at all" before anything is tuned.
    """

    enabled: bool = False
    axe_wood_cost: int = 1
    axe_stone_cost: int = 1
    # Yield per successful chop while holding an axe. 2 is the rung-1 setting;
    # the multiplier is a config knob so the adoption question can be asked at a
    # price the agents can actually see.
    chop_multiplier: int = 2
    # An axe is made AT A SITE -- the household's own building spot doubles as the
    # workshop. Somewhere rather than anywhere, because a tool you can make while
    # standing in a berry patch is not a technology, it is an inventory slot.
    craft_radius: float = 2.0
    observe_axe: bool = True


@dataclass(frozen=True)
class MixConfig:
    """Train on TWO worlds at once: a share of the envs run a second config.

    The lever the subset-steering result points at. Navigation fails here because
    every sub-~20-tick deviation toward far food is correctly priced <= 0 under a
    wandering policy, so one-step improvement never proposes the excursion that
    pays +173. Sequencing worlds does not fix it -- `spread-nav` and `nav-refork`
    both show the scarce world erasing a navigator it was handed. Interleaving
    puts states where crossings COMPLETE into every batch instead, so a positive
    navigation gradient is present in the same update as the scarce world's
    negative one.

    Both configs must agree on the observation layout, the action set and the
    agent count -- one policy trains on both, and a mismatch would feed trained
    weights the wrong features. VecWorld raises rather than reshaping anything.

    Only the PRIMARY config's episodes reach the metrics log, and evaluation
    builds a primary World, so every reported number stays a statement about the
    world the run is named for. Blending two worlds' lifespans into one mean
    would describe neither (rule 6).
    """

    config: str | None = None   # path to the second world config
    fraction: float = 0.0       # share of envs running it


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
    # Milestone 5. 0.0 is the brief-faithful default, and the same call as
    # reward.steal: a gift has to pay for itself through what the receiver then
    # does with it, or it is not exchange, it is us paying agents to hand things
    # over. config/m5_shaped.yaml raises it as a labelled ablation.
    give: float = 0.0


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
    threads: int | None = 4   # CPU threads torch may use; null = all cores.
                              # 4 measured as fast as 8 -- these nets are small
                              # enough that the env step dominates, so extra
                              # cores buy heat and nothing else. Bit-identical
                              # results either way (tested).


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
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    society: SocietyConfig = field(default_factory=SocietyConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    mix: MixConfig = field(default_factory=MixConfig)
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
    "exchange": ExchangeConfig,
    "society": SocietyConfig,
    "tools": ToolsConfig,
    "mix": MixConfig,
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
