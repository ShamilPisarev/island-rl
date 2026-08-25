"""Run an Island 2.0 population and report on it (stage 2).

Drives a world with the utility arbiter, prints a diagnostic report, and
optionally writes a replay for the viewer. The report deliberately leads with the
design doc's THREE PRE-REGISTERED FAILURE MODES, because a 100-agent sim is easy
to watch and hard to judge -- "it looks alive" is not a result, and each of these
is a specific way a utility population goes visibly wrong:

  1. INEQUALITY SNOWBALL -- early-rich agents run away with the food. Reported as
     the lifespan Gini and the best/worst agent split. The doc's own guess is
     that this is good drama; it only needs capping if it goes degenerate.
  2. PERMANENT WAR -- if the steal goal is scored too cheaply, nobody forages and
     the population eats its own tail (the M3 lesson: theft redistributes, it
     never creates). Reported as the steal share of goal-ticks and the harvest
     total against what the island produced.
  3. MEGA-CAMP -- too few clusters for the population and everyone piles onto one
     spot. Reported as mean distance to the nearest neighbour against the
     random-placement expectation for the same island.

Usage:
    python -m sim.society --config config/island2/society100.yaml --episodes 3
    python -m sim.society --config config/island2/society100.yaml --replay
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from .agents import action_names, night_phase, num_actions
from .config import Config, load_config
from .economy import subsistence
from .policy import random_actions
from .replay import ReplayRecorder
from .utility import GOAL_NAMES, N_GOALS, ArbiterConfig, utility_runner
from .world import World


@dataclass
class SocietyReport:
    episodes: int = 0
    ticks: list[int] = field(default_factory=list)
    lifespans: list[np.ndarray] = field(default_factory=list)
    deaths: list[int] = field(default_factory=list)
    berries: list[int] = field(default_factory=list)
    shelters: list[int] = field(default_factory=list)
    steals: list[int] = field(default_factory=list)
    gifts: list[int] = field(default_factory=list)
    builds: list[int] = field(default_factory=list)
    night_in: list[int] = field(default_factory=list)
    night_out: list[int] = field(default_factory=list)
    goal_ticks: np.ndarray = field(default_factory=lambda: np.zeros(N_GOALS, dtype=np.int64))
    decisions: list[float] = field(default_factory=list)
    nn_distance: list[float] = field(default_factory=list)
    day_positions: list[float] = field(default_factory=list)
    night_positions: list[float] = field(default_factory=list)


def gini(values: np.ndarray) -> float:
    """Standard Gini coefficient. 0 = everyone equal, 1 = one agent has it all."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = v.size
    if n == 0 or v.sum() <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * (index * v).sum()) / (n * v.sum()) - (n + 1.0) / n)


def mean_nearest_neighbour(x: np.ndarray, z: np.ndarray, alive: np.ndarray) -> float:
    """Mean distance from a living agent to its nearest living neighbour."""
    idx = np.flatnonzero(alive)
    if idx.size < 2:
        return float("nan")
    d2 = (x[idx][:, None] - x[idx][None, :]) ** 2 + (z[idx][:, None] - z[idx][None, :]) ** 2
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min(axis=1)).mean())


def _distance_to_shelter(world: World) -> float:
    """Mean distance from a living agent to the nearest FINISHED shelter.

    ``nan`` when nothing is finished yet, which the caller averages away with
    nanmean -- an episode's early ticks legitimately have no shelter to measure
    against, and counting them as zero would fake a commute.
    """
    pool = world.pool
    done = np.flatnonzero((world.site_wood_needed == 0) & (world.site_stone_needed == 0))
    idx = np.flatnonzero(pool.alive)
    if done.size == 0 or idx.size == 0:
        return float("nan")
    d2 = ((world.site_x[done][None, :] - pool.x[idx][:, None]) ** 2
          + (world.site_z[done][None, :] - pool.z[idx][:, None]) ** 2)
    return float(np.sqrt(d2.min(axis=1)).mean())


def expected_nearest_neighbour(radius: float, n: int) -> float:
    """Mean nearest-neighbour distance for n uniform points in a disc of `radius`.

    The 2-D Poisson result, 0.5 / sqrt(density). This is the control for failure
    mode 3: clustering only means something against what chance would give on the
    same island, which is the same discipline `sim.navigation --baselines` applies
    to toward-food shares.
    """
    if n < 2:
        return float("nan")
    density = n / (np.pi * radius ** 2)
    return float(0.5 / np.sqrt(density))


def run_episodes(cfg: Config, episodes: int, seed: int, acfg: ArbiterConfig | None = None,
                 policy: str = "utility") -> SocietyReport:
    rep = SocietyReport(episodes=episodes)
    n_act = num_actions(cfg)
    for e in range(episodes):
        world = World(cfg, seed=seed + e)
        runner = utility_runner(cfg, seed=seed + e, acfg=acfg) if policy == "utility" else None
        rng = np.random.default_rng(seed + e + 99991)
        obs = world.observations()
        day_r: list[float] = []
        night_r: list[float] = []
        nn: list[float] = []
        while True:
            mask = world.action_mask()
            if runner is not None:
                actions = runner.act(obs, mask)
            else:
                actions = random_actions(obs, rng, n_act)
            res = world.step(actions)
            obs = res.obs
            pool = world.pool
            if world.tick % 10 == 0 and pool.alive.any():
                _, is_night = night_phase(world.tick, cfg)
                # Rhythm is measured as distance to the nearest FINISHED shelter,
                # not as radius from the island centre. Radius cannot see a
                # commute when shelters are spread over twenty clusters: agents
                # walk to different homes in different directions and the mean
                # radius barely moves (measured 51.4 by day against 51.5 at
                # night, while 96% of them were actually indoors).
                d_home = _distance_to_shelter(world)
                (night_r if is_night else day_r).append(d_home)
                # Crowding is a DAY statistic. At night everyone is deliberately
                # packed into shelters, so including night ticks makes the
                # intended behaviour read as the degenerate failure mode.
                if not is_night:
                    nn.append(mean_nearest_neighbour(pool.x, pool.z, pool.alive))
            if res.episode_done:
                break
        stats = world.stats()
        rep.ticks.append(stats.ticks)
        rep.lifespans.append(world.alive_ticks)
        rep.deaths.append(stats.deaths)
        rep.berries.append(stats.berries_gathered)
        rep.shelters.append(stats.shelters_completed)
        rep.steals.append(stats.steals)
        rep.gifts.append(stats.gifts)
        rep.builds.append(stats.builds)
        rep.night_in.append(stats.night_ticks_sheltered)
        rep.night_out.append(stats.night_ticks_exposed)
        if runner is not None:
            rep.goal_ticks += runner.goal_ticks
            rep.decisions.append(float(runner.decisions.mean()))
        rep.nn_distance.append(float(np.nanmean(nn)) if nn else float("nan"))
        rep.day_positions.append(float(np.mean(day_r)) if day_r else float("nan"))
        rep.night_positions.append(float(np.mean(night_r)) if night_r else float("nan"))
    return rep


def format_report(cfg: Config, rep: SocietyReport, label: str) -> str:
    lives = np.concatenate(rep.lifespans)
    econ = subsistence(cfg)
    total_goal = max(int(rep.goal_ticks.sum()), 1)
    night_total = max(sum(rep.night_in) + sum(rep.night_out), 1)
    out = [
        f"=== {label}: {cfg.world.num_agents} agents x {rep.episodes} episodes ===",
        f"mean lifespan      {lives.mean():7.1f} of {cfg.world.max_ticks} "
        f"({100 * lives.mean() / cfg.world.max_ticks:.0f}%)",
        f"deaths / episode   {np.mean(rep.deaths):7.2f} of {cfg.world.num_agents}",
        f"berries / episode  {np.mean(rep.berries):7.1f} of {econ.supply:.0f} the island makes "
        f"({100 * np.mean(rep.berries) / max(econ.supply, 1):.0f}% harvested, "
        f"demand {econ.demand_sheltered:.0f})",
        f"shelters / episode {np.mean(rep.shelters):7.2f} of {cfg.construction.num_sites}",
        f"nights indoors     {100 * sum(rep.night_in) / night_total:7.1f}%",
        f"steals / episode   {np.mean(rep.steals):7.1f}",
        f"gifts / episode    {np.mean(rep.gifts):7.1f}",
    ]
    if rep.decisions:
        out.append(f"goal decisions/agent {np.mean(rep.decisions):5.1f} "
                   f"(i.e. one every {np.mean(rep.ticks) / max(np.mean(rep.decisions), 1e-9):.0f} ticks)")

    out.append("\n-- what the population spent its time wanting --")
    order = np.argsort(-rep.goal_ticks)
    for g in order:
        share = 100 * rep.goal_ticks[g] / total_goal
        if share >= 0.05:
            out.append(f"  {GOAL_NAMES[g]:<15} {share:5.1f}%")

    out.append("\n-- the three pre-registered failure modes --")
    g = gini(lives)
    best, worst = lives.max(), lives.min()
    verdict = "OK" if g < 0.25 else ("WATCH" if g < 0.4 else "DEGENERATE")
    out.append(f"  1. inequality   Gini {g:.3f} [{verdict}]  "
               f"best {best:.0f} / worst {worst:.0f} ticks")
    steal_share = 100 * rep.goal_ticks[5] / total_goal
    forage_share = 100 * rep.goal_ticks[0] / total_goal
    war = "DEGENERATE" if steal_share > forage_share else ("WATCH" if steal_share > 15 else "OK")
    out.append(f"  2. war          steal {steal_share:.1f}% vs forage {forage_share:.1f}% "
               f"of goal-ticks [{war}]")
    nn = float(np.mean([v for v in rep.nn_distance if v == v] or [float("nan")]))
    chance = expected_nearest_neighbour(cfg.world.island_radius, cfg.world.num_agents)
    ratio = nn / chance if chance == chance else float("nan")
    camp = "DEGENERATE" if ratio < 0.35 else ("WATCH" if ratio < 0.6 else "OK")
    out.append(f"  3. mega-camp    daytime nearest neighbour {nn:.2f} vs {chance:.2f} "
               f"at random ({ratio:.2f}x) [{camp}]")

    def mean_or_nan(values: list[float]) -> float:
        """nanmean over a list that may be empty or all-nan without warning.

        A policy that never finishes a shelter (the random floor does not, in
        daylight) legitimately has nothing to measure here.
        """
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        return float(arr.mean()) if arr.size else float("nan")

    day = mean_or_nan(rep.day_positions)
    night = mean_or_nan(rep.night_positions)
    out.append(f"\n-- day/night rhythm --\n  mean distance to the nearest finished shelter: "
               f"day {day:.1f}, night {night:.1f} ({night - day:+.1f} at dusk)")
    return "\n".join(out)


def record_replay(cfg: Config, seed: int, path: str, label: str,
                  acfg: ArbiterConfig | None = None) -> str:
    """One episode, written as a replay the viewer can load."""
    world = World(cfg, seed=seed)
    runner = utility_runner(cfg, seed=seed, acfg=acfg)
    rec = ReplayRecorder(world, cfg, label=label, source="sim.society", seed=seed)
    rec.snapshot()
    obs = world.observations()
    while True:
        mask = world.action_mask()
        res = world.step(runner.act(obs, mask))
        obs = res.obs
        rec.snapshot()
        if res.episode_done:
            break
    written = rec.save(path)
    return str(written)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/island2/society100.yaml")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--commit", type=int, default=None, help="option commitment in ticks")
    ap.add_argument("--softmax", type=float, default=None, help="goal sampling temperature")
    ap.add_argument("--random", action="store_true", help="also run the random-action floor")
    ap.add_argument("--replay", action="store_true", help="write a replay for the viewer")
    ap.add_argument("--replay-path", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    overrides = {}
    if args.commit is not None:
        overrides["commit_ticks"] = args.commit
    if args.softmax is not None:
        overrides["softmax_temp"] = args.softmax
    acfg = ArbiterConfig(**overrides)

    rep = run_episodes(cfg, args.episodes, args.seed, acfg)
    print(format_report(cfg, rep, "utility agents"))

    if args.random:
        floor = run_episodes(cfg, args.episodes, args.seed, acfg, policy="random")
        print()
        print(format_report(cfg, floor, "random actions (the floor)"))
        a = np.concatenate(rep.lifespans).mean()
        b = np.concatenate(floor.lifespans).mean()
        print(f"\nutility / random lifespan: {a / max(b, 1e-9):.2f}x")

    if args.replay:
        path = args.replay_path or f"{cfg.logging.replay_dir}/society100.json"
        written = record_replay(cfg, args.seed, path, "island2 utility agents", acfg)
        print(f"\nreplay written: {written}")
        print(f"open viewer/index.html?replay={written.split('/')[-1]}")


if __name__ == "__main__":
    main()
