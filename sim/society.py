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
from pathlib import Path

import numpy as np

from .agents import action_names, night_phase, num_actions
from .config import Config, load_config
from .economy import subsistence
from .policy import random_actions
from .replay import ReplayRecorder
from .utility import (GOAL_NAMES, GOAL_RAID, N_GOALS, ArbiterConfig,
                      utility_runner)
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
    # --- stage 4
    deposits: list[int] = field(default_factory=list)
    withdrawals: list[int] = field(default_factory=list)
    raids: list[int] = field(default_factory=list)
    storms: list[int] = field(default_factory=list)
    damaged: list[int] = field(default_factory=list)
    blight_ticks: list[int] = field(default_factory=list)
    # per-household, concatenated over episodes
    house_lifespan: list[np.ndarray] = field(default_factory=list)
    house_stock_food: list[np.ndarray] = field(default_factory=list)
    house_stock_material: list[np.ndarray] = field(default_factory=list)
    # (raider household, victim household) counts, summed over episodes
    raid_matrix: np.ndarray | None = None
    # mean stockpile food over the episode, sampled -- the level, not the endpoint,
    # because a pile that filled and was drained twice ends where it started.
    stock_trace: list[float] = field(default_factory=list)
    home_distance: list[float] = field(default_factory=list)
    displaced: list[float] = field(default_factory=list)
    # of the agents holding the `raid` goal at a sample tick, the share that were
    # actually hungry -- desperation against revenge, which is what separates a
    # raid economy that moves food to someone who needs it from pure churn.
    raid_hungry: list[float] = field(default_factory=list)
    # --- mixed population (stage 5's first lever): who is learned, and the
    # per-subset splits of the numbers the free-ride-or-contribute question
    # turns on. None/empty unless the runner is a MixedArbiter.
    learn_mask: np.ndarray | None = None
    night_in_agent: list[np.ndarray] = field(default_factory=list)
    night_out_agent: list[np.ndarray] = field(default_factory=list)
    goal_ticks_learned: np.ndarray = field(
        default_factory=lambda: np.zeros(N_GOALS, dtype=np.int64))
    goal_ticks_scripted: np.ndarray = field(
        default_factory=lambda: np.zeros(N_GOALS, dtype=np.int64))


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


def household_dispersion(world: World) -> tuple[float, float]:
    """(mean distance to own household site, share of agents nearer a foreign one).

    The household-world replacement for failure mode 3's control, and it exists
    because the stage-2 control stopped applying the moment households became
    places. Nearest-neighbour distance against uniform placement asks "has the
    population piled up?" -- and a stage-4 population is SUPPOSED to have piled
    up, twenty times over, one pile per household. Measured, that reads 0.47x of
    chance and trips a WATCH on behaviour the design asked for. Rule 5: when you
    change a mechanic, re-derive the arithmetic that justified the check.

    What "mega-camp" means once households exist is that the piles stop being
    SEPARATE -- everyone drifts onto one cluster and household stops predicting
    location. The second number is the direct test of that: an agent closer to
    somebody else's home than to its own has left its household behind, and a
    population where most agents have done so has one camp, not twenty.
    """
    pool = world.pool
    idx = np.flatnonzero(pool.alive)
    if idx.size == 0 or world.stock_x.size < 2:
        return float("nan"), float("nan")
    dx = world.stock_x[None, :] - pool.x[idx][:, None]
    dz = world.stock_z[None, :] - pool.z[idx][:, None]
    d = np.sqrt(dx ** 2 + dz ** 2)
    rows = np.arange(idx.size)
    own = d[rows, world.household[idx]]
    foreign = d.copy()
    foreign[rows, world.household[idx]] = np.inf
    return float(own.mean()), float((foreign.min(axis=1) < own).mean())


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


def _nanmean(values: list[float]) -> float:
    """Mean of the non-nan entries, or nan if there are none. No warning."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def make_runner(cfg: Config, policy: str, seed: int,
                acfg: ArbiterConfig | None = None,
                checkpoint: str | None = None) -> "OptionRunner | None":
    """The arbiter zoo, one place. `None` means raw random actions (the floor).

    Every non-random entry is an `OptionRunner` over the SAME controllers and the
    same option menu; they differ only in the chooser. That is the stage-5
    contract, and building them all here is what keeps a comparison between two
    of them a comparison of choosers rather than of programs.
    """
    from .utility import OptionRunner, UtilityArbiter
    if policy == "random":
        return None
    if policy == "utility":
        return utility_runner(cfg, seed=seed, acfg=acfg)
    if policy == "randomgoal":
        from .arbiter import RandomGoalArbiter
        return OptionRunner(RandomGoalArbiter(cfg, acfg, seed=seed), cfg, seed=seed)
    if policy == "learned":
        from .arbiter import load_arbiter
        if not checkpoint:
            raise ValueError("--arbiter learned needs --checkpoint")
        arb = load_arbiter(checkpoint, cfg)
        arb.acfg = acfg or arb.acfg
        return OptionRunner(arb, cfg, seed=seed)
    if policy == "mixed":
        from .arbiter import MixedArbiter, load_arbiter
        if not checkpoint:
            raise ValueError("--arbiter mixed needs --checkpoint")
        arb = load_arbiter(checkpoint, cfg)
        arb.acfg = acfg or arb.acfg
        ids = getattr(arb, "learn_agents", None)
        if ids is None:
            raise ValueError(f"{checkpoint} was not trained mixed (no "
                             f"learn_agents in the checkpoint); use --arbiter "
                             f"learned for an all-learned population")
        learn_mask = np.zeros(cfg.world.num_agents, dtype=bool)
        learn_mask[np.asarray(ids, dtype=np.int64)] = True
        scripted = UtilityArbiter(cfg, acfg or arb.acfg, seed=seed)
        return OptionRunner(MixedArbiter(scripted, arb, learn_mask), cfg, seed=seed)
    if policy == "mixedrandom":
        # The mixed experiment's own floor: the same scripted 80 carrying a
        # RANDOM-goal minority in the same slots. If this minority already
        # survives well, "learned matches scripted" would mean the society
        # carries any passenger -- so the learned run is read against this.
        from .arbiter import MixedArbiter, RandomGoalArbiter
        n_learn = max(cfg.society.num_households, 1)
        learn_mask = np.zeros(cfg.world.num_agents, dtype=bool)
        learn_mask[:n_learn] = True
        scripted = UtilityArbiter(cfg, acfg, seed=seed)
        rand = RandomGoalArbiter(cfg, acfg, seed=seed)
        return OptionRunner(MixedArbiter(scripted, rand, learn_mask), cfg, seed=seed)
    raise ValueError(f"unknown arbiter {policy!r}")


def run_episodes(cfg: Config, episodes: int, seed: int, acfg: ArbiterConfig | None = None,
                 policy: str = "utility", checkpoint: str | None = None) -> SocietyReport:
    rep = SocietyReport(episodes=episodes)
    n_act = num_actions(cfg)
    for e in range(episodes):
        world = World(cfg, seed=seed + e)
        runner = make_runner(cfg, policy, seed + e, acfg, checkpoint)
        rng = np.random.default_rng(seed + e + 99991)
        learn_mask = (getattr(getattr(runner, "arbiter", None), "learn_mask", None)
                      if runner is not None else None)
        if learn_mask is not None:
            rep.learn_mask = learn_mask
        obs = world.observations()
        day_r: list[float] = []
        night_r: list[float] = []
        nn: list[float] = []
        stock_trace: list[float] = []
        home_d: list[float] = []
        displaced: list[float] = []
        raid_hungry: list[float] = []
        while True:
            mask = world.action_mask()
            if runner is not None:
                actions = runner.act(obs, mask)
            else:
                actions = random_actions(obs, rng, n_act)
            res = world.step(actions)
            obs = res.obs
            pool = world.pool
            if learn_mask is not None:
                # Same counting rule as OptionRunner.goal_ticks (every agent,
                # every tick), split by who owns the agent, so the two subsets'
                # shares are comparable with the population line above them.
                np.add.at(rep.goal_ticks_learned, runner.goals[learn_mask], 1)
                np.add.at(rep.goal_ticks_scripted, runner.goals[~learn_mask], 1)
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
                if cfg.society.enabled:
                    stock_trace.append(float(world.stock_food.mean()))
                    # HOUSEHOLD COHESION IS A NIGHT STATISTIC, for the mirror of
                    # the reason crowding is a day one. By day an agent is out
                    # foraging and the nearest foreign home is often closer than
                    # its own, which is not defection -- it is a commute. Where an
                    # agent actually sleeps is what "my group is who sleeps where
                    # I sleep" is a claim about. Measured over all ticks this read
                    # 37.3% and tripped a WATCH on agents who went home every
                    # single night.
                    if is_night:
                        own, away = household_dispersion(world)
                        home_d.append(own)
                        displaced.append(away)
                    if runner is not None:
                        raiders = (runner.goals == GOAL_RAID) & pool.alive
                        if raiders.any():
                            hungry = pool.hunger[raiders] < cfg.hunger.eat_threshold
                            raid_hungry.append(float(hungry.mean()))
            if res.episode_done:
                break
        stats = world.stats()
        if cfg.society.enabled:
            rep.deposits.append(stats.deposits)
            rep.withdrawals.append(stats.withdrawals)
            rep.raids.append(stats.raids)
            rep.storms.append(stats.storms)
            rep.damaged.append(stats.shelters_damaged)
            rep.blight_ticks.append(stats.blight_ticks)
            h = cfg.society.num_households
            lives_by_house = np.zeros(h)
            np.add.at(lives_by_house, world.household, world.alive_ticks)
            counts = np.bincount(world.household, minlength=h)
            rep.house_lifespan.append(lives_by_house / np.maximum(counts, 1))
            rep.house_stock_food.append(stats.stock_food_final.astype(np.float64))
            rep.house_stock_material.append(stats.stock_material_final.astype(np.float64))
            rep.stock_trace.append(float(np.mean(stock_trace)) if stock_trace else 0.0)
            rep.home_distance.append(_nanmean(home_d))
            rep.displaced.append(_nanmean(displaced))
            rep.raid_hungry.append(_nanmean(raid_hungry))
            if rep.raid_matrix is None:
                rep.raid_matrix = np.zeros((h, h), dtype=np.int64)
            for _tick, raider_h, victim_h, _item in stats.raid_ledger:
                rep.raid_matrix[raider_h, victim_h] += 1
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
        if learn_mask is not None:
            rep.night_in_agent.append(world.night_sheltered_agent.copy())
            rep.night_out_agent.append(world.night_exposed_agent.copy())
        if runner is not None:
            rep.goal_ticks += runner.goal_ticks
            rep.decisions.append(float(runner.decisions.mean()))
        rep.nn_distance.append(float(np.nanmean(nn)) if nn else float("nan"))
        # nanmean, not mean: a storm can knock every shelter back to incomplete,
        # and a tick with nothing finished to measure against legitimately has no
        # distance. One such sample used to poison the whole episode's mean.
        rep.day_positions.append(_nanmean(day_r))
        rep.night_positions.append(_nanmean(night_r))
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
        f"shelters / episode {np.mean(rep.shelters):7.2f} of {cfg.construction.num_sites}"
        + (" COMPLETIONS, not distinct sites: a storm knocks finished shelters back "
           "to incomplete and they are rebuilt, so this counts rebuilds and can "
           "exceed num_sites" if cfg.society.enabled and cfg.society.shock_interval
           else ""),
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
    if cfg.society.enabled and rep.displaced:
        # The stage-2 verdict band does not apply once households are places, so
        # it is not printed: the check is whether the piles are still SEPARATE.
        # See household_dispersion for why the old control had to be retired.
        #
        # THE CONTROL, and it is not 50%. If sleeping position told you nothing
        # about household, an agent's own home would be the nearest of H by chance
        # alone, so 1 - 1/H of the population would be displaced -- 95% at twenty
        # households. Note also what the RANDOM-ACTION floor gives: 14.8%, and
        # LOWER than the utility agents', because random agents barely leave the
        # spawn point they were placed on. So the floor is not the control here;
        # the uniform-position expectation is.
        away = _nanmean(rep.displaced)
        h = max(cfg.society.num_households, 1)
        chance_away = 1.0 - 1.0 / h
        camp = "DEGENERATE" if away > 0.6 else ("WATCH" if away > 0.35 else "OK")
        out.append(f"  3. mega-camp    AT NIGHT, {100 * away:.1f}% of agents are nearer a "
                   f"FOREIGN household's home than their own, against "
                   f"{100 * chance_away:.0f}% if position told you nothing [{camp}]")
        out.append(f"                  (mean night distance to own home "
                   f"{_nanmean(rep.home_distance):.1f}; daytime nearest neighbour "
                   f"{nn:.2f} vs {chance:.2f} at random -- that ratio no longer has "
                   f"a meaningful band, see household_dispersion)")
    else:
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

    if cfg.society.enabled and rep.house_lifespan:
        out.append(_household_section(cfg, rep))
    if rep.learn_mask is not None:
        out.append(_mixed_section(cfg, rep))
    return "\n".join(out)


def _mixed_section(cfg: Config, rep: SocietyReport) -> str:
    """The free-ride-or-contribute split for a mixed population.

    Three columns of the same statistic, learned vs scripted, because the mixed
    experiment's question is not "did the population survive" (the scripted 80
    guarantee most of that) but what the learned minority DID with a society
    around it: did it shelter at all (the thing every all-learned run refused),
    and did it put anything in -- build/deliver/store shares -- or only draw out.
    """
    m = rep.learn_mask
    lives = np.stack(rep.lifespans)                     # (episodes, agents)
    ni = np.stack(rep.night_in_agent).sum(axis=0).astype(np.float64)
    no = np.stack(rep.night_out_agent).sum(axis=0).astype(np.float64)

    def night_share(mask: np.ndarray) -> float:
        tot = ni[mask].sum() + no[mask].sum()
        return 100.0 * ni[mask].sum() / max(tot, 1.0)

    gl, gs = rep.goal_ticks_learned, rep.goal_ticks_scripted
    tl, ts = max(int(gl.sum()), 1), max(int(gs.sum()), 1)
    out = [f"\n-- mixed population: {int(m.sum())} learned among "
           f"{int((~m).sum())} scripted --",
           f"  {'':<16} {'learned':>9} {'scripted':>9}",
           f"  {'mean lifespan':<16} {lives[:, m].mean():9.1f} {lives[:, ~m].mean():9.1f}",
           f"  {'nights indoors':<16} {night_share(m):8.1f}% {night_share(~m):8.1f}%",
           "  goal shares (% of the subset's own goal-ticks):"]
    for g in np.argsort(-(gl / tl + gs / ts)):
        a, b = 100.0 * gl[g] / tl, 100.0 * gs[g] / ts
        if max(a, b) >= 0.5:
            out.append(f"    {GOAL_NAMES[g]:<14} {a:8.1f}% {b:8.1f}%")
    return "\n".join(out)


def _household_section(cfg: Config, rep: SocietyReport) -> str:
    """Stage 4: the household economy, and its own pre-registered failure modes.

    Read this INSTEAD OF the per-agent inequality line above when households are
    on. Stage 2's lifespan Gini was 0.058 and the write-up said the honest thing
    about it -- "not much drama either; households and stockpiles are what would
    give inequality something to accumulate in". The between-household Gini is the
    number that claim has to be judged on, and it is not the same statistic: a
    world can be perfectly equal between individuals and starkly unequal between
    groups, which is the shape of inequality stage 4 was built to produce.
    """
    sc = cfg.society
    houses = np.stack(rep.house_lifespan)          # (episodes, households)
    food = np.stack(rep.house_stock_food)
    mat = np.stack(rep.house_stock_material)
    per_house = houses.mean(axis=0)
    out = ["\n-- the household economy (stage 4) --",
           f"  {sc.num_households} households of "
           f"{cfg.world.num_agents / max(sc.num_households, 1):.0f}",
           f"  deposits / episode      {np.mean(rep.deposits):7.1f}",
           f"  withdrawals / episode   {np.mean(rep.withdrawals):7.1f}",
           f"  raids / episode         {np.mean(rep.raids):7.1f}",
           f"  stockpile food, mean level over the episode "
           f"{np.mean(rep.stock_trace):5.2f} of {sc.stockpile_food_capacity}",
           f"  stockpile at the end    food {food.mean():5.2f}, "
           f"material {mat.mean():5.2f} per household"]
    if rep.storms:
        out.append(f"  shocks: {np.mean(rep.blight_ticks):.0f} blighted ticks, "
                   f"{np.mean(rep.storms):.1f} storms damaging "
                   f"{np.mean(rep.damaged):.1f} shelters")

    out.append("\n-- two more pre-registered failure modes, from the design doc --")
    # 4. INEQUALITY SNOWBALL, the version stage 2 could not test. The doc's guess
    # is that this is good drama and only needs capping if it goes degenerate, so
    # the verdict bands are deliberately looser than a moral judgement would be.
    hg = gini(per_house)
    verdict = "OK" if hg < 0.20 else ("WATCH" if hg < 0.35 else "DEGENERATE")
    order = np.argsort(-per_house)
    out.append(f"  4. household inequality  Gini {hg:.3f} [{verdict}]  "
               f"richest household {per_house[order[0]]:.0f} ticks / "
               f"poorest {per_house[order[-1]]:.0f}")
    # 5. PERMANENT WAR at household scale. The doc's own forecast: "raid goal
    # scored too cheap -> permanent war, nobody forages, collapse". A raid is
    # gated on motive rather than on distance precisely because stage 2 watched
    # the un-gated version of this fire, so the number to watch is raids against
    # deposits -- a store that is raided more often than it is filled is a store
    # nobody will keep filling.
    dep = max(np.mean(rep.deposits), 1e-9)
    ratio = np.mean(rep.raids) / dep
    war = "DEGENERATE" if ratio > 1.0 else ("WATCH" if ratio > 0.4 else "OK")
    out.append(f"  5. raid economy          {np.mean(rep.raids):.1f} raids per "
               f"{dep:.1f} deposits ({ratio:.2f}x) [{war}]")
    hungry = _nanmean(rep.raid_hungry)
    if hungry == hungry:
        out.append(f"     of agents holding the raid goal, {100 * hungry:.0f}% were "
                   f"below the eat threshold (the rest are settling grudges)")

    if rep.raid_matrix is not None and rep.raid_matrix.sum():
        m = rep.raid_matrix
        raiders = m.sum(axis=1)
        victims = m.sum(axis=0)
        top_r = int(np.argmax(raiders))
        top_v = int(np.argmax(victims))
        # Directionality is what separates a feud from noise. Symmetric raiding
        # is everyone helping themselves; a household that raids far more than it
        # is raided is a predator, and one raided far more than it raids is prey.
        # `reciprocity` here is the same idea sim.exchange applies to gifts.
        pairs = np.minimum(m, m.T).sum() / max(m.sum(), 1)
        out.append(f"  raiding: household {top_r} took {raiders[top_r]} times "
                   f"(most), household {top_v} was hit {victims[top_v]} times "
                   f"(most); reciprocity {pairs:.2f}")
    return "\n".join(out)


def record_replay(cfg: Config, seed: int, path: str, label: str,
                  acfg: ArbiterConfig | None = None, policy: str = "utility",
                  checkpoint: str | None = None) -> str:
    """One episode, written as a replay the viewer can load."""
    world = World(cfg, seed=seed)
    runner = make_runner(cfg, policy, seed, acfg, checkpoint)
    assert runner is not None, "record_replay drives an arbiter, not raw random"
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
    ap.add_argument("--arbiter", default="utility",
                    choices=["utility", "learned", "randomgoal", "mixed", "mixedrandom"],
                    help="which chooser drives the population ('mixed' rebuilds "
                         "the learned/scripted split recorded in the checkpoint; "
                         "'mixedrandom' is its floor -- a random-goal minority in "
                         "the first num_households slots)")
    ap.add_argument("--checkpoint", default=None, help="for --arbiter learned/mixed")
    ap.add_argument("--vs", default=None,
                    choices=["utility", "learned", "randomgoal", "mixed", "mixedrandom"],
                    help="also run this arbiter on the SAME seeds and print the "
                         "paired per-island differences (rule 7)")
    ap.add_argument("--vs-checkpoint", default=None)
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

    rep = run_episodes(cfg, args.episodes, args.seed, acfg,
                       policy=args.arbiter, checkpoint=args.checkpoint)
    print(format_report(cfg, rep, f"{args.arbiter} arbiter"))

    if args.vs:
        other = run_episodes(cfg, args.episodes, args.seed, acfg,
                             policy=args.vs, checkpoint=args.vs_checkpoint)
        print()
        print(format_report(cfg, other, f"{args.vs} arbiter"))
        # Paired per-island differences: same seed block, so island noise
        # cancels -- the same discipline sim.evaluate applies (rule 7).
        a = np.array([float(np.mean(l)) for l in rep.lifespans])
        b = np.array([float(np.mean(l)) for l in other.lifespans])
        d = a - b
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"\npaired, {args.arbiter} - {args.vs}: {d.mean():+.1f} +- {se:.1f} "
              f"ticks (better on {int((d > 0).sum())}/{len(d)} islands)")
        if rep.learn_mask is not None:
            # The population diff above is diluted 4:1 by scripted agents who are
            # near-identical in both runs. The learned SLOTS are the experiment:
            # same agent indices under the other arbiter, same seeds, so the
            # comparison is what those twenty lives cost or gained.
            m = rep.learn_mask
            a = np.array([float(np.mean(l[m])) for l in rep.lifespans])
            b = np.array([float(np.mean(l[m])) for l in other.lifespans])
            d = a - b
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"paired at the LEARNED slots only: {d.mean():+.1f} +- {se:.1f} "
                  f"ticks (better on {int((d > 0).sum())}/{len(d)} islands)")
            a = np.array([float(np.mean(l[~m])) for l in rep.lifespans])
            b = np.array([float(np.mean(l[~m])) for l in other.lifespans])
            d = a - b
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"paired at the SCRIPTED slots (spillover): {d.mean():+.1f} "
                  f"+- {se:.1f} ticks (better on {int((d > 0).sum())}/{len(d)} islands)")

    if args.random:
        floor = run_episodes(cfg, args.episodes, args.seed, acfg, policy="random")
        print()
        print(format_report(cfg, floor, "random actions (the floor)"))
        a = np.concatenate(rep.lifespans).mean()
        b = np.concatenate(floor.lifespans).mean()
        print(f"\nutility / random lifespan: {a / max(b, 1e-9):.2f}x")

    if args.replay:
        # Named after the CONFIG, not hardcoded: stage 4 has its own world and
        # overwriting stage 2's replay with it would quietly destroy the thing the
        # two are meant to be compared against.
        stem = Path(args.config).stem
        path = args.replay_path or f"{cfg.logging.replay_dir}/{stem}.json"
        written = record_replay(cfg, args.seed, path,
                                f"island2 {args.arbiter} arbiter ({stem})", acfg,
                                policy=args.arbiter, checkpoint=args.checkpoint)
        print(f"\nreplay written: {written}")
        print(f"open viewer/index.html?replay={written.split('/')[-1]}")


if __name__ == "__main__":
    main()
