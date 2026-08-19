"""Opportunity versus uptake, per action: is a rare behaviour the world or the policy?

Every "the policy does not do X often enough" finding in this project has turned
out to be one of two completely different problems, and a raw count of X cannot
tell them apart:

  * **no opportunity** -- X is almost never legal, so the policy is not declining
    anything. `m4f` was this: `chop` was reachable on 0.2% of ticks and taken on
    82-93% of those, because trees were nowhere near where agents lived. The fix
    was geography and it was worth 20x the shelters.
  * **declined opportunity** -- X is legal and the policy passes. `m4h` was this:
    `build` legal on 1.5% of ticks and taken on 28.6% of them.

The action mask is what makes this measurable: it is the world's own statement of
what is reachable this tick, so "legal" here is not a heuristic. Worlds without
`competition.mask_invalid_actions` have an all-ones mask and every action reads as
100% legal -- the report says so rather than pretending otherwise.

Uptake is a share of *legal* ticks, so it is directly comparable across actions
with wildly different opportunity rates. Note that a taken action can still fail
(a thief can empty the victim first, two givers can race for one slot), so uptake
is what the policy chose, not what it achieved.

**READ THE PER-SPAN COLUMN, NOT THE PER-TICK ONE.** An action that consumes what
made it legal -- `build` spends the material, `gather` spends the berry -- stays
legal for a run of consecutive ticks and only the first take does anything. So the
per-tick share divides one opportunity by the number of ticks it happened to
persist. Measured on `m4h`: `build` is taken on 30.6% of legal ticks and on
**77.4%** of distinct opportunities (spans average 4.3 ticks). `gather` spans last
1.2 ticks, so there the two agree (100.0% and 100.0%) -- which is exactly why the
error hid for a milestone. This is rule 6 in CLAUDE.md, arriving in a new place.

`--force ACTION` answers the question a low uptake only *suggests*: is the
remaining opportunity worth anything? It re-runs the same weights with the action
forced whenever legal, and again suppressed entirely, paired per island against
the unmodified policy. On both `m4h`'s `build` and `m3-masked`'s `steal` the answer
was no -- forcing bought nothing while suppressing was catastrophic.

    python -m sim.opportunity --checkpoint checkpoints/m4h/latest.pt
    python -m sim.opportunity --checkpoint checkpoints/m4h/latest.pt --force build
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .agents import N_MOVE_ACTIONS, action_names
from .config import Config
from .evaluate import ActFn, baseline_kinds, load_checkpoint, paired_lines, policy_act_fn
from .world import World


def measure(cfg: Config, act_fn: ActFn, episodes: int = 10, seed: int = 10000) -> dict:
    """Legal-tick share, uptake per tick, and uptake per distinct opportunity.

    A "span" is a maximal run of consecutive ticks on which the action stayed legal
    for one agent. It is the unit that corresponds to an opportunity a human would
    count: one loaded agent standing at a site that wants its material is ONE chance
    to build, however many ticks it lingers there.
    """
    names = action_names(cfg)
    n_act, n_ag = len(names), cfg.world.num_agents
    legal = np.zeros(n_act)
    taken = np.zeros(n_act)
    taken_legal = np.zeros(n_act)
    spans = np.zeros(n_act)
    spans_taken = np.zeros(n_act)
    span_ticks = np.zeros(n_act)
    ticks = 0

    for e in range(episodes):
        w = World(cfg, seed=seed + e)
        obs = w.observations()
        # Per (action, agent): is a span open, did it see a take, how long is it.
        open_span = np.zeros((n_act, n_ag), dtype=bool)
        hit = np.zeros((n_act, n_ag), dtype=bool)
        run = np.zeros((n_act, n_ag), dtype=np.int64)
        for _ in range(cfg.world.max_ticks):
            mask = w.action_mask()
            actions = act_fn(obs, mask)
            live = w.pool.alive
            rows = np.flatnonzero(live)
            ticks += rows.size
            legal += mask[live].sum(axis=0)
            for i in rows:
                a = int(actions[i])
                taken[a] += 1
                # A masked action should never be sampled; counted separately so a
                # regression in the mask plumbing shows up as a number rather than
                # hiding inside the uptake share.
                taken_legal[a] += bool(mask[i, a])
            # Span bookkeeping. A dead agent's open spans are closed by the
            # ~live term, so a corpse cannot hold one open to the horizon.
            legal_now = mask & live[:, None]
            for j in range(n_act):
                opening = legal_now[:, j] & ~open_span[j]
                open_span[j] |= opening
                hit[j] &= ~opening                   # a new span starts unhit
                run[j] = np.where(opening, 0, run[j])
                run[j] += open_span[j] & legal_now[:, j]
                hit[j] |= open_span[j] & (actions == j) & legal_now[:, j]
                closing = open_span[j] & ~legal_now[:, j]
                if closing.any():
                    spans[j] += closing.sum()
                    spans_taken[j] += (closing & hit[j]).sum()
                    span_ticks[j] += run[j][closing].sum()
                    open_span[j] &= ~closing
            result = w.step(actions)
            obs = result.obs
            if result.episode_done:
                break
        # Spans still open when the episode ends are real opportunities and are
        # counted, not discarded: dropping them would bias uptake toward whichever
        # spans happened to close.
        for j in range(n_act):
            still = open_span[j]
            spans[j] += still.sum()
            spans_taken[j] += (still & hit[j]).sum()
            span_ticks[j] += run[j][still].sum()

    return {
        "ticks": ticks,
        "masked": bool(cfg.competition.mask_invalid_actions),
        "actions": [
            {"name": names[j],
             "legal_frac": float(legal[j] / ticks) if ticks else float("nan"),
             "uptake": float(taken[j] / legal[j]) if legal[j] else float("nan"),
             "taken": int(taken[j]), "legal": int(legal[j]),
             "spans": int(spans[j]),
             "uptake_per_span": float(spans_taken[j] / spans[j]) if spans[j] else float("nan"),
             "mean_span_ticks": float(span_ticks[j] / spans[j]) if spans[j] else float("nan"),
             "taken_while_masked": int(taken[j] - taken_legal[j])}
            for j in range(n_act)
        ],
    }


def forced_comparison(cfg: Config, policy, target: str, episodes: int = 40,
                      seed: int = 10000) -> list[tuple[str, list]]:
    """Is the declined opportunity worth anything? Force it, suppress it, compare.

    Same weights throughout; only the action is overridden -- taken whenever legal,
    or replaced by `idle` whenever it was chosen. `idle` is always legal, so the
    suppressed variant never manufactures a doomed action.

    torch is re-seeded before each variant (common random numbers) because sampling
    noise alone moves a 40-island mean by ~10 ticks here, which is larger than the
    effects being looked for.
    """
    import torch                        # local: the analysis path, not the env path

    from .evaluate import make_act_fn, policy_act_fn, run_episodes

    names = action_names(cfg)
    if target not in names:
        raise ValueError(f"{target!r} is not an action in this world: {names}")
    tgt, idle = names.index(target), names.index("idle")
    base = policy_act_fn(policy)

    def variant(mode: str | None) -> ActFn:
        def act(obs: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
            a = base(obs, mask)
            if mode == "always" and mask is not None:
                return np.where(mask[:, tgt], tgt, a)
            if mode == "never":
                return np.where(a == tgt, idle, a)
            return a
        return act

    runs: list[tuple[str, list]] = []
    variants = [("learned policy", variant(None)),
                (f"ALWAYS {target}", variant("always")),
                (f"NEVER {target}", variant("never"))]
    for kind, label in baseline_kinds(cfg):
        if kind != "random":
            variants.append((label, make_act_fn(kind, cfg, None, seed)))
    for label, fn in variants:
        torch.manual_seed(0)
        runs.append((label, run_episodes(cfg, fn, episodes, seed)))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--all-actions", action="store_true",
                    help="include the eight moves and idle, which are always legal")
    ap.add_argument("--force", metavar="ACTION",
                    help="also run the counterfactual: this action taken whenever legal, "
                         "and suppressed entirely, paired against the unmodified policy")
    ap.add_argument("--force-episodes", type=int, default=40,
                    help="islands for --force (paired, so 40 is cheap and much tighter)")
    ap.add_argument("--report", default="viewer/reports/opportunity.json")
    args = ap.parse_args()

    policy, cfg, blob = load_checkpoint(args.checkpoint)
    report = measure(cfg, policy_act_fn(policy), args.episodes, args.seed)

    print(f"loaded {args.checkpoint} (update {blob.get('update', '?')})")
    print(f"{report['ticks']:,} living agent-ticks over {args.episodes} episodes\n")
    if not report["masked"]:
        print("  competition.mask_invalid_actions is OFF in this world: the mask is")
        print("  all ones, so every action reads as legal and `uptake` is just the")
        print("  action mix. Only the ordering means anything here.\n")
    print(f"  {'action':10} {'legal on':>9} {'taken/TICK':>11} "
          f"{'spans':>7} {'taken/SPAN':>11} {'mean span':>10}   counts")
    for a in report["actions"]:
        if not args.all_actions and (a["name"] == "idle"
                                    or a["name"] in action_names(cfg)[:N_MOVE_ACTIONS]):
            continue
        print(f"  {a['name']:10} {a['legal_frac']:8.1%} {a['uptake']:10.1%} "
              f"{a['spans']:7,} {a['uptake_per_span']:10.1%} "
              f"{a['mean_span_ticks']:9.1f}t   ({a['taken']:,} of {a['legal']:,} legal)")
        if a["taken_while_masked"]:
            print(f"    !! {a['taken_while_masked']} of those were MASKED when taken "
                  f"-- the mask plumbing is broken, not the policy")
    print("\n  taken/SPAN is the honest one: one take ends a run of legal ticks for any")
    print("  action that consumes what made it legal, so taken/TICK divides a single")
    print("  opportunity by however long it happened to persist.")

    if args.force:
        runs = forced_comparison(cfg, policy, args.force, args.force_episodes, args.seed)
        print(f"\n  --- counterfactual: `{args.force}` forced and suppressed, "
              f"{args.force_episodes} paired islands ---")
        print("  the paired lines below are LEARNED MINUS THE ROW, so a negative "
              "`vs ALWAYS`\n  means forcing the action helped and a positive "
              "`vs NEVER` is what the\n  action is currently worth.")
        for label, stats in runs:
            life = np.array([e.mean_lifespan for e in stats])
            print(f"  {label:22} lifespan {life.mean():6.1f} +- {life.std(ddof=1):5.1f}")
        print()
        for line in paired_lines(runs, "learned policy"):
            print(line)

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, **report}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
