# CLAUDE.md — working notes

Assume you are starting cold with only this file and `PROJECT_BRIEF.md`. This is
the state of the project, the decisions behind it, and the things that will bite
you.

## START HERE — handoff for the next session

**Milestones 1–5 are all trained, verified and written up.** Every milestone the
brief puts in scope for the simulation is done; M6 (an LLM narration layer) is
explicitly parked by the brief itself, so there is no obvious "next milestone" to
start. Read the results sections below before touching anything — several of them
record experiments that cost real time and should not be repeated.

Where each milestone landed, in one line each:

| | result | canonical checkpoint |
|---|---|---|
| M1 | 1.97× random, at the scripted forager's ceiling | `checkpoints/m1` |
| M2 | specialisation: action divergence 19× the shared-brain control | `checkpoints/m2` |
| M3 | 2.11× random; theft emerged *unpaid* after action masking | `checkpoints/m3-masked` |
| M4 | construction partially emerged; shaping annealed away cleanly | `checkpoints/m4c-anneal` |
| M5 | exchange did **not** emerge unpaid; paying for it made survival worse | `checkpoints/m5` |

### If you are picking this up, the honest open problems

In rough order of how much they would teach:

1. **The learned policy still loses to the scripted references in M3–M5.** M1 hit
   its ceiling; nothing since has. M5's best is 413 lifespan against the scripted
   trader's 544. The gap is not doomed actions (masking removed those), not
   entropy, not budget, and not perception of any mechanic we could name — see
   "Why the learned policy loses to the scripted forager" below for the seven
   interventions that all came back inside noise. This is the real open problem.
2. ~~**Nobody finishes a shelter.**~~ **This was not a real open problem, and the
   note that said it was had a stale premise.** Under `partial_shelter` the night
   drain falls *linearly* with build progress, so the final unit of a site is
   worth exactly what the first one was, and a partial shelter persists just as a
   finished one does. In the canonical annealed world nothing pays for completion
   at all. 0.1 shelters an episode is the policy correctly declining to buy
   something worthless — not a failure. The claim that finishing "is worth much
   more than the marginal unit suggests" was written for `m4`/`m4b`, where
   protection was binary, and was never re-derived after `m4c` turned
   `partial_shelter` on. Full arithmetic in the Milestone 4 section.
   `m4d` (cheaper sites) was run against the stale premise and came back flat, as
   it had to. `m4e` (`completion_premium`) tests the corrected one. Do not re-run
   `m4d`, and do not raise the shaping.
3. **Exchange needs a mechanism, not a bigger number.** M5 showed the unpaid
   chain is too weak and a flat payment produces a gift farm. If you want trade,
   the thing to change is the *mechanic* — see "What would actually be worth
   trying" in the Milestone 5 section.

Whatever you do next: an experiment here costs about six minutes (200 updates),
so run the control. Every result in this file that turned out to be wrong was
wrong because it had no control, and every one that survived had one.

## Current state

**Milestones 1–5 are complete and verified.** M6 is parked by the brief (an LLM
narration layer, explicitly out of scope for the simulation loop). Read the M3
shaping ablation and the M4 economy sizing notes before touching any config.

- M1: survival + foraging, one shared brain. 1.97× the random baseline, at the
  ceiling set by a hand-written forager.
- M2: six individual brains forked from the M1 checkpoint and trained
  independently, plus a behavioural-divergence view. Specialisation appeared:
  action divergence is 19× the shared-brain control.
- M3: scarcity, contested bushes, stealing. After the action-masking fix:
  **2.11× random** (471.6), theft emerging *unpaid* at 115 steals/ep, zero doomed
  actions — still honestly below the scripted forager (495.5). Territoriality
  appeared as inequality rather than as spatial partitioning. Canonical
  checkpoint: `checkpoints/m3-masked`.
- M4: wood/stone/shelter/night mechanics, day/night hazard, replay schema v2.
  Construction partially emerged (22% of nights sheltered against a control's 2%)
  but shelters almost never complete — which turned out to be *correct play*
  rather than a failure, because `partial_shelter` makes the last unit worth
  exactly what the first one is. The shaping annealed away cleanly. Canonical
  checkpoint: `checkpoints/m4c-anneal`. Halving the site cost (`m4d`) was tested
  afterwards and bought the policy nothing, as the corrected arithmetic says it
  had to.
- M5: `give_food`/`give_material`, a transfer ledger, replay schema v3, an
  exchange analysis tool and viewer page, and a scripted trader reference.
  **Exchange did not emerge unpaid** — giving was mildly selected *against* — and
  paying for it produced a gift farm that cost 54 ticks of life. Canonical
  checkpoint: `checkpoints/m5`.
- `pytest` passes (231 tests).
- All three viewer pages verified in a browser against real data, including their
  schema-mismatch failure paths.

Reproduce end to end:

```bash
python -m sim.train --run-name m1
python -m sim.train --run-name m2 --policy-mode individual \
    --init-from checkpoints/m1/latest.pt
python -m sim.divergence --checkpoint checkpoints/m2/latest.pt

# the M3 result (masking is what fixed it; see below)
python -m sim.train --config config/m3_masked.yaml --run-name m3-masked \
    --policy-mode individual --init-from checkpoints/m2/latest.pt
python -m sim.divergence --checkpoint checkpoints/m3-masked/latest.pt

# the labelled ablation, not the M3 result
python -m sim.train --config config/m3_shaped.yaml --run-name m3-shaped \
    --policy-mode individual --init-from checkpoints/m2/latest.pt

# Milestone 4: shaped and its unshaped control, from the M3 checkpoint
python -m sim.train --config config/m4.yaml --run-name m4 --updates 400 \
    --policy-mode individual --init-from checkpoints/m3-masked/latest.pt
python -m sim.train --config config/m4_unshaped.yaml --run-name m4-unshaped \
    --updates 400 --policy-mode individual --init-from checkpoints/m3-masked/latest.pt

# the cheap-sites test and its control — a NEGATIVE result, kept so nobody
# spends the six minutes finding it again
python -m sim.train --config config/m4d.yaml --run-name m4d --updates 200 \
    --policy-mode individual --init-from checkpoints/m3-masked/latest.pt
python -m sim.train --config config/m4d_unshaped.yaml --run-name m4d-unshaped \
    --updates 200 --policy-mode individual --init-from checkpoints/m3-masked/latest.pt

# Milestone 5: gifts unpaid (the result) and gifts paid (the ablation), both
# continued from the annealed M4 policy
python -m sim.train --config config/m5.yaml --run-name m5 --updates 200 \
    --policy-mode individual --init-from checkpoints/m4c-anneal/latest.pt
python -m sim.train --config config/m5_shaped.yaml --run-name m5-shaped \
    --updates 200 --policy-mode individual --init-from checkpoints/m4c-anneal/latest.pt
python -m sim.exchange --checkpoint checkpoints/m5/latest.pt
```

## Compute budget — runs are longer than they need to be

Measured on the shipped configs: the point at which a run's trailing 40-update
mean is within 3% of where it finishes.

| run | updates used | actually settled by |
|---|---|---|
| m1 | 300 | 175 |
| m3-masked | 300 | 89 |
| m4c | 400 | 79 |
| m5 | 200 | ~120 |

A 200-update run is **about six minutes** on this laptop (4.9M agent-steps at
~14k steps/s), which is the number to have in mind when deciding whether to run a
control. You can always afford the control.

**200 updates is enough for anything in this project**, and 250 is generous. The
300/400 figures are historical, not tuned. Cutting to 200 halves the wall-clock
and the heat for no loss of signal — the curves are flat long before the end.

Two related notes for anyone running this on a laptop:

* **`ppo.threads` defaults to 4, and that is not a compromise.** Measured, 4
  threads is as fast as 8 (40.0s vs 40.8s over 20 updates): these nets are small
  enough that the numpy env step dominates, so extra cores produce heat and
  nothing else. Results are bit-identical at any thread count and a test pins it.
* **Run experiments sequentially, not in parallel.** Two concurrent runs do not
  finish sooner in total, they just concentrate the same work into a hotter
  window — and on a fanless machine, thermal throttling can make the pair slower
  than running them back to back. Several results in these notes were produced
  by parallel pairs; that was for my convenience, not because it was faster.

**And do not repeat the long-run experiment.** `scarce-long` deliberately spent
1000 updates (3.3× budget) to test whether M3 was compute-starved. It was flat
from update 150. That question is answered; more compute is never the fix here.

## Layout notes

Built at the repo root rather than in a nested `island/` directory as the brief's
tree suggested — the repo *is* the project. Two files exist that the brief's tree
did not list:

- `sim/config.py` — YAML into frozen dataclasses. Needed by world, ppo, train and
  evaluate, so it did not belong inside any of them.
- `sim/make_fake_replay.py` — generates a replay without training, so the viewer
  could be built and verified before a policy existed (brief §5.2).
- `sim/divergence.py` + `viewer/divergence.html` — the M2 behavioural-divergence
  view.

**Checkpoints are scoped by run** (`checkpoints/<run_name>/latest.pt`). They were
flat until M2, at which point a run that forks from `checkpoints/latest.pt`
promptly overwrote the file it had just forked from. Do not flatten this again.

## Decisions, and why

**Auto-eat rather than an explicit eat action.** The brief left this open. Agents
eat automatically when `hunger < eat_threshold` and they are carrying food. Two
reasons: it keeps the action space at exactly the 10 actions the brief pins down,
and — the real reason — the eat reward is *scaled by hunger deficit*. With an
explicit action, an agent would be paid more for eating later, i.e. rewarded for
starving itself closer to death before acting. Auto-eat removes the pathology
instead of tuning around it.

**`hunger` counts down.** Per the brief it starts at 100, drains per tick, and
`<= 0` is death. So it is a satiety meter with a misleading name. The name is
kept for consistency with the brief; every docstring flags it. Do not "fix" the
direction without changing the observation normalisation and the viewer's colour
thresholds together.

**Observation distances use `observation.distance_scale` (20), not the island
diameter.** The brief said "distance-normalised". Dividing by the diameter (80)
squashes a bush 5 units away to 0.06 and destroys exactly the near-field gradient
the policy needs. Offsets are divided by 20 and clipped to ±1: full resolution
where it matters, saturation far away. Tunable in config.

**Padding is zeros, and magnitudes are [0, 1] not [-1, 1].** Berry counts and
neighbour hunger are normalised to [0, 1] so a zero-padded slot reads as "an
entity with nothing in it at zero offset", which no real entity is. Had those
channels been centred at zero, padding would have been indistinguishable from a
half-full bush sitting on top of you.

**Walking into the sea is impossible, not fatal.** Movement is projected back
onto the disc; the agent slides along the shoreline and loses the tick. The brief
wanted agents to "learn not to walk into the sea"; drowning would teach that
harder but adds a death mode the brief never specified. If foraging near the edge
looks bad, this is the knob.

**Bush layout is resampled every episode** (`bushes.resample_each_episode`).
Observations are egocentric with no absolute coordinates, so a fixed map cannot
be memorised anyway, but resampling stops the policy overfitting to one
arrangement of clusters. Set false if you want a stable map for eyeballing.

**Death is termination, the tick limit is truncation.** An agent that starves has
genuinely zero future value. An episode that hits `max_ticks` does not — those
agents were alive and would have carried on. `ppo.py` bootstraps `V(final_obs)`
into the reward at a truncation boundary. Collapsing the two teaches the policy
that the world ends at tick 600 and quietly poisons every value estimate near the
horizon.

**Dead agents keep their slot.** They are stepped every tick with a zero
observation and a forced `idle`, and dropped from GAE and every loss term via an
`active` mask. Masking the *loss* rather than the buffer keeps every tensor
rectangular. This is the part of `ppo.py` most likely to break under a change;
`tests/test_ppo.py` pins it from three directions (activity monotonicity, exact
row count reaching the update, forced-idle actions).

## Gotchas

**Berries gathered is a weak metric once a policy is competent.** Auto-eat makes
consumption homeostatic: a competent forager eats exactly as often as hunger
arithmetic demands and gathers exactly enough to top its inventory back up. The
scripted forager returns *identical* totals across wildly different noise levels
for this reason. Judge policies on lifespan and deaths; berries only separates
the incompetent from the competent, not the good from the great.

**Value loss rising is not a bug.** As the policy improves, returns grow, so the
critic's targets grow, so absolute value loss climbs. It only falls once the
return scale settles. Use `explained_variance` to judge the critic — it is
scale-free. (On the smoke-test bandit EV pins to 0 no matter what, because that
task's return is i.i.d. and genuinely unpredictable from the observation.)

**The viewer cannot list a directory.** It is a static page: no directory
listing over http, and `fetch` on `file://` is blocked outright. `sim/replay.py`
maintains `viewer/replays/index.json` and the viewer reads that. Anything that
writes a replay outside `ReplayRecorder.save` must call `update_manifest` or the
file will not appear in the dropdown. Drag-and-drop always works, including from
`file://`.

**The viewer needs network on first load** — Three.js comes from a CDN via an
import map, as the brief specified. There is no bundler and no vendored copy.

**Replay schema is versioned and the viewer fails loudly.** `SCHEMA_VERSION` in
`sim/replay.py`, `SUPPORTED_SCHEMA` in `viewer/main.js`. The viewer also
cross-checks the file's own `tick_fields.agent` against the column order it
expects, so reordering columns cannot silently shift what gets rendered. Bump the
version on *any* change to the tick encoding, and update both sides.

**Determinism is CPU-only and covers the whole stack.** Same seed + same config
gives byte-identical replays and identical training curves. `PPOTrainer` owns its
own `np.random.Generator` for minibatch shuffling — do not reach for global numpy
state anywhere in the training path, it silently breaks the guarantee.
`VecWorld` seeds per-env streams via `SeedSequence`, not `seed + i`.

## Reference points

Measured on the shipped `config/default.yaml`, 20 episodes, seeds 10000+.
Training run `m1`: 300 updates, 7.37M agent-steps, 4.2 min on CPU.

| policy | mean lifespan | of 600 ticks | deaths/ep | berries |
|---|---|---|---|---|
| random actions | 302.3 ± 61.6 | 50% | 5.10 | 11.5 |
| **learned (300 updates)** | **595.6 ± 19.0** | **99%** | **0.10** | **64.0** |
| scripted greedy forager | 600.0 ± 0.0 | 100% | 0.00 | 66.0 |

**1.97× the random baseline**, effectively at the scripted ceiling. Reproduce
with `python -m sim.train --run-name m1` (seed 0) and
`python -m sim.evaluate --checkpoint checkpoints/latest.pt --baselines`.

The scripted forager (`policy.greedy_forager_actions`) reads *only* the 26-dim
observation, not world state. That is on purpose: it proves the observation is
sufficient for the task, so any failure to learn is the algorithm's fault rather
than the sensor's. Treat it as a soft ceiling for memoryless reactive foraging —
it is not optimal, it just never wastes a tick.

Learning curve shape, for recognising a healthy run: lifespan sits at the random
floor for ~60 updates while explained variance climbs to ~0.8 (the critic learns
first), then lifespan rises steeply between updates 80 and 170, and pins at 600
from ~240 onward. Entropy only falls from 2.30 to ~2.04 — the policy stays
noticeably stochastic even when solving the task, because with bushes everywhere
many actions are near-equivalent and nothing punishes the indifference.

### Observed behaviour (not rewarded, worth knowing)

- **The learned policy spams `gather`** — 34% of its actions, against 1.9% for
  the scripted forager, which gathers only when it needs to. Gathering pays +1.0
  whenever an inventory slot is free, so camping a bush and grabbing
  opportunistically is straightforwardly worth more than walking away. It is
  rational given the reward, not a bug, but it means the learned policy looks
  *busier* than an optimal one.
- **Agents cluster on bush hotspots** and travel much less than the scripted
  forager (mean distance to nearest bush 2.25 vs 7.05 for random; 53% of ticks
  spent inside gather range against 11% by chance). No reward encourages
  proximity to other agents — this falls out of everyone independently wanting
  the same clusters. Do not read it as social behaviour yet; M3 is where
  competition gets a real test.

## Milestone 2 — individual brains

`PolicyGroup` (in `policy.py`) holds one `ActorCritic` per agent and dispatches on
an `agent_ids` tensor that `ppo.py` threads through collect and update. A shared
`ActorCritic` accepts the same argument and ignores it, which is the whole reason
there is still only one training loop rather than two that drift apart.

**Gradient clipping is per-brain, not global.** `PolicyGroup.clip_grad_norm`
clips each policy separately. Clipping the union would mean one agent's bad
update scales down every other agent's gradient that step — quietly coupling six
policies whose entire purpose is to be independent. If you add another optimiser
concern, ask the same question of it.

**Forking, not fresh initialisation.** M2 starts from the trained M1 shared
checkpoint copied six ways, so every agent begins competent and diverges from
there. Six randomly-initialised brains would also "diverge", but you would mostly
be measuring initialisation noise.

One Adam over all six brains is exactly equivalent to six separate Adams — its
state is per-parameter and nothing couples the policies — so the optimiser stays
simple. Throughput drops from ~30k to ~26k agent-steps/s (six small matmuls where
there was one large one), which was judged an acceptable price for a readable loop.

### Results

300 updates forked from `checkpoints/m1/latest.pt`. Both rows below are 20
episodes on the *same fixed island*, seed 10000:

| | M1 shared (control) | M2 individual |
|---|---|---|
| mean lifespan | 594.5 | 592.9 |
| **action JS divergence** | **0.0012 bits** | **0.0233 bits (19×)** |
| territory JS divergence | 0.6722 | 0.9080 |
| gather-share spread | 2.2 pts | 7.4 pts |
| mean-radius spread | 5.5 | 21.1 |
| time-in-gather-range spread | 10.5 pts | 22.8 pts |

Individual brains genuinely specialised. Agent 5 became a bush-camper (37.8%
gather, 77.3% of ticks in range, closest to bushes); agent 2 a rover (68.2%
travel, lowest gather share, best hit rate); agent 4 stayed near the island centre
(mean radius 9.0) while agent 3 ranged to 30.1.

**Read the action matrix, not the territory matrix.** Territory divergence is
already 0.67 bits for six agents *sharing one brain* — they spawn apart and each
walks to whichever cluster is nearest, so different ground is the default, not a
finding. Action divergence is near-zero under sharing and is the signal that
survives the control.

**Lifespan is saturated, so it cannot show M2 working.** Both milestones sit at
the 600-tick ceiling; M2 is not "better", it is *differentiated*. Foraging
proximity did improve (time in gather range went from 41–51% to 55–77%), but the
headline survival number has no room left to move. Any future milestone that
wants a survival signal needs a harder world first.

## Gotchas (Milestone 2)

**Territory heatmaps need a fixed map.** With `bushes.resample_each_episode` on
(the training default), every episode scatters clusters somewhere new, so
averaging positions in absolute coordinates smears every agent toward the same
centred blob. `sim/divergence.py` pins the layout by default; `--no-fixed-map`
exists but makes the territory section meaningless, and the report carries a
`fixed_map` flag that the viewer turns into a warning banner. The
map-independent statistics (action mix, hit rate, bush distance) are fine either way.

**Old checkpoints have no `mode` key.** `policy_from_config_dict` defaults them to
shared, which is what pre-M2 checkpoints are. Do not make the key required.

## Milestone 3 — competition

### Configs are layered now

`extends:` in a YAML config inherits from a parent and overrides only the keys it
restates. The chain is `m3.yaml -> scarce.yaml -> default.yaml`. This exists so
the M1/M2 world stays reproducible instead of being edited out from under the
results already documented above. Don't collapse it back into one file.

### The M3 world sits at exactly 100% of subsistence

Measured, not estimated (`config/m3.yaml`, 600 ticks):

| | berries |
|---|---|
| demand: 8 meals x 6 agents | 48 |
| supply: 12 initial + 6 bushes x 6 regrowths | 48 |

There is **no slack at all**. Perfect play feeds everyone exactly, and any
inefficiency starves somebody. The scripted forager harvests 37.4 of the 48 the
island produces (78%) and still loses 2.2 agents an episode; nothing in this
world keeps six agents alive.

This was not deliberate — when sizing `scarce.yaml` I estimated 6 meals per agent
from `eat_restore / drain_per_tick` and the true figure is 8, because an agent
eats at the *threshold* (60) rather than at empty, so each meal only buys
`(60 + 35 - 60) / 0.5 = 70` ticks rather than a full tank. Recompute demand with
`sim` rather than by hand before changing the bush economy again.

Two consequences worth carrying forward:

* **Survival here is dominated by distribution, not production.** That is exactly
  why the scripted thief beats the scripted forager despite harvesting slightly
  *less*: theft moves berries to whoever is about to eat one.
* **It is a knife-edge testbed.** Judge a policy on berries harvested as a share
  of the 48 the island produces, not on lifespan alone — lifespan compresses
  every policy into a narrow band because the food simply is not there.

### The world had to get harder first

M1 and M2 were oversupplied by roughly 7×: ~360 berries against the ~48 six
agents actually eat. Everything survived, every policy pinned to the 600-tick
ceiling, and survival stopped being able to register any effect. `scarce.yaml`
brings supply down to meet demand (6 bushes, capacity 2, regrow 100 → 48 berries)
and the scripted forager falls from 600 to ~498 with 2.4 deaths per episode. Six
bushes for six agents is deliberate: one each, if they can hold it.

### Mechanics, and one that did not work

* **`contest_bushes`** — one taker per bush per tick. **Nearly inert on its own**,
  and this is the interesting part: agents are crowded onto the same bush for
  ~2500 of 4800 ticks, yet only 3 gather attempts per 8 episodes were ever
  blocked. In a scarce world bushes are *empty* most of the time, so two agents
  almost never manage a *successful* gather on the same tick even while both
  parked on it. What agents actually compete over is who is standing there when a
  berry regrows.
* **`exclusive_bushes`** — only the agent closest to a bush may take from it. This
  is the mechanic with teeth: blocked attempts went from 3 to 1456. Standing on a
  bush now denies it. Dead agents cannot block (tested — a corpse holding a bush
  forever would be a nasty silent bug).
* **`enable_steal`** — action 10, take one berry from a neighbour within
  `steal_radius` who has some. Appended, never inserted, so every earlier action
  keeps its index and an M1/M2 checkpoint means the same thing here.
* **`observe_neighbour_food`** — widens the observation 26 → 29. Required: a policy
  that cannot tell a loaded neighbour from an empty one could only learn "rob at
  random", which resembles the behaviour without being it.

**Theft pays no reward.** Gathering pays +1.0, stealing pays 0. The brief allows
no reward terms beyond survival, so robbery has to earn its keep through the food
it yields and the eating that food enables — deliberately the harder option. If
it emerges anyway, it emerged from survival pressure rather than from us paying
for it. There is a test pinning this; if it ever starts paying out, that test
should fail loudly.

The `scripted thief` baseline (`policy.greedy_thief_actions`) is the reference:
opportunistic theft only, never chasing a victim, so it is a floor on what theft
is worth rather than a ceiling.

### Growing a policy across a milestone boundary

M3's observation and action space are both wider than M2's, so an M2 checkpoint no
longer fits. `grow_policy` copies every trained weight and **zero-initialises the
new ones**: a zeroed input column contributes nothing and a zeroed action row
gives `steal` a logit of 0 beside trained logits, so the grown policy starts out
behaving as it did and then learns to use what it has been given. The new action
is reachable rather than masked, which is what lets PPO find out whether it is
worth taking. Shrinking is refused outright.

### Results — and two things that did not work

All 20 episodes, `config/m3.yaml`, seeds 10000+:

| policy | mean lifespan | vs random | deaths/ep | steals/ep |
|---|---|---|---|---|
| random actions | 223.0 | 1.00× | 5.85 | 0 |
| **learned, forked from M2** | **452.1** | **2.03×** | 3.40 | 28.9 |
| learned, from scratch | 223.3 | 1.00× | 5.95 | 0.1 |
| scripted forager | 495.5 | 2.22× | 2.20 | 0 |
| scripted thief | 553.3 | 2.48× | 1.45 | 1099.3 |

**1. Theft did not emerge.** The learned policy steals 29 times an episode where the
scripted thief manages ~1100, and it finishes *below both* scripted references.
This is not "theft is useless" — the scripted thief proves theft is worth ~58 ticks
of extra lifespan and a death per episode. It is a credit-assignment failure: with
`reward.steal = 0.0` the chain is steal → carry → auto-eat some ticks later → don't
starve, and at γ=0.99 that signal is too weak and too delayed to compete with the
+1.0 a gather pays immediately. This is exactly the failure mode the brief predicts
for M4's construction rewards, arriving early.

### The shaping ablation, and why it is a warning for M4

`config/m3_shaped.yaml` pays a successful steal the same +1.0 a gather earns.
Everything else is identical. 20 episodes:

| | M3 (theft unpaid) | M3 shaped (theft paid) |
|---|---|---|
| steals / episode | 28.9 | **140.0** |
| berries gathered / episode | 26.4 | 23.4 |
| policy entropy (last 40 updates) | 2.167 | 1.742 |
| **mean lifespan** | **452.1** | **428.6** |
| deaths / episode | 3.40 | 3.70 |

**The shaping worked and the outcome got slightly worse.** Theft emerged — 4.8×
more of it, and the entropy drop shows the policy genuinely committing rather than
sampling it by accident, which confirms the credit-assignment diagnosis. But
survival did *not* improve: lifespan fell 452 → 429 and deaths rose.

The mechanism is worth understanding before M4 leans on shaping. Stealing moves
food between agents; it never creates any. Paying for it buys ticks spent
redistributing the same berries instead of harvesting new ones — gathering fell
26.4 → 23.4 — so the population ends up with less food overall and dies sooner.
The policy learned to steal *because stealing pays*, not because stealing helps.

Note also that the scripted thief steals ~8× more than the shaped policy and
*does* survive better (553): opportunistic theft targeted at loaded neighbours is
useful, indiscriminate theft-for-reward is not. Volume was never the point.

**The lesson for M4:** a shaped reward reliably produces the behaviour it pays
for. That is not evidence the behaviour helps. When M4 adds intermediate rewards
for gathering materials and partial construction, the shaped run has to be
compared against the unshaped one *on the terminal metric* — and the brief's
instruction to test whether shaping can be annealed away is the right instinct.

**2. The scarce world cannot be learned from scratch.** A from-scratch run lands on
*exactly* the random baseline (223.3 vs 223.0, 1.00×) after the full 7.4M steps,
with entropy still at ~2.28 of a possible ln(11)=2.40 and negative mean reward.
Agents starve before they can discover foraging, the −10 death term dominates
everything, and the policy never escapes. **The milestone chain is load-bearing,
not a narrative convenience** — M3 only works because M1 and M2 transferred
competence into it. If you make the world harder again, expect to need a
curriculum, not a bigger budget.

### Competition produced convergence, not partitioning

The prediction going in was that territoriality would show up as territory
divergence *rising*. It fell — hard: 0.9080 in M2 to **0.3628** in M3. With 6
bushes instead of 20, agents pile onto the same few spots rather than spreading
out. Territory divergence measures how *differently* agents are distributed, and
scarcity makes them all want the same ground.

What did appear is inequality. Lifespan spread went from 0 ticks in M2 (everyone
hit the ceiling) to **144 ticks**:

| | agents 3, 5 | agents 2, 4 |
|---|---|---|
| lifespan | 388.6, 392.1 | 276.0, 248.0 |
| distance to nearest bush | 1.50, 1.55 | 4.97, 3.21 |

A dominant pair holds bushes and survives; the excluded ones are pushed to the
margins and starve. **That is the territorial result — it is just expressed as who
eats rather than as who stands where.** Read the lifespan spread, not the
territory matrix, as M3's headline behavioural number.

Gather hit rate also collapsed from ~5.5% to 0.8–3.4%, which is `exclusive_bushes`
working: most gather attempts now lose to a closer agent.

### Why the learned policy loses to the scripted forager

M3's headline sits below both scripted references. That is worth understanding
before building on it, so here is the investigation, including the parts that
found nothing — they are the expensive ones to repeat.

**What the policy actually does wrong.** Run the learned policy, and at each tick
ask what the scripted forager would have done from the same observation:

| forager wanted | policy did | share of living ticks |
|---|---|---|
| move (81% of ticks) | move | 47.6% |
| | **gather** | **17.7%** |
| | **steal** | **12.8%** |
| | idle | 2.5% |

Exact agreement is 9.3%. **The policy does not travel.** Where the forager would
be walking to a berry-bearing bush, the policy stands still and mashes `gather`
or `steal`, which pay nothing from where it is standing. That is 30% of every
tick it lives.

**This is inherited, not a tuning failure.** M1's abundant world taught exactly
this: "camp a bush and spam gather" is *optimal* when there are 20 bushes and one
is always underfoot, and it is written up as an observed M1 behaviour above (34%
gather actions). M2 kept it and M3 forked from M2. In a six-bush world where only
the closest agent may harvest, that prior is actively wrong — and a policy that
never travels cannot find the four bushes nobody is standing on. It explains the
harvest gap (26 berries of 48 against the forager's 37) and the *fall* in
territory divergence at the same time.

**Seven interventions, all within noise (437–452 lifespan):**

| intervention | result |
|---|---|
| baseline (as first run) | 452 |
| fixing the feature-misalignment bug | 447 |
| `observe_bush_contested` | 450 |
| `ent_coef` 0.01 → 0.002 | 446 |
| `gamma` 0.99 → 0.995 | 445 |
| annealing `ent_coef` to 0.0005 | 437 |
| annealed entropy + `gamma` 0.997 | 442 |
| one shared brain instead of six | 449 |

Two hypotheses were killed outright rather than merely failing to help:

* **Entropy is not the constraint.** Annealing the bonus to 0.0005 left policy
  entropy at 2.03 of a possible 2.40. The policy is near-uniform *by choice* — it
  has no confident preference to express — so "the bonus forbids commitment" is
  simply wrong.
* **Distance clipping is not the constraint.** `observation.distance_scale` (20)
  was tuned for the abundant world, and a scarce island has bushes much further
  away, so the ±1 clip looked like a suspect. Measured: 0.3% of nearest-bush
  offsets saturate on one axis and 0.0% on both. Direction is intact.

**The gap survives removing competition entirely.** In `scarce.yaml` with no
blocking and no stealing — the pure foraging task — the policy still plateaus at
~27 berries against the forager's 38.7. So this is not about M3's mechanics at
all; it is about foraging in a world where food is far apart, and the failure was
simply invisible in M1 because food was never far apart. A 1000-update run
(3.3× budget) was flat from update 150, so more compute is not the answer either.

### The fix that worked: action masking

`competition.mask_invalid_actions` hides `gather` and `steal` when they cannot
possibly succeed (no berry in range / no loaded neighbour in reach; moving and
idling are always available, dead agents keep `idle` so no row is ever fully
masked). It is not a reward term and not a hint about what is best — the
observation says what is *there*, the mask says what is *reachable*. The masked
logits use −1e8 rather than −inf so a fully-masked row cannot mint NaN gradients,
and PPO stores the rollout masks because the ratio must be computed against the
behaviour policy, which was masked.

| | unmasked (m3-fork) | masked (m3-masked) |
|---|---|---|
| mean lifespan | 452.1 | **471.6 (2.11×)** |
| deaths / ep | 3.40 | 3.10 |
| berries / ep | 26.4 | 29.1 |
| steals / ep — **still unpaid** | 28.9 | **115.5** |
| doomed gathers (% of ticks) | 17.7% | **0.0%** |
| gather hit rate | 0.8–3.4% | **50–98%** |

Two things worth reading twice. **Theft finally emerged without being paid for**
— masking made `steal` only ever appear when a loaded victim is in reach, so its
empirical return became visible to PPO, and usage rose 4× with `reward.steal`
still 0.0. And the behavioural profile flipped from stand-and-mash to travel
(60% → 87% of ticks moving).

Stacking the contested-bush channel and entropy annealing *on top of* masking
(600 updates, `m3-final`) gave 456.9 — nothing again. **`checkpoints/m3-masked`
is the canonical M3 checkpoint.**

**The honest residual: 471.6 still trails the scripted forager (495.5) and thief
(553.3).** The remaining gap is not doomed actions (there are none left), not
entropy, not budget, not perception of any mechanic we could name. Lifespan
spread on the fixed map is 307–550: one agent roves at a 98% hit rate while
others get excluded and starve, so the shortfall lives in the crowding/exclusion
dynamics. Left as the open problem it is; masking is where principled
single-change fixes stopped paying.

## Milestone 4 — multi-resource + construction

Wood and stone as depletable nodes, communal shelter sites, and a day/night
cycle where the unsheltered drain hunger at 3×. `chop`/`mine`/`build` appended as
actions 11–13; observation 29 → 55. Replay schema v2. All off by default, so
M1–M3 worlds stay bit-identical.

### The economy, sized in code

Per the M3 postmortem rule, demand was computed with `sim` rather than by hand:

| | berries |
|---|---|
| sheltered demand (8 meals × 6 agents) | 48 |
| **supply** (8 bushes × 2 cap + 8 × 6 regrowths) | **64** |
| exposed demand (sleeping rough every night) | 72 |

Supply sits *between* the two, so a population that shelters can afford the ticks
construction costs and one that does not starves. Shelter is load-bearing by
arithmetic, not decoration.

### Results — construction partially emerged

20 episodes each. The scripted builder is the bar; it forages first, runs home at
dusk, and feeds the most-finished site (targeting the *nearest* site instead
spreads material across three and completes almost nothing — worth knowing).

| policy | lifespan | deaths | shelters/ep | nights indoors |
|---|---|---|---|---|
| scripted builder | **529.8** | 2.05 | **1.9** | **92%** |
| scripted forager | 457.6 | 3.12 | 0 | 0% |
| **learned, shaped (m4c)** | **393.5** | 4.70 | 0.1 | **22%** |
| learned, unshaped control | 362.5 | 5.05 | 0.0 | 2% |

**The shaped run beat its control on the terminal metric** — +31 ticks of life,
22% of night ticks under shelter against 2% — which is the *opposite* of the M3
shaping ablation, where paying for theft produced more theft and less survival.
Here the shaping bought a real outcome, not just the behaviour it paid for.

**But full shelters essentially never complete** (0.1 per episode against the
builder's 1.9), and the learned policy remains far below the scripted reference.
Agents deliver materials and shelter under half-built walls; they do not finish
the job. That is the honest headline.

### Three attempts, and what each one taught

| attempt | deliveries/ep | shelters/ep | nights in |
|---|---|---|---|
| `m4` — sites scattered, cheap shaping | 0.56 | 0.012 | 0.3% |
| `m4b` — sites on the berry clusters | 2.00 | 0.056 | 2.5% |
| `m4c` — + partial shelter protection | 1.94 | 0.064 | **16%** |

Neither fix touched the shaping coefficients, deliberately — the M3 ablation is
the standing warning against that lever. Both were the same *kind* of move that
solved M3: **change the shape of the problem, do not pay more at the summit.**

* **`m4b` deleted the uncreditable walk.** With sites scattered independently, a
  loaded agent had to cross open ground to a place it otherwise never went, and
  that leg earned nothing — structurally identical to M3's doomed gathers.
  Putting sites on the clusters agents already live at raised deliveries 3.6×.
* **`m4c` removed the cliff.** A site costs four units and only the fourth bought
  anything, so three quarters of the work was invisible to the value function.
  Scaling protection with build progress made the landscape continuous, and
  night protection went 2.5% → 16%. **It also removed the summit, which nobody
  noticed for two milestones** — a linear protection curve makes the last unit
  worth exactly what the first one is, so "nobody completes a shelter" stopped
  being a defect and became correct play. See "What is left" below.

An earlier sizing (6-unit sites, shaping 0.3/0.5/2.0) produced *zero* completions
in 200 updates; CSVs in `runs/_m4_probe1`. At 0.3 a material action loses to a
+1.0 gather everywhere the two compete.

### The annealing test — the shaping was a real bootstrap

The brief asks whether M4's shaping can be annealed away once it has done its
job. `config/m4c_anneal.yaml` continues the shaped policy for 200 more updates
with all four shaping terms at 0.0, changing nothing else.

| | deliveries/ep | shelters/ep | nights indoors | lifespan |
|---|---|---|---|---|
| m4c shaped | 1.94 | 0.064 | 16.1% | 377.6 |
| **m4c annealed (nothing paid)** | **2.21** | **0.091** | **18.0%** | **382.3** |
| m4c unshaped control | 0.73 | 0.011 | 4.7% | 368.8 |

**The behaviour survived, and slightly improved.** Two hundred updates with
nothing paying for wood, stone, delivery or completion, and construction stayed
at the shaped level rather than decaying toward the control. Final evaluation:
403.1 lifespan, 17% of nights sheltered — the best M4 number of any run.

So the shaping here was genuine scaffolding that could be removed, not a
subsidy the behaviour depended on. Once the policy has *found* shelter, the
survival benefit alone sustains it — which is the answer the brief was asking
for, and the reason the shaping is defensible.

**Read this against the M3 shaping ablation, which went the other way.** Paying
for theft produced 4.8× the theft and *worse* survival, because stealing moves
food without creating any. Paying for construction produced building that pays
for itself and persists unpaid, because a shelter genuinely reduces the drain.
The difference is not the shaping technique — it is whether the shaped behaviour
was actually worth doing. That is the test, and only the terminal metric
answers it.

### What is left, and a correction to what this section used to say

**This section was wrong for two milestones, so read the correction before the
open problem.** It used to say: "the remaining gap is the last unit — finishing a
site is worth much more than the marginal unit suggests (a complete shelter
protects fully and permanently), and the policy stops at *good enough* partial
cover." That was true of `m4`/`m4b`, where `partial_shelter` was off and
protection was binary. **It stopped being true the moment `m4c` turned
`partial_shelter` on, and nobody re-derived it.** Measured, `m4c`, 4-unit site,
night drain 1.5 exposed / 0.5 sheltered:

| delivered | protection | night drain | marginal gain |
|---|---|---|---|
| 1/4 | 0.25 | 1.25 | +0.25 |
| 2/4 | 0.50 | 1.00 | +0.25 |
| 3/4 | 0.75 | 0.75 | +0.25 |
| **4/4** | **1.00** | **0.50** | **+0.25** ← the last unit |

Exactly linear, and a partial shelter persists exactly as a finished one does, so
"fully and permanently" separates nothing. In the canonical annealed world
(`reward.complete` 0.0) **laying the final unit buys precisely what the first one
bought and not one tick more.**

So "nobody finishes a shelter" was never a perception failure or a credit-
assignment failure. **It was correct play**, and m4c's own fix caused it:
`partial_shelter` removed the cliff, and the reason to reach the top went with
it. The only thing that ever paid for completion was `reward.complete: 3.0`,
which is shaping, and which is zero in the canonical checkpoint.

Two things follow, and both are load-bearing for whoever picks this up:

* **0.1 shelters an episode is not a defect to fix.** Read it as the policy
  correctly declining to pay for something worth nothing. The comparison to the
  scripted builder's 1.9 is not like-for-like: the builder finishes because it
  was *written* to finish, not because finishing pays.
* **A fix that removes a cliff can remove the summit with it.** That is the
  general lesson, and it is why this went unnoticed — `m4c` was a success on
  every metric anyone looked at (nights indoors 2.5% → 16%), and the thing it
  quietly deleted was only visible by re-deriving the arithmetic.

The standing advice that survives the correction:

* **Do not raise the shaping.** `m4` already showed 3.5× the material activity of
  its control with no completions; volume was never the constraint.
* Longer training is *not* indicated — the M3 investigation burned 3.3× budget
  for nothing, and these curves are flat by update ~250.
* **Cheaper sites were the untried structural lever. They were tried, and they
  did not pay** — see below. Note that the correction above explains *why* they
  could not have: m4d halved the distance to a summit that was worth nothing on
  arrival.
* If you want completion, the mechanic has to pay for it. That is what
  `construction.completion_premium` is for — it withholds a slice of the
  protection until a site is finished, so the continuous landscape survives and
  the last unit is worth more than the others. `config/m4e_premium.yaml` (the
  mechanic alone) and `config/m4e.yaml` (mechanic + the `site{j}.finishes`
  observation channel) are the pair; both headers carry their predictions.

### The cheap-sites test (`m4d`) — the lever did not pay

`config/m4d.yaml` cuts a site from 4 units to 2 (1 wood + 1 stone), which at
`material_capacity` 2 is exactly **one round trip**: an agent already carrying a
full load can finish a shelter without ever forming a multi-trip intention. Every
other thing is m4c. 1+1 rather than 2+0 so `mine` does not become a permanently
doomed action and the rock observation channels do not become noise.

The trap this run was designed around, written into the config header before it
ran: under `partial_shelter`, protection is the *fraction* of units delivered and
"indoors" is `protection >= 0.5`. That is 2 deliveries at 4-unit sites and **1**
at 2-unit sites, so `shelters` and `night_sheltered_frac` both get mechanically
cheaper along with the world. Neither can be compared across sizings.
`tests/test_construction.py::test_indoors_statistic_scales_with_site_cost` pins
this so the result cannot be misread later.

Matched budget, trailing 40 updates of a 200-update run — the tightest estimate
available, since it averages far more episodes than a 20-episode evaluation:

| | m4c (4-unit) | **m4d (2-unit)** |
|---|---|---|
| **mean lifespan** | **374.6** | **373.6** |
| shelters / ep *(not comparable)* | 0.064 | 0.083 |
| nights indoors *(not comparable)* | 17.9% | 24.7% |
| deliveries / ep *(not comparable — 6 units exist, not 12)* | 1.91 | 0.96 |

**Dead flat on the only metric that transfers.** And the 20-episode evaluation
says who did benefit:

| | m4c world | m4d world (cheap sites) |
|---|---|---|
| learned, shaped | 393.5 | 379.6 ± 61.9 |
| learned, unshaped control | 362.5 | 363.6 ± 58.9 |
| **scripted builder** | **529.8** | **545.0 ± 45.7** (2.5 shelters, 95% nights in) |
| **gap, learned → builder** | **136** | **165** |

The scripted builder converted cheaper sites into more shelters and ~15 more
ticks of life. The learned policy converted them into nothing, so the gap did not
narrow — if anything it widened. Neither individual delta clears ~1.5 SE on 20
episodes, so the honest claim is the conservative one: **cheaper sites bought the
learned policy no survival, while being clearly usable by something that knows
how to use them.**

The sharpest way to see it: the "indoors" bar was **halved** — one delivery
instead of two — and the shaped policy still cleared it on 22% of nights, exactly
what it managed at m4c. The control drifted 2% → 10% on the same halving, which
is the metric inflating with no behaviour behind it. That is the whole result in
two numbers.

Shaping still beat its own control (379.6 vs 363.6, +16), so nothing about m4c is
retracted. What is retracted is the hypothesis that completion was out of reach
because it was *too far*. It is not the distance to the summit. **No annealing
run was done, deliberately** — there was no gain to remove the scaffolding from,
and m4c's own header calls that kind of run a ritual.

Where this leaves the milestone: the three structural levers that worked (site
placement, partial protection, action masking) all removed something *uncreditable*
from the chain. Cheap sites removed *length*, not uncreditability, and length was
never what was broken.

**And the deeper reason they could not have worked** — found while writing up the
next experiment, not while running this one — is the correction above: under
`partial_shelter` the last unit is worth exactly what every other unit is worth.
m4d halved the distance to a summit that pays nothing on arrival. Any lever that
only shortens the chain is arguing with the wrong premise.

## Milestone 5 — exchange

Two appended actions, `give_food` (14) and `give_material` (15), each moving one
unit to the nearest neighbour in reach with room for it. Observation 55 → 61
(neighbours' carried wood and stone). Replay schema v3 carries a per-tick list of
transfers. All off by default, so M1–M4 worlds stay bit-identical.

The world is `m5.yaml` = the annealed M4 world plus exchange, so an M5 world pays
for exactly three things, all from Milestone 1: staying alive, gathering, eating.

### Design decisions, and why

**Food and materials are separate actions; wood and stone are not.** They are two
different economies — one keeps you alive, the other builds shelter — and an
agent carrying both would otherwise be unable to choose which it is taking part
in. Wood versus stone is a much narrower distinction (the receiver's `build`
already resolves which the site needs), so collapsing those two saved an action
slot that PPO would have had to discover the value of separately.

**The giver does not choose the recipient.** A gift goes to the *nearest*
neighbour with room, exactly as `steal` takes from the nearest loaded victim.
What the giver actually controls is where it stands, so **positioning is the
targeting mechanism**. This bit the scripted trader first: a rule that asked "is
anyone near me hungry?" gave away 80 berries an episode of which 3 were eaten,
because the berry kept going to somebody else. The rule has to be evaluated on
the neighbour the *world* would pick.

**Nothing pays for a gift** (`reward.give = 0.0`), the same call as
`reward.steal`. `m5_shaped.yaml` raises it as a labelled ablation, and its header
contains the prediction it was run to test, written before the run.

**The transfer ledger is its own artefact.** A per-episode average of "gifts"
cannot answer who gave to whom or whether it came back, so `sim/exchange.py`
writes a JSONL of `(episode, tick, giver, receiver, item)` plus an aggregate
report that `viewer/exchange.html` renders. `exchange.log_transfers` is off by
default: training runs 32 worlds at once and does not need the ledger.

### Results — exchange did not emerge

20 episodes, seeds 10000+, all on the same islands:

| policy | lifespan | deaths | berries | gifts/ep | nights indoors |
|---|---|---|---|---|---|
| random actions | 176.6 | 6.00 | 1.6 | — | — |
| scripted forager | 463.6 | 3.05 | 50.4 | 0 | 0% |
| scripted builder | 529.8 | 2.05 | 38.5 | 0 | 92% |
| **scripted trader** | **543.8** | **1.85** | 38.0 | 8.3 | 93% |
| M4 policy grown into M5, untrained (control) | 406.4 | 4.60 | 34.9 | 11.3 | 15% |
| **learned, gifts unpaid (`m5`)** | **413.3** | 4.60 | 34.6 | **7.9** | 20% |
| learned, gifts paid (`m5-shaped`) | 359.1 | 5.30 | 26.9 | **330.9** | 15% |

**Read the control row.** A grown-but-untrained policy already gives 11.3 times
an episode, because a zero-initialised action row makes `give` just another thing
to sample. After 200 updates with nothing paying for it, that fell to 7.9. Giving
was not merely un-learned — it was mildly selected *against*, which is correct:
a gift costs a tick and hands the payoff to somebody else.

**The mechanic is not worthless, which is what makes this a finding.** The
scripted trader beats the scripted builder on the same islands by **+14.0 ± 5.2
ticks (paired, better on 12 of 20 islands)** and 0.2 fewer deaths, on about
**eight** well-aimed gifts an episode. So a handful of gifts genuinely buys
survival, and PPO still cannot find them. This is the credit-assignment wall the
M4 handoff predicted, arriving exactly where it said it would.

### The shaping ablation — a gift farm, as predicted

`m5_shaped.yaml` pays a successful transfer +1.0, what a gather pays. The
prediction written into its header beforehand was: many gifts, fewer berries,
lifespan at or below the unpaid run. All three:

| | unpaid (`m5`) | paid (`m5-shaped`) |
|---|---|---|
| gifts / episode | 7.9 | **330.9 (42×)** |
| berries gathered / ep | 34.6 | 26.9 |
| **mean lifespan** | **413.3** | **359.1** |
| deaths / episode | 4.60 | 5.30 |

The ledger says exactly what went wrong, which a lifespan number alone could not:

| ledger (20 episodes) | unpaid | paid |
|---|---|---|
| transfers / episode | 11.1 | 355.2 |
| share of living ticks spent giving | 0.4–1.5% | 16–20% |
| net flow per agent (gave − received) | −13 … +14 | −23 … +31 of ~1200 |
| reciprocity | 0.851 | **0.942** |
| gifted food eaten within 50 ticks | 56% | **5%** |
| gifted material delivered within 50 ticks | 3% | **0%** |

Every agent gives and receives about 1200 times and ends up net flat, and almost
nothing gifted is ever used. It is a circulation farm: two agents standing next
to each other pass a berry back and forth and collect a gather's worth of reward
each time, without a berry being created or eaten. **A shaped reward reliably
produces the behaviour it pays for, and that is still not evidence the behaviour
helps** — the M3 theft ablation, restated with a bigger multiplier.

Note the asymmetry with M4, which is the useful comparison: paying for
construction produced building that survived annealing, because a shelter really
does reduce the drain. Paying for gifts produced motion, because a transfer
creates nothing. The technique did not change; the mechanic did.

### What would actually be worth trying

Not a bigger coefficient, and not more updates (the curves are flat by ~120).
The unpaid chain is too weak, so change the *mechanic*:

* **Make gifts non-fungible.** If wood could only be harvested by an agent
  standing far from the sites, and building only worked near them, a relay would
  be the *cheapest* way to build rather than a nicety. Specialisation would then
  be forced by geography rather than hoped for.
* **Let the giver choose the recipient** (nearest is currently forced). Adding a
  target choice widens the action space, but it is the difference between
  "positioning as targeting" and actual directed trade.
* **Pay only gifts the receiver uses** — reward the giver when the receiver eats
  or delivers within N ticks. That is a much narrower shaping than a flat payment
  and would answer whether the farm is the only thing a payment can buy. It needs
  a deferred-reward mechanism the trainer does not currently have.

### Gotchas (Milestone 5)

**A thief can take the berry you were about to hand over.** Theft resolves in
phase 1c and gifts in 1e, so a robbery lands first and the gift silently does not
happen — the giver's `gave` stays 0 even though the mask had said yes. This is
**75% of every failed give the learned policy makes**, and it is not a mask bug:
you cannot hand over what was just taken from you. Pinned by a test. The other
failure mode is the same-tick race two givers can have for one free slot, which
`gather` has had since M3 (the mask is a start-of-tick promise).

**Give hit rate is therefore not a mask-quality metric.** In M3 a low gather hit
rate meant doomed actions; here a give hit rate of 63–100% is mostly other agents
acting first. Look at the ledger's utilisation figures instead.

**`reciprocity` near 1.0 is not good news by itself.** Perfectly balanced pairs
are what both mutual aid and a reward farm look like. Only utilisation separates
them, which is why the report prints both and the viewer shows them together.

## Gotchas (Milestone 3)

**Don't edit `metrics.py` while a run is in flight.** A run holds its CSV header
from the moment it opens the file, so fields added mid-run are missing from that
run's CSV and read as blanks forever. Cost me a run.

**`contests_lost ≈ 0` is behavioural, not a broken code path.** It is tested
directly. See above for why the same-tick rule cannot fire in a scarce world.

## The five rules that survived five milestones

Written down because each one was learned the expensive way, and because every
result in this file that ignored one of them turned out to be wrong.

1. **A shaped reward reliably produces the behaviour it pays for. That is never
   evidence the behaviour helps.** Theft, paid: 4.8× the theft, worse survival.
   Gifts, paid: 42× the gifts, 54 fewer ticks of life. Construction, paid: more
   building *and* better survival, and it survived annealing. Same technique,
   three different verdicts — decided entirely by whether the shaped behaviour
   was worth doing. Only the terminal metric can tell you which case you are in,
   so every shaped run ships with an unshaped control and is read on lifespan.
2. **Fix the shape of the problem, not the size of the number.** Every real
   improvement here came from changing what the agent could perceive or reach —
   action masking (M3), siting shelters where agents already live and making
   partial walls give partial protection (M4). Every attempt to buy the outcome
   with a bigger coefficient produced activity without result.
   **The refinement, learned from `m4d`:** not every structural change qualifies.
   The three that worked all deleted a step that *could not be credited* — a
   doomed action, an unpaid approach walk, three-quarters of a build invisible to
   the value function. Halving the site cost shortened the chain without making
   any part of it more creditable, and bought nothing. Ask what the agent cannot
   perceive or cannot be paid for, not what is merely far away.
3. **The milestone chain is load-bearing.** The scarce world is unlearnable from
   scratch — a from-scratch run lands on *exactly* the random baseline after the
   full budget. Each milestone works because the previous one transferred
   competence into it via `--init-from` and `grow_policy`.
4. **More compute is never the fix.** `scarce-long` spent 3.3× budget and was
   flat from update 150. Runs settle by ~120–175 updates. If a run is not
   working, the world or the observation is wrong, not the budget.
5. **When you change a mechanic, re-derive the arithmetic that justified the
   open problems around it.** `m4c` made shelter protection linear in build
   progress. That fixed the cliff it was aimed at, and in the same stroke made
   the last unit of a site worth exactly what the first one was — deleting the
   reason to complete a shelter. The note calling incomplete shelters M4's
   failure was written before that change and was carried forward, unexamined,
   through the whole of M5 and into `m4d`, an experiment that could not have
   worked because it was arguing with a premise that had already expired. Cost:
   one run and two milestones of a wrong headline. The M3 economy postmortem
   already says "recompute demand with `sim` rather than by hand" — this is the
   same rule, applied to *incentives* rather than supply. A fix that removes a
   cliff can remove the summit with it.
