# CLAUDE.md — working notes

Assume you are starting cold with only this file and `PROJECT_BRIEF.md`. This is
the state of the project, the decisions behind it, and the things that will bite
you.

## Current state

**Milestones 1 and 2 are complete and verified.** M3–M6 have not been started —
do not start them without reading §3 of the brief.

- M1: survival + foraging, one shared brain. 1.97× the random baseline.
- M2: six individual brains forked from the M1 checkpoint and trained
  independently, plus a behavioural-divergence view. Specialisation appeared:
  action divergence is 19× the shared-brain control.
- `pytest` passes (112 tests).
- Both viewer pages verified in a browser against real data, including their
  schema-mismatch failure paths.

Reproduce end to end:

```bash
python -m sim.train --run-name m1
python -m sim.train --run-name m2 --policy-mode individual \
    --init-from checkpoints/m1/latest.pt
python -m sim.divergence --checkpoint checkpoints/m2/latest.pt
```

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

## Gotchas (Milestone 3)

**Don't edit `metrics.py` while a run is in flight.** A run holds its CSV header
from the moment it opens the file, so fields added mid-run are missing from that
run's CSV and read as blanks forever. Cost me a run.

**`contests_lost ≈ 0` is behavioural, not a broken code path.** It is tested
directly. See above for why the same-tick rule cannot fire in a scarce world.
