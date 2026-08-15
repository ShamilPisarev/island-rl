# Island

A 3D observer-mode multi-agent RL simulation. Six agents forage for berries on a
circular island and try not to starve. Their brains are small reinforcement
learning policies — not language models — trained with a hand-rolled PPO. You do
not play it; you watch it.

The project is deliberately split in two halves that never run at the same time:

```
sim/  (Python)                         viewer/  (Three.js)
environment + PPO training   ──JSON──▶ loads a replay, renders it in 3D
writes replay files                    playback + camera controls
```

Training needs millions of steps; rendering needs 30 fps. Trying to do both at
once gets you neither, so training runs headless and drops replay files that the
viewer plays back afterwards.

## Quickstart

```bash
make watch
```

Then open <http://localhost:8000>, pick a replay from the dropdown, and press play.
Trained replays ship in `viewer/replays/`, so there is nothing to train first.

`make help` lists the rest. If you have no `.venv` yet, run `make venv` once.

> **Everything must run through `.venv/bin/python`, not plain `python`.**
> A bare `python -m sim.train` will fail with `ModuleNotFoundError: No module
> named 'torch'`, because `python` on this machine is the conda base environment,
> which does not have the project's dependencies. Either use the `make` targets,
> or activate the environment first:
>
> ```bash
> source .venv/bin/activate
> ```
>
> After activating, plain `python -m sim.train` works. Run commands from the repo
> root — `sim` is imported as a package relative to the working directory.

## Install

The repo already has a `.venv`. To build one from scratch:

```bash
make venv
```

or by hand:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Python 3.11+. Dependencies are `torch`, `numpy`, `pyyaml`, and `pytest`.

## Train

```bash
.venv/bin/python -m sim.train --run-name m1
```

Reports the random-action baseline before it starts, then logs a metrics row per
update. Checkpoints land in `checkpoints/<run-name>/`, metrics in
`runs/<name>/metrics.csv`, and replays in `viewer/replays/`. `Ctrl-C` saves a
checkpoint on the way out, and `--resume checkpoints/m1/latest.pt` picks it back up.

Useful flags: `--updates N`, `--num-envs N`, `--seed N`, `--no-baseline`,
`--threads N`.

**On a laptop:** `ppo.threads` defaults to 4, which measures as fast as 8 — these
nets are small enough that the environment step dominates, so extra cores are
pure heat. Results are bit-identical at any thread count. And 200 updates is
enough for every result in this project (the curves settle by update 79–175);
the 300–400 figures in the reproduce commands are historical. Run experiments
back to back rather than in parallel — concurrent runs don't finish sooner
overall, they just make more heat at once.

To give every agent its own brain (Milestone 2), fork a trained shared policy:

```bash
.venv/bin/python -m sim.train --run-name m2 --policy-mode individual \
    --init-from checkpoints/m1/latest.pt
```

Forking rather than starting fresh means each agent begins competent and diverges
from there, instead of six agents independently rediscovering how to walk to a bush.

Milestone 3 turns the food scarce and lets agents block and rob each other:

```bash
.venv/bin/python -m sim.train --config config/m3.yaml --run-name m3 \
    --policy-mode individual --init-from checkpoints/m2/latest.pt
```

Configs are layered — `m3.yaml` extends `scarce.yaml` extends `default.yaml`, each
overriding only what it restates — so the M1/M2 world stays exactly as it was
while later milestones change it. The canonical M3 config is `m3_masked.yaml`
(see the results section for why masking matters).

Milestone 4 adds wood, stone, communal shelter-building, and a night hazard,
trained as a shaped run plus an unshaped control:

```bash
.venv/bin/python -m sim.train --config config/m4.yaml --run-name m4 --updates 400 \
    --policy-mode individual --init-from checkpoints/m3-masked/latest.pt
.venv/bin/python -m sim.train --config config/m4_unshaped.yaml --run-name m4-unshaped \
    --updates 400 --policy-mode individual --init-from checkpoints/m3-masked/latest.pt
```

Milestone 5 lets agents hand food and materials to each other, again as a pair —
the brief-faithful run where a gift pays nothing, and the ablation where it pays
what a gather pays:

```bash
.venv/bin/python -m sim.train --config config/m5.yaml --run-name m5 --updates 200 \
    --policy-mode individual --init-from checkpoints/m4c-anneal/latest.pt
.venv/bin/python -m sim.train --config config/m5_shaped.yaml --run-name m5-shaped \
    --updates 200 --policy-mode individual --init-from checkpoints/m4c-anneal/latest.pt
```

## Measure divergence

```bash
.venv/bin/python -m sim.divergence --checkpoint checkpoints/m2/latest.pt
```

Prints a per-agent behaviour table and writes `viewer/reports/divergence.json`,
which the divergence view renders as territory heatmaps, action mixes, and
pairwise Jensen–Shannon divergence matrices.

Run it against the shared M1 checkpoint too — that is the control. Six agents
sharing one brain still occupy different ground, so territory divergence alone
proves nothing; it is the *action* divergence that has to clear the control.

## Analyse the transfer ledger

```bash
.venv/bin/python -m sim.exchange --checkpoint checkpoints/m5/latest.pt
```

Writes `viewer/reports/transfers.jsonl` — one JSON object per transfer,
`{episode, tick, giver, receiver, item}` — plus `viewer/reports/exchange.json`
for the exchange view. Prints who gave to whom, how much of what was given got
*used* (eaten, or delivered to a building site) within a window, and a
reciprocity score.

Read utilisation next to reciprocity, never on its own: two agents passing one
berry back and forth score a perfect 1.0 for reciprocity, and it is a reward
farm, not a trade.

## Evaluate

```bash
.venv/bin/python -m sim.evaluate --checkpoint checkpoints/m1/latest.pt --baselines
.venv/bin/python -m sim.evaluate --policy random --episodes 20
```

Every policy is evaluated on the same block of seeds, so the islands are
identical across comparisons. Add `--replay viewer/replays/out.json` to record
one episode while you are at it.

## Watch

The viewer is a static page — no build step, no bundler — but it does need to be
served, because `fetch()` on `file://` is blocked by browsers:

```bash
python -m http.server 8000 --directory viewer
```

Then open <http://localhost:8000>. It also needs network access on first load:
Three.js comes from a CDN via an import map, per the project brief.

Pick a replay from the dropdown (populated from `viewer/replays/index.json`,
which the sim maintains) or drag any replay `.json` onto the page — drag-and-drop
works from `file://` too, if you would rather not run a server.

Controls: orbit/pan/zoom with the mouse, space to play/pause, arrow keys to step
a tick, speed buttons for 0.5×–16×, and the scrubber to seek. Click an agent —
in the scene or in the side panel — to have the camera follow it.

Two more views are linked from the side panel. `divergence.html` renders a
divergence report as territory heatmaps, action mixes, and divergence matrices.
`exchange.html` renders a transfer ledger as a flow matrix, a per-agent
gave/received table, and each agent's share of what the population harvested,
delivered and handed over.

In a Milestone 5 replay, every transfer is drawn in the scene as the item arcing
from giver to receiver, colour-coded by what changed hands.

All three pages refuse to render a file whose schema version they do not
recognise, rather than drawing something plausible and wrong.

To get something to look at before training anything:

```bash
.venv/bin/python -m sim.make_fake_replay
```

That runs the real world under a hand-written greedy forager (plus a couple of
random wanderers, so there are deaths to look at), which is also how the viewer
was verified before any policy existed.

## Test

```bash
.venv/bin/python -m pytest
```

230 tests covering world stepping, hunger and death, resource regrowth,
observation shape and bounds, replay round-trip and schema versioning (v1 to v3),
seed determinism, GAE correctness, dead-agent masking, action masking, per-agent
brain dispatch and gradient isolation, cross-milestone policy growth by feature
name, bush contention and theft, construction and the night hazard, transfers and
the ledger they leave behind, divergence maths, and a PPO smoke test on a task
with a known optimum.

## Results

### Milestone 1 — survival and foraging

300 updates, 7.37M agent-steps, 4.2 minutes on a laptop CPU. Twenty evaluation
episodes on identical seeds:

| policy | mean lifespan | of 600 ticks | deaths/episode | berries |
|---|---|---|---|---|
| random actions | 302.3 ± 61.6 | 50% | 5.10 | 11.5 |
| **learned** | **595.6 ± 19.0** | **99%** | **0.10** | **64.0** |
| scripted greedy forager | 600.0 ± 0.0 | 100% | 0.00 | 66.0 |

**1.97× the random baseline**, effectively at the ceiling set by a hand-written
forager. And they are genuinely navigating rather than surviving by luck: the
learned agents sit 2.25 units from the nearest bush on average against 7.05 for
random play, and spend 53% of their ticks within gathering range where chance
would give 11%.

### Milestone 2 — individual brains

Six brains forked from the M1 checkpoint and trained independently for 300
updates. Both columns are 20 episodes on the same fixed island:

| | M1 shared (control) | M2 individual |
|---|---|---|
| mean lifespan | 594.5 | 592.9 |
| **action divergence** | **0.0012 bits** | **0.0233 bits (19×)** |
| territory divergence | 0.6722 bits | 0.9080 bits |
| gather-share spread | 2.2 pts | 7.4 pts |
| mean-radius spread | 5.5 | 21.1 |

They specialised. Agent 5 became a bush-camper (37.8% of its actions gathering,
77.3% of its ticks within range); agent 2 a rover (68.2% travelling, the best
gather hit rate); agent 4 stayed near the island centre while agent 3 ranged to
the shore.

Two caveats worth stating plainly. **Survival is saturated** — both milestones sit
at the 600-tick ceiling, so M2 is differentiated rather than better, and any
future milestone wanting a survival signal needs a harder world first. And
**territory divergence is not by itself evidence**: six agents sharing a single
brain already score 0.67 bits on it, because they spawn apart and each heads for
the nearest cluster. Action divergence is the number that clears its control.

### Milestone 3 — competition

Food supply cut to match demand, only the closest agent may harvest a bush, and
agents can rob each other. Twenty episodes on `config/m3.yaml`:

| policy | mean lifespan | vs random | deaths/ep | steals/ep |
|---|---|---|---|---|
| random actions | 223.0 | 1.00× | 5.85 | 0 |
| **learned (forked from M2)** | **452.1** | **2.03×** | 3.40 | 28.9 |
| learned (from scratch) | 223.3 | 1.00× | 5.95 | 0.1 |
| scripted forager | 495.5 | 2.22× | 2.20 | 0 |
| scripted thief | 553.3 | 2.48× | 1.45 | 1099.3 |

Three findings, two of them negative and worth stating plainly.

**Theft did not emerge.** Stealing pays no reward by design, and PPO never found
it — 29 steals an episode against the scripted thief's ~1100, finishing below both
scripted references. Theft is clearly *worth* having (the thief buys 58 ticks of
life and a death per episode); the problem is credit assignment. Steal → carry →
auto-eat later → don't starve cannot compete with the +1.0 a gather pays now.

So `config/m3_shaped.yaml` pays a steal the same +1.0 as a gather, as an explicitly
labelled bootstrap, to test that diagnosis:

| | theft unpaid | theft paid |
|---|---|---|
| steals / episode | 28.9 | **140.0** |
| berries gathered | 26.4 | 23.4 |
| **mean lifespan** | **452.1** | **428.6** |

**The shaping worked and the outcome got slightly worse.** Theft emerged, 4.8× more
of it — but survival fell. Stealing moves food between agents and never creates
any, so paying for it buys ticks spent redistributing berries instead of harvesting
new ones, and the population ends up with less food. The policy learned to steal
because stealing pays, not because stealing helps. Worth remembering before
Milestone 4 leans on shaped rewards for construction.

**What actually fixed M3: action masking.** The policy was spending 30% of its
living ticks firing `gather` and `steal` from positions where they could not
possibly succeed — a local optimum that survived eight interventions including
3.3× the training budget. `mask_invalid_actions` hides an action when it cannot
do anything (the observation says what is there; the mask says what is
reachable), and with the trap gone:

| | unmasked | masked |
|---|---|---|
| mean lifespan | 452.1 | **471.6 (2.11×)** |
| steals/ep — still unpaid | 28.9 | **115.5** |
| doomed actions | 30% of ticks | **0%** |
| gather hit rate | 0.8–3.4% | **50–98%** |

Theft finally emerged *without being paid for*: once `steal` only ever appears
with a real victim in reach, its true return becomes visible to PPO. The learned
policy still trails the scripted forager (495.5) — the residual gap lives in
crowding and exclusion dynamics and is documented as an open problem rather than
tuned at.

**The scarce world cannot be learned from scratch.** A from-scratch run lands on
exactly the random baseline after the full 7.4M steps. M3 only works because M1
and M2 transferred competence into it — the milestone chain is load-bearing.

**Competition produced convergence, not partitioning.** Territory divergence
*fell* (0.908 → 0.363): with six bushes instead of twenty, every agent wants the
same ground. Territoriality appeared as inequality instead — lifespan spread went
from 0 to 144 ticks, a dominant pair holding bushes 1.5 units away and living ~390
ticks while the excluded sit 3–5 units out and starve around 250.

### Milestone 4 — multi-resource and construction

Wood, stone, communal shelter sites, and a day/night cycle where anyone caught
outside drains hunger at 3×. The food economy is sized so shelter is load-bearing:
supply (64 berries) sits between what a sheltered population needs (48) and what
one sleeping rough needs (72).

This is the milestone the brief says needs reward shaping, so it ships as a
shaped run *and* an unshaped control, compared on the terminal metric:

| policy | lifespan | deaths/ep | shelters/ep | nights indoors |
|---|---|---|---|---|
| scripted builder | **529.8** | 2.05 | **1.9** | **92%** |
| scripted forager | 457.6 | 3.12 | 0 | 0% |
| **learned, shaped** | **393.5** | 4.70 | 0.1 | **22%** |
| learned, unshaped control | 362.5 | 5.05 | 0.0 | 2% |

**Construction partially emerged.** The shaped run beat its control by 31 ticks of
life with 22% of night ticks under shelter against 2% — the opposite of the M3
ablation, where shaping bought the behaviour and cost survival. Here it bought a
real outcome.

**But agents do not finish.** They deliver materials and huddle under half-built
walls; completed shelters run at 0.1 per episode against the scripted builder's
1.9, and the learned policy stays well below that reference. That is the honest
headline, and it is the failure the brief predicted for this milestone.

Getting even that far took two fixes, neither of which touched the shaping
coefficients — both were the move that solved M3, reshaping the problem rather
than paying more at the summit. Siting shelters on the berry clusters deleted an
approach walk that earned nothing (deliveries ×3.6), and making protection scale
with build progress removed a cliff where three of every four delivered units
were invisible to the value function (nights sheltered ×6).

**The shaping annealed away cleanly.** The brief asks whether an M4 bootstrap can
be removed once it has done its job. Continuing the shaped policy for 200 more
updates with every shaping term at zero:

| | deliveries/ep | nights indoors | lifespan |
|---|---|---|---|
| shaped | 1.94 | 16.1% | 377.6 |
| **annealed (nothing paid)** | **2.21** | **18.0%** | **382.3** |
| unshaped control | 0.73 | 4.7% | 368.8 |

The behaviour held rather than decaying toward the control — so the shaping was
genuine scaffolding, not a subsidy the behaviour depended on. Read that against
M3, where the same technique produced 4.8× the theft and *worse* survival. Same
method, opposite verdict: what differs is whether the shaped behaviour was worth
doing, and only the terminal metric tells you.

**And the obvious next lever did not work.** If agents stop one unit short, make
the summit nearer: `m4d` halves a site to 2 units — one round trip instead of
two. It bought the learned policy nothing (373.6 lifespan against m4c's 374.6 at
matched budget), while the *scripted* builder in the same cheap world gained ~15
ticks and 0.6 shelters. So the last unit is not out of reach because it is far.

This run is also a small lesson in reading your own metrics. Protection scales
with the *fraction* of a site delivered, and "indoors" means half-built, so
halving the site cost halves the deliveries needed to count as sheltered. The
construction statistics duly improved — and meant nothing. The bar was halved and
the policy cleared it on exactly the same 22% of nights as before. The prediction
and the confound were both written into the config header before the run, and the
threshold arithmetic is pinned by a test.

### Milestone 5 — exchange

Agents can hand over one unit of food or material to the nearest neighbour in
reach. Every transfer is logged so the economics can be read back afterwards.
A gift pays nothing, exactly as a theft pays nothing: it creates no food and no
wood, so it has to earn its place through what the receiver does with it.

| policy | lifespan | deaths/ep | berries | gifts/ep |
|---|---|---|---|---|
| scripted builder | 529.8 | 2.05 | 38.5 | 0 |
| **scripted trader** | **543.8** | **1.85** | 38.0 | 8.3 |
| M4 policy grown into M5, untrained | 406.4 | 4.60 | 34.9 | 11.3 |
| **learned, gifts unpaid** | **413.3** | 4.60 | 34.6 | 7.9 |
| learned, gifts paid (ablation) | 359.1 | 5.30 | 26.9 | **330.9** |

**Exchange did not emerge.** The row that makes this readable is the untrained
control: a policy grown into the wider action space already gives 11.3 times an
episode just by sampling an action it has no opinion about. Two hundred updates
later that had fallen to 7.9 — giving was mildly selected *against*, which is the
correct response to an action that costs you a tick and pays somebody else.

**And it is not because gifts are worthless.** The scripted trader — the builder
plus two opportunistic rules, hand food to someone about to eat it, hand material
to someone standing on the site — beats the plain builder by **+14.0 ± 5.2 ticks
(paired, better on 12 of 20 islands)** on about eight gifts an episode. A few
well-aimed gifts really do buy survival; PPO just cannot find them through a
credit chain that runs give → someone else eats → they don't die.

**Paying for gifts produced a farm.** The ablation pays a transfer what a gather
pays. The prediction was written into the config header before the run, and all
of it landed: 42× the gifts, fewer berries, and 54 fewer ticks of life. The
ledger shows what those 355 transfers an episode actually were:

| | unpaid | paid |
|---|---|---|
| transfers / episode | 11.1 | 355.2 |
| ticks spent giving | 0.4–1.5% | 16–20% |
| reciprocity | 0.851 | 0.942 |
| gifted food eaten within 50 ticks | 56% | **5%** |
| gifted material delivered within 50 ticks | 3% | **0%** |

Every agent gives and receives roughly equally and ends net flat, and almost
nothing handed over is ever used — berries circulating between neighbours for the
reward, while the population harvests less and starves sooner. That is the M3
theft ablation again with a bigger multiplier, and the contrast with M4 is the
whole lesson: paying for construction bought an outcome because a shelter really
does reduce the drain; paying for gifts bought motion because a transfer creates
nothing.

## Configuration

Everything tunable lives in `config/default.yaml` — world size, hunger rates,
bush clustering, reward weights, network shape, PPO hyperparameters. If you find
a magic number in `sim/*.py`, that is a bug. Unknown keys are rejected at load
time rather than silently ignored.

## How it works

**World.** A disc of radius 40 with continuous `(x, z)` positions. Berry bushes
are scattered in gaussian clusters, not uniformly — clusters create hotspots
several agents want at once. Bushes deplete when gathered and regrow one berry on
a cooldown. Agents have `hunger` (0–100), `food_carried`, and an `alive` flag.

Note that `hunger` counts *down*: it starts at 100, drains every tick, and `<= 0`
is death. It is a satiety meter despite the name (which the brief fixed).

**Actions** (10, discrete): eight compass directions, `idle`, `gather`. Later
milestones append and never insert, so an early checkpoint keeps its meaning:
`steal` is the 11th (M3), `chop`/`mine`/`build` the 12th–14th (M4), and
`give_food`/`give_material` the 15th–16th (M5). Eating is automatic when hunger
drops below the threshold and the agent is carrying food. That is a deliberate
choice — see `sim/world.py` for why an explicit eat action would have paid agents
to starve themselves.

**Observations** (26 floats, egocentric, all in [-1, 1]): own hunger and food;
the four nearest bushes as `(dx, dz, berries)`; the three nearest living agents
as `(dx, dz, hunger)`; and distance plus direction to the shoreline. No absolute
coordinates, so nothing can be memorised — only navigated.

**Competition** (Milestone 3, off by default): food supply cut to match demand,
only the agent closest to a bush may harvest it, and agents can rob a neighbour
who is carrying berries. Stealing pays **no** reward — it has to be worth taking
for the food alone.

**Exchange** (Milestone 5, off by default): an agent can hand one unit to the
nearest neighbour in reach with room for it. Like theft it pays nothing, and like
theft it moves things rather than making them. The giver picks *when* and *what*,
never *who* — the recipient is whoever is standing closest, so positioning is the
targeting mechanism.

**Reward:** `+0.01` per tick alive, `+1.0` per gather, `+2.0` for eating scaled
by how hungry, `-10.0` on death. Nothing else. In particular there is no reward
for anything social — if clustering or competition appears, it has to appear on
its own.

**Algorithm:** PPO with GAE, a 2×128 tanh MLP with a separate value head,
vectorised over `num_envs` parallel islands. Milestone 1 shares one policy across
all six agents (parameter sharing); Milestone 2 gives each agent its own,
dispatched by agent id through the same training loop, with gradients clipped per
brain so the six stay independent. Agents die at different ticks, so dead slots are
masked out of the loss rather than removed from the batch, and the time limit is
treated as truncation — bootstrapped from `V(final_obs)` — rather than as death.

## Layout

```
config/default.yaml   every tunable (the M1/M2 world)
config/scarce.yaml    supply cut to meet demand
config/m3.yaml        + contested bushes and stealing
config/m4*.yaml       + wood, stone, shelter, night (and the shaping ablations)
config/m5.yaml        + giving, with the paid-gift ablation beside it
sim/world.py          environment, tick order, resources
sim/agents.py         agent state, action space, observation construction
sim/policy.py         the networks (shared + per-agent), plus reference policies
sim/ppo.py            the training algorithm
sim/divergence.py     per-agent behavioural divergence analysis (M2)
sim/exchange.py       the transfer ledger and its economics (M5)
sim/replay.py         replay schema (versioned) and the viewer manifest
sim/metrics.py        aggregation, CSV, console table
sim/train.py          training CLI
sim/evaluate.py       evaluation CLI and the baselines
viewer/index.html     the replay viewer (static, no build step)
viewer/main.js
viewer/divergence.html  the behavioural-divergence view (M2)
viewer/divergence.js
viewer/exchange.html    the transfer-ledger view (M5)
viewer/exchange.js
tests/
```

`CLAUDE.md` carries the working notes: current state, decisions and why, and the
gotchas worth knowing before changing anything.
