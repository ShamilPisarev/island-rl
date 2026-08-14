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

Useful flags: `--updates N`, `--num-envs N`, `--seed N`, `--no-baseline`.

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
while later milestones change it.

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

`divergence.html` (linked from the side panel) is the second view: it renders a
divergence report as territory heatmaps, action mixes, and divergence matrices.

Both pages refuse to render a file whose schema version they do not recognise,
rather than drawing something plausible and wrong.

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

112 tests covering world stepping, hunger and death, resource regrowth,
observation shape and bounds, replay round-trip and schema versioning, seed
determinism, GAE correctness, dead-agent masking, per-agent brain dispatch and
gradient isolation, divergence maths, and a PPO smoke test on a task with a
known optimum.

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

**Actions** (10, discrete): eight compass directions, `idle`, `gather`, plus
`steal` as an 11th in Milestone 3. Eating is automatic when hunger drops below the
threshold and the agent is carrying food. That is a deliberate choice — see
`sim/world.py` for why an explicit eat action would have paid agents to starve
themselves.

**Observations** (26 floats, egocentric, all in [-1, 1]): own hunger and food;
the four nearest bushes as `(dx, dz, berries)`; the three nearest living agents
as `(dx, dz, hunger)`; and distance plus direction to the shoreline. No absolute
coordinates, so nothing can be memorised — only navigated.

**Competition** (Milestone 3, off by default): food supply cut to match demand,
only the agent closest to a bush may harvest it, and agents can rob a neighbour
who is carrying berries. Stealing pays **no** reward — it has to be worth taking
for the food alone.

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
sim/world.py          environment, tick order, resources
sim/agents.py         agent state, action space, observation construction
sim/policy.py         the networks (shared + per-agent), plus reference policies
sim/ppo.py            the training algorithm
sim/divergence.py     per-agent behavioural divergence analysis (M2)
sim/replay.py         replay schema (versioned) and the viewer manifest
sim/metrics.py        aggregation, CSV, console table
sim/train.py          training CLI
sim/evaluate.py       evaluation CLI and the baselines
viewer/index.html     the replay viewer (static, no build step)
viewer/main.js
viewer/divergence.html  the behavioural-divergence view (M2)
viewer/divergence.js
tests/
```

`CLAUDE.md` carries the working notes: current state, decisions and why, and the
gotchas worth knowing before changing anything.
