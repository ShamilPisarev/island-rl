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

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Python 3.11+. Dependencies are `torch`, `numpy`, `pyyaml`, and `pytest`.

## Train

```bash
.venv/bin/python -m sim.train --run-name my-run
```

Reports the random-action baseline before it starts, then logs a metrics row per
update. Checkpoints land in `checkpoints/`, metrics in `runs/<name>/metrics.csv`,
and replays in `viewer/replays/`. `Ctrl-C` saves a checkpoint on the way out, and
`--resume checkpoints/latest.pt` picks it back up.

Useful flags: `--updates N`, `--num-envs N`, `--seed N`, `--no-baseline`.

## Evaluate

```bash
.venv/bin/python -m sim.evaluate --checkpoint checkpoints/latest.pt --baselines
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

Covers world stepping, hunger and death, resource regrowth, observation shape
and bounds, replay round-trip and schema versioning, seed determinism, GAE
correctness, dead-agent masking, and a PPO smoke test on a task with a known
optimum.

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

**Actions** (10, discrete): eight compass directions, `idle`, `gather`. Eating is
automatic when hunger drops below the threshold and the agent is carrying food.
That is a deliberate choice — see `sim/world.py` for why an explicit eat action
would have paid agents to starve themselves.

**Observations** (26 floats, egocentric, all in [-1, 1]): own hunger and food;
the four nearest bushes as `(dx, dz, berries)`; the three nearest living agents
as `(dx, dz, hunger)`; and distance plus direction to the shoreline. No absolute
coordinates, so nothing can be memorised — only navigated.

**Reward:** `+0.01` per tick alive, `+1.0` per gather, `+2.0` for eating scaled
by how hungry, `-10.0` on death. Nothing else. In particular there is no reward
for anything social — if clustering or competition appears, it has to appear on
its own.

**Algorithm:** PPO with GAE, one shared policy across all six agents (parameter
sharing), a 2×128 tanh MLP with a separate value head, vectorised over
`num_envs` parallel islands. Agents die at different ticks, so dead slots are
masked out of the loss rather than removed from the batch, and the time limit is
treated as truncation — bootstrapped from `V(final_obs)` — rather than as death.

## Layout

```
config/default.yaml   every tunable
sim/world.py          environment, tick order, resources
sim/agents.py         agent state, action space, observation construction
sim/policy.py         the network, plus random and scripted reference policies
sim/ppo.py            the training algorithm
sim/replay.py         replay schema (versioned) and the viewer manifest
sim/metrics.py        aggregation, CSV, console table
sim/train.py          training CLI
sim/evaluate.py       evaluation CLI and the baselines
viewer/index.html     the viewer (static, no build step)
viewer/main.js
tests/
```

`CLAUDE.md` carries the working notes: current state, decisions and why, and the
gotchas worth knowing before changing anything.
