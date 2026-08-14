# PROJECT BRIEF — "Island" : A 3D Observer-Mode Multi-Agent RL Simulation

## 0. Context for you (the coding agent)

This project came out of a design conversation. Key decisions already made — do not re-litigate them, build to them:

- **This is not a game you play. It is a world you watch.** Think digital ant farm / aquarium. No player input beyond camera control and playback controls.
- **The agents' brains are reinforcement learning policies (small neural networks), NOT large language models.** The whole point is *emergent* behaviour discovered through trial and error, in the spirit of OpenAI's multi-agent hide-and-seek work. No LLM calls anywhere in the simulation loop.
- **Conceptually one brain per agent.** Milestone 1 uses a single shared policy as a training shortcut (standard practice, much faster to get off the ground); Milestone 2 splits them into individual policies so distinct strategies can diverge.
- **The viewer is genuinely 3D** — Three.js, low-poly, orbiting camera above an island. Simple shapes are fine (capsules, cubes, cones). It must not be top-down 2D.
- **Ambition is layered, not attempted all at once.** Long-horizon behaviours like shelter-building and trade are known-hard for RL (sparse reward, long action chains). Milestone 1 targets survival and foraging only, because that emerges reliably. Later milestones add reward shaping to bootstrap the harder stuff.
- The developer is a data scientist / AI engineer, comfortable with Python, PyTorch, and Three.js. Write for a competent peer: no hand-holding comments, but do explain non-obvious design choices in docstrings.

---

## 1. Architecture

Two decoupled halves. **Do not try to render live at training speed** — training needs millions of steps, rendering needs 30 fps. Separate them:

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│  sim/  (Python)             │        │  viewer/  (Three.js)         │
│                             │        │                              │
│  environment + RL training  │ ──────▶│  loads replay, renders 3D    │
│  writes replay files        │  JSON  │  playback + camera controls  │
└─────────────────────────────┘        └──────────────────────────────┘
```

- Training runs headless, fast, and writes **replay files**: a compact JSON/JSONL record of one episode (every agent's position, state, and action per tick, plus resource state).
- The viewer is a **single static HTML page** that loads a replay file and plays it back. No build step, no bundler, no framework. Three.js from CDN via import map. `OrbitControls` from the Three.js examples module.
- Optional, only if Milestone 1 is fully done: a `--live` mode using a small WebSocket bridge so a *trained* (not training) policy can be watched acting in real time.

### Deliverable layout

```
island/
  README.md               # how to install, train, and view
  CLAUDE.md               # working notes for future agent sessions
  pyproject.toml
  config/
    default.yaml          # ALL tunables live here, nothing hardcoded
  sim/
    world.py              # environment: state, step(), resources, spawning
    agents.py             # agent state, observation construction, action decoding
    policy.py             # the neural net (small MLP)
    ppo.py                # training loop
    replay.py             # replay recording + schema versioning
    metrics.py            # logging: mean lifespan, food gathered, deaths, etc.
    train.py              # CLI entrypoint
    evaluate.py           # run a checkpoint, record a replay, print stats
  viewer/
    index.html
    main.js
    replays/              # training drops replays here
  tests/
    test_world.py
    test_observations.py
    test_replay.py
    test_ppo.py
```

---

## 2. Milestone 1 — Survival and Foraging (build this first, completely)

### World

- **Continuous 2D positions** (`x`, `z` floats) on a circular island, radius ~40 units. Rendered in 3D on a flat-ish disc; agents walk on the surface.
- **Discrete action space** (9 actions): move in 8 compass directions by a fixed step, or `idle`. Plus one action `gather` → 10 total. Discrete keeps PPO stable and training fast; continuous positions keep motion smooth to watch.
- **Resources: berry bushes.** Fixed positions, scattered with clustering (not uniform — clustering creates competition hotspots). Each bush holds N berries, depletes when gathered, and **regrows on a cooldown**. Finite + respawning = genuine scarcity.
- **Tick-based.** Episode ends when all agents are dead or after `max_ticks`.
- **6 agents** by default.

### Agent state

- `position (x, z)`, `hunger` (0–100), `alive` flag, `food_carried` (int, small cap).
- Hunger decreases by a fixed amount per tick. Gathering adds food to inventory. Eating (auto-eat when hunger below a threshold and food carried > 0, or make it an explicit action — your call, document it) restores hunger.
- `hunger <= 0` → death, agent is removed from the world but stays in the replay as a marker.

### Observations (fixed-size vector, egocentric — no CNN, keep it fast)

Per agent, concatenate:
- own `hunger` (normalised), `food_carried` (normalised)
- relative position + berry count of the **K nearest bushes** (K=4), as `(dx, dz, berries)`, distance-normalised, zero-padded if fewer
- relative position + hunger of the **K nearest other agents** (K=3)
- distance and direction to island edge (so they learn not to walk into the sea)

Normalise everything to roughly [-1, 1]. Write a test that asserts observation shape and bounds.

### Reward

Deliberately light shaping — enough to bootstrap, not enough to dictate strategy:

- `+0.01` per tick alive
- `+1.0` for a successful gather
- `+2.0` for eating while hunger is below the threshold (scaled by how hungry)
- `-10.0` on death
- No reward for anything else. Do **not** add rewards for social behaviour — if clustering or competition emerges, it must emerge.

### Algorithm

- **PPO, hand-rolled in PyTorch** (~200–300 lines). Do not pull in Stable-Baselines3 — multi-agent doesn't fit its API cleanly and we need control over policy splitting in Milestone 2.
- **Shared policy across all agents** in Milestone 1 (parameter sharing). Each agent still acts independently on its own observation; they just share weights while learning.
- Small MLP: two hidden layers, 128 units, tanh. Separate value head.
- Vectorised across parallel worlds (`num_envs` in config) for throughput.
- Checkpoints to `checkpoints/`, resumable.

### Metrics (print and log to CSV every N updates)

Mean episode lifespan · deaths per episode · total berries gathered · mean final hunger · policy entropy · value loss · explained variance.

**Definition of done for Milestone 1:** mean agent lifespan rises clearly above a random-action baseline (compute and report that baseline explicitly), agents visibly move toward bushes rather than wandering, and a replay renders in the viewer end to end.

### Viewer (Milestone 1 scope)

- Low-poly island disc, water plane, soft directional light, subtle shadows.
- Agents = capsules or cubes, one colour per agent, with a small floating hunger bar above each. Dead agents fade out / drop flat.
- Bushes = simple spheres/cones, colour or scale reflecting remaining berries so depletion and regrowth are visible.
- **Playback controls:** play/pause, speed (0.5× / 1× / 4× / 16×), tick scrubber, tick counter.
- **Side panel:** per-agent list showing hunger, food carried, alive/dead, and last action. Click an agent to have the camera follow it.
- `OrbitControls` — free orbit, pan, zoom.
- Replay file chosen via a dropdown listing files in `viewer/replays/`, or drag-and-drop.

---

## 3. Later milestones (do NOT start these until Milestone 1 is done and verified)

**M2 — Individual brains.** Split the shared policy into per-agent policies (initialise from the shared checkpoint, then train independently). Add a metrics view showing behavioural divergence between agents — e.g. share of time spent gathering vs travelling, territory heatmaps. This is where specialisation should start to appear.

**M3 — Competition.** Agents can steal from or block each other; bushes become contested. Watch for territorial behaviour. No new reward terms beyond survival.

**M4 — Multi-resource + construction.** Add wood/stone. Introduce shelter, which reduces hunger drain or protects from a periodic hazard (night, storm). **This is the milestone that needs reward shaping** — a small intermediate reward for gathering materials and for partial construction progress, because the terminal reward is far too sparse otherwise. Document the shaping honestly as a bootstrap, and test whether it can be annealed away later.

**M5 — Exchange.** Agents can transfer items. See whether anything resembling trade or specialisation-plus-exchange appears. Log all transfers so the economics can be analysed after the fact.

**M6 (separate idea, parked).** An LLM-driven narration or dialogue layer *on top* of the simulation — commentary on what the agents are doing, or characters that talk. Explicitly out of scope for the simulation loop; noted here so it isn't forgotten.

---

## 4. Engineering requirements

- Python 3.11+. Dependencies: `torch`, `numpy`, `pyyaml`, `pytest`. Nothing else without a reason.
- **Everything tunable lives in `config/default.yaml`.** No magic numbers in the simulation code.
- **Full seeding and determinism.** Same seed + same config = identical replay. Write a test for it.
- **Replay schema is versioned** and documented in `replay.py`. The viewer must fail loudly on a version mismatch, not silently render garbage.
- `pytest` must pass before any milestone is called done. Tests cover: world stepping, hunger/death logic, resource regrowth, observation shape and bounds, replay round-trip, and a short PPO smoke test that asserts loss decreases on a trivial task.
- Type hints throughout. Docstrings on anything where the *why* isn't obvious.
- Keep the training loop readable over clever. This code is going to be read and modified a lot.

---

## 5. How to work

1. **Plan first.** Read this whole brief, then produce a written plan with a task breakdown before writing code. Flag anything you think is wrong or under-specified — push back rather than guessing.
2. Build Milestone 1 in this order: world → tests → observations → replay format → viewer against a **hand-written fake replay** (so the visuals are verified before training exists) → policy → PPO → train → evaluate.
3. **Verify as you go.** Run the tests. Run a short training job. Actually check the metrics move. Don't declare something working because it imports.
4. **Keep going without asking** for routine decisions — file names, helper structure, minor parameter choices. Only stop for genuine forks in the road (a design decision that changes the architecture, or something in this brief that turns out to be a bad idea).
5. Maintain `CLAUDE.md` as you go: current state, what's done, what's next, decisions made and why, and any gotchas. Assume a future session starts cold with only that file.
6. Commit at each meaningful step with clear messages.

Start with the plan.
