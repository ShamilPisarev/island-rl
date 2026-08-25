# Island 2.0 — design document, and the log of building it

Written 2026-08-25, after the session that found the viewer was auto-loading a
stripped-down probe replay (fixed: `?replay=` deep link in `viewer/main.js`).
Island 1.0 (milestones 1-5, all verified) stays intact whatever happens; 2.0 is
a new code path, new configs, and must not touch the existing results.

**STATUS: stages 1-4 are built (all 2026-08-25). Stage 5, the learned arbiter,
is next.** The sections below are the original design as written; where a stage
has since been run, an inline note says what actually happened. **Section 8
carries the stage 2/3 results and section 9 the stage 4 results**, each with the
corrections its build forced. Three of the design's own forecasts have now been
refuted by measurement -- the spatial hash (section 4), the schema bump for
animation state (section 5), and the guess that stockpiles would produce runaway
hoarding (section 3) -- so read the notes before trusting a prediction here.

Run it:

```bash
# stage 4, the current world
python -m sim.economy  --config config/island2/society4.yaml     # size the world
python -m sim.society  --config config/island2/society4.yaml --episodes 5 --random
python -m sim.society  --config config/island2/society4.yaml --replay
python -m sim.exchange --config config/island2/society4.yaml --policy utility \
    --episodes 2 --out viewer/reports/exchange.json     # household ledger view

# stage 2's world, kept as the control the stage-4 numbers are read against
python -m sim.society  --config config/island2/society100.yaml --episodes 5 --random
python -m sim.profile_engine --config config/island2/engine100_full.yaml --profile
```

## 0. The goal has changed, and it is worth saying plainly

Island 1.0 asked a science question: does social behaviour emerge UNPAID from
survival pressure, with six agents? Answered: foraging, specialisation, theft
and construction did; exchange did not. One deep open problem remains (PPO will
not take a ~20-step trip worth +173 because every one-step prefix is correctly
priced <= 0).

Island 2.0 asks a different question: can we get a WATCHABLE SOCIETY at 50-100
agents -- groups, conflict over resources, trade, maybe trade wars -- on this
laptop, without giving up learning entirely?

Those are different projects. The honest trade at the centre of 2.0: every
piece of scripted competence we hand the agents makes the sim more watchable
and makes "it emerged" mean less. The design below tries to draw that line
deliberately instead of by accident: SCRIPT THE MUSCLES, LEARN THE CHOICES.

## 1. What games actually do for NPCs (the landscape)

None of the big titles use RL for shipped NPCs. Designers need authorability
and predictability; a trained policy gives neither. What they use instead, in
rough order of appearance:

### Finite State Machines (FSM)
The classic: states (patrol, chase, flee) with hand-written transitions.
Simple, fast, and it is what most GTA-era ambient pedestrians effectively run:
walk-along-navmesh, react-to-event, flee. Scales to hundreds of agents because
each tick is a switch statement. Breaks down when behaviours multiply --
transitions grow quadratically.

### Behavior Trees (BT)
Popularised by Halo 2, now the default in Unreal and most AAA AI. A tree of
composites (sequence, selector, parallel) over condition and action leaves,
re-evaluated top-down every tick. The win over FSMs is modularity: "flee"
is a subtree you graft anywhere, and priority is just ordering under a
selector. A BT is still 100% authored -- it does exactly what you wrote,
which is the point.

### GTA specifically: hierarchical tasks + scenario points
Rockstar's system (worth knowing because it is what "GTA crowds" actually
are): every ped runs a stack of hierarchical TASKS (a task decomposes into
subtasks down to motor primitives), plus the world is sprinkled with
SCENARIO POINTS -- markers saying "an NPC here can lean on this railing /
sweep this floor / sit and smoke". Ambient life is peds attaching to nearby
scenarios. The crowd looks alive because the WORLD carries the behaviour,
not because the agents are smart. That trick transfers directly to us:
bushes, sites and stockpiles can advertise what can be done at them.

### Utility AI / needs systems -- THE SIMS, and what "Maslow arbiter" means
The Sims is the real ancestor of what I called a "Maslow arbiter". Each Sim
has needs/motives (hunger, energy, social, fun...) that decay over time.
Every object in the world ADVERTISES what it restores ("fridge: +hunger").
Each tick-ish, the Sim scores every advertised action:

    score(action) = sum over needs of (how low the need is) x (how much this restores it)
                    x distance discount x personality weight

and takes the best one. That is utility AI. The "Maslow" part is just a
priority shaping on top: survival needs dominate the score until satisfied,
then safety, then social -- a hierarchy of needs, hence the name. It is not
an algorithm from the literature; it is a weighting scheme over a utility
scorer. Concretely for us:

    needs:  hunger (exists), safety (night exposure), shelter-stock,
            wealth (inventory), social (near group), ...
    goals:  eat / forage-near / TRAVEL-TO-FOOD / harvest-wood / harvest-stone /
            deliver / build / go-home / steal / give / raid / idle
    arbiter: score each goal from needs + world state, pick argmax
             (or softmax for variety), commit until done or interrupted

### GOAP and HTN (for completeness)
GOAP (F.E.A.R.): actions have preconditions/effects, a planner chains them
backwards from a goal at runtime. HTN (Killzone): hand-authored task
decompositions. Both give plan-shaped behaviour. Overkill for us -- our action
chains are short (harvest -> deliver -> build) and a utility arbiter over
whole-chain goals covers them.

### The colony sims -- the actual model for Island 2.0
RimWorld and Dwarf Fortress are the proof that AUTHORED agents + rich world
mechanics = emergent-LOOKING societies. Their agents are utility-scored job
pickers, nothing more. The drama (sieges, tantrum spirals, economies) emerges
at the POPULATION level from the interaction of simple agents with deep world
mechanics. Lesson: if 2.0 wants trade wars, the leverage is in the WORLD
MECHANICS (ownership, scarcity, asymmetry), not in agent brain size. Island
1.0's own data agrees: M5 doubled the value of trading and PPO still could
not find it -- headcount and incentive were never the missing piece,
mechanics were.

## 2. Architecture options, and the recommendation

The spectrum, from no-learning to all-learning:

### Option A -- pure utility agents (no RL)
The Maslow arbiter picks a goal; scripted controllers execute it. We ALREADY
HAVE the controllers: the scripted forager, builder, thief and trader in
`sim/policy.py` are exactly the goal-executors this needs, tested and priced.
Per-agent personality = a small random weight vector over the needs (agent 7
values wealth 1.3x, agent 12 is a coward about night...). Cost per agent per
decision: scoring ~12 goals = trivial. Scales to hundreds.

  + Watchable on day one. Deterministic. Debuggable. Fast.
  + Validates the 100-agent ENGINE before any training exists.
  - Nothing is learned. "Emergence" moves up a level: you author individual
    behaviour and study population dynamics (the RimWorld deal).

### Option B -- hybrid: RL CHOOSES THE GOAL, scripts execute it  << recommended
Same controllers as A, but the arbiter is a LEARNED policy: PPO over the
~12 goals instead of the 16 micro-actions. An option runs until it terminates
(ate / delivered / arrived / interrupted / timeout ~20-30 ticks); PPO trains
on one transition per OPTION with the discounted rewards summed -- semi-MDP
style, and `world.decision_interval` already built 80% of that machinery
(sum-rewards-per-decision, episode-boundary handling, tested).

Why this is not just a compromise but the RIGHT move given 1.0's findings:
the entire remaining open problem is that PPO cannot take a 20-step trip
whose every one-step prefix prices <= 0. Make "travel to the nearest loaded
bush" ONE action and the +173 prize becomes a one-step decision -- exactly
the "travel option" the CLAUDE.md plan already ranks as the most defensible
fix. Option B is that fix, generalised to every behaviour. The science
question survives in a sharper form: given the muscles, does PPO learn WHEN
to travel, build, steal, trade? M5's answer for trade was "no" at the
micro-level; the option level is a genuinely new experiment.

  + Keeps RL, and aims it at the one level where 1.0 proved it fails.
  + Credit assignment collapses: a trade is 1 decision, not a 40-tick chain.
  + One shared policy + per-agent trait embedding scales to 100 agents
    (see section 4 -- 100 individual brains is dead on arrival).
  - What emerges is when-to-X, never X itself. Say so in every write-up.
  - Off-policy wrinkle: ticks inside an option are not policy decisions.
    Handled the decision_interval way (the world enforces the commitment,
    PPO sees one transition), so the ratio stays on-policy. Interruption
    (a higher-priority need fires mid-option) must terminate the option and
    write the transition -- design this before coding, it is the one place
    the semi-MDP bookkeeping can silently rot.

### Option C -- inverted hybrid: scripted arbiter, RL executes
A BT/utility layer picks the need, an RL policy does the low-level control.
Rejected: low-level local competence is the one thing 1.0's PPO is GOOD at,
and this puts all the interesting decisions in the scripted layer. Maximum
cost, minimum science.

### Option D -- population heterogeneity tricks (orthogonal, cheap)
Whatever the brain, per-agent trait vectors (needs weights for A, an
embedding fed to the shared policy for B) give visible individual character
at 100 agents without 100 brains. M2's specialisation result suggests
differentiation will also be learned into the embedding.

RECOMMENDATION: build A first (1-2 sessions, validates engine + world +
watchability), then swap the arbiter for B on the same interface. The
arbiter is one function: `goal = decide(agent_state, world_view)`. Design
that interface once and A and B are interchangeable -- and directly
comparable, which becomes the headline experiment:

    same world, same controllers:
      scripted Maslow arbiter  vs  learned option-policy
    -> does the learned one beat the authored one, and where do they differ?

## 3. World mechanics needed for groups and trade wars

1.0's data says these do not come from headcount. What has to be in the world:

* RESOURCE ASYMMETRY BY REGION. Wood-rich north, stone-rich south, berries
  in between. Forces travel; makes trade the cheap alternative to a long
  walk. (M5's own postmortem: a relay needs its chain shortened by
  GEOGRAPHY, not its deliveries made fungible.)
* OWNED STOCKPILES. A household stockpile that members deposit into and
  draw from. Gives theft a target worth raiding (trade-WAR needs something
  to fight over; a berry in a pocket is not it).
* HOUSEHOLDS AS THE GROUP PRIMITIVE. Shared shelter = household = shared
  stockpile. Group identity for free, no new abstract channel: "my group"
  is "who sleeps where I sleep". Observation carries same-household flags.
* REPUTATION, minimal version: per-pair theft memory decaying over time,
  visible in the observation. Enables retaliation and guarding without any
  scripted "war" logic.
* SHOCKS (the RimWorld storyteller, tiny version): a bad berry season, a
  storm that damages shelters. Populations that never get stressed never
  visibly cooperate. Deterministic from the seed, like everything else.
* Scale housekeeping: keep supply per agent at ~1.2-1.5x subsistence (1.0's
  knife-edge sizing was for measurement, not watchability). Recompute with
  `sim`, never by hand (CLAUDE.md rule).

Forecast of failure modes to watch for (pre-registered):
* Utility agents + stockpiles -> runaway hoarding by early-rich households
  (inequality snowball). Probably GOOD drama; cap only if degenerate.
* Raid goal scored too cheap -> permanent war, nobody forages, collapse.
  The M3 lesson (theft redistributes, never creates) at population scale.
* 100 agents on too-few clusters -> one mega-camp. Region count must scale
  with population (rough rule: 1 cluster per 4-6 agents, from 1.0's data).

## 4. Feasibility forecast at 50-100 agents (before writing any code)

**STAGE 1 RAN (2026-08-25), AND THE HEADLINE FORECAST BELOW IS REFUTED.**
The spatial hash is NOT needed at n=100: the neighbour queries were already
fully vectorised, and a 100x100 pairwise matrix is trivial for numpy.
Measured with `sim.profile_engine` (random actions, mask computed per tick as
VecWorld does): 100 agents with EVERY mechanic on (m5b's world, scaled --
`config/island2/engine100_full.yaml`) ran at **98k agent-steps/s** before any
change, ~20x the 5k exit bar. The one real hotspot was the full stable argsort
in `_k_nearest` (~32% of the tick); replaced with argpartition + a stable sort
of the k winners, **bit-identical** to the old path (checksummed against the
stashed original across default/m3_masked/m5b worlds; `tests/test_scale.py`
pins the equivalence and 100-agent determinism). After: **144k agent-steps/s**
full-mechanics, 345k on the plain world. Configs: `config/island2/engine100.yaml`
and `engine100_full.yaml` (neither economy is sized yet -- stage 2 must
recompute subsistence with `sim`). Rule 2 held again: the pre-registered lever
was not the constraint. Revisit the hash only if the population goes well past
a few hundred.

Measured base (1.0): ~14k agent-steps/s, env step = 92.7% of cost, network
7.3%. Six agents x 32 envs = 192 concurrent agent-streams already run fine.

* NEIGHBOUR QUERIES ARE THE WALL. Current world does O(n^2)-ish scans
  (nearest bushes, neighbours in radius). 6 -> 100 agents is ~278x pairwise
  pairs. REQUIRED: a spatial hash grid (cell size = max query radius), which
  makes it O(n). Must iterate cells in deterministic order -- CPU determinism
  is a project invariant and several results depend on it.
* OBSERVATION STAYS FIXED-WIDTH: k-nearest everything (k unchanged), plus
  household/reputation channels. Width grows by ~10 dims, not with n.
* ONE SHARED POLICY + trait embedding. M2's six brains cost 15% throughput
  at n=6; 100 brains is absurd, and a shared policy is also the only way a
  100-agent batch is one matmul. Individual character comes from the
  embedding (option D).
* TRAINING BUDGET (option B): 100 agents x 8 envs = 800 streams, ~4x today's
  192. Env cost per step also grows ~linear in n with the spatial hash. Rough
  honest guess: a 200-update-equivalent run goes from ~6 min to ~40-60 min on
  this laptop. Tolerable, fanless throttling noted. Options help again here:
  one DECISION per ~10-20 ticks cuts PPO's transitions 10-20x.
* UTILITY AGENTS (option A): no training at all, and scoring is trivial next
  to the env step. 100 agents real-time or faster. This is why A goes first.
* REPLAYS: 100 agents x 600 ticks is ~17x today's replay size (~2MB JSON).
  Fine. Viewer rendering is the graphics section's problem.
* GPU still irrelevant (7.3% network share), CUDA still breaks determinism.

## 5. Graphics: agents need bodies

Current agents are capsules. Two routes, and the recommendation is the one
with zero external dependencies:

### Route 1 -- procedural low-poly biped (recommended)
Build a ~150-triangle humanoid in code from Three.js boxes/cylinders: torso,
head (with a 2-triangle nose or painted eyes via canvas texture), two arms,
two legs. Animate procedurally from replay data -- walk = sinusoidal arm/leg
swing keyed to speed, gather = lean forward, build = arm hammer loop, night =
sit. No rigging, no assets, no new CDN dependency, works offline like the
rest of the viewer, and 100 of these instanced is nothing for Three.js.
Colour = agent identity; a coloured headband/shirt = household.

### Route 2 -- rigged GLB characters (Quaternius / Kenney, CC0)
Real modelled characters with skeletal animations via AnimationMixer. Looks
better, costs: asset files in the repo, SkinnedMesh x100 needs care
(~500-tri models are fine, but it is the first real render cost), and clip
management. Do this later if Route 1 looks too crude -- the replay schema
work below is identical either way.

Either route needs one schema change: the replay should carry a per-tick
ANIMATION STATE per agent (walking/gathering/building/eating/sleeping/
stealing/giving -- derivable from the action already recorded, so possibly
zero schema change, just viewer-side mapping). Check before bumping
SCHEMA_VERSION; if the action column is enough, do not bump it.

Also worth it at 100 agents, cheap: name labels on hover, household banner
colours, a "follow this agent" camera, and a population sidebar replacing
the per-agent list (6 fit; 100 do not).

## 6. Staged plan (each stage ends runnable and watchable)

0. DECIDE SCOPE (you, now): is 2.0 a fork (new repo/dir) or a mode
   (config-gated, like every 1.0 mechanic)? Recommendation: config-gated
   mode, `island2/` configs -- 1.0's zero-shot/fork tooling is too useful
   to leave behind.
1. ENGINE SCALE PASS. Spatial hash, 100-agent config, deterministic order,
   profile it. No AI changes -- random policies. Exit: 100 agents at >= 5k
   agent-steps/s, tests green, byte-identical at n=6 with hash on.
   **DONE 2026-08-25, with one deviation the measurement forced: no spatial
   hash (see section 4 -- 144k agent-steps/s at 100 agents full-mechanics,
   byte-identity verified against the original code, all tests green).**
2. UTILITY AGENTS (option A). Needs + goal scoring + controller reuse +
   trait vectors. Exit: a 100-agent replay a stranger finds watchable;
   day/night rhythm visible; the pre-registered failure modes checked.
   **DONE 2026-08-25.** `sim/utility.py` (needs, goal scoring, option
   commitment), `sim/obsview.py`, `sim/society.py` (runner + the failure-mode
   report), `sim/economy.py`, `config/island2/society100.yaml`. Result: 2.80x
   the random floor, 94% of the tick limit, 20/20 shelters, 96% of nights
   indoors, and a visible commute (11.1 units from shelter by day, 3.7 at
   night). All three pre-registered failure modes clear -- see section 8 for
   what had to be corrected to get there, including the one that fired.
3. GRAPHICS. Procedural biped + animation mapping + follow-cam + population
   sidebar. Parallel with 2.
   **DONE 2026-08-25.** `viewer/biped.js`: route 1, instanced. NO SCHEMA BUMP
   was needed -- section 5 asked that this be checked, and the action column
   plus the night cycle already in the file are enough to drive every animation
   state. The viewer's 6-agent assumptions are addressed: instanced bodies (8
   draw calls for the population, against ~1300 meshes), a golden-angle colour
   ramp that survives a crowd, a population panel (aggregates + action
   histogram + swatch grid) above 24 agents with the old roster kept below it,
   hover labels, single-pass death ticks, and the gift-line cap raised from 32.
4. SOCIETY MECHANICS. Regions, households, stockpiles, reputation, shocks.
   Iterate on utility agents (fast loop, no training). Exit: raids and
   deliveries between households visible in a replay; an exchange-ledger
   view at household level.
   **DONE 2026-08-25.** All five mechanics, config-gated behind `society.*`, so
   every 1.0 world and stages 1-3 stay bit-identical. Five appended actions
   (deposit/withdraw x food/material, and raid), observation 61 -> 77 dims,
   replay schema v4 (stockpiles and raids per tick, households at top level),
   five new goals and two household needs in the arbiter, and household sections
   in both the society report and the exchange view. Exit met: raids render as
   red arcs from the robbed store to the raider, the household raid matrix is
   sparse and directional (household 12 raided household 9 134 times in two
   episodes), and `config/island2/society4.yaml` is the world. **Six corrections
   the build forced are in section 9, and one of the doc's own forecasts above
   is refuted** -- hoarding did not run away, the store churned instead.
5. LEARNED ARBITER (option B). Swap scoring for a shared PPO policy over
   goals, trait embedding, semi-MDP transitions. Train, then run the
   headline comparison: learned vs scripted arbiter, same world, paired
   islands (rule 7: paired, always).
6. WRITE IT UP against 1.0's findings -- especially whether the option-level
   policy finally takes the travel/trade decisions the micro-level one
   priced away.

Session-sized: 1 is one session; 2-3 together maybe two; 4 is open-ended by
design; 5 is one to build + the usual 6-minute-x-many experiment loop.

## 7. What this design deliberately gives up, so nobody rediscovers it

* "Trade emerged" can no longer mean what it meant in 1.0. The trade
  CONTROLLER is scripted; only the DECISION to trade can emerge. Every
  write-up must say which level it is claiming.
* The one-step-vs-20-step-prize problem is not solved, it is ROUTED AROUND
  by making the 20 steps one option. The pure-RL question stays open in
  1.0 (mix-anneal is still the cheapest next probe there, ~6 min).
* Scripted controllers read world state today (steering reads the true
  nearest loaded bush). Port them to read only the OBSERVATION before they
  become option-executors, or 2.0's agents are quietly psychic. The scripted
  forager already proves observation-only is enough for foraging; do the
  same audit for builder/thief/trader.

## 8. Stage 2/3 results, and the four corrections the build forced

Written 2026-08-25, right after the runs. Numbers are 5 episodes of
`config/island2/society100.yaml` unless stated.

### Where it landed

| | utility agents | random floor |
|---|---|---|
| mean lifespan | **563.2** of 600 (94%) | 201.0 (34%) |
| deaths / episode | **16.8** of 100 | 98.0 |
| berries / episode | 782 of the 1120 the island makes | 82 |
| shelters / episode | **20.0** of 20 | 2.8 |
| nights indoors | **96.3%** | 6.0% |

**2.80x the random floor.** The economy was sized with `sim.economy` rather
than by hand (the M3 postmortem rule): supply 1120 berries against 857
sheltered demand (1.31x, the doc's watchable band) and 1286 exposed demand
(0.87x), so shelter stays load-bearing by arithmetic exactly as it is in M4.

The three pre-registered failure modes, each against a control:

* **inequality snowball** -- lifespan Gini **0.058**. Not degenerate, and
  honestly not much drama either; households and stockpiles (stage 4) are
  what would give inequality something to accumulate in.
* **permanent war** -- steal 5.8% of goal-ticks against forage 13.5%. **This
  one fired on the first run** and is written up below.
* **mega-camp** -- daytime nearest-neighbour 5.81 against 8.86 for uniform
  placement on the same island (0.66x). Clustered, not collapsed. Measured by
  DAY only: at night the population is deliberately packed into shelters, and
  including those ticks made the intended behaviour read as the failure.

Watchability, which is the actual exit condition: by day the population is
spread over the whole island foraging; at night it is gathered into ~14 lit
settlements. Verified in a browser at 100 agents, v1/v2/v3 replays all
rendering, no console errors. The CPU side of a tick is 0.246 ms at 100 agents
(pose composition plus the panel). Rendered frame rate was NOT measured -- the
automation pane reports `document.hidden`, which throttles rAF -- so the
draw-call claim rests on construction (8 instanced meshes) rather than on a
timing.

### The four corrections, because each looked right in code

Three of these are the same mistake in different clothes: **confusing what a
need IS with what it costs to satisfy.** They are pinned by tests named after
the symptom in `tests/test_utility.py`.

1. **Theft as a travelling goal produced the pre-registered war.** 6932 steals
   an episode, foraging down to 10.9% of intentions, the population harvesting
   685 berries against a demand of 857. Cause, measured: a loaded victim sits
   at a median 2.8 units against a loaded bush at 4.2, so with 100 agents
   packed together the distance discount handed theft every contest. The fix is
   1.0's own measurement, not a weight: theft is available only when a victim
   is ALREADY in reach, never as somewhere to walk to -- the scripted thief
   never chases, and forcing every legal steal on the learned policy measured
   -8.3 +- 4.1 ticks. Theft redistributes and never creates; a utility scorer
   over selfish needs cannot see that, so the mechanic is where the correction
   belongs.
2. **The Maslow gate suppressed `explore`, the goal that SERVES the unmet
   need.** An agent with an empty inventory had tier-2 urgency at 1.0, which
   zeroed every tier above it -- including the search that was the only way to
   fix the shortage. 39.6% of all intentions went to `rest`: hungry agents
   standing still because wanting food had suppressed looking for it. Searching
   is never a luxury, so `explore` sits at tier 0.
3. **`distance_scale` did not grow with the island.** 1.0 set it to 20 on a
   radius-40 island; Island 2.0 is radius 100, so everything past a fifth of
   the way to the shore clipped to the same value and agents could not tell a
   shelter 25 units off from one 90 units off. 50 restores the ratio. This is
   the "perception beyond 20 units" problem from CLAUDE.md arriving on a bigger
   map, and it is worth checking on every future world resize.
4. **The safety need was wrong twice.** First it was scaled by
   distance-to-shelter, which made an agent 9.7 units from cover at night only
   19% unsafe -- it is 100% exposed, the drain is 3x whatever the distance is.
   Then, once corrected, zeroing it on arrival emptied the shelter halfway
   through the night: satisfied need, shelter scores 0, agent wanders back out.
   Being under cover is not a state that discharges the need, it is how the
   need goes on being met. Nights indoors: 22.7% -> 28.2% -> **96.3%**.
   The dusk lead time is now derived from `distance_scale / move_step` rather
   than hardcoded, so it is always long enough to actually walk home.

### What stage 2 says about stage 4, before anyone builds it

* **Construction is over by the first nightfall.** 20 sites x 4 units = 80
  units against 100 agents who each carry one, so `deliver` is 0.6% of
  goal-ticks and `shelter_stock` sits at 0 for the rest of the episode. The
  building economy has no *sustained* demand. Stockpiles, shelter decay, or
  storm damage (the doc's shocks) are what would give it one -- more sites is
  capacity, not depth, which is exactly the `m4g` lesson.
* **The `social` need is still not modelled, deliberately.** Nothing in the
  world satisfies one, so it would score goals against an appetite the world
  cannot feed. Households are what give it teeth.
* **Giving is trait-driven, not need-driven**, and it barely fires (11.8
  transfers an episode). That is the honest encoding -- handing a berry away
  restores nothing of the giver's own -- and it is the same wall M5 hit. A
  household stockpile changes the arithmetic rather than the weight.
* **The option interface is ready for stage 5.** `arbiter.choose(view, mask,
  rng) -> goal ids` and `execute_goals(...) -> actions` are the only seam; a
  learned chooser reuses `execute_goals`, `goal_viable` and `OptionRunner`
  untouched, which is what keeps the headline comparison honest. `OptionRunner`
  already tracks decisions per agent for the one-transition-per-option
  bookkeeping.

## 9. Stage 4 results, and the six corrections the build forced

Written 2026-08-25, right after the runs. Numbers are 5 episodes of
`config/island2/society4.yaml` against the random floor measured in the SAME
world, seeds 10000+.

### Where it landed

| | utility agents | random floor |
|---|---|---|
| mean lifespan | **578.3** of 600 (96%) | 314.8 (52%) |
| deaths / episode | **10.4** of 100 | 90.4 |
| berries / episode | 882 of the 1080 the island makes (82%) | 315 (29%) |
| nights indoors | **88.6%** | 28.0% |
| deposits / episode | **2795** | 198 |
| withdrawals / episode | 2031 | 163 |
| raids / episode | **629** | 3.8 |
| stockpile food, mean level | **5.40** of 12 | 0.49 |

**1.84x the random floor, and that ratio is NOT comparable with stage 2's
2.80x.** The floor rose, not the ceiling: stage 4 spawns a household together at
its own shelter site, so random agents start next to cover and sleep indoors on
28% of night ticks against stage 2's 6%. Absolute lifespan went **569.0 ->
578.3** on the same tick limit. Read the absolute number and the floor together;
the ratio between two different worlds' floors says nothing.

Two more figures that do not transfer, flagged because the M4 `m4d` write-up was
wrong for exactly this reason (rule 5):

* **`shelters / episode` is now a FLOW, not a stock.** A storm knocks finished
  shelters back to incomplete and they get rebuilt, so the counter reads 52 of 20
  sites. It counts completions including rebuilds and the report says so inline.
  Stage 2's "20.0 of 20" was a stock and cannot be set beside it.
* **Nights indoors fell 98.1% -> 88.6%**, and that is the storms working rather
  than the population getting worse: the roof over a household is periodically
  removed, and 11% of night ticks are spent in a house that is being rebuilt.

The five pre-registered failure modes, each against a control:

* **inequality snowball (per agent)** -- Gini **0.035** [OK].
* **permanent war (theft)** -- steal 3.5% of goal-ticks against forage 12.7%
  [OK]. It fired first at 13.8%; see correction 2.
* **mega-camp** -- the stage-2 measure had to be retired and replaced, because a
  stage-4 population is *supposed* to have piled up, twenty times over. See
  correction 4. At night **26.7%** of agents are nearer a foreign household's
  home than their own, against **95%** if position told you nothing about
  household [OK].
* **household inequality** (the doc's own version, which stage 2 could not test)
  -- Gini **0.016** across households, richest 599 ticks against poorest 537
  [OK]. So the doc's forecast of a hoarding snowball is **refuted**: the store
  turned out to be the opposite problem, a treadmill (correction 1).
* **raid economy** -- 629 raids per 2795 deposits (0.23x) [OK]. Of agents holding
  the raid goal, **19% were below the eat threshold**; the other 81% are settling
  grudges, which is the reputation mechanic producing feuds rather than famine
  relief. The household raid matrix is sparse and directional (reciprocity 0.61,
  one household raided 302 times and another was hit 350).

Watchability, the actual exit condition: twenty ringed settlements, each with a
hut, a food crate and a material crate whose heights move as the store fills and
empties; a visible dusk commute (mean distance to the nearest finished shelter
9.2 by day, 4.5 at night); red arcs when a store is robbed. Verified in a
browser on the v4 replay, no console errors, and v1/v2/v3 replays render
unchanged. **Frame rate was not measured** -- the automation pane reports
`document.hidden`, which throttles rAF -- so the cost claim rests on
construction (20 stockpiles is 60 extra meshes) rather than on a timing.

**Engine cost, measured.** `sim.profile_engine` at 100 agents: **101k
agent-steps/s** with the stage-4 mechanics on against 137k with them off, i.e.
the five mechanics cost ~26% of the tick and the world still runs at 20x stage
1's 5k exit bar. Most of that is the per-agent Python loops the stockpile and
raid phases use, in the same style as the existing gather/steal/build phases;
vectorising them is available if stage 5's training budget ever needs it, and
stage 1's lesson (measure before choosing a lever) says not to bother until it
does.

### The economy had to be resized, and the tool had to be taught why

CLAUDE.md rule 5, and it fired before the world ran once. A blight suspends berry
regrowth, so three blights over an episode cost the island 180 growing ticks of
600 -- at `regrow_ticks` 100 that is two of every bush's six regrowths, 280 of
1120 berries. Stage 2's 7 bushes per cluster would have run this world at
**0.98x sheltered subsistence**, where nothing behavioural can be read off it.

`sim.economy` now models blights and says so out loud, which is why this was
caught by the sizing tool rather than by a puzzling result. 9 bushes per cluster
puts it back in the doc's band: supply 1080, **1.26x** sheltered demand and
**0.84x** exposed, so shelter stays load-bearing by arithmetic exactly as it is
in M4.

The surplus is the mechanism, not slack. A cluster earns 9 berries per 100 ticks
against its 5 residents needing 7.1, so a household has ~1.26x its own
subsistence to put in a pile -- which is where a stockpile's contents come from.
During a blight that income is zero and the pile is the only thing between the
household and the night.

### The six corrections, because each looked right in code

Every one is pinned by a test named after the symptom in `tests/test_society.py`,
and four of them are an old lesson arriving in new clothes.

1. **The stockpile became the M5 gift farm.** 5858 deposits and 5475 withdrawals
   an episode with the pile never rising above **1.14 of 12**. An agent deposited
   its surplus, which raised its own `food_stock` deficit, which made drawing the
   best-scoring goal -- at the same location -- forever. The fix is the same shape
   as every fix that has worked here: not a smaller weight, a mechanic that cannot
   loop. A deposit needs a real SURPLUS (keep one unit back, and be above the eat
   threshold); a draw needs a real SHORTAGE (be empty-handed and actually getting
   hungry). The two are now mutually exclusive by construction. Pile level
   1.14 -> **5.40**.
2. **Theft exploded, and it was stage 2's correction 1 in a new world.** Spawning
   a household together keeps five agents permanently inside `steal_radius`, so an
   opportunistic steal fires every tick: **8105 steals an episode**, steal
   overtaking forage as a share of intentions (13.8% against 12.0%), and the
   grudges from all that theft then saturating every raid gate. `household_theft_immunity`
   is the mechanic-level answer and it is also the coherent reading -- a household
   shares a store, so you do not rob a housemate's pocket, you `withdraw`. Steals
   8105 -> 1960, steal share -> 3.5%.
3. **A household is not a place unless the shelter goal says so.** With `shelter`
   targeting the nearest finished hut, **55.7%** of agents slept closer to a
   foreign home than their own, so "my group is who sleeps where I sleep" was
   quietly false and every household statistic was describing a round-robin index.
   The fix needed a new observation channel, `home.complete`: an agent certainly
   knows whether its own roof is on, and after a storm its home site may not rank
   inside the `k_sites` nearest. It now sleeps at home when the roof is on and at
   the nearest finished shelter when it is not. 55.7% -> 26.7%.
4. **The mega-camp control expired when households became places.** Stage 2
   measured daytime nearest-neighbour distance against uniform placement on the
   same island, and stage 4 reads **0.47x** of chance -- tripping a WATCH on
   behaviour the design asked for. What "mega-camp" means once households exist is
   that the piles stop being SEPARATE, so the measure is now the share of agents
   who sleep nearer a foreign home than their own. Two details that matter:
   * it is a **NIGHT** statistic, the mirror of crowding being a day one. By day an
     agent is out foraging and a foreign home is often closer, which is a commute,
     not defection. Measured over all ticks it read 37.3% instead of 26.7%.
   * **the control is not 50% and it is not the random floor.** If position told
     you nothing, an agent's own home would be nearest by chance alone, i.e.
     1 - 1/H = **95%** displaced. The random-action floor gives **14.6%** --
     *lower* than the utility agents' -- because random agents barely leave the
     spawn point they were placed on. The uniform-position expectation is the
     control; the action floor is not.
5. **A committed `explore` outlived its own reason.** 28.3% of all goal-ticks went
   to exploring while the mean forage score among the explorers was 0.40 against
   explore's 0.17 -- they were not choosing to wander, they were serving out a
   25-tick commitment after a bush had come into view on tick three. Searching is
   the one option whose purpose is a perception, so its termination test is one
   too. This is a stage-2 bug found in stage 4, and **it changes a stage-2 number**:
   that world's lifespan is now 569.0 where section 8 records 563.2.
6. **One nan poisoned a whole episode's commute figure.** A storm can leave no
   finished shelter to measure distance against, and a single such sample turned
   the day/night rhythm into `nan`. `np.mean` -> a nanmean helper.

### What stage 4 says about stage 5, before anyone builds it

* **The option interface did not have to change**, which is the load-bearing
  claim for stage 5. Five new goals went in as five more rows of `RESTORE`,
  `GOAL_TIER` and the `execute_goals` dispatch; `arbiter.choose(view, mask, rng)`
  and `execute_goals(...)` are untouched, so a learned chooser still drops in
  beside the scripted one. The goal count is 10 -> 15 and the append-never-insert
  rule held for both goals and actions, so a stage-5 trait embedding or goal head
  keeps its meaning.
* **Trade across a household boundary still does not happen: 5 gifts in two
  episodes**, against 1337 raids. That is M5's wall standing exactly where it
  stood, now with geography and a store on top of it -- and it is the sharpest
  question stage 5 gets to ask, because at the option level "give this to a
  neighbour who will use it" is one decision rather than a forty-tick chain.
  Note honestly what stage 4 did NOT do: it did not make trade pay. `region_split`
  makes one material a journey, but `fungible_materials` is inherited from the
  m4h lineage, so a household can still substitute whichever material it has. A
  world where a site genuinely needs the other region's material is the next
  mechanic-level lever, and it is the one M5's own postmortem asked for.
* **`raid` is the goal a learned arbiter will find most interesting**, because
  its availability is gated on motive (desperation or a grudge) while its VALUE is
  a distance-discounted score. That is the one place stage 4 encodes a judgement
  the scripted arbiter cannot revise and a learned one could: 81% of raid
  intentions are revenge rather than hunger, and whether that is good play is
  exactly the sort of thing the headline comparison should settle.
* **The `social` need is still not modelled.** Households give it teeth in
  principle, but nothing in the world yet rewards standing near a housemate --
  shelter protection is a radius around a site, not around a group. Adding the
  need before adding the mechanic would score goals against an appetite the world
  cannot feed, which is the same call section 8 made.
