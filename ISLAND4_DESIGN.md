# Island 4.0 — a population the array does not bound, ground that can be taken,
# and an island where one place is worth more than another

Written 2026-08-30, the same day Island 3.0 stage 3 landed. Read
`ISLAND3_DESIGN.md` first: this is built on the frontier island and every
comparison here is against a one-key partner in `config/island4/`, never against
a 3.0 world.

**Nothing here is trained.** Every read below is the scripted utility arbiter, as
the whole of 3.0 was. That is deliberate and it is the working rule this stage
was built under: *mechanics first, scripted and measured; training last, once the
mechanic set is frozen.* The one training run 3.0 did (`arb7-village`) used
600-tick episodes and never reached agriculture, teaching or the granary — which
is what happens when you train a world that is not finished yet.

---

## 0. Why this stage exists, in three sentences from the previous one

* **`world.num_agents` was never an island, it was an array width.** Read R6
  called a village "extinct at tick 12,000"; stage 2 showed that was numpy, and
  slot reuse hid the ceiling (200 rows held 606 lives) without removing it —
  the number ALIVE AT ONCE was still `num_agents` in every run this project has
  published.
* **Tribes have been refuted three times, and stage 3 named the reason.**
  Quarter Gini 0.27 vs 0.23, then 0.31 vs 0.27, then 0.24 vs 0.25. Expansion
  here is SYMMETRIC — every household that fills its house founds a daughter, so
  the ratios between tribes never move — and raiding moves berries but cannot
  move a SITE. Stage 3's write-up ends: *"Conquest is the missing ingredient and
  nothing in this world expresses it."*
* **The ground is uniform, and three separate nulls sit downstream of that.**
  M5's exchange verdict (its own postmortem asked for geography), the tribe
  reads (an arc of an island where every arc is identical), and every
  carrying-capacity number (an island-wide mean over ground with no variance).

---

## 1. The blocks, all config-gated and all off by default

| block | key | what it does |
|---|---|---|
| growth | `world.grow_slots` | a birth with no free row WIDENS every per-agent array |
| conquest | `conquest.enabled` | a site changes hands under sustained foreign pressure, and the conqueror learns what it knew |
| terrain | `terrain.enabled` | clusters get a fertility; material is dealt to the poor half |
| culture | `reproduction.culture_weight` | a child's traits blend its parents' with its VILLAGE's |
| skill | `skills.enabled` | you get better at what you do; a practised hand brings back more |

Five 2.0 checksums, six 3.0 checksums and the stage-2/3 pins all reproduce
exactly with every key at its default — `tests/test_island4.py` opens with that
assertion and `pytest` is green at 503 tests.

---

## 2. Growth — the array stops being the ceiling

A birth that finds no free row grows every per-agent array by `slot_growth`
instead of being refused. Three things make it safe rather than fiddly:

* **A new row is UNBORN, not dead.** `born = False` is a state the world has had
  since Island 3.0 and every consumer already handles it, so growth introduces
  no new case anywhere — it reuses the one that unused founding slots already
  occupied.
* **It happens at the END of a tick (phase 7d), not before the births that need
  the room.** Every array inside `step` is sized from `pool.n` at the top of it,
  so growing mid-step would leave `rewards` one width and `hunger` another. A
  tick's births use the rows the PREVIOUS tick provided, and the headroom is one
  tick of maximum demand: at most one birth per household.
* **The registry is EXPLICIT**, and a test builds a world whose agent count
  matches nothing else in it and compares the registry against what is actually
  on the object. "Widen every array whose first axis equals `num_agents`" sweeps
  up the bushes the moment a world has as many bushes as agents.

`world.max_slots` is a **memory guard, not a design cap**, and the difference is
reported rather than assumed: the grudge matrix is (n, n) float64, so 20,000 rows
is 3.2GB on an 8GB machine. `stats().slot_cap_hits` counts the ticks on which it
bound, `sim.society` prints a `**` line when it did, and the line says outright
that every carrying-capacity read in such a run is void. That is read R6's
mistake with the sign reversed — there, 200 rows looked like an island.

**Training is not supported with growth on**, and that is honest rather than
lazy: PPO's buffers are fixed-width. The scripted arbiter, `sim.society` and the
replay recorder all handle a changing width; `sim.arbiter` does not.

### S1 — what the array was costing: 49 agents

`empire.yaml` vs `empire_capped.yaml`, one key apart, 5 paired episodes of 8,000
ticks on the same seeds:

| | grow | capped | paired diff | |
|---|---|---|---|---|
| **alive at the end** | **106.6** | **58.0** | **+48.6 ± 8.1** | **5/5** |
| rows in the array | 203.0 | 60.0 | +143.0 | 5/5 |
| villages | 26.0 | 21.2 | +4.8 ± 1.9 | 5/5 |
| ever born | 285.0 | 179.6 | +105.4 ± 15.2 | 5/5 |
| farming households | 20.6 | 13.6 | +7.0 ± 1.4 | 5/5 |
| granary households | 18.6 | 11.8 | +6.8 ± 2.6 | 4/5 |
| fissions | 6.0 | 1.2 | +4.8 ± 1.9 | 5/5 |
| conquests | 19.6 | 10.8 | +8.8 ± 3.5 | 5/5 |
| **mean lifespan** | **2101** | **2538** | **−437 ± 58** | **0/5** |
| `slot_cap_hits` | 0 | 0 | — | — |

**The sizing is the part to read first.** This island's carrying capacity is
~98 sheltered / ~65 exposed by `sim.economy`, plus ~33 more if every field slot
is planted. `empire.yaml` starts at **60 rows, deliberately below all of those**,
so in the capped control the array is what stops the population. At the frontier
world's own 200 rows neither arm reaches the ceiling and `grow_slots` correctly
does nothing — which is a fact worth knowing and is *not* an experiment.

**The honest cost is in the lifespan row and it is large.** A population twice
the size is a hungrier one: mean life falls 2538 → 2101, on 0 of 5 islands. Growth
does not make anybody better off; it makes there be more of them, which is a
different thing and the write-up should not blur it.

`cap_hits = 0` in every episode, so the memory guard never bound and the
population was limited by the island. That is the condition under which the other
rows mean anything.

---

## 3. Conquest — the missing ingredient, and the first tribe result that survives

A site changes hands when adults of another tribe outnumber its defenders inside
`radius`, for `hold_ticks` consecutive ticks. There is **no action, no combat and
no reward**: a siege measures PRESENCE, and presence is something a household has
to spend grown members on. Progress decays when the pressure lifts and a
different challenger restarts the clock, so two tribes taking turns at one
village never add up to a capture between them.

**The prize is the technology.** On a capture every household that had an adult
at the gate learns whatever the village knew — `TECH_CONQUERED`, the fourth way a
technology travels after invention, teaching and migration. That is the channel
by which a tribe that cannot invent has a move, which is the version of the
question worth asking.

**Two authored choices, stated rather than buried:**

* **Assimilation.** The residents change flag with the ground rather than dying.
  A mechanic that killed them would be measuring a massacre; this one measures a
  border.
* **`hold_ticks` and `radius` are a sizing**, picked the way the predator's count
  and reach were.

**A `conquer` goal, and its control.** `conquest.goal` gives the scripted arbiter
a goal that serves a new need, `NEED_TERRITORY`. The need is deliberately NOT
`land`: `land` reads zero for a household without farming, which is exactly the
household that should want somebody else's farm. `NEED_TERRITORY` is the larger
of "that village holds a technology I lack" and "my larder is short and I have no
way to grow more", both read off observation channels. The goal's muscle walks to
the village and stands there; it has no action, because a siege is presence.

**`conquest.threat` has no scripted consumer, on purpose.** The predator rung
shipped three channels and no `flee` goal on the argument that a scripted
response scores the response and then measures the score. A scripted `defend`
would do exactly that. The channel is there so a LEARNED chooser can see a siege
coming — the second place in this project where a learned arbiter is offered
something the scripted one is blind to.

### S2 — the tribe claim, refuted three times, finally survives its control

`empire.yaml` vs `empire_nocon.yaml`, one key apart, 5 paired episodes of 8,000
ticks:

| | conquest | control | paired diff | |
|---|---|---|---|---|
| **village Gini** | **0.31** | **0.22** | **+0.09 ± 0.04** | **4/5** |
| top tribe's share of villages | 0.46 | 0.39 | +0.07 ± 0.04 | 4/5 |
| tribe-population Gini | 0.36 | 0.29 | +0.08 ± 0.05 | 3/5 |
| villages changing hands | 19.6 | 0.0 | +19.6 ± 3.5 | 5/5 |
| households holding a tech BY CONQUEST | 4.4 | 0.0 | +4.4 ± 1.3 | 4/5 |
| alive at the end | 106.6 | 106.0 | +0.6 ± 6.8 | 3/5 |
| villages (total) | 26.0 | 26.0 | +0.0 ± 1.8 | 2/5 |
| mean lifespan | 2101 | 2016 | +85 ± 34 | 4/5 |

**Read the first row against the three that came before it.** Stage 1:
+0.04 ± 0.03. Stage 2: +0.03 ± 0.08. Stage 3: −0.02 ± 0.04. Now **+0.09 ± 0.04,
4/5** — the first time a tribe mechanic has moved inequality outside its own
noise, and it moved the number that means GROUND rather than the number that
means people.

**The honest size: +0.09 ± 0.04 is about 2.2 standard errors on five seeds.** It
is real and it is not overwhelming; it is larger than the three previous attempts
put together and it should be re-run at ten seeds before anybody calls it
settled.

**Two rows say what conquest is NOT.** The population is unchanged (+0.6 ± 6.8)
and so is the total number of villages (+0.0 ± 1.8): conquest moves ground
between tribes without creating or destroying any. That is the mechanic behaving
exactly as specified, and it is also why the *village* Gini is the right
measurement and the population Gini is the confounded one.

---

### S3 — did they go there on purpose? Mostly, and the knowledge channel entirely

`empire.yaml` vs `empire_nogoal.yaml` — conquest on in both, the `conquer` goal
off in the control. Raiding already puts enemy adults at enemy stockpiles, so
captures can happen without anybody intending one; this pair prices the
intention. Same question `--arm-all` asks of the axe, one level up.

| | goal on | goal off | paired diff | |
|---|---|---|---|---|
| villages changing hands | 19.6 | 5.4 | +14.2 ± 3.7 | 5/5 |
| **households holding a tech BY CONQUEST** | **4.4** | **0.0** | **+4.4 ± 1.3** | **4/5** |
| village Gini | 0.31 | 0.28 | +0.03 ± 0.04 | 3/5 |
| top tribe's share of villages | 0.46 | 0.43 | +0.03 ± 0.02 | 4/5 |
| alive at the end | 106.6 | 100.2 | +6.4 ± 8.1 | 3/5 |
| granary households | 18.6 | 16.4 | +2.2 ± 1.1 | 4/5 |
| mean lifespan | 2101 | 2034 | +67 ± 33 | 4/5 |

**Three readings, and the second is the surprising one.**

* **73% of captures are deliberate** (19.6 with the goal, 5.4 without). The
  remaining quarter is the grudge economy doing it by accident: a raiding party
  that lingers is a siege whether it meant to be one or not.
* **The technology channel is ENTIRELY goal-driven: 4.4 against 0.0, exactly.**
  Accidental conquest moves ground and no knowledge at all. Read against S2's
  headline, that is the sharper version of the claim: taking a village on purpose
  is how a tribe learns something; taking one by accident is just a border
  moving.
* **Most of the inequality survives without the goal.** Village Gini 0.28 with
  accidental conquest alone against 0.22 with no conquest at all (S2) and 0.31
  with the goal — so roughly two thirds of S2's +0.09 comes from captures nobody
  intended. Worth knowing before crediting the scorer for the result.

### S4 — the prize does not motivate conquest. It decides who WINS it.

`empire.yaml` vs `empire_notech.yaml` — conquest takes ground in both, and in the
control the conqueror learns nothing from the village it took.

**The pre-registered read, written into the config header before the run:** *"if
capture rates are the same in both arms, the technology transfer is decoration
and conquest is purely about land."* Half of that is confirmed and the
conclusion is wrong.

| | prize on | prize off | paired diff | |
|---|---|---|---|---|
| villages changing hands | 19.6 | 16.6 | +3.0 ± 2.9 | 3/5 |
| households holding a tech BY CONQUEST | 4.4 | 0.0 | +4.4 ± 1.3 | 4/5 |
| **village Gini** | **0.31** | **0.37** | **−0.05 ± 0.02** | **0/5** |
| **top tribe's share of villages** | **0.46** | **0.51** | **−0.05 ± 0.02** | **0/5** |
| alive at the end | 106.6 | 103.0 | +3.6 ± 1.8 | 4/5 |
| mean lifespan | 2101 | 2050 | +51 ± 20 | 4/5 |

**The rate is flat (+3.0 ± 2.9, inside noise): the prize is not why they go.**
The `conquest.prize` channel raises `NEED_TERRITORY`, and it turns out the other
term — a short larder with no way to farm — is what actually drives the goal.

**And the transfer is very far from decoration: without it the strongest tribe
gets STRONGER.** Village Gini 0.37 against 0.31 and the top tribe's share 0.51
against 0.46, both on 0 of 5 islands and both at ~2.5 SE — the tightest numbers
in this stage. **Capturing technology is a CATCH-UP mechanism, not a motive**: a
weak tribe that takes a farming village closes the gap, and with the channel off
the leader runs away.

That is the user-facing version of the question — *"the other ones have to take
their technology to beat them"* — measured, and the answer is subtler than the
question: they do not attack MORE for it, but it is what stops the leader
compounding.

---

## 4. Terrain — the ground stops being the same everywhere

Every island this project has run is uniform: clusters are interchangeable, so
one village's ground is worth exactly what another's is. Three separate nulls sit
downstream of that (§0), and the third one is why terrain belongs in the same
stage as conquest — taking a village is only interesting if villages differ.

Two knobs, doing different jobs:

* **`fertility_spread`** gives each cluster a lognormal multiplier, clipped, on
  its bush capacity and (inversely) on its regrow time. A fertile patch holds
  more and refills faster. Measured on this island: fertility spans
  **0.35–2.50** against a uniform 1.00, capacity 1–5 berries and regrow 40–286
  ticks.
* **`material_anticorrelated`** deals trees and rocks to the POOR half of each
  region. Measured, mean fertility where material stands: **0.59 (trees) /
  0.44 (rocks) against an island mean of 0.98** — the fertile half of each region
  has no material on it. That is M5's own prescription in place for the first
  time: *"a relay needs its chain shortened by GEOGRAPHY"*.

**It composes with the stage-4 region split rather than replacing it.** The
split says which economy lives on which side of the island; fertility says which
patches on that side are worth farming. Both keys are one-key controls and the
bias is applied inside each side.

**NO NEW OBSERVATION CHANNEL, deliberately.** A rich bush already reads as a bush
with more berries in it. A `fertility` input would tell a scripted scorer where
to go, and a mechanic that scores its own adoption measures the score — the
lesson the axe's write-up records.

**A field is not scaled by the ground under it.** A field is a thing a household
MADE, and scaling it would silently make agriculture a geography mechanic too,
which is a second variable nobody asked for. Pinned by a test.

### S5 — a 69% richer island that supports 17 fewer people, and no inequality

`empire_terrain.yaml` vs `empire_terrain_flat.yaml`, one key apart, 5 paired
episodes of 8,000 ticks.

**Read the supply line FIRST, because the pre-registered assumption about it was
wrong.** "Fertility is multiplicative about 1.0, so the island holds roughly what
it did" — it does not. A cluster's berry RATE is `capacity × f / (regrow / f)`,
i.e. proportional to **f²**, and E[f²] = 1.70 against E[f] = 1.20. Measured over
five seeds:

| | flat | terrain |
|---|---|---|
| berries per tick, whole island | **4.00** | **6.75 (+69%)** |

| | terrain | flat | paired diff | |
|---|---|---|---|---|
| **alive at the end** | **88.8** | **106.6** | **−17.8 ± 15.7** | **1/5** |
| rows the array grew to | 148.6 | 203.0 | −54.4 ± 13.6 | 0/5 |
| ever born | 239.2 | 285.0 | −45.8 ± 28.9 | 1/5 |
| **village Gini** | **0.34** | **0.31** | **+0.02 ± 0.07** | **3/5** |
| top tribe's share of villages | 0.45 | 0.46 | −0.01 ± 0.08 | 2/5 |
| villages changing hands | 14.8 | 19.6 | −4.8 ± 2.9 | 1/5 |
| farming households | 17.8 | 20.6 | −2.8 ± 3.5 | 1/5 |
| mean lifespan | 2122 | 2101 | +21 ± 40 | 3/5 |

**Two readings, and both are honest negatives.**

* **A much richer island supports fewer people, and the mechanism is
  CONCENTRATION rather than quantity.** The extra 69% sits on a handful of very
  rich clusters, and an agent harvests one bush at a time: a bush holding 5 that
  refills every 40 ticks cannot be drained by the villagers who live next to it,
  while the families beside a bush holding 1 that refills every 286 starve. Per-
  bush rates span **0.0035 to 0.125 berries/tick, a factor of 36.** Supply that
  nobody can reach is not supply. The population read is −17.8 ± 15.7, i.e. 1.1
  SE and inside noise on its own — but `slots` (−54.4 ± 13.6, 0/5) and `born`
  are not, and all three point the same way.
* **It produces NO inequality between tribes: +0.02 ± 0.07, 3/5. Nothing.** And
  the reason is a design flaw that is worth more than the result: **fertility is
  drawn per cluster, independently, so it is spatially UNCORRELATED — and a tribe
  is an angular arc.** Every arc contains a mix of good and bad ground, so no
  tribe systematically got the better half. **For terrain to make tribes unequal
  it has to be a fertile REGION, not a fertile patch** — a smooth field over the
  island rather than 40 independent draws. That is a concrete next mechanic and
  it is the thing this run actually taught.

**What is verified regardless of the reads:** the mechanic does what it says
(fertility 0.35–2.50 against a uniform 1.00; material on ground of mean fertility
0.59/0.44 against an island mean of 0.98), and it changes neither the observation
layout nor the action space, so a 3.0 arbiter runs in it unchanged.

## 5. Culture — a refuted prediction, and it is the interesting kind

`reproduction.culture_weight` blends a child's inherited traits with the
geometric mean of the household it is born into. 0.0 is pure heredity, which is
every world before this one and is the default.

**The prediction, written before the run: villages would DIVERGE from each
other while staying uniform inside — which is what a culture looks like from
outside.** Measured on one 4,000-tick episode, between-village variance of the
mean log-trait against within-village variance:

| | between villages | within a village | ratio |
|---|---|---|---|
| `culture_weight: 0.0` (heredity only) | 0.0813 | 0.0201 | 4.04 |
| `culture_weight: 0.7` | 0.0465 | 0.0161 | **2.89** |

**Both fell, and the ratio fell with them.** Culture made villages more uniform
inside *and* more like each other — the opposite of the prediction, on the axis
the prediction was about.

**The mechanism, and it is arithmetic rather than a surprise once seen.**
Blending toward a household mean is a SHRINKAGE operator. Under pure heredity a
child inherits from two parents, so each lineage performs its own random walk and
between-village differences accumulate. Under culture it inherits from the mean
of ~7 people, which carries about a seventh of the variance, so the village's own
walk takes much smaller steps and drifts less far. Averaging does not create
distinctness; it destroys it in both directions at once.

**What that says about what is actually missing.** For cultures to diverge you
need selection or drift AMPLIFICATION, not averaging — something that pushes a
village's average somewhere in particular. Nothing in this world does. That is a
concrete next mechanic and it is more interesting than the one that was built.

**Scope, stated plainly: this is ONE seed and one episode.** It is enough to
refute the direction of the prediction (the effect is large and the mechanism is
arithmetic) and it is not enough to quote the numbers as measurements. Re-run
paired at five seeds before using them for anything.

---

## 6. Skill — the mechanic works and the specialisation does not appear

`skills.enabled` gives each agent a proficiency per resource (berries, wood,
stone) that rises with SUCCESSFUL use and buys yield. Swinging at an empty tree
teaches nothing — paying for the attempt would make standing at a stump a way to
get good at chopping, which is the doomed-action failure the M3 mask deletes,
wearing a learning curve's clothes.

**Deterministic, with no rng anywhere.** A yield is an integer and a skill bonus
is a fraction, so the bonus banks in a per-agent CARRY and pays a whole extra
unit when it crosses 1.0. Rolling for the extra unit was the obvious
implementation and it would have put a random stream inside the harvest phase,
which is how a replay stops reproducing.

**`rate` was sized against a measured number rather than picked** (rule 5). At
0.002 the best forager on this island reached 0.154 in 4,000 ticks — about 77
successful gathers — so a whole career bought a 15% bonus and the mechanic was
decorative. It ships at 0.008, which reaches ~0.6 over the same 77 uses.

### The read: one profession, not a division of labour

6,000 ticks, `skills.enabled`, 111 agents alive at the end:

| | forage | wood | stone |
|---|---|---|---|
| mean skill | 0.341 | 0.066 | 0.029 |
| best agent | 0.704 | 0.280 | 0.184 |
| skill Gini | 0.342 | 0.544 | **0.733** |
| **agents whose BEST skill is this** | **111** | **0** | **0** |

**The mechanic fires and competence is very unevenly distributed — and every
single agent on the island is best at foraging.** There is one profession here,
not a division of labour.

**Why, and it is the same shape as the construction wall.** The scripted arbiter
spends most of its time foraging whatever an agent's traits say, because hunger
is tier 0 and everything else is prudence. Practice therefore accumulates where
the time already goes, and the trait differences that were supposed to seed
specialisation are swamped: `corr(trait_forage, skill_forage) = +0.207` and
`corr(trait_wood, skill_wood) = +0.049`. Practice tracks preference a little for
the job everybody does and not at all for the job nobody does.

**So specialisation does not emerge from practice alone when one task dominates
the time budget** — which is a finding about the world, not about the mechanic,
and it is exactly what the `sim.society` report now says out loud
(`[ONE PROFESSION -- no division of labour]` when one resource holds >90%).

**The channels are in the observation and nothing scores them.** A scripted "do
what you are best at" rule would score the specialisation and then measure the
score. `own.skill_*` exists so a LEARNED chooser can notice its own competence
and lean into it, which is the third place in this project where a learned
arbiter is offered something the scripted one cannot see.

**Scope: one seed, one episode, no paired control.** The Gini and the "one
profession" row are qualitative enough to survive that; the correlations are not,
and should not be quoted without a five-seed pair.

---

## 7. Graphics — schema v9, and the bug that had made a whole mechanic invisible

### The bug first, because it is the largest single thing in this stage

`ReplayRecorder.to_dict` wrote the header's `sites` off `world.site_x`. A DORMANT
site is parked at `inf`; `json.dump` writes the literal `Infinity`; `JSON.parse`
rejects it outright. **So every frontier replay ever written was unloadable in a
browser, and village fission — the mechanic that makes villages MULTIPLY, and the
headline of stage 3 — had never once been watched.** It is the same trap the
fields fix already caught (ISLAND3_DESIGN.md), in the one place that fix did not
reach.

Sites now come off `site_layout_*`, which is always finite, and `site_active`
says who lives where. `blob["households"]` was widened to every slot for the same
reason and would have written `Infinity` next.

Pinned by `test_a_frontier_replay_is_valid_json_with_no_infinity`, which asserts
on the serialised TEXT rather than on the dict — the dict was always fine.

### What v9 adds, and why each key is not derivable

| key | what | why the viewer cannot compute it |
|---|---|---|
| `hh` | households whose (tribe, farming, granary, active) changed | four things a v8 reader could not show at all |
| `hm` | agents whose (household, tribe) changed | a founding party changes household with NO birth, so v8's `u` never fired |
| `cq` | captures this tick | otherwise a colour quietly changes |
| `fs` | foundings this tick | same |
| `sg` | siege progress per household, sparse | not in any post-step state |
| `y` / `yt` | the per-tick aggregate series | population is recomputable; tech adoption is not, because a household's history arrives as sparse events |

Every one is sparse or a handful of integers: a 12,000-tick village pays a few
hundred KB for a history it previously could not show.

**A tick's agent list may now be SHORTER than the header's**, because
`grow_slots` widens the world mid-episode and the header is sized from the final
width. Rows past the end are unborn, which is a state every v7 reader already
handles — the only new thing is that the row is absent rather than
present-and-unborn. Four call sites in the viewer are guarded and the guards say
so.

### What the viewer draws that it did not

* **Colour by TRIBE, and it is the default when a world has tribes.** The
  golden-angle per-agent palette is right for six agents and wrong for four
  tribes: at n=100 the hues sit 3.6° apart, so "which tribe holds the north" is a
  question the default colouring physically cannot answer. Three modes —
  agent / tribe / village — and the rig gained a `recolour` hook because identity
  now changes DURING an episode (a settler joins a new village, a conquered
  village changes tribe).
* **A territory field**, not a disc per village. The disc version was written
  first and answered the wrong question: a ring under each hut says where the
  huts are. A 72×72 vertex-coloured plane tints every patch of ground by the
  tribe of the nearest inhabited village, with the reach scaled from how many
  villages there are, so a tribe taking villages visibly spreads. Unclaimed
  ground is left unclaimed — a field that tinted the whole disc would say four
  tribes had carved up an island they have barely settled.
* **A siege reads as a red arc that CLOSES** round the village under it. A number
  in a panel is a number nobody watching sees.
* **A history strip.** A civilisation is a TRAJECTORY and every number this
  viewer showed before was one tick's, so "the population overshot its carrying
  capacity and fell back" — the headline of 3.0 — was a thing you could read in a
  write-up and never see. Five series, or the per-tribe populations when
  colouring by tribe, with the playhead marked; clicking it scrubs.
* **An event feed**: foundings, captures and technology unlocks, derived at load
  from the sparse stream.
* **Dormant sites are hidden**, so a daughter settlement reads as APPEARING
  rather than as one of forty foundations slowly filling in.

Verified in a browser against a real 1,500-tick frontier replay: loads, scrubs,
no console errors, and the territory tint, history strip and event feed all
populate.

---

## 8. What is still missing to make this behave like real civilisations

Written as a list of mechanisms, not a wish list, and ordered by how much each
would change the story. Nothing here is built.

1. **Something that pushes a village's average somewhere.** §5 is the argument:
   averaging cannot make cultures diverge, and this world has no selection
   pressure that differs BY PLACE. Fertility (§4) is the obvious carrier — a
   village on poor ground should be selected toward different behaviour than one
   on rich ground — and nothing currently connects the two.
2. **A task budget that is not dominated by hunger**, or specialisation cannot
   appear (§6). Every agent forages most of the time because hunger is tier 0, so
   practice accumulates in one place for everybody. A household that could FEED a
   non-forager — a real division of labour rather than a preference — needs the
   larder to be reliable enough that somebody can stop eating out of it directly.
3. **Trade with a price.** Exchange exists as gift, steal and raid; there is no
   mechanism by which two households agree terms. §4's geography is the
   precondition (one village has food and no stone), and it is now in place;
   the mechanic on top of it is not.
4. **A technology that changes a RULE, not a number.** Three rungs exist (axe,
   farming, granary) and four routes now carry them (invention, teaching,
   migration, conquest), but every rung so far multiplies a quantity. Writing,
   record-keeping, a road — something that changes what the world can express —
   is a different kind of rung and none has been tried.
5. **Defence.** `conquest.threat` is in the observation and no scripted goal
   reads it, deliberately (§3). A world where a besieged village can respond is a
   different world from one where it cannot, and it is the natural first thing to
   hand a learned chooser here.
6. **Death by violence.** Conquest assimilates; raiding moves berries. Nothing in
   this project can kill an agent except hunger and old age, so "war" is a word
   the world does not currently support.

---

## 9. Corrections this stage forced, each pinned by a test named after the symptom

1. **`sites` written as `Infinity`.** §7. Every frontier replay was unloadable
   and the fission mechanic had never been seen. `test_a_frontier_replay_is_valid_json_with_no_infinity`.
2. **`blob["households"]` stopped at `num_households`.** A daughter settlement
   had no row in the table, and reading its position off `stock_x` would have
   written `Infinity` next. Widened to every slot, off the layout.
3. **The agent table was sized from `cfg.world.num_agents`.** With growth that is
   only where the array STARTED, so the table silently dropped every grown row —
   and `zip` truncated the household column to match without saying so. Sized
   from `world.pool.n`. `test_a_grown_world_writes_an_agent_table_the_size_it_ENDED`.
4. **`n_tribes` was derived as `max() + 1`.** Fine while tribe membership never
   changed; with conquest a tribe can be wiped off the map, and a per-tribe array
   that loses its last column reports the annihilated tribe's zero as the NEXT
   tribe's total. Taken from the config when tribes are on.
   `test_a_tribe_wiped_off_the_map_keeps_its_column`.
5. **The growth registry was nearly discovered by shape.** "Widen every array
   whose first axis equals `num_agents`" sweeps up the bushes the moment a world
   has as many bushes as agents. Explicit table, and a test that builds a world
   whose agent count matches nothing else in it.
   `test_every_per_agent_array_is_in_the_growth_registry`.
6. **Growth mid-step would have split the tick in half.** Every array inside
   `step` is sized from `pool.n` at the top of it, so a birth-time grow leaves
   `rewards` one width and `hunger` another. Moved to phase 7d, and everything in
   the StepResult is padded to the new width in one place.
   `test_step_result_arrays_match_the_new_width_after_growth`.
7. **`material_anticorrelated` was inert as first written.** Dealing round-robin
   over a fertility-sorted list biases only the SURPLUS: with 50 trees over 40
   clusters just ten clusters get a second one, and the measured fertility where
   trees stood came out ABOVE the island mean (1.04 against 0.98). It now deals
   to the poorest half of each region, which measures 0.59 / 0.44 against 0.98.
   That is the difference between a geography and a rounding error.
8. **The `conquer` goal appended to `GOAL_NAMES`**, which reshuffles every
   agent's traits and rewidens a learned head — the trap rung 1 sprang twice.
   `N_GOALS_ISLAND3` freezes the 3.0 draw and `goal_width` returns it for a 3.0
   world. `test_the_goal_rung_is_frozen_so_island3_traits_do_not_move`.
9. **A skill bonus could have overdrawn a bush or a pack.** Bounded by the same
   two things a normal take is. `test_a_skill_bonus_can_never_overdraw_a_bush_or_a_pack`.
10. **`sim.society` sampled runner state against pool state after `step`.**
    `world.grow_slots` widens the world INSIDE `step`, and `OptionRunner.act`
    widens itself on its NEXT call — so the per-tick diagnostics ran in between
    and `(runner.goals == GOAL_RAID) & pool.alive` broadcast a (135,) against a
    (203,). Found by recording a replay, not by a test, which is the honest
    order. The runner is now widened immediately after `step`.
11. **Two pre-registered predictions were wrong**, and both write-ups lead with
    that rather than burying it: culture was supposed to make villages diverge
    (§5, it does the opposite) and the technology prize was supposed to be
    decoration if capture rates were flat (§3 S4 — the rates ARE flat and the
    prize decides who wins anyway).

---

## 10. Reproduce

```bash
# size the world FIRST (rule 5) -- and read the carrying capacity, because
# `world.num_agents` is deliberately set below it
.venv/bin/python -m sim.economy --config config/island4/empire.yaml

# the five one-key pairs, sequentially -- never in parallel on this laptop
.venv/bin/python -m sim.society --config config/island4/empire.yaml         --episodes 5 --ticks 8000
.venv/bin/python -m sim.society --config config/island4/empire_capped.yaml  --episodes 5 --ticks 8000  # S1
.venv/bin/python -m sim.society --config config/island4/empire_nocon.yaml   --episodes 5 --ticks 8000  # S2
.venv/bin/python -m sim.society --config config/island4/empire_nogoal.yaml  --episodes 5 --ticks 8000  # S3
.venv/bin/python -m sim.society --config config/island4/empire_notech.yaml  --episodes 5 --ticks 8000  # S4
.venv/bin/python -m sim.society --config config/island4/empire_terrain.yaml      --episodes 5 --ticks 8000  # S5
.venv/bin/python -m sim.society --config config/island4/empire_terrain_flat.yaml --episodes 5 --ticks 8000

# a watchable replay: the frontier world (which v9 made loadable at all), and an
# empire where villages change hands
.venv/bin/python -m sim.society --config config/island3/village_frontier.yaml --ticks 1500 --replay
.venv/bin/python -m sim.society --config config/island4/empire.yaml --ticks 4000 --replay

# the two blocks with no shipped config, both single-key on top of empire
.venv/bin/python -m sim.society --config config/island4/empire.yaml --episodes 3 --ticks 6000 \
    --set skills.enabled=true
.venv/bin/python -m sim.society --config config/island4/empire.yaml --episodes 3 --ticks 6000 \
    --set reproduction.culture_weight=0.7
```

**Do not re-run S1 or S2.** S1's mechanism is arithmetic and its sizing is
documented; S2 is the one worth extending, and the extension is MORE SEEDS
(ten, not five), not another configuration.
