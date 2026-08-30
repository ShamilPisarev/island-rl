# Island 3.0 — families, births, farms, tribes

Written 2026-08-30, before a line of it was implemented. Island 2.0 ended with a
society that survives 600 ticks and, run longer, dies out: `ISLAND2_DESIGN.md`
§15 measured 100 agents falling to 16 by tick 24,000 because trees and rocks
never regrow, so the shelters storms knock down are never rebuilt.

That collapse is the door into 3.0. A world that cannot sustain itself cannot be
asked who *wins* it — and "who wins" is the question this stage exists to make
askable. The 2.0 population is a fixed cast of 100 that only ever shrinks. The
3.0 population is born, grows up, and dies, so a household that feeds and houses
itself well leaves more descendants than one that does not, and the island's
composition at tick 20,000 is a measurement rather than a coincidence.

## What is being asked for, and what each ask becomes

| the ask | the mechanic |
|---|---|
| reproduction | births at the family's own house, automatic, conditioned on prosperity |
| a family houses its own people in one house | houses have OCCUPANCY, kin get the beds first |
| build the house bigger when the family grows | `build` at a finished house adds a room |
| trees regrow like a real world | `tree_regrow_ticks` / `rock_regrow_ticks` |
| invent agriculture when there is no food | a field, unlocked per household by hunger |
| families fight and rob each other for food | steal/raid/grudge already exist; tribes make them collective |
| technology | the axe (rung 1) is the first; farming is the second |
| the strongest tribe survives and reproduces | tribe population share, measured over a long run |

## The five design decisions that carry the stage

**1. Reproduction is automatic, not an action and not a goal.** This is the
project's auto-eat call (`CLAUDE.md`, "Decisions, and why") applied to the thing
the whole stage is about. A birth happens when two grown, well-fed members of one
household stand at their own house, the house has a free bed, the family larder
can pay for the child, and the household's cooldown has elapsed. Nothing chooses
it and nothing is rewarded for it.

That is not laziness, it is the only version of this that can answer the
question. If `reproduce` were a goal in the arbiter, the scorer would decide how
much a family wants children, and "the strongest tribe reproduces" would be a
restatement of the weight we typed. Made automatic and conditioned on prosperity,
births become a **measurement of** the household's success at feeding and housing
itself — the thing the previous two milestones spent their whole length trying to
produce. Rule 1, in the one place where getting it wrong would be invisible.

**2. A house has a capacity, and kin get the beds.** Until now `shelter_radius`
was a disc that protected everyone standing in it, so a hundred agents could
shelter under one hut. A house now shelters `base_occupants`, plus
`occupants_per_room` for every room built onto it, and when more agents are in
range than there are beds the household's own members are admitted first and the
rest sleep out. That is the ask ("every family should house their own people")
and it is also what makes the next decision load-bearing.

**3. Building at a finished house makes it bigger.** No new action: `build` at a
complete site adds a unit to `site_extra`, and every `expand_units` of those adds
a room. So a family whose children have outgrown the house has somewhere for its
labour to go, and the material economy acquires the sustained demand that stage 4
had to invent storms to create. This also means the construction wall stage 5
could never climb (`ISLAND2_DESIGN.md` §10) now has a *recurring* prize rather
than a one-off one — which is a fact worth measuring, not a fix anyone should
assume works.

**4. Fields are bushes.** A planted field is appended to the bush arrays with its
own capacity and regrow rate. Everything downstream — the observation, the mask,
`gather`, the `forage` goal, `sim.navigation` — works on it unchanged, which is
why agriculture costs one action and no new machinery. Field slots are
preallocated inactive at reset (position (0,0), zero berries, `bush_active`
false), so a world with agriculture off is bit-identical to one without the
block, and an unplanted slot is invisible because every consumer already tests
`berries > 0`.

**5. A tribe is a place.** Households are assigned to tribes by the ANGLE of
their site around the island, not round-robin, so a tribe is a contiguous arc of
the coast and inter-tribe raiding is a border phenomenon rather than a lottery.
Round-robin tribes would have produced tribes that are everywhere and therefore
nowhere, and "the strongest tribe" would have meant nothing spatial at all.

## The honest costs, stated before the run

* **The invention of agriculture is scored, not discovered.** A household unlocks
  farming when its members have spent `unlock_hunger_ticks` agent-ticks below
  `unlock_hunger`. Necessity is the mother of invention *because we wrote that
  down*. What is genuinely measured is (a) whether the unlock fires more in a
  hungry world than a fed one, which is a fact about the world rather than the
  rule, and (b) whether a field, once available, is used and whether it pays.
  Same honesty the axe rung carries: adoption is scored by the utility arbiter's
  `RESTORE` row, and "agents worked out it was worth it" is a claim only a
  learned chooser can earn.
* **Fights between families are the 2.0 mechanics wearing a tribe.** Nothing new
  is added to steal or raid; what tribes change is who is immune and who
  remembers. A war that appears here is the stage-4 grudge economy at a larger
  grain, not a new capability.
* **Nothing here is trained.** Rungs 1 and 2 were run on the scripted utility
  arbiter, and so is this. Every number below is a statement about a scripted
  population in a new world. Training a learned arbiter on top is the follow-up,
  and it is the first world where a learned chooser would have something a
  scripted one cannot score (a house that is full).

## Pre-registered acceptance criteria

Written before implementation. Every behavioural read is paired on the same seed
block against a control that differs in exactly one config key (rule 7).

**Correctness**

* **C1 — bit-identity.** With every 3.0 block off, five 2.0 worlds
  (society4, ramp, predator, axe, trade) reproduce their pre-3.0 trajectory
  checksums exactly. Pinned by a test with the hashes taken from the previous
  commit.
* **C2 — the suite is green**, existing tests included, plus new tests named
  after the thing they protect.
* **C3 — sized before it is run** (rule 5). `sim.economy` is taught about
  regrowing material, child drain and field yield BEFORE any 3.0 config is
  written, and the config is sized against its output.

**Behaviour** — each with its prediction and its failure mode.

* **R1 — regrowing trees turn the collapse into a carrying capacity.** society4
  run 6,000 ticks with `tree_regrow_ticks` on against the same world with it off,
  same seeds. Prediction: survivors materially higher with regrowth. *Failure
  mode:* if survivors do not move, material was not what was killing them and
  §15's mechanism is wrong.
* **R2 — births happen and track prosperity.** Births per 6,000 ticks > 0, and
  the per-household birth count correlates with that household's mean food stock.
  Control: the same world with `reproduction.enabled: false` (population can only
  fall). *Failure mode:* zero births means the birth conditions never co-occur —
  read the four conditions separately before touching any number.
* **R3 — a family that grows builds a bigger house.** Expansion units delivered
  per episode > 0, and the share of night ticks spent outside for want of a BED
  (as distinct from for want of a house) falls after expansions begin. Control:
  `housing.enabled: false`. *Failure mode:* if capacity never binds, the world is
  not crowded enough to test the mechanic — check mean occupancy first.
* **R4 — agriculture is invented under scarcity, not under plenty.** Fields
  planted per episode in a food-scarce world against the same world with the
  berry economy at 2.0's 1.26x subsistence. Prediction: scarce >> plenty. Plus an
  adoption floor: a random-goal population in the same scarce world (the axe
  rung's rule — an adoption number without its floor is not a number).
* **R5 — tribes fight, and the strongest ends up biggest.** Raids per episode
  across tribe borders against raids within a tribe; final population share by
  tribe and its Gini. Control: `tribes.enabled: false`. *Failure mode:* if tribe
  sizes stay equal, either nothing differentiates them or the run is too short —
  read births and deaths by tribe before concluding anything.

---

# Results

Every number below is the scripted utility arbiter, paired on the same seed block
against a control that differs in exactly one config key (rule 7). Nothing here
is trained; a learned chooser in this world is the follow-up, and §"What is left"
says why it is now the first genuinely new question in the project.

**Read the population TRACE, not the mean lifespan.** In a world where agents are
born, mean lifespan is bounded by the ticks each agent had left to live: an agent
born at tick 5,000 of a 6,000-tick run cannot score above 1,000 whatever it does,
so a village with 160 births reports a "mean lifespan" of 2,181 while being
perfectly healthy. It is not a survival statistic here and the report says so on
the line itself. The trace is.

## R1 — regrowing trees turn the collapse into a carrying capacity. CONFIRMED.

`society4_regrow.yaml` against `society4.yaml`: same 100 fixed agents, same
bushes, same shocks, same seeds, one key changed. §15's diagnosis was that the
material runs out and the shelters storms keep levelling are never rebuilt.

**Population at tick 24,000: 48 with regrowth, 16 without** (1 episode). The
shape is the result:

| tick | 2,000 | 4,000 | 6,000 | 10,000 | 16,000 | 24,000 |
|---|---|---|---|---|---|---|
| trees regrow | 59 | 49 | 49 | 49 | 48 | **48** |
| trees do not (2.0) | 46 | 36 | 30 | 25 | 18 | **16** |

The regrowing island is **flat for twenty thousand ticks**. The 2.0 island is
still falling at 24,000 and its own §15 write-up says so.

At 6,000 ticks with 3 paired episodes the effect and its mechanism are both
outside noise:

| | regrow | 2.0 | paired |
|---|---|---|---|
| survivors | 51.3 | 33.0 | **+18.3 ± 1.2 (3/3)** |
| nights indoors | 88.5% | 38.1% | **+50.4 ± 1.2 (3/3)** |
| shelters completed | 522 | 135 | +387 ± 31 (3/3) |
| berries gathered | 5711 | 6054 | −342 ± 361 (1/3) |

**The mechanism is confirmed and it is not food.** Berries do not move (inside
noise, and if anything the collapsing island gathers *more*, because there are
fewer mouths per bush). What moves is the roof: 38% of nights indoors against
88%. §15 named the right resource.

## R2 — births happen, and the village overshoots its own carrying capacity. CONFIRMED.

`village.yaml` against `village_static.yaml` (40 fixed agents, everything else
identical), 5 paired episodes of 6,000 ticks.

| | village | static | paired |
|---|---|---|---|
| births | **159.6** | 0 | +159.6 ± 0.4 (5/5) |
| population at the end | 64.8 | 32.0 | +32.8 ± 1.6 (5/5) |
| berries gathered | 6914 | 3444 | +3469 ± 292 (5/5) |
| rooms added | 58.4 | 0.8 | +57.6 ± 2.4 (5/5) |
| fields planted | 32.2 | 0 | +32.2 ± 1.2 (5/5) |

**The trace is the finding, and it was not pre-registered:**

| tick | 500 | 1,000 | 2,000 | 3,000 | 4,000 | 4,500 | 5,000 | 6,000 |
|---|---|---|---|---|---|---|---|---|
| village | 64 | 71 | 65 | 67 | 86 | **92** | 73 | 65 |
| static | 39 | 37 | 35 | 34 | 34 | 33 | 33 | 32 |

`sim.economy` sizes the island at **88 sheltered / 59 exposed**. The village
climbs to 92, past the sheltered ceiling, and falls back to 65 — a Malthusian
overshoot, with 134.8 deaths against 159.6 births. Nothing in the arbiter models
a carrying capacity; the ceiling is arithmetic and the population finds it by
dying at it.

Read the static column as the control it is: a fixed 40 decays gently to 32 and
does nothing else. Every one of the three other 3.0 mechanics is *unreachable*
without reproduction — 0.8 rooms and 0 fields, because a household of two never
fills a house and never gets hungry enough to invent farming.

## R3 — a full house gets a room built onto it, and the beds cost more than they buy. MIXED.

`village.yaml` against `village_nohousing.yaml` (the 2.0 rule: a shelter radius
protects everyone standing in it), 5 paired episodes of 6,000 ticks.

**The mechanic works.** 58.4 rooms are built per episode against 0.0 in the
control, and the capacity genuinely binds: **2,446 agent-ticks a night are spent
outside for want of a BED** rather than for want of a house.

**And it does not pay for itself.**

| | with beds | uncapped | paired |
|---|---|---|---|
| nights indoors | 84.6% | 86.5% | **−1.88 ± 1.01 (1/5)** |
| population at the end | 64.8 | 67.8 | −3.0 ± 3.8 (2/5) |
| mean lifespan | 2181 | 2252 | −71 ± 33 (1/5) |
| rooms added | **58.4** | 0.0 | +58.4 ± 3.1 (5/5) |
| tribe size Gini | 0.27 | 0.18 | +0.09 ± 0.06 (3/5) |

So occupancy is a **net tax**: the village answers it (58 rooms) and still sleeps
outside slightly more often and lives slightly less long. The population effect is
inside noise; the two smaller ones are not, and both point the same way.

**One measurement caveat, stated because it would otherwise read as a result:**
`bed_denied` and `roofless` are only computed inside the housing branch, so the
control's 0.0 on both is an artefact of instrumentation and not a claim that
nobody in it slept out. The comparable number across the two worlds is
`nights indoors`, and that is the row above.

**What it did buy is inequality.** Tribe size Gini rises 0.18 → 0.27 (+0.09 ±
0.06, 3/5 — weak, and the spread crosses zero at two SE). The direction is what a
bed limit should do: a tribe whose households get their rooms up breeds, and one
that does not, does not.

## R1b — at village scale the forest is not an improvement, it is the precondition.

The same key again (`village_noregrow.yaml`), but in the world where the
population is an outcome. 5 paired episodes of 6,000 ticks:

| | forest regrows | dead forest | paired |
|---|---|---|---|
| population at the end | 64.8 | 23.2 | **+41.6 ± 0.9 (5/5)** |
| births | 159.6 | 47.4 | +112.2 ± 2.5 (5/5) |
| nights indoors | 84.6% | 32.6% | +52.0 ± 3.9 (5/5) |
| shelters completed | 490 | 112 | +378 ± 45 (5/5) |
| rooms added | 58.4 | 24.0 | +34.4 ± 2.8 (5/5) |
| **fields planted** | 32.2 | **2.6** | +29.6 ± 1.8 (5/5) |
| farming households | 16.4 | 6.0 | +10.4 ± 1.9 (5/5) |

Read the last two rows twice. **A dead forest cannot afford agriculture either**,
because a field costs a unit of material to put in and the material is gone. The
three 3.0 mechanics are not independent: births need beds, beds need rooms, rooms
need wood, and so does the field that would have fed the children. On a 2.0
island every one of those chains terminates at tick ~2,000.

## R4 — a village invents farming when it outgrows its food. CONFIRMED, and the first control failed in the most interesting way.

**The pre-registered control did not work, and why is a finding.**
`village_fed.yaml` doubles the berry economy (carrying capacity 176 against 88).
Prediction: farming should be invented far less. Measured, it is invented **just
as much** — 16.4 farming households in the scarce world against 14.4 in the fed
one (+2.0 ± 1.1, 3/5), i.e. nothing.

Because the population is an outcome. Fed twice as well, the village simply
breeds to 144 and its households are crowded and hungry again:

| tick | 1,000 | 2,000 | 3,000 | 4,000 | 6,000 |
|---|---|---|---|---|---|
| village | 71 | 65 | 67 | 86 | 65 |
| **twice the food** | 79 | 121 | **144** | 129 | **117** |

**You cannot feed a village out of scarcity when its population is an outcome.**
That is the Malthusian mechanism eating its own control, and it is worth more
than the reading it spoiled.

**The control that works caps the population below the food.**
`village_fed_capped.yaml` is the fed island with 60 slots on a 176-agent
carrying capacity, so the village cannot crowd itself. One key, and it settles at
50. 5 paired episodes:

| | hungry village | fed and capped | paired |
|---|---|---|---|
| **fields planted** | **32.2** | **0.0** | +32.2 ± 1.2 (5/5) |
| **farming households (of 20)** | **16.4** | **0.2** | +16.2 ± 0.6 (5/5) |
| first household to invent it | **tick 1,487, in 5/5 episodes** | **tick 5,588, in 1/5** | — |
| berries off fields | 830 of 6,914 (12%) | 0 | +830 ± 77 (5/5) |
| population at the end | 64.8 | 50.2 | +14.6 ± 1.6 (5/5) |

**Farming is invented in a village that has outgrown its food, and essentially
never in one that cannot.** One household of a hundred episode-households
invented it in the fed world, five sixths of the way through the run.

**The honest cost, unchanged and load-bearing:** the unlock RULE is authored —
`unlock_hunger_ticks` agent-ticks below `unlock_hunger`, per household — so
"necessity is the mother of invention" is a sentence in `world.py`. What the
control establishes is a fact about the world rather than about the rule: hunger
is common in one of these islands and rare in the other, the same rule fires in
one and not the other, and the field then supplies 12% of everything the village
eats. Whether agents would *choose* to farm is a question only a learned chooser
can answer, exactly as it is for the axe.

## R5 — tribes cut fighting nearly in half, and do NOT produce a winner. PARTLY REFUTED.

`village_notribes.yaml`: households, stockpiles, steals, raids and grudges all
unchanged, with the tribe layer off. 5 paired episodes of 6,000 ticks.

**What tribes do to the fighting is large and one-directional.**

| raids per episode | with tribes | no tribes |
|---|---|---|
| across a tribe border | **796** | — (there is one tribe) |
| within a tribe | 1,038 | 4,306 |
| **total** | **1,834** | **4,306** |

A wider circle of people you do not rob cuts raiding by **57%**, and 43% of what
survives crosses a border. The mechanism is indirect and worth stating: tribe
immunity blocks *steals* within a tribe, which produces fewer grudges, and the
raid goal is gated on desperation-or-grudge — so most of the raiding that
disappears is vengeance that was never provoked.

**And it costs the island.** Population 64.8 against 77.2 (−12.4 ± 7.3, 1/5 —
weak, the spread crosses zero), mean lifespan −128.7 ± 32.3 (0/5 — real).
Solidarity is not free here.

**The "strongest tribe" claim is where this has to be careful, and the obvious
number is a trap.** Tribe-size Gini reads 0.27 with tribes and 0.00 without —
which is not a result at all, because a world with one tribe has a Gini of zero
by construction. Rule 6, in a fresh disguise.

The control that works partitions households into the **same four angular
quarters whether or not tribes are switched on**, and counts the living in each:

| | with tribes | no tribes (same quarters) | paired |
|---|---|---|---|
| quarter-population Gini | 0.27 | **0.23** | **+0.04 ± 0.03 (4/5)** |
| episode 0 | 17 / 29 / 16 / 6 | 17 / 27 / 34 / 18 | — |

**Most of the inequality is the geography, not the tribe.** Quarters of this
island differ in bushes, sites and material whether or not anyone calls them
tribes, and that alone produces a Gini of 0.23. Making them social groups adds
0.04 ± 0.03 — the right sign, about 1.3 SE, and not something to call a result.

So the pre-registered read "the strongest tribe ends up biggest" is **not
supported**. Tribes differentiate, and so does the ground under them by nearly
as much. What tribes demonstrably do is change who fights whom.

## R6 — old age, and the generational horizon that is an array bound

Not pre-registered; run because "who survives and reproduces" is only a selection
question once a founder can be replaced rather than simply accumulating.

**The first run measured an extinction that was not mortality.**
`village_mortal.yaml` (`max_age: 4000`) on the 200-slot village goes to **zero by
tick 12,000**. The cause is in the engine's own contract, not in the mechanic:
**`world.num_agents` is a slot capacity and a slot is used once.** A dead agent
keeps its row forever, which is what makes every per-agent statistic in this repo
mean something — and it puts a hard ceiling on how many lives an episode can
contain. 40 founders plus 160 births exhausts 200 slots, and after that nobody
can be born while `max_age` keeps killing.

**Re-run against the same slot budget, the effect of old age is real and much
smaller.** `village_mortal` against `village_long` (both 300 slots), 2 paired
episodes of 10,000 ticks:

| tick | 4,000 | 6,000 | 7,000 | 8,000 | 9,000 | 10,000 |
|---|---|---|---|---|---|---|
| with old age | 80 | 88 | **100** | 84 | 54 | **38** |
| no old age | 94 | 96 | **115** | 92 | 71 | **66** |

Old age costs 91 deaths and 28 of the final population (−28.0 ± 5.0, 0/2). But
**both columns fall after tick 7,000**, and both worlds report `born: 300` of 300
— they are both up against the slot cap, and the shared late decline is the
array, not the island.

**So the honest statement is that Island 3.0 has a generational horizon, and it
is a property of the implementation.** Every result above is measured inside it
(6,000 ticks, ≤200 of 200 slots used at the end), which invalidates none of them
— they are paired inside that window against their own controls, exactly as
ISLAND2_DESIGN.md §15's window works. What it does mean is that this stage has
not measured a village sustaining itself across generations, and cannot until
slots are recycled. See "What is left".

---

# Corrections this stage forced

Each is pinned by a test named after the symptom, in the style the previous two
stages use. Four of the six are the same shape: **a denominator that stopped
meaning what it meant**, which is CLAUDE.md rule 6 arriving in a world where the
population is an outcome.

1. **The founders were born as infants.** `initial_agents` set `born` and `alive`
   and left `age` at 0, so the whole starting population was children for
   `maturity_ticks`. Nobody could chop, build or plant for the first 200 ticks of
   every episode, so no house was ever finished — and because a birth needs a
   bed, nobody could be born either. A world that starts with a generation of
   infants and no parents. Found by a masked `plant` in a unit test, not by a run.
2. **Capping an unfinished site deleted `partial_shelter`.** `site_capacity`
   returns 0 for an incomplete house, because a birth needs a real bed in a real
   house — and using that same 0 at night meant a half-built wall sheltered
   nobody, which is the M4 cliff put straight back (`partial_shelter` exists
   because three quarters of a build bought nothing). The two uses now compute
   separately. This is rule 5 catching a mechanic before the run, which is the
   only time it is cheap.
3. **Every per-agent statistic had to divide by the BORN.** With 160 unborn rows
   carrying a lifespan of 0, a mean over slots reports a thriving village as
   nearly dead. Lifespan, deaths, the household Gini and `sim.society`'s headline
   all changed denominator; every 2.0 world has `born` all-True, so none of them
   moved.
4. **The goal histogram was counting agents that did not exist** — and this one
   is a correction to numbers this repo has already published. `OptionRunner`
   accumulated `goal_ticks` over every row including the dead. At 2.0's death
   rates that is a small bias; in a 200-slot village with 40 alive it put **80% of
   all goal-ticks into `explore`** and reported it as what the population wanted.
   It now counts the living only ("can I move" is an exact aliveness test off the
   mask). **Every goal share in ISLAND2_DESIGN.md is very slightly overstated for
   whichever goal dead agents defaulted to; the effect is small (society4 loses
   ~12 of 100 agents by tick 600) and no comparison in that document is between
   worlds with different death rates.**
5. **`expand` keyed on overflow could never fire.** A birth needs a free bed, so a
   household stops AT capacity and can only exceed it if a storm takes the roof
   off. The need now rises as the beds fill and saturates when the last one goes.
6. **The replay header emitted `Infinity`.** Unplanted field slots are parked at
   infinity so the engine treats them as absent; the header was built from the
   whole bush array, `json.dump` wrote the literal `Infinity`, and `JSON.parse`
   rejects it — so every village replay was unloadable in a browser. Wild bushes
   only, and the test now parses its own output strictly.

Two more that are sizing rather than code, both `sim.economy` doing the job rule
5 gives it, before any run:

* **Three fields per household would have deleted the world.** The first sizing
  added **168 agents** of carrying capacity to an island whose wild ceiling is 88
  — agriculture would have solved everything and left nothing to fight over. Two
  fields at a 60-tick clock adds ~33.
* **The stock ratios stop meaning anything.** `supply / demand` divides by
  `num_agents`, which in a 3.0 world is a slot capacity, so a comfortably-fed
  village reads as a famine and the tool's WARNING would have been a lie. It now
  prints a carrying capacity instead and says which number to read.

---

# What is left

**1. Slot reuse, which is the only thing between this and a generational world.**
R6 is bounded by an array, not by an island. Recycling a dead agent's row would
let a village run indefinitely — and the cost is exactly why it has not been done
casually: `alive_ticks` is per slot and would sum two lives, the replay's agent
ids would revive mid-episode, and per-agent counters (`night_sheltered_agent`,
`attacks_per_agent`, the grudge matrix's row and column) all assume one life per
row. Done properly it needs a per-life ledger and a `generation` column, and the
replay needs to say a row changed occupant. It is a day's careful work and it
unlocks the one question this stage set out to ask and did not answer.

**2. A learned arbiter in the village, which is the first place one would know
something the scripted scorer cannot score.** The observation carries
`home.beds_free`, `home.overflow`, `home.expandable`, `own.farming`,
`home.field_room`, `own.age`, `own.adult` and a per-neighbour `same_tribe` flag.
The scripted arbiter reads all of them, so that is not yet the asymmetry the
predator rung created. **The asymmetry is that a birth is not a goal**: no scorer
anywhere decides to have a child, so a learned chooser optimising its own
survival has no term for "my household's next generation" — and a policy trained
on a HOUSEHOLD reward (`arb5-hh` already exists) does. Whether kin-shared reward
produces agents that go home well-fed and stand still is a question this world
can ask and 2.0 could not.

**3. The construction wall, offered a recurring prize for the first time.**
ISLAND2_DESIGN.md §10 spent three levers on the fact that learned agents will not
contribute to construction: `deliver` sat on their menu at 3.9% of decision points
and was taken zero times of ~2,700. In 2.0 the prize was one-off — twenty sites,
finished before the first nightfall. Here a house is never finished: storms take
it down, the family grows, and `expand` is available for the whole episode. That
does not make the credit chain shorter, and nothing above suggests it will work.
It does make it a genuinely different question, and it costs one training run.

**4. Do NOT re-run.** R1 at 6,000 and 24,000 ticks; R1b; R2; R3; R4 with the
`village_fed` control (it fails for the reason above) and with
`village_fed_capped` (it works); R5 with the geographic-quarter control; R6 at
200 slots (extinction is the array) and at 300 (both arms slot-bound).
