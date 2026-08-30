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

---

# Stage 2 — generations, heredity, and technology that spreads

Written 2026-08-30, before implementation, in the same shape as the first stage's
criteria. It answers queue item 3.0-1 (slot reuse) and adds the three mechanics
that turn a village that *lasts* into one that can *change*.

## Why these four and not more

R6 left this stage with one hard blocker and one soft one. The hard blocker is
the array: a slot is used once, so no village here has ever run past its
founders' great-grandchildren. The soft one is that **nothing in Island 3.0
changes over a run except the numbers**. Households invent farming, build rooms,
raid each other — and a household in generation four behaves exactly as its
founders did, because its traits were drawn at reset and its technology arrives
by a private clock. There is no channel through which anything can be inherited,
taught, or selected for, so there is nothing for a long run to be long *for*.

1. **Slot reuse** removes the horizon. A dead agent's row is recycled, with
   per-LIFE bookkeeping so no statistic silently sums two lives.
2. **Heritable traits** give selection something to act on. A child's arbiter
   trait vector is the geometric mean of its parents' with lognormal mutation,
   so a household that breeds passes on how it behaves. This is the one mechanic
   here that can produce behaviour nobody wrote: **the population's mean
   preferences at tick 20,000 are an outcome, not a config value.**
3. **Technology spreads by contact.** A household adjacent to one that farms
   learns to farm. Invention stays hunger-driven; adoption becomes social, which
   is how technology actually moves, and it makes the tech front a thing you can
   watch cross the island.
4. **A technology with a prerequisite.** The granary needs farming *and* a full
   larder, and doubles the household's food store. It exists to make the ladder a
   ladder: the first thing in this project that cannot be reached at all until
   something else has been.

## The honest costs, stated before the run

* **Selection acts on the arbiter's TRAIT VECTOR, not on its behaviour rules.**
  A trait is a multiplier on a goal's score. So what can evolve is *how much a
  lineage wants* each of the 18 goals — never a new goal, never a new way of
  pursuing one. "The village evolved a strategy" would be a claim about
  weightings inside a scorer somebody wrote, and every write-up has to say so.
* **Both new unlock rules are still authored**, exactly as farming's was. What
  the controls can establish is whether contact spreads a technology *faster than
  independent invention would*, and whether a prerequisite actually gates.
* **A reused row is a different person in the same slot.** Per-agent counters
  that mean "this life" are reset (age, lifespan, the grudge row and column);
  ones that mean "this row's contribution to the episode" are not (berries,
  builds). The replay says when a row changed occupant. Anything that reads a
  per-agent array across a reuse boundary without checking is wrong, and the
  test suite pins the ones that exist.

## Pre-registered acceptance criteria

**Correctness**

* **C1** — the five 2.0 checksums AND six new Island 3.0 checksums (taken from
  the previous commit) reproduce exactly with every stage-2 block off.
* **C2** — full suite green, new tests named after what they protect.
* **C3** — the granary changes a household's food capacity, so `sim.economy` is
  told about it before any world is sized (rule 5).

**Behaviour** — each paired on the same seed block, one config key apart.

* **S1 — slot reuse removes the generational horizon.** `village_mortal` with
  reuse against the same world without it, 20,000 ticks. Prediction: the no-reuse
  arm reaches `born == num_agents` and decays toward zero; the reuse arm sustains
  a population near the carrying capacity. *Failure mode:* if the reuse arm also
  decays, the horizon was never the array and R6's diagnosis was wrong.
* **S2 — heredity produces selection.** Heritable traits against a control where
  a newborn gets a fresh draw from the founding distribution. Reads: the
  population's mean trait per goal at the end against 1.0 (the founding median);
  whether the DIRECTION of the largest drifts is consistent across seeds; and
  population/lifespan. *Failure mode:* drift that is large but seed-inconsistent
  is drift, not selection, and must be reported as such.
* **S3 — technology spreads by contact.** Teaching on against off. Reads: how
  many households acquired farming by being TAUGHT against by INVENTING; ticks to
  full adoption. *Failure mode:* if adoption is already complete before teaching
  can act, the world is too hungry to test diffusion — read the invention count
  first.
* **S4 — a prerequisite gates.** Granaries appear only in farming households (by
  construction, so the real reads are *when* and *how many*), and granary
  households hold more food and produce more children than their neighbours on
  the same island.
* **S5 — the payoff read: do tribes differentiate across generations?** With
  reuse and mortality over 20,000 ticks, tribe population share against the
  geographic-quarter control that killed the first version of this claim. *This
  is the question the whole stage exists for*, and the first-stage answer
  (+0.04 ± 0.03, i.e. nothing) is the number to beat.

---

# Stage 2 — results

Same discipline as stage 1: scripted arbiter, paired seed blocks, one config key
between each arm and its control.

## S1 — slot reuse removes the horizon, and R6 was right about what was killing them. CONFIRMED.

`village_gen.yaml` against `village_gen_noreuse.yaml`, 3 paired episodes of
**20,000 ticks**, everything identical but whether a dead agent's row can be
lived in again.

| tick | 1,000 | 5,000 | 7,000 | 9,000 | 13,000 | 19,000 |
|---|---|---|---|---|---|---|
| rows recycled | 60 | 76 | 96 | 84 | 85 | **76** |
| rows used once | 60 | 76 | 54 | 13 | — | **extinct** |

| | reuse | no reuse | paired |
|---|---|---|---|
| **alive at tick 20,000** | **82.7** | **0.0** | +82.7 ± 7.5 (3/3) |
| lives lived (in 200 rows) | **605.7** | 200.0 | +405.7 ± 30.9 (3/3) |
| rows lived in twice or more | 405.7 | 0 | — |
| berries gathered | 24,283 | 8,802 | +15,481 ± 1,441 (3/3) |
| deaths of old age | 236.7 | 120.3 | +116.3 ± 16.3 (3/3) |

**200 rows now hold 606 lives, and the village is still there at tick 20,000
where the same village without reuse is gone by 11,000.** R6's diagnosis is
confirmed exactly: nothing was wrong with the island, the array was full.

*Two caveats that ride with it.* `mean_lifespan` is 2,508 against 2,678 — lower
with reuse, because the reuse arm keeps producing newborns whose lives are still
running when the episode ends, and the no-reuse arm's last agents simply grow
old. And the tribe-Gini row in this comparison reads 0.27 against 0.00, which is
not a result: the control has nobody left to be unequal.

**A second cause of R6's extinction, found by a test rather than a run:** every
founder was created at the same age, so with `max_age` the entire first
generation died on the *same tick*. `stagger_founders` spreads them over
[maturity, max_age); it is off by default, so R6's own world still reproduces,
and it is on in every stage-2 world.

## S3 — technology spreads by contact, and teaching beats inventing. CONFIRMED.

`village_gen.yaml` against `village_gen_noteach.yaml`, 5 paired episodes of
8,000 ticks. One key: whether a household can learn a technology from a
neighbour who has it.

| | teaching on | teaching off | paired |
|---|---|---|---|
| **farming households (of 20)** | **14.2** | **10.8** | +3.4 ± 0.8 (5/5) |
| ...invented it | 4.6 | 10.8 | −6.2 ± 0.8 (0/5) |
| ...**were taught it** | **9.6** | **0.0** | +9.6 ± 1.2 (5/5) |
| **granary households** | **13.0** | **6.4** | **+6.6 ± 0.8 (5/5)** |
| ...invented / taught | 4.0 / 9.0 | 6.4 / 0.0 | — |
| fields planted | 27.2 | 20.8 | +6.4 ± 1.5 (5/5) |
| berries off fields | 1,465 | 970 | +495 ± 146 (5/5) |
| median household's adoption tick | 2,744 | 3,367 | −623 ± 280 (0/5) |

**Two thirds of the farming households never invented anything — they were
taught.** And note the mechanism in the row that goes the *other* way:
independent invention falls from 10.8 to 4.6, because a household taught before
it gets desperate never has to invent. That is diffusion doing exactly what
diffusion does, and it is the reason `_tech_source` exists: a single has-it flag
would have shown 14.2 against 10.8 and said nothing about why.

**The second technology is where it matters most.** A granary needs farming
*and* a full larder, so its adoption is gated twice — and teaching **doubles**
it (13.0 against 6.4, 5/5). The deeper a ladder gets, the more it depends on
being taught rather than rediscovered, which is the realistic result and not one
this world was tuned to produce.

**What diffusion does not buy is population.** +2.0 ± 3.9 agents (2/5), inside
noise, over 8,000 ticks. The technology spreads; the island is still the island.

## S2 — heredity produces persistent lineages, and the direction is suggestive. CONFIRMED, with a stated limit.

`village_gen.yaml` against `village_gen_nohered.yaml` — a newborn gets a fresh
draw from the founding distribution instead of its parents' traits, and
everything else, mutation included, is identical. **10 paired episodes** of 8,000
ticks, because 5 was not enough to test a direction.

| | heredity | fresh draw | paired |
|---|---|---|---|
| **mean absolute trait drift** | **0.08** | **0.05** | **+0.04 ± 0.01 (10/10)** |
| population at the end | 86.6 | 85.1 | +1.5 ± 5.7 (6/10) |
| agents turned away from a full house | 2,586 | 1,880 | +707 ± 302 (9/10) |
| quarter-population Gini | 0.28 | 0.21 | +0.06 ± 0.04 (7/10) |

**The magnitude is decisive: heritable variation produces 1.6× the drift the same
world produces with fresh draws, on every one of ten seeds.** That is what
inheritance is *for* — a lineage's preferences persist instead of being
re-rolled — and it is the only number here that is outside noise by a wide
margin.

**The direction is interpretable and not statistically established.** The largest
drifts, with the share of seeds agreeing on their sign:

| | drift | seeds agreeing |
|---|---|---|
| `explore` | **−10%** | 8/10 |
| `forage` | **+9%** | 8/10 |
| `raid` | **−8%** | 9/10 |
| `draw_food` | **+7%** | 8/10 |
| `steal` | **−6%** | 7/10 |
| the control's largest | ±3% | similar rates |

Four of the five point the same way as a fitness story would: **the village
drifts toward feeding itself (forage, draw_food) and away from wandering and
fighting (explore, raid, steal)** — and stage 1's R5 measured that raiding costs
lifespan, so "away from raiding" is the direction survival should favour. At
20,000 ticks (the S1 arm, 3 seeds) the same five move further and agree 100%.

**The honest limit, and it is why the control's row is in the table.** Ten seeds
give P(≥8 agreeing) ≈ 0.11 per goal under pure drift, and there are 18 goals, so
two such goals are expected by chance. The *agreement* is not significant on its
own. What is significant is that A's drifts are three times the size of B's on
the same island and the same seeds. **The claim this run supports is "heredity
makes lineages persist and the population's preferences move"; the claim it does
not yet support is "and they move because of selection rather than drift."**
Settling that needs many more seeds or an explicit fitness measurement, and it is
in "What is left".

**A five-seed version of this run reported the quarter Gini at +0.14 ± 0.04
(5/5); at ten seeds it is +0.06 ± 0.04 (7/10).** The first number was optimistic
and is retracted — which is exactly why S5 below is run at its own length rather
than read off this table.

## S5 — tribes still do not produce a winner, now tested across generations. REFUTED again.

`village_gen.yaml` against `village_gen_notribes.yaml`, 5 paired episodes of
**12,000 ticks** — long enough for four or five generations to turn over, which
is the condition stage 1's version of this read did not have.

| | tribes | no tribes | paired |
|---|---|---|---|
| **quarter-population Gini** | **0.31** | **0.27** | **+0.03 ± 0.08 (3/5)** |
| raids across a border | 1,116 | — | — |
| raids within a group | 611 | 5,968 | −5,357 ± 1,337 (0/5) |
| **total raids** | **1,727** | **5,968** | **−71%** |
| population at the end | 80.6 | 87.6 | −7.0 ± 7.5 (2/5) |
| mean lifespan | 2,301 | 2,396 | −95 ± 52 (1/5) |
| rooms added | 62.6 | 69.8 | −7.2 ± 1.9 (0/5) |

**+0.03 ± 0.08 is nothing** — 0.4 of a standard error, and the sign flips on two
of five islands. Making a spatial group a social group still does not make the
island's quarters diverge, and now that has been tested with lineages persisting
across 350 births rather than 160. Worth noting the episode-0 detail, which
points the other way from the hypothesis: with tribes the four quarters held
17/18/16/11 and *without* them the same quarters held 19/31/29/6. **If anything a
tribe equalises the ground it sits on**, because the people you may not rob are
the people next to you.

What tribes do, they do to the fighting, and at generational length it is larger
than stage 1 measured: **raiding falls 71%**. The cost is a little of everything —
7 fewer people, 95 ticks of life, 7 fewer rooms — with only the rooms outside
noise.

**So the answer to "who will survive and reproduce and be the strongest tribe" is
that nobody does, and the reason is geographic rather than social.** A quarter of
this island is unequal to the others by about 0.27 of a Gini whatever you call
its households, and belonging to a tribe adds nothing measurable on top.

## S4 — a technology with a prerequisite, and it pays. CONFIRMED, after the first pair was thrown away.

**The first version of this read had no control and is discarded.**
`village_seasons.yaml` against `village.yaml` differs in **two** keys — the shock
ramp and the granary — so it could only say that a seasons world is harder
(population 17.6 against 64.8), which is a fact about the ramp. It also extends
`village.yaml` rather than `village_gen.yaml`, so it has neither teaching nor
slot reuse: the granary could appear and never spread, and the village could not
outlive its array. Both are why `village_gen_seasons.yaml` exists.

`village_gen_seasons.yaml` against `village_gen_seasons_notech.yaml` — one key,
the granary — 5 paired episodes of 10,000 ticks:

| | granary | none | paired |
|---|---|---|---|
| **alive at tick 10,000** | **34.4** | **23.8** | **+10.6 ± 3.1 (5/5)** |
| births | 193.0 | 179.4 | +13.6 ± 5.6 (4/5) |
| berries gathered | 8,003 | 7,706 | +297 ± 154 (4/5) |
| **granary households (of 20)** | **9.8** | 0 | 3.4 invented, **6.4 taught** |
| farming households | 15.2 | 15.8 | −0.6 ± 0.8 (1/5) |
| mean lifespan | 2,106 | 2,176 | −69 ± 12 (0/5) |

**A village with granaries ends the hard season 44% larger.** The mechanism is
visible in the trace, and it is the one the sizing predicted: the two arms track
each other exactly until tick ~6,000 and separate only as the ramped blights get
long enough to matter.

| tick | 2,000 | 4,000 | 6,000 | 8,000 | 10,000 |
|---|---|---|---|---|---|
| granary | 57 | 51 | 42 | **53** | **34** |
| none | 57 | 53 | 42 | 44 | 24 |

`sim.economy` sized it before the run: a full larder feeds a full house 56 ticks,
112 with a granary, against a worst blight of 120. Neither bridges the season
fully — the granary buys twice as far into it, and twice as far is worth ten
people.

**The prerequisite is doing real work, not decoration.** Farming households are
the same in both arms (15.2 against 15.8), so the granary is not smuggling in
extra agriculture: it is a second rung reached from the first, and **two thirds
of the households that have it were taught rather than invented it**. Read that
with S3: the deeper the ladder, the more it depends on being taught.

*One number that goes the other way and is not a regression:* mean lifespan is 69
ticks lower with granaries. More children are born and their lives are still
running when the episode ends — the same bound that makes mean lifespan useless
as a survival statistic in any world that grows.

---

# Stage 2 — corrections, and what is left

## Corrections

Each is pinned by a test named after it.

1. **Every founder was the same age**, so with `max_age` the whole first
   generation died on the same tick — a synchronised die-off no real population
   has, and **a second, unnamed cause of R6's extinction**. `stagger_founders`
   spreads them over [maturity, max_age); off by default, so R6's own world still
   reproduces exactly.
2. **The viewer drew every UNBORN row as a corpse** at its spawn point — up to
   160 of them from the first frame — because death was derived from a row's
   FIRST `alive: 0`, and an unborn row is dead from tick 0. With reuse it would
   also have drawn a reused row as a corpse that walks. Death now comes from
   every stretch a row was alive.
3. **A reused row must not sum two lifespans.** `alive_ticks` is per row; a
   finished life is pushed to a ledger and the counter zeroed, so `mean_lifespan`
   and `born` are over LIVES. Getting this wrong would have understated every
   stage-2 lifespan by roughly the reuse rate.
4. **A newborn owes nothing and is owed nothing** — the grudge matrix's row AND
   column are cleared. Clearing only the row would leave the village avenging a
   robbery on a child.
5. **A runner given no world raises** rather than silently handing newborns their
   row's founding traits. That failure would have produced a clean null on the
   whole selection read with nothing in the output to explain it.
6. **The blend is geometric.** Traits are lognormal about 1.0, so an arithmetic
   mean of two parents is biased upward and the population would climb every
   generation with no selection at all — and it would look exactly like evolution.
7. **`sim.economy` sized the store against the wrong household.** Dividing the
   larder by a FOUNDING pair reported 420 ticks of cover against a 60-tick
   blight, so a granary could never have mattered and S4 would have measured
   noise. Against a full house it is 56 — and that is what `village_seasons`
   exists for.
8. **S4's first pair had no control.** `village_seasons` against `village`
   differs in two keys (the ramp and the granary) and in two more by inheritance
   (no teaching, no reuse). Discarded and re-run one key apart.

## What is left

**1. Selection versus drift.** S2 establishes that heredity makes lineages
persist (1.6× the trait drift, 10/10 seeds). It does not establish that the
*direction* is selection: ten seeds give ~0.11 per goal under a drift null and
there are 18 goals. The cheap discriminator is a fitness measurement rather than
more seeds — regress a lineage's realised number of descendants on its trait
vector, which the pedigree already in `world.last_births` makes possible without
a single new run.

**2. Households never split.** A family grows to its house's capacity and stops;
twenty households is twenty households forever, because a household owns the site
of its own index and there are no spare sites. Village fission — a grown
household founding a new site — is what would let a successful lineage actually
*expand territorially*, and it is the most likely reason S5 keeps coming back
null: a tribe cannot win ground it has no way to occupy.

**3. A learned arbiter, and now there are two things it would know that the
scorer does not.** A birth is not a goal (stage 1's item), and neither is
teaching: nothing in `RESTORE` values standing next to somebody who knows
something. A chooser trained on a household or lineage reward has a term for both.

**4. Do NOT re-run.** S1 (20,000 ticks, extinct control), S2 at 5 seeds (the
quarter-Gini number there is retracted — use the 10-seed one) and at 10, S3, S4's
first pair (no control) and S4b, S5 at 12,000 ticks.
