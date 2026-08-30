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
