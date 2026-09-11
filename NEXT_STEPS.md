# Next steps — queue for future sessions

Written 2026-08-25, at the end of the session that ran the mixed population
(`arb5-mix`), the household baseline (`arb5-hh`) and the seasons world's first
run. Each item below is one session or less, ordered cheapest-first within its
thread. Every item carries THE EXACT PROMPT to paste into a fresh session --
the prompts assume the session starts cold with CLAUDE.md and ISLAND2_DESIGN.md
in context, which they always are.

The standing rules travel with every item: pre-register the reads before the
run, size any economy change with `sim.economy` first (rule 5), pair on the
same seed block (rule 7), and 150-200 updates is always enough (rule 4).

Do NOT re-run: arb4/b/c/d (all four all-learned variants), arb5-mix, arb5-hh,
the mixedrandom floor, anything on the §10 do-not-re-run list, the learned-share
sweep (item 3), or the axe (item 4).

## ISLAND 4.0 IS BUILT (2026-08-30). Read this queue first.

Growth, conquest, terrain, culture and skill are in, all off by default, all
measured with the SCRIPTED arbiter. Read `ISLAND4_DESIGN.md` before touching
anything. Do NOT re-run S1 (growth) or S3 (the goal). The queue:

### 4-1. S2 at ten seeds (cheapest, and it is the one that matters)

The first tribe result that survives its control reads **village Gini
+0.09 +- 0.04, 4/5** -- about 2.2 SE on five seeds. Three previous attempts came
back at +0.04, +0.03 and -0.02, so this is the first positive one and it is not
yet settled. Ten paired seeds of `empire.yaml` vs `empire_nocon.yaml`, 8,000
ticks, nothing else changed. ~25 minutes.

Pre-register: if it holds above 2 SE at ten seeds, "a tribe can take ground"
is earned. If it falls inside noise, S2 joins the other three refutations and
the honest headline becomes "conquest fires 19.6 times an episode and still does
not make a winner", which is a result too.

### 4-2. Culture and skill need paired controls

Both currently rest on ONE seed and one episode (ISLAND4_DESIGN.md §5, §6). The
culture read refutes its own pre-registered direction, which is enough to change
the story and not enough to quote; the skill read's "all 111 agents are best at
foraging" is qualitative enough to stand, its correlations are not. Five paired
seeds each, one key on top of `empire.yaml`. ~20 minutes for both.

### 4-3. Something that pushes a village average somewhere

§5's conclusion: averaging cannot make cultures diverge, and this world has no
selection pressure that differs BY PLACE. Terrain (§4) is now the obvious
carrier -- a village on poor ground should be selected toward different behaviour
than one on rich ground -- and nothing connects the two. This is a design item,
not a run.

### 4-4. The learned chooser, but only once the mechanic set is frozen

Three things are now in the observation that no scripted goal reads:
`conquest.threat` (a siege coming), `own.skill_*` (my own competence) and the
whole conquest block. That is the largest gap between what a learned arbiter can
see and what the scripted one uses that this project has ever had. **But 3.0's
lesson stands: `arb7-village` used 600-tick episodes and never reached
agriculture.** Episodes must be long enough to meet the mechanics, and
`world.grow_slots` is NOT supported under training (PPO's buffers are fixed
width) -- so a learned run here is a capped-array run, which is a one-key
difference from every number in ISLAND4_DESIGN.md and has to be said out loud.

## ISLAND 3.0 STAGE 2 IS BUILT (2026-08-30). Read this queue first.

Stage 2 answered the old item 3.0-1 (slot reuse) and added heredity, technology
diffusion and a technology with a prerequisite. Read `ISLAND3_DESIGN.md` (stage 2
sections) and CLAUDE.md's Island 3.0 blocks before touching anything. Do NOT
re-run S1, S2 (either seed count), S3, S4/S4b or S5.

### 3.1-1. Selection or drift? (cheapest, and it needs NO new runs)

**What**: regress a lineage's realised number of descendants on its trait vector,
using the pedigree `world.last_births` already records. **Why**: S2 establishes
that heredity makes lineages persist -- trait drift 0.08 against a fresh-draw
control's 0.05, +0.04 +- 0.01 on 10/10 seeds -- and does NOT establish that the
direction is selection. Ten seeds give ~0.11 per goal under a drift null and
there are 18 goals, so the 80-90% direction agreement is not significant. A
fitness regression settles it out of runs that already exist. **Reads**: the sign
and size of each goal's fitness coefficient, against the drift directions
(`explore` -10%, `forage` +9%, `raid` -8%, `draw_food` +7%, `steal` -6%). If they
agree, the drift IS selection and the write-ups can say so.

### 3.1-2. Village fission -- DONE 2026-08-30 (stage 3). Result: 20 households become 32 and the population reaches 107.2 against 85.2 without it -- but S7 refutes the tribe claim a THIRD time (quarter Gini 0.24 vs 0.25). Expansion is symmetric, so nobody pulls ahead; conquest is the missing ingredient. Original item below.

### 3.1-2-orig. Village fission (the most likely reason S5 keeps coming back null)

**What**: let a household at capacity found a NEW household at a new site.
**Why**: twenty households is twenty households forever -- a household owns the
site of its own index and there are no spare sites -- so a successful lineage
cannot expand territorially and "the strongest tribe" has no way to win ground.
S5 has now returned null twice, at 6,000 and 12,000 ticks, with the geographic
control saying most of the inequality is the ground. Fission is the mechanic that
would let a tribe actually take some. **Design care**: `num_sites` and
`num_households` are coupled through the stockpile-owns-its-site rule, and
`num_households <= num_sites` is enforced at reset; a new household needs a site
that does not exist yet, so this is a real change to the world's layout, not a
config knob. **Reads**: households at the end; quarter-population Gini against
the same geographic control (0.27-0.31 is the number to beat); tribe share.

### 3.1-3. A learned arbiter -- DONE 2026-08-30 (`arb7-village`). Result: learned slots +20.3 +- 3.6 (10/10) against a -28.9 floor, and a SPILLOVER of +21.1 +- 3.5 (10/10) -- the largest the project has measured, where arb5-mix's was nil. Construction still exactly 0.0% for the fifth time. The chooser never met agriculture or teaching, because a 600-tick episode is too short to unlock either. Original item below.

### 3.1-3-orig. A learned arbiter in the generational village

**What**: `sim.arbiter` on `config/island3/village_gen.yaml`, mixed, household
reward. **Why**: there are now TWO things a learned chooser would face that no
scorer scores -- a birth is not a goal, and neither is teaching (nothing in
`RESTORE` values standing next to somebody who knows something). **Reads**:
births and taught-technologies in learned households against scripted ones;
`expand` uptake, where any sustained nonzero is the construction wall finally
moving.

---

### 3.2-1. Conquest, the thing S7 has now shown is missing three times

**What**: let a raid take a SITE, not just a berry -- burn a house back to
incomplete, or claim an emptied household's site. **Why**: S5 (twice) and S7 all
return null on tribe inequality, and S7 names the reason: expansion is symmetric,
so every tribe grows at the rate its ground supports and the ratios never move.
Raiding moves food; nothing in this world moves TERRITORY. Until something does,
"the strongest tribe" is not expressible, however long the run.
**Reads**: quarter-population Gini against the same geographic control
(0.24-0.31 is the number to beat, three times over now); households per tribe;
whether a conquered site is rebuilt or abandoned. **Pre-registered failure
mode**: mutual destruction -- everyone burns everyone and the island loses its
housing stock, which would show as nights indoors collapsing while the Gini
stays flat.

### 3.2-2. Train on episodes long enough to contain a technology

**What**: `arb7-village` was trained and evaluated on 600-tick episodes, and
farming needs 300 cumulative hungry household-ticks -- so `0.0 of 20 households`
invented anything and the learned chooser never met agriculture, teaching or the
granary. **Why**: those were two of the three things a learned chooser was
supposed to know that no scorer scores. **Design care**: PPO needs episodes that
end, so this is a trainer question (truncation and bootstrapping at a longer
horizon), not a config change.

### 3.2-3. Per-env traits in the trainer, so heredity can be trained through

`ArbiterTrainer` holds ONE trait vector per agent row shared across all
environments, so heredity cannot be on during training. Making it per-env would
let the learned chooser be trained in a world where lineages persist -- and
would let selection act on a LEARNED policy's inputs, which is the closest this
project could get to evolving behaviour rather than weights.

---

## The stage-1 queue (item 1 is DONE; 2 and 3 are DONE too)

Read `ISLAND3_DESIGN.md` first, then CLAUDE.md's Island 3.0 section. Do NOT
re-run any of R1, R1b, R2, R3, R4 (either control), R5 or R6 -- the do-not-re-run
list is at the end of the design doc.

### 3.0-1. Slot reuse -- DONE 2026-08-30 (stage 2). Result: 200 rows now hold 606 lives, and a village that went extinct at tick 12,000 is at 82.7 agents at 20,000 (+82.7 +- 7.5, 3/3). R6's extinction was the array. Original item below for the record.

### 3.0-1-orig. Slot reuse (the biggest, ~1 session)

**What**: recycle a dead agent's row so a village can run past its generational
horizon. **Why**: R6 measured an extinction that was an ARRAY BOUND -- 40
founders plus 160 births exhausts 200 slots and nobody can be born again, so
nothing in 3.0 has measured a village sustaining itself across generations, which
is the whole question the stage set out to ask. **The cost, which is why it was
not done casually**: `alive_ticks` is per slot and would sum two lives; the
replay's agent ids would revive mid-episode; `night_sheltered_agent`,
`attacks_per_agent` and the grudge matrix's row AND column all assume one life
per row. It needs a per-life ledger, a `generation` column, and a replay that
says a row changed occupant. **Reads**: population trace over 40,000 ticks
against the same world without reuse; tribe population share at the end, which
is the first time "the strongest tribe" can mean a selection result.

    Build slot reuse from NEXT_STEPS.md item 3.0-1: recycle a dead agent's row
    for a new birth, behind a config flag so every existing result stays
    bit-identical (pin it with the ISLAND3 checksums). Keep per-LIFE lifespan
    bookkeeping (a ledger of completed lives, not a per-slot counter), give the
    replay a way to say a row changed occupant, and pin both with tests. Then
    run village_mortal for 40,000 ticks against the same world without reuse and
    read the population trace and the tribe share. Pre-register the failure mode:
    if the population still falls, the horizon was never the array.

### 3.0-2. A learned arbiter in the village (~1 session)

**What**: `sim.arbiter` on `config/island3/village.yaml`, mixed 20 learned among
the scripted rest, household reward (the `arb5-hh` recipe). **Why**: this is the
first world where a learned chooser faces something no scorer scores -- **a birth
is not a goal**, so nothing in `RESTORE` decides to have a child, and a policy
trained on its HOUSEHOLD's reward has a term for the next generation that a
selfish one does not. **Reads**: births in learned households against scripted
ones; store_food share (arb5-hh's provisioner was 8.8%); nights indoors; and
construction share, where any sustained nonzero `expand`/`deliver` is the
headline -- see 3.0-3.

### 3.0-3. The construction wall, offered a RECURRING prize (folds into 3.0-2)

**What**: read `expand` uptake off the same run. **Why**: ISLAND2_DESIGN.md §10
spent three levers on learned agents refusing to build, with `deliver` on their
menu at 3.9% of decision points and taken **zero times of ~2,700**. In 2.0 the
prize was one-off -- twenty sites, finished before the first nightfall. In the
village a house is never finished: storms take it down, the family grows, and
`expand` is available all episode. That does not shorten the credit chain and
nothing so far suggests it will work. It does make it a different question, and
it costs nothing on top of 3.0-2.

---

## The 2.0 queue (all done, kept for the record)

Items 1-6 were all done by 2026-08-29. What the two tech rungs left behind:

**Run the learned arbiter in the predator world.** Rung 1 (the axe) attacked the
COST of contributing and moved nothing, because cost was not what binds. Rung 2
(the predator) moves plenty -- deaths +3.4, lifespan -10.8 -- and the population
does not respond, because responding would need a need the scripted scorer does
not have. The predator's three observation channels ARE in the observation, so
**a learned chooser in that world is being offered a hazard the scripted one is
blind to.** That is the first rung where the learned arbiter knows something the
scripted arbiter does not, which makes it the first place a learned-over-scripted
win would mean something new rather than repeating arb5-mix.

    Train the mixed arbiter in the predator world: sim.arbiter --config
    config/island2/society4_predator.yaml --run-name arb6-pred --learn-agents 20
    --gamma 0.997 --updates 150, then evaluate paired vs utility on 10 episodes
    and against the size-matched mixedrandom floor. Pre-register: nights indoors
    of the learned 20 against the scripted 80's 84.5%, attacks per learned agent
    against attacks per scripted agent (the direct test of whether the channels
    are used at all), and learned-slot paired lifespan. Construction share is
    expected 0.0% again -- any nonzero is news.

The tempting alternative -- adding predator proximity to NEED_SAFETY -- would
certainly make the population hide, and would be scoring the response and then
reporting the score. Rule 1 in scripted clothing. Do not start there.

---

## 1. Provisioner in the seasons world — DONE 2026-08-25 (`arb5-ramp`)

Result: transfers but does not intensify (store_food 8.3% vs 8.8%, theft
11.2% vs 11.4%); piles fuller all season (8.63/8.62/6.69 vs 8.24/7.60/6.12,
deposits +25%); learned edge +5.0 +- 2.2 (6/10), compressed by the 599/600
ceiling, not a regression; construction 0.0% again. Do not re-run.
Write-up: ISLAND2_DESIGN.md §11. Original item kept below for the record.

## 1-orig. Provisioner in the seasons world (cheapest, ~30 min total)

**What**: train the mixed 20-learned setup in `society4_ramp.yaml` instead of
`society4.yaml`. **Why**: arb5-hh produced a food-banking provisioner in a
world where banking barely matters; the seasons world is the first where
banking early is arithmetically load-bearing. Does the provisioner bank
harder when winter is coming, and does the learned edge grow?
**Reads**: store_food share vs arb5-hh's 8.8%; stock level by episode third at
the learned households vs scripted-only households; learned-slot paired
lifespan vs all-scripted in the same world; construction share (expect 0.0% --
any nonzero is news).

**Prompt**:

    Run the seasons-world provisioner experiment from NEXT_STEPS.md item 1:
    train the mixed arbiter in the ramp world
    (.venv/bin/python -m sim.arbiter --config config/island2/society4_ramp.yaml
    --run-name arb5-ramp --learn-agents 20 --gamma 0.997 --updates 150
    --household-reward), then evaluate paired with sim.society --arbiter mixed
    --vs utility on 10 episodes, read against arb5-hh's numbers (store_food
    8.8%, nights 91.7%, +22.6 +- 7.0), write it up in ISLAND2_DESIGN.md and
    CLAUDE.md, and commit. Pre-register the reads before the run.

## 2. Persist-until-goal options — DONE 2026-08-25 (`arb5-persist`)

Result: the wall stands with the whole programme collapsed into one decision.
Learned 20 spent 0 of 120,000 goal-ticks on deliver/harvest, with `deliver` on
their menu at 3.9% of their own decision points and `harvest_wood` at 20.5% —
zero taken of ~2,700 chances (scripted: 33%/27%; a RANDOM minority contributes
2-3% and dies 37 ticks sooner). Learned-slot edge +18.2 +- 6.5 (7/10), level
with arb5-mix. Nights indoors 80.5% (small regression from 84.0%). Two
corrections the lever forced: the dusk curfew, and excluding explore/raid from
persistence. Do not re-run. Write-up: ISLAND2_DESIGN.md §10. All three levers
against the construction wall are now spent; what is left is mechanic-level
(make shelter excludable, or make contributing a single act with an immediate
personal return). Original item kept below for the record.

## 2-orig. Persist-until-goal options (the one untried move against the building wall, ~1 session)

**What**: an option that runs until its GOAL STATE (site complete, delivery
made) instead of a 25-tick budget, so a whole build programme is ONE semi-MDP
decision. **Why**: every level of abstraction tried (micro-actions, 25-tick
options, household-shared reward) refuses the compound construction programme;
this is the §10 lever that turns a compound prize into a single-trip prize,
the one shape PPO reliably takes. **Honest cost, unchanged**: what emerges is
when-to-build, never building. **Design care**: interruption (tier-0
emergency) must still terminate and write the transition -- the semi-MDP
bookkeeping is "the one place this can silently rot" (utility.py's warning).
**Reads**: construction shares of the learned 20 (any sustained nonzero
deliver/harvest is the headline); learned-slot paired lifespan; nights
indoors must not regress from 84-92%.

**Prompt**:

    Build persist-until-goal options from NEXT_STEPS.md item 2: extend
    OptionRunner and the arbiter trainer so a goal can commit until its goal
    state (site complete / delivery made / timeout as a backstop) instead of
    a fixed 25 ticks, gated behind an ArbiterConfig flag so every existing
    result stays bit-identical (pin it). Keep the tier-0 interruption rule and
    the one-transition-per-decision contract, pin both with tests. Then train
    the mixed 20-learned setup with it on society4.yaml (gamma 0.997, 150
    updates), evaluate paired vs utility and vs arb5-mix's numbers, write up,
    commit. Pre-register: any sustained nonzero deliver/harvest share from the
    learned 20 is the headline; the honest cost (when-to-build, never
    building) goes in every write-up.

## 3. The learned-share sweep — DONE 2026-08-26 (`arb5-mix40/60/80/90/100`)

Result: free-riding scales to 90% and the "threshold" does not exist. The
pre-registered prediction (collapse between 60 and 80 learned) is refuted —
completions/episode fall gently 59.3 / 58.2 / 57.6 / 52.7 / 48.1 / 41.6 at
0/20/40/60/80/90 learned with population lifespan FLAT at 582-589 (all at or
above the all-scripted 579.2), then collapse to 0.8 completions / 473.8 / 45.5
deaths / 0.35% nights at 100. Ten scripted builders house a hundred agents,
because the shrinking minority works ~3.5x harder per agent (harvest_wood share
3.0% -> 10.6%). The cliff is arb4's option-level chicken-and-egg at exactly
zero builders, not a commons degrading. Learned edge +19.1 -> +7.3 across the
interior, each point against its own size-matched random floor. Do not re-run
these six points. Write-up: ISLAND2_DESIGN.md §10. Original item kept below.

## 3-orig. The learned-share sweep (where does free-riding collapse? ~1.5h of runs)

**What**: mixed training at 40, 60, 80 learned agents (20 and 100 already
measured -- 100 is `arb4c`, the all-learned collapse). **Why**: 20 free-riders
cost the society nothing; at 100 nobody builds and everyone dies young.
Somewhere in between the shelters stop getting built -- the tragedy-of-the-
commons threshold, measured. **Reads**: learned-subset nights indoors and
population lifespan as functions of the learned share; the share at which
completions/episode falls off the ~50-58 plateau.

**Prompt**:

    Run the learned-share sweep from NEXT_STEPS.md item 3: train the mixed
    arbiter on society4.yaml at --learn-agents 40, 60, 80 (gamma 0.997, 150
    updates each, SEQUENTIALLY -- never parallel on this laptop), evaluate
    each paired vs utility on 10 episodes, and plot/tabulate nights indoors,
    completions and population lifespan against the learned share alongside
    the measured endpoints (20: arb5-mix, 100: arb4c). Find where free-riding
    stops scaling. Write up, commit.

## 4. Tech ladder rung 1: the craftable axe — DONE 2026-08-29 (`society4_axe`)

Result: **the mechanic works perfectly and buys nothing, because wood was never
the constraint.** Against its own control on 5 paired seeds: +0.8 +- 3.2 ticks
(1/5), shelters identical (55.20 both). The counterfactual is what makes that a
result -- arm every agent at spawn and chop actions fall 264.4 -> 141.8
(-122.6 +- 9.3) while wood gathered does not move (+1.0 +- 3.9). Adoption 8.80
of 100 against a random-goal floor of 3.20, limited by m4h's composition
deadlock at a workbench (1.6% of unarmed-loaded-at-a-site ticks hold one of
each). M2's specialisation prediction is REFUTED in the interesting direction:
axe-owners harvest LESS (0.53% vs 3.31% of their goal-ticks). Honest cost:
adoption here is scored, not emergent. No training was needed -- the scripted
arbiter runs it. Do not re-run. Write-up: ISLAND2_DESIGN.md §12. Original item
kept below for the record.

## 4-orig. Tech ladder rung 1: the craftable axe (first true "technology", ~1 session)

**What**: an axe crafted at a site (costs 1 wood + 1 stone), carried in a new
inventory slot, doubling chop yield. Config-gated, off by default, appended
action + observation channels. **Why**: the cheapest authored "technology"
rung -- a short, creditable chain -- and M2's specialisation result predicts
tool-owners become the village lumberjacks, which would be a genuine
emergence headline at the population level. **Rules that bite**: size with
`sim.economy` first (an axe changes wood supply); adoption claims need the
random floor and a no-axe control on the same seeds; the utility arbiter
needs a craft goal whose RESTORE row serves a real need (wealth/house
material), never a paid reward.

**Prompt**:

    Build the craftable axe from NEXT_STEPS.md item 4: a config-gated tool
    (craft at a site for 1 wood + 1 stone, doubles chop yield), appended
    action and observation channels so every existing world stays
    bit-identical (pin it), a craft goal in the utility arbiter serving a
    real need, sim.economy taught about tool-multiplied wood supply BEFORE
    the config is sized (rule 5). Pre-register the reads: adoption share vs
    the random floor, whether tool-owners specialise into harvesting (M2's
    result predicts yes), and completions/lifespan vs the same world without
    the axe on paired seeds. Run 5-episode utility evals, write up, commit.

## 5. Tech ladder rung 2: the night predator — DONE 2026-08-29 (`society4_predator`)

Result: **the hazard is real and the population does not adapt to it.** 5 paired
episodes against the same world without it: lifespan -10.8 +- 2.8 (1/5), deaths
+3.40 +- 0.40 (11.8 -> 15.2), while nights indoors moves -1.08 +- 0.76 and
berries and shelters do not move at all. 484 attacks an episode, 29.8 of 100
agents caught. Pre-registered failure mode 2 fired: NEED_SAFETY is driven by the
CLOCK and knows nothing about a wolf. Sizing note: sim.economy's estimate (0.2%
of an exposed night) is a LOWER bound and was fiftyfold low against the measured
10.2%; the shipped 12/reach-6.0 came from a four-row sweep against a
pre-registered 20% target, with lifespan flat down the column. Replay schema
went to v6 so the predators are visible. No training needed. Do not re-run.
Write-up: ISLAND2_DESIGN.md §14. Original item kept below for the record.

## 5-orig. Tech ladder rung 2: the night predator (~1 session)

**What**: a scripted hostile that hunts agents outside shelter radius at
night (drains hunger on contact, deterministic from the seed). Config-gated.
**Why**: escalating danger with a face -- forces guarding/grouping the way
the night drain forced construction, and it is the most watchable rung.
**Rules that bite**: it is a demand-side change, so re-size (exposed demand
rises); the mega-camp and war failure modes need re-deriving (rule 5: a
predator legitimately packs people together).

**Prompt**:

    Build the night predator from NEXT_STEPS.md item 5: a config-gated
    scripted hostile that hunts agents outside shelter radius at night,
    deterministic from the seed, observation channels appended, every
    existing world bit-identical (pin it). Teach sim.economy the extra
    exposure cost first, re-derive the mega-camp control (a predator packs
    people together on purpose -- rule 5), pre-register the failure modes
    (everyone hides all night and starves; predator ignored entirely), then
    run 5-episode utility evals vs the same world without it, write up,
    commit.

## 6. Watch it — DONE 2026-08-29 (replays + replay schema v5)

Done, and it turned into a schema bump because the replays could not answer the
questions the runs were about. v5 adds `o` (one arbiter goal per agent), `n`
(blight / storm), and a `learn` flag per agent; the viewer counts GOALS in its
histogram, rings the learned agents in cyan, and banners a storm or a blight.
Replays written and verified in a browser with no console errors:
`society4_ramp.json`, `society4_mixed_arb5hh.json`, `society4_axe.json`.
Write-up: ISLAND2_DESIGN.md §13. Original item kept below for the record.

## 6-orig. Watch it (no science, ~15 min)

**What**: replays of the seasons world and the mixed population for the
viewer. The seasons world especially -- late storms levelling settlements
during a blight is the drama the ramp was built for.

**Prompt**:

    Write viewer replays for the current worlds: sim.society --replay on
    config/island2/society4_ramp.yaml (utility arbiter) and on society4.yaml
    with --arbiter mixed --checkpoint checkpoints/arb5-hh/latest.pt. Verify
    both render in the browser with no console errors, then tell me the
    viewer URLs to open.
