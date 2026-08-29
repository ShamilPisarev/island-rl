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

**What is left in this queue: item 5 (the night predator) only.** Rung 1 is the
reason to read it differently than it was written: the axe attacked the COST of
contributing and the cost was not what binds. §10's sweep says what binds is the
DECISION to contribute at all, so a demand-side rung (a predator, or making
shelter excludable) is the one with something to move.

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

## 5. Tech ladder rung 2: the night predator (~1 session)

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
