# CLAUDE.md — working notes

Assume you are starting cold with only this file and `PROJECT_BRIEF.md`. This is
the state of the project, the decisions behind it, and the things that will bite
you.

## How to talk to me in this repo

**Short and skimmable. Answer first, then at most a few bullets or one small
table.** No narrative build-up, no restating the question, no summarising what you
are about to do before doing it. The detail belongs in this file and in commit
messages, which is where it already goes — not in chat.

What brevity does **not** licence, because this project runs on it:

* Say when a result is inside noise. "+7 ticks on a ±73 spread" is the honest
  form; "improved" is not.
* Say when a previous claim was wrong, in one line, and move on. Two metric
  errors in this file were caught that way (rules 5 and 6).
* Keep the control. A number without its baseline is not a result, and every
  wrong answer in this file was wrong for that reason.

Measure before choosing a lever (rule 2), and if a claim is unverified, say so
rather than smoothing it over.

**End every reply with `**Summary**` (two or three bullets) and `**Next steps**`
(one line per option, cheapest first).** Requested 2026-08-19; the response style in
`.claude/output-styles/skimmable.md` carries the same rule so it survives a machine
switch.

## Island 2.0 is on the table — read `ISLAND2_DESIGN.md` first

Decided 2026-08-25: the next likely direction is a watchable 50–100-agent
society (utility/"Maslow" arbiter + scripted controllers first, then a learned
option-level policy on the same interface). The plan below remains the 1.0
research thread and is unaffected.

**Stages 1-5 are DONE, same day.** The headline stage-5 comparison is run and
**the scripted arbiter wins outright** -- see the stage-5 bullet and
ISLAND2_DESIGN.md §10 before re-running anything. Stage 2/3/4/5 in one line
each:

* **Stage 2, utility agents.** A needs arbiter over goal-level options
  (`sim/utility.py`), with the interface stage 5 swaps a learned chooser into:
  `arbiter.choose(view, mask, rng) -> goals` and `execute_goals(...) -> actions`.
  100 agents reach **2.80× the random floor** (563.2 of 600 ticks against
  201.0; stage 4's explore-termination fix later moved this same world to 569.0
  and 98.1% nights indoors), 20/20 shelters, **96.3% of nights indoors**, and a visible commute
  (11.1 units from shelter by day, 3.7 at night). All three pre-registered
  failure modes clear — one of them (permanent war) fired first and was fixed at
  the mechanic, not the weight. New: `sim/obsview.py`, `sim/society.py`,
  `sim/economy.py` (size a world's subsistence in code, per rule 5),
  `config/island2/society100.yaml`.
  **Four corrections are written up in ISLAND2_DESIGN.md §8 and pinned by tests
  named after the symptom** — read them before touching the scorer; three are
  the same confusion (what a need IS vs what it costs to satisfy).
* **Stage 3, graphics.** Instanced procedural bipeds (`viewer/biped.js`):
  **8 draw calls for 100 agents** where the old per-agent groups were ~1300
  meshes. **No schema bump was needed** — the design doc asked that this be
  checked, and the action column plus the night cycle already in the replay
  drive every animation state. Also: golden-angle agent colours (evenly spaced
  hues put agents 3.6° apart at n=100), a population panel above 24 agents
  (aggregates + action histogram + swatch grid) with 1.0's roster kept below it,
  hover labels, single-pass death ticks, gift-line cap 32 → 256. Verified in a
  browser on v1/v2/v3 replays, no console errors; six-agent 1.0 replays render
  unchanged in roster mode.

* **Stage 4, society mechanics.** Regions, households, stockpiles, reputation
  and shocks, all config-gated behind `society.*` so every 1.0 world and stages
  1-3 stay bit-identical (`config/island2/society4.yaml`, 281 tests green). Five
  appended actions (deposit/withdraw x food/material, and raid), observation
  61 -> 77, replay schema **v4**, five new goals and two household needs in the
  arbiter, household sections in both `sim.society`'s report and the exchange
  view. 100 agents in 20 households of 5: **578.3 of 600 ticks** against a
  314.8 random floor, 82% of the island harvested, 2795 deposits and 629 raids an
  episode, and a stockpile holding 5.4 of 12 berries on average. The mechanics
  cost ~26% of the tick (**101k agent-steps/s** at 100 agents against 137k with
  them off), still 20x stage 1's exit bar.
  **Three numbers do NOT transfer from stage 2 and the write-up says so**: the
  1.84x floor ratio (the *floor* rose from 201 to 315, because a household spawns
  at its own shelter site), `shelters/episode` (now a flow including storm
  rebuilds, 52 of 20 sites), and nights indoors (98.1% -> 88.6%, which is storms
  removing roofs). **Six corrections are written up in ISLAND2_DESIGN.md §9 and
  pinned by tests named after the symptom** -- read them before touching the
  scorer; two of them are old lessons in new clothes (the M5 gift farm rebuilt
  out of a stockpile, and stage 2's theft correction firing again once households
  spawn together), and one retired a pre-registered failure-mode measure whose
  control had expired (rule 5). The economy had to be resized because blights cut
  supply to 0.98x subsistence -- caught by `sim.economy`, which was taught about
  blights first. One design-doc forecast is refuted: stockpiles produced churn,
  not runaway hoarding. **The trade lever is also run**
  (`config/island2/society4_trade.yaml`, fungibility off on the split island):
  the demand is real (lifespan 578 -> 559, deaths 2x) and the cross-region flow
  happens by RAID (774/ep, 28% taking material, stone 2.6x over wood), not by
  gift (~4/ep) -- three treadmill/deadlock corrections are in ISLAND2_DESIGN.md
  §9 under "The trade lever". That world is stage 5's sharpest testbed.

* **Stage 5, the learned arbiter.** `sim/arbiter.py`: semi-MDP PPO over the 15
  goals (one transition per decision, per-agent asynchronous, gamma^k
  bootstraps), one shared net + the scripted arbiter's own trait vector, menu
  parity enforced by a shared `goal_availability`, and an arbiter zoo in
  `sim.society` (`--arbiter`, `--vs` paired). **The scripted arbiter beats every
  learned variant by ~100 ticks on 10/10 paired islands.** Four runs, all
  converging to the same hyper-competent pure forager (93% harvest, 0-3% of
  nights indoors): from scratch (456.7, and BELOW the random-over-menu floor of
  533.9), imitation-warm-started (464.1 -- PPO erases the teacher in flight,
  spread-nav's shape one level up), warm-started at gamma 0.997 (476.3, in a
  regime where its own measured objective prefers the scripted behaviour, 8.24
  vs 7.22), and with a critic warm-up (479.3). At gamma 0.99 the forager
  genuinely wins the objective (4.47 vs 4.41) -- reward-alignment, not
  algorithm; above that it is optimisation. **Verdict: the option level routes
  around the one-step wall for single-trip prizes (travel, forage) and the wall
  reappears intact for compound multi-agent prizes (construction).** §10 has
  the do-not-re-run list and the three levers worth trying (mixed populations,
  household-level baseline, persist-until-goal options).
  **The first lever is RUN (`arb5-mix`), and it produced the project's first
  learned-over-scripted number.** 20 learned agents (one per household,
  `--learn-agents 20`, gamma 0.997, from scratch, no imitation) trained among
  80 scripted: they shelter **84.0% of nights** where every all-learned run
  managed 0-3.3% -- the state distribution, not the objective, was what blocked
  sheltering -- and beat the scripted arbiter in its own slots
  **+19.1 +- 7.4 paired (7/10)**, by abandoning the grudge economy (raid and
  steal both 0.0% against the scripted 80's 6.5%/3.6%). The floor holds it
  honest: a random-goal minority in the same slots is **-55.9 +- 8.3 (0/10)**,
  so the society is not carrying a passenger. And they free-ride on
  construction totally (harvest_wood/deliver/store_material all 0.0%, though
  they do bank food at 4.4%; spillover cost to the 80 is -0.5 +- 4.7, nil at
  this ratio) -- the compound wall stands, now isolated to exactly the credit
  a household-level baseline would assign. New: `--learn-agents` in
  `sim.arbiter`, `MixedArbiter`, `--arbiter mixed|mixedrandom` and a
  per-subset section plus learned-slot paired diffs in `sim.society`,
  per-agent night counters in `world.py`, five pinning tests. Do not re-run
  this configuration; the open follow-ups are the household-level baseline,
  persist-until-goal options, and the learned-share sweep (20 -> 50 -> 100,
  where free-riding must collapse back into `arb4c`).
  **The second lever is RUN too (`arb5-hh`, `--household-reward`): the
  compound wall survives its own credit being handed over.** Same mixed
  setup, one change -- the learned agent trains on its HOUSEHOLD's mean
  reward. Construction contribution is **still exactly 0.0%**
  (harvest/deliver/store_material), so the wall is the OPTIMISATION, not
  credit assignment. What kin-shared reward did buy is a provisioner:
  store_food doubled (4.4% -> 8.8%) and stealing from strangers appeared at
  11.4% (arb5-mix: 0.0%; housemates are immune, so theft is pure import for
  the larder). Nights indoors 91.7%, learned slots +22.6 +- 7.0 (9/10) --
  level with arb5-mix's +19.1 within one SE. Honest cost, stated in the
  write-up: household reward changes what "unpaid" means. Do not re-run; the
  one untried move against the construction wall is persist-until-goal
  options. Both write-ups: ISLAND2_DESIGN.md §10.

  **The third lever is RUN (`arb5-persist`, `--persist`): with the whole build
  programme collapsed into ONE decision, construction is still exactly zero --
  and this time it is priced per OPPORTUNITY, so it is a refusal, not an
  absence.** `ArbiterConfig.persist_until_goal` lets a goal run to its goal
  state (site fed, inventory full, night over) instead of a 25-tick budget,
  with a 150-tick backstop; off by default and pinned bit-identical by a golden
  trajectory checksum and a trainer-buffer checksum taken against the previous
  commit. Same mixed recipe as arb5-mix plus the flag: learned slots
  **+18.2 +- 6.5 (7/10)**, level with arb5-mix's +19.1 +- 7.4, 55 ticks clear
  of the random-goal floor (-37.4 +- 11.1). Over 10 episodes the learned 20
  spent **0 of 120,000 goal-ticks** on deliver/harvest_wood/harvest_stone while
  `deliver` sat on their menu at 3.9% of their own decision points and
  `harvest_wood` at 20.5% -- **zero taken of ~2,700 chances**, against the
  scripted arbiter's 33%/27% in the same slots, and against a RANDOM minority
  that does contribute 2-3% and dies 37 ticks sooner. Nights indoors 80.5%, a
  small regression from 84.0%. Two corrections the lever forced: a **curfew**
  (safety is tier 1, so dusk now interrupts a running option once, exactly as
  tier-0 hunger does -- without it the scripted population fell 585.8 -> 525.3
  and nights indoors to 60.9%), and **`explore`/`raid` may not persist** (a
  goal state must be REACHABLE: a full-handed explorer can never see "a bush I
  have room for", so it wandered 150 ticks). Mechanism check: 28.8% of
  persisted deliver options contain a harvest leg against 0.0% without, shelter
  options run 73.8 ticks against 22.4 -- but society4's programmes are SHORT
  (eighty scripted builders finish 60 sites an episode), so the decision on
  offer is mostly join-a-build, not build-from-scratch. Honest cost, unchanged:
  what can emerge is when-to-build, never building. Do not re-run. All three
  §10 levers against the construction wall are now spent. Write-up:
  ISLAND2_DESIGN.md §10.

* **The seasons world (`config/island2/society4_ramp.yaml`, design doc §11).**
  `society.shock_ramp` scales shock SEVERITY with episode progress (cadence
  and rng untouched; ramp 0 bit-identical, pinned; storm damage clamped at
  site cost so progress cannot go negative). `sim.economy` models the ramp
  and rejected two sizings before any run (9 bushes/cluster starves everyone
  at 0.84x; 13 makes even exposure survivable at 1.01x); shipped at
  11/cluster, **1.28x sheltered / 0.86x exposed**. First run confirms every
  pre-registered read: stockpiles fill in the fat summer and drain in the
  hard winter (**8.19 / 7.99 / 5.64** by episode third -- the first seasonal
  banking in the project), deaths concentrate 62% in the last third, nights
  indoors 88.6% -> 84.0% as late storms level settlements. Utility agents
  586.8 against a 336.6 random floor (1.74x). The escalation ladder beyond it
  (night predator, shelter tiers, craftable tool, gated unlocks) is sketched
  at the end of §11 -- every rung gets sized with `sim.economy` before it
  runs, and every adoption claim needs its floor.
  **The provisioner was then trained IN this world (`arb5-ramp`, same recipe
  as arb5-hh: 20 learned, household reward, gamma 0.997, 150 updates), and it
  transfers but does not intensify.** Goal mix is arb5-hh's within a point
  (store_food 8.3% vs 8.8%, theft-import 11.2% vs 11.4%, raid 0.0%), so
  "banks harder when winter is coming" is falsified -- the policy is
  climate-invariant; what rose is the pile (stock by third 8.63/8.62/6.69
  vs the all-scripted 8.24/7.60/6.12 on the same seeds, deposits +25%).
  Learned-slot paired edge **+5.0 +- 2.2 (6/10)** -- NOT a regression from
  arb5-hh's +22.6: the learned slots sit at 599.0 of 600, so the seasons
  world caps the winnable edge (the scripted arbiter is simply better here
  in those slots). Construction still exactly 0.0%, third world running,
  now with storms levelling shelters all episode. Nights indoors 86.6%,
  above the scripted 80's 83.6% in the same world (the drop from 91.7% is
  the storms, not the policy). One measurement note: household = i % 20 and
  the learned set is agents 0..19, so every household hosts a learned agent
  -- there are no scripted-only households inside a mixed run. Do not
  re-run. Write-up: ISLAND2_DESIGN.md §11.

**Stage 1 (engine scale pass) detail.** The design doc's forecast
that O(n²) neighbour queries would need a spatial hash was refuted by the
profile: 100 agents with every mechanic on already ran at 98k agent-steps/s
(~20× the exit bar) because the queries were vectorised all along. The one
measured hotspot — the full argsort in `_k_nearest` — was replaced with
argpartition + a stable k-sort, **bit-identical** to the old path (checksummed
against the stashed original on default/m3_masked/m5b; `tests/test_scale.py`).
After: **144k agent-steps/s** full-mechanics at 100 agents. New: `sim/profile_engine.py`,
`config/island2/engine100{,_full}.yaml` (economies NOT sized — stage 2 must
recompute subsistence with `sim`). No spatial hash exists; do not build one
below a few hundred agents.

## Plan for the next session

Written 2026-08-19, at the end of the session that measured the advantage signal
(`sim.advantage`), ran the k-tick commitment lever (`nav-commit`), killed the
equilibrium explanation (`spread-nav`), refuted the capacity corollary
(`spread-nav512`), priced the individual navigation prize
(`sim.steering --subset`), and finally showed the competence and the refusal
living in one set of weights (`spread-mix`). Ordered by what it would teach;
each item says what has already been ruled out so nothing gets re-run.

**If you read one thing, read this.** The project's oldest open problem has
changed shape. It is not "PPO cannot learn to navigate here" — a policy trained
half on a probe geography navigates at 98.3% and, in a zero-shot 2×2, applies
that navigation in every food layout except one: far *and* thin, which is the
scarce world. There a sustained crossing is worth +173 and every single step of
it is correctly priced ≤ 0. The remaining problem is an optimisation-operator
problem, and the file's older framings ("the chain loses navigation", "the world
erases it", "the signal is too small") are all superseded by it.

### 0. What last session settled, so it is not reopened

**The per-step learning signal for direction is now measured, and the arithmetic
this plan carried was half wrong.** The noise figure was right: the std of the GAE
advantage — what PPO's normalisation divides by — is 0.851. The signal is
band-local, not "~0.06 everywhere": positive near food (GAE gap toward − away
**+0.083 ± 0.045** at 3–6, stable across runs, and that is the one band
nav-spread2 clears its floor in), ~zero at 6–20 (δ +0.013 ± 0.006, GAE
+0.010 ± 0.021), and **negative beyond 20** (δ −0.028 ± 0.007, the stablest cell
in the table). m3-masked has the same shape. 82% of far-field move ticks are
agents camped at an *empty* bush, for whom stepping toward distant food abandons a
regrowth queue (δ −0.045 ± 0.008): the critic prices the camping basin correctly
under a wandering continuation policy, so the on-policy gradient points home from
everywhere far. Write-up: "The advantage check and `nav-commit`" in the M3 section.

**The k-tick lever is run, and it is a clean negative with the mechanism
verified.** `world.decision_interval: 4` (an action persists 4 ticks, enforced by
`World.step` itself; PPO trains one transition per decision) multiplied the signal
exactly as designed — δ gap at 10–20 +0.013 → **+0.061 ± 0.033** (×4.7 ≈ k),
noise ×2.7 ≈ √k, the critic's V-fall to 10–20 doubled to −2.24 — and PPO still learned no
direction: trained vs zero-shot control **−1.5 ± 7.5** paired, toward-food bands
identical to the untrained control's, steering headroom intact at **+93.6 ± 7.5**.
What commitment did buy arrived zero-shot: **+50** (436.3 → 485.9) from ballistic
diffusion alone, collapsing the forager gap to −11.6 ± 8.6 with no direction
learned — and it does NOT generalise (m4h: +6.2, inside noise, nights indoors
83% → 69%).

**The equilibrium explanation is dead too.** `spread-nav` forks `nav-probe` (86%
toward-food far-field) into the spread world, where camping verifiably cannot feed
even one agent. The far bands decay **+6.3 → +2.1** over floor between 200 and 400
updates — nearly the refork's trajectory (+7.1 → +1.4) in the world built to
prevent it — while survival climbs (334.5 → 376.2, still below the m2 lineage's
433.4; rule 3 cuts both ways). Making navigation worth +140.9 and camping fatal
does not make PPO retain it.

**The capacity corollary is already tested, and it is refuted.** A 512×512 probe
learns navigation *better* (99.9% toward-food at 20+ against the 128-wide probe's
86.1%) and the spread world strips it *faster* (+0.7 over floor at 200 updates,
where the narrow lineage still held +6.3) — while banking **+67.7 ± 8.3** of
survival (36/40 islands) out of pure local competence. So the unlearning is not
passive interference in a too-small trunk; it is **gradient-following**: the
on-policy advantage actively trains far-field navigation away at any capacity,
exactly as the measured negative 20+ δ gap says it should. See "The capacity
probe" in the M3 section.

**And the individual-vs-collective question is answered: the prize is
individual, first-mover-largest, with positive spillover.** `sim.steering
--subset` steers only the first n agents and pairs each agent against itself
unsteered: a LONE navigator gains **+173.0 ± 19.0** in the spread world (more
than the +140.9 each gets when all six navigate) and **+138.3 ± 23.0** on m4h,
while the unsteered five *gain* +19.5 ± 6.4 rather than lose. Reconciled with
the advantage check, this names the barrier exactly: a one-step toward-move is
correctly priced ≤ 0 under a wandering π, a sustained walk pays +173, and
**one-step policy improvement cannot see a ~20-step prize**. Not the world, not
the signal, not the critic, not capacity, not coordination — the improvement
operator itself.

**Interleaving proved that in the strongest possible form: the competence and
the refusal now live in ONE set of weights.** Training half the envs on the probe
geography (`spread-mix`, `cfg.mix`) leaves the scarce world's bands flat
(+0.3 over floor at 20+) while the *same policy* navigates at **98.3%** in the
probe half — and a zero-shot 2×2 shows the switch is conditioned on whether a
trip pays: three of four cells navigate at +46 to +50, and the only one that
does not is food that is both far and thin, i.e. the real scarce world. **So the
policy is not missing navigation. It has it, and correctly declines to use it one
step at a time, in the one world where using it sustained is worth +173.** Every
"the chain loses navigation" framing in this file is retired by that.

Do not re-run: `exclusive_bushes` (`nav-compete`, flat), `move_step` (`nav-move`,
flat), entropy, γ, contest perception, brain sharing, 3.3× budget (all flat, see
the seven-intervention table); re-forking navigators into scarce worlds — both
destinations are done (`nav-refork` where camping pays, `spread-nav` where it
cannot, decay either way), and at both widths (`spread-nav512`, faster decay);
`decision_interval` at any k (k=4 verified the mechanism and moved nothing; the
anti-gradient multiplies with k, and k=8 commits 6.4 units past a 2.0 gather
radius). Do not chase `build` or `steal` uptake.

### 1. The policy HAS navigation and declines to use it — the remaining question is how to move a decision it makes correctly one step at a time (start here)

This is no longer "why can't it learn to navigate". `spread-mix` holds a
navigation controller worth 98.3% toward-food in the probe half and applies it
in three of four zero-shot 2×2 cells; the only cell it withholds in is food that
is far *and* thin — the scarce world — where a sustained crossing pays
**+173.0 ± 19.0** and every one-step prefix is correctly priced ≤ 0. The
competence, the world's incentive, the critic and the capacity are all in place.

**Interleaving is answered — do not re-run it at another fraction.** 50% left
the scarce bands flat (+0.3 at 20+) while costing nothing in lifespan or in the
probe leg. A smaller fraction cannot do more; the failure is not dilution.

What is genuinely left, ordered by how little scripted competence it injects:

* **Temporally-extended exploration** — the last lever that changes nothing
  about the world or the reward. ~20-tick correlated exploration puts *completed*
  crossings into the batch, which is the only thing a one-step operator has never
  been offered in the scarce world. The PPO wrinkle to design around first: held
  actions are off-policy for the ratio, so they must either repeat the policy's
  own sampled move (like `decision_interval` but stochastic and per-agent) or be
  excluded from the policy loss and used only to fit the critic. Worth doing
  properly rather than quickly.
* **Anneal the mix, and measure at every step.** `spread-mix` is a policy that
  navigates when a trip looks worth it. Lower `mix.fraction` toward zero over
  training and watch the scarce-world bands: if they ever rise before the probe
  branch decays, the crossing behaviour transferred and the barrier is
  surmountable from inside PPO. If the branch simply decays (the `spread-nav`
  outcome), that is a clean negative and the one-step story is complete. Cheap —
  the mechanism already exists and this is a schedule on one number.
* **A travel option** — an appended action "walk toward the nearest visible
  loaded bush for up to k ticks" — crosses the wall in one decision, and is the
  shape of fix that has worked here before (the mask says what is *reachable*; an
  option says how long to *persist*). The honest cost, unchanged: it hands the
  policy a scripted micro-controller, so what emerges is *when* to travel rather
  than travel itself. Given that the policy already has the controller and only
  the persistence is missing, this is now the most defensible version of the
  idea — but decide whether it is still the project's question before building it.
* **No reward-side lever can work.** The one-step pricing is *correct*; paying
  more does not lengthen the deviation PPO can evaluate. And **not shaping**:
  paying for approach would produce approach; rule 1.
* **Not bigger frame-skip.** k=4 shortened the desert to 5–6 decisions and moved
  nothing; each decision is still priced by the same one-step rule, and k=16
  destroys near-field control (12.8 units per commitment, gather radius 2.0).
* **Not another world shape.** Six have been tried (`nav_compete`, `nav_move`,
  `nav_spread`, deep/thin × tight/spread in the 2×2). The 2×2 in particular says
  a world where navigation *is* one-step-rational already gets navigation, for
  free, from weights that were never trained in it.

### 2. Nights are NOT the gap — closed, do not reopen

The night channel already exists (`night.phase`, `night.is_night`, plus a
`site{j}.complete` flag per site), the nearest finished shelter is in the
observation on 99.3% of night ticks, and the policy ignores all of it — it holds
~3.2 units from shelter all day and drifts outward at night, where the builder
closes to 2.8 by phase 0.5 and holds. Forcing the home run lifts nights indoors
71% → 94% and buys **−0.3 ± 7.6 ticks**, because berries fall 34.7 → 29.2 and
shelters 2.77 → 2.15. It is worth ~+24 only *on top of* navigation. Full write-up
in the M4 section under "The night channel already existed".

### 3. Uptake is closed — do not reopen it

Both "the mask says yes and the policy shrugs" items from the last plan are
answered, and they were the same error twice.

* **Per legal tick was the wrong denominator.** Per distinct opportunity (a span of
  consecutive legal ticks for one agent), `build` uptake is 72.6% not 28.6%, and
  `steal` is 85.0% not 46.4%. `sim.opportunity` prints both and says which to read.
* **The residue is worth nothing.** `--force` runs the action taken-whenever-legal
  and suppressed-entirely, paired: forcing `build` gives +3.6 ± 7.7, forcing `steal`
  gives −8.3 ± 4.1. Suppressing them is catastrophic (−104.5 ± 8.7 and −15.6 ± 5.9),
  so both behaviours are valuable *and* saturated.
* **The refusals were sampling, not preference**, in both cases: mean P(steal) 45.1%
  against 44.4% uptake. Hunger does not select the declined steals either (95.6%
  above the eat threshold, against 95.7% of the taken ones).

What is left of the thief's +87.8 in the M3 world is opportunity *generation*:
forcing every legal steal yields 423 an episode against the thief's 1111, because
the thief positions itself beside loaded neighbours. Same shape as item 1 — going
somewhere, not choosing differently.

### 4. Perception beyond 20 units, if you want the navigation thread anyway

The one place perception *is* deficient: at 20+ units the target is absent from
the observation 31% of the time and its direction is clipped on 90% of ticks
(15.9% on both axes, which loses direction to a diagonal). Two levers, and the
cheap one is safe:

* **`observation.distance_scale` 20 → 40** changes no dimension, so an existing
  checkpoint forks straight into it. It also halves near-field resolution, which
  is the reason it was set to 20 in the first place — read the *near* bands as
  carefully as the far ones.
* **`observation.k_bushes` 4 → 6** is safe, and an earlier version of this plan
  said otherwise. `--init-from` maps columns by **name**: `train.py` builds
  `column_map(observation_layout(source), observation_layout(target))` and hands it
  to `grow_policy`, so inserted channels are handled and new columns are zeroed.
  The trigger now keys off the layout rather than the width, so a same-width feature
  swap raises instead of misaligning silently.

Expect little from either: the 3–20 bands, where the collapse lives, have neither
problem.

### 5. Exchange, if you want to push M5 further

`m5b` retested exchange on the working economy: giving stopped being selected
against and the flow turned directional (reciprocity 0.85 → 0.65), but it still
buys no survival and the scripted trader's edge *doubled* to +28.1. Do not raise
`reward.give` — that produces a gift farm (42× the gifts, 54 fewer ticks of life).
The three mechanic-level ideas are in "What would actually be worth trying".
Note the measured surprise: **material** giving was selected against (2%
utilisation) while **food** giving rose (48%), so a relay needs its chain
shortened, not its deliveries made fungible.

### Housekeeping

* **Back up `checkpoints/` before switching machines.** `./backup_checkpoints.sh <dest>`
  copies the 44 `latest.pt` files (~69MB) plus `runs/`. Both directories are
  gitignored and the chain is unlearnable from scratch, so this is the one piece
  of state git will not save for you.
* A GPU will not help: measured, the network is **7.3%** of per-tick cost and the
  numpy env step is the rest. Determinism is CPU-only, so CUDA would also break
  bit-identical reproduction of every result here. A faster CPU with real cooling
  would help; this machine is fanless and throttles.
* Response style lives in `.claude/output-styles/terse.md`, selected by
  `.claude/settings.json`. Project-scoped on purpose, so it travels with the repo.

## START HERE — handoff for the next session

**Milestones 1–5 are all trained, verified and written up.** Every milestone the
brief puts in scope for the simulation is done; M6 (an LLM narration layer) is
explicitly parked by the brief itself, so there is no obvious "next milestone" to
start. Read the results sections below before touching anything — several of them
record experiments that cost real time and should not be repeated.

Where each milestone landed, in one line each:

| | result | canonical checkpoint |
|---|---|---|
| M1 | 1.97× random, at the scripted forager's ceiling | `checkpoints/m1` |
| M2 | specialisation: action divergence 19× the shared-brain control | `checkpoints/m2` |
| M3 | 2.11× random; theft emerged *unpaid* after action masking | `checkpoints/m3-masked` |
| M4 | construction emerged; **490.4 lifespan, 3.0 shelters/ep**, beats forager + thief | `checkpoints/m4h` (annealed lineage: `m4c-anneal`) |
| M5 | exchange did **not** emerge unpaid, on a fair retest either; paying for it made survival worse | `checkpoints/m5b` |

### If you are picking this up, the honest open problems

In rough order of how much they would teach:

1. **The milestone chain LOSES navigation — but that is NOT the gap to the
   scripted references, and every explanation for it has now failed.** Measured,
   toward-food share by current distance (50% = a coin flip):

   | | 0–3 | 3–6 | 6–10 | 10–20 | 20+ |
   |---|---|---|---|---|---|
   | `nav-probe` | 48.7% | 66.8% | 74.6% | **80.1%** | **86.1%** |
   | m1 | 47.4% | **60.9%** | **71.1%** | — | — |
   | m3-masked | 50.6% | 49.3% | 56.5% | 54.3% | **50.5%** |
   | m4h | 50.3% | 53.8% | 56.3% | 52.3% | **51.2%** |
   | random | 47.8% | 48.6% | 50.3% | 51.0% | 49.0% |
   | scripted forager | — | — | — | **100%** | **100%** |

   **Navigation is learnable here and M1 had it.** `nav_probe.yaml` strips the
   world to "walk to the food" and PPO reaches 86% from long range, so this is
   not a limitation of the observation, the action encoding or the algorithm.
   But m3-masked and m4h sit within a couple of points of a random walk *at every
   range*, despite forking from a policy that could navigate. The chain carried
   the opportunism forward and dropped the navigation.
   That reframes everything else in this file: every fix that has ever worked
   here — action masking, sites on the clusters, materials on the clusters, a
   fourth site, fungible deliveries — works by bringing things TO an agent whose
   navigation has decayed to chance. The seven M3 interventions all failed
   because none of them touched navigation.
   **Why: six explanations tried, six dead.** *Not* competition —
   `nav-compete` turns `exclusive_bushes` off and changes nothing (53.0% / 54.4% /
   57.0% against m3-masked's 50.8% / 55.8% / 53.6%). *Not* the target expiring
   during the walk, which was the leading candidate and was quantitative: a berry
   lasts ~27 ticks and takes ~22 to reach, a ratio of 1.22× against 22× in the M1
   world. `nav-move` doubled `move_step` and took the measured ratio to **2.96×**;
   navigation stayed flat (+6 / +11 / +3 / +1 over that world's floor). *Not*
   perception below 20 units — the target is in the observation 94–100% of the
   time there and the ±1 offset clip cannot bite under 20 units by arithmetic, yet
   the bands are flat with the target plainly visible. *Not* entropy, γ, contest
   perception, brain sharing or budget (the seven-intervention table). *Not* a weak
   starting policy — `nav-refork` grows the chain from a navigator and the world
   trades the competence away in flight. *Not* the equilibrium: `nav-spread` makes
   camping insufficient, verified, and navigation stays flat while steering headroom
   rises to +140.9 ± 8.2. *Not* the critic: V falls ~1.2 from "on food" to 10–20 units
   away, so the gradient exists.
   **What it is worth, measured: +89 to +113 ticks.** `sim.steering` keeps every
   decision the policy makes and replaces only the direction of its moves. On m4h,
   40 paired islands, that is **+89.1 ± 11.4** steering at food and **+113.2 ± 10.7**
   with a night home-run — beating the scripted builder by ~40, with deaths 3.35 →
   0.62. Nothing else in this file is that large. It reads world state, so treat it
   as an upper bound on perfect navigation rather than as what the current
   observation supports.
   **Two scope corrections that go with it.** `nav-move`'s finding stands as
   written — the lifetime/travel ratio does not make navigation learnable, and the
   paired gap to the forager halved (−32.7 ± 10.8 → −15.4 ± 10.0) while the
   toward-food share stayed flat, so that *share* does not track the gap across
   worlds. But the stronger sentence it invited, that navigation is not the gap, is
   wrong: steering says it is. And the theft headroom that was promoted in its place
   was a counting artefact (open problem 3).
   **The per-step-signal explanation now has its measurement and its lever, and the
   lever is dead too.** `sim.advantage` shows the toward-food learning signal is
   band-local: real inside 6 units, ~zero at 6–20, **negative beyond 20** — the
   critic correctly prices leaving a regrowth camp as a loss under a policy that
   wanders, so the on-policy gradient points home from everywhere far. Multiplying
   the per-decision signal 4× with `world.decision_interval` (`nav-commit`)
   verified the mechanism — SNR ×√k, the critic twice as steep — and produced no
   navigation: trained vs zero-shot −1.5 ± 7.5, bands identical to the untrained
   control, steering headroom intact at +93.6 ± 7.5. The interface change alone is
   worth **+50 zero-shot** (ballistic wandering) and closes two thirds of the
   forager gap with no direction learned. See "The advantage check and
   `nav-commit`" in the M3 section.

   *Metric warning, learned the hard way.* An aggregate toward-food share is
   **confounded and must not be used** — an agent parked on its target scores
   ~50%, identical to a random walker, so a competent forager and a drunkard are
   indistinguishable. The first version of this finding claimed the project had
   *never* learned to navigate; that was the artefact, not the result. Always
   bucket by current distance.
2. ~~**Nobody finishes a shelter.**~~ **Solved — see `m4f`.** The cause was
   geography, not the last unit: `m4b` moved the shelter *sites* onto the berry
   clusters and left trees and rocks scattered, so the uncreditable walk it
   deleted simply moved upstream to the harvest leg. Agents stood in harvest
   range on 2.7% of ticks. `construction.materials_at_clusters` fixes it and
   completions go **0.10 → 1.10 an episode** with every shaping term still at
   zero. The control worth knowing: the **unchanged** `m4e-premium` policy scores
   1.00 shelters in the new world with *no retraining at all* — **the policy
   already knew how to build and had nowhere to do it.**
   Three earlier explanations are dead and should not be revisited: summit too
   far (`m4d`), summit pays nothing (`m4e-premium`), summit invisible (`m4e`). The
   old framing also rested on a stale premise; see rule 5.
   `num_sites: 4` (one per cluster) followed as **`m4g`** — capacity, not
   competence: completion rate per site flat, gap to the builder unchanged.
   Then **`m4h`** found the real second constraint. Agents stand at a site
   *holding material* on 29.7% of ticks, and on **97.5% of those the site does
   not want what they carry** — sites need 3 wood + 1 stone, everyone feeds their
   nearest site whatever they hold, and a fifth of sites end one stone short while
   agents carry 3.7 stone. `construction.fungible_materials` lets any unit take
   any material: **490.4 lifespan, 3.0 shelters/ep (87% of the builder's rate),
   83% of nights indoors, and the gap to the builder halved from 82 to 50 while
   the builder stood still.** This is the best policy in the project and the first
   since M1 to beat any scripted reference.
   **What is left is NOT `build` uptake** — that read 28.6% per legal tick and is
   **72.6% per distinct opportunity**, with forcing the remainder worth +3.6 ± 7.7.
   It is the night: m4h is exposed on 18–19% of night ticks and **a finished
   shelter existed for 100% of them**, 13.6 units away on average, while the
   builder is exposed 0% and is +72.6 ± 10.5 ahead. See "What is left in M4".
3. **Exchange needs a mechanism, not a bigger number.** M5 showed the unpaid
   chain is too weak and a flat payment produces a gift farm. **`m5b` retested it
   on `m4h`'s working economy** — the original verdict was reached in a world
   where the policy finished 0.09 shelters an episode, so there was nothing worth
   trading. On the fair retest giving is no longer selected *against* (7.9 → 11.3
   an episode) and the flow turns directional (reciprocity 0.85 → 0.65), but it
   still buys no survival, and the scripted trader's edge *doubled* to +28.1 —
   so the mechanic is worth twice as much as before and PPO still cannot find it.
   If you want trade, change the *mechanic* — see "What would actually be worth
   trying" in the Milestone 5 section.

Whatever you do next: an experiment here costs about six minutes (200 updates),
so run the control. Every result in this file that turned out to be wrong was
wrong because it had no control, and every one that survived had one.

## Current state

**Milestones 1–5 are complete and verified.** M6 is parked by the brief (an LLM
narration layer, explicitly out of scope for the simulation loop). Read the M3
shaping ablation and the M4 economy sizing notes before touching any config.

- M1: survival + foraging, one shared brain. 1.97× the random baseline, at the
  ceiling set by a hand-written forager.
- M2: six individual brains forked from the M1 checkpoint and trained
  independently, plus a behavioural-divergence view. Specialisation appeared:
  action divergence is 19× the shared-brain control.
- M3: scarcity, contested bushes, stealing. After the action-masking fix:
  **2.11× random** (471.6), theft emerging *unpaid* at 115 steals/ep, zero doomed
  actions — still honestly below the scripted forager (495.5). Territoriality
  appeared as inequality rather than as spatial partitioning. Canonical
  checkpoint: `checkpoints/m3-masked` — `nav-move` (faster travel) is a negative
  and does not replace it.
- M4: wood/stone/shelter/night mechanics, day/night hazard, replay schema v2.
  Construction emerged and shelters now complete — **1.10 an episode against the
  scripted builder's 2.80**, with every construction shaping term at zero. The
  shaping annealed away cleanly first (`checkpoints/m4c-anneal`), and the thing
  that finally produced completions was geography: `materials_at_clusters` put
  trees and rocks where the agents already live, `m4g` added a fourth site so
  every cluster has one, and `m4h` made deliveries fungible so sites stop
  deadlocking on composition. Best checkpoint: `checkpoints/m4h` — **490.4
  lifespan, 3.0 shelters/ep (87% of the scripted builder's rate), 83% of nights in
  a finished shelter**, and the first learned policy since M1 to beat a scripted
  reference. Three earlier levers aimed at "the last unit" (`m4d`, `m4e-premium`,
  `m4e`) all came back flat and are written up as negatives.
- M5: `give_food`/`give_material`, a transfer ledger, replay schema v3, an
  exchange analysis tool and viewer page, and a scripted trader reference.
  **Exchange did not emerge unpaid**, and paying for it produced a gift farm that
  cost 54 ticks of life. Retested as `m5b` on M4's now-working economy, since the
  original verdict was reached where nothing was worth trading: giving stops being
  selected against and the flow turns directional, but it still buys no survival.
  Best checkpoint: `checkpoints/m5b` (497.5 lifespan).
- `pytest` passes (363 tests).
- All three viewer pages verified in a browser against real data, including their
  schema-mismatch failure paths.

Reproduce end to end:

```bash
python -m sim.train --run-name m1
python -m sim.train --run-name m2 --policy-mode individual \
    --init-from checkpoints/m1/latest.pt
python -m sim.divergence --checkpoint checkpoints/m2/latest.pt

# the M3 result (masking is what fixed it; see below)
python -m sim.train --config config/m3_masked.yaml --run-name m3-masked \
    --policy-mode individual --init-from checkpoints/m2/latest.pt
python -m sim.divergence --checkpoint checkpoints/m3-masked/latest.pt

# the labelled ablation, not the M3 result
python -m sim.train --config config/m3_shaped.yaml --run-name m3-shaped \
    --policy-mode individual --init-from checkpoints/m2/latest.pt

# Milestone 4: shaped and its unshaped control, from the M3 checkpoint
python -m sim.train --config config/m4.yaml --run-name m4 --updates 400 \
    --policy-mode individual --init-from checkpoints/m3-masked/latest.pt
python -m sim.train --config config/m4_unshaped.yaml --run-name m4-unshaped \
    --updates 400 --policy-mode individual --init-from checkpoints/m3-masked/latest.pt

# the cheap-sites test and its control — a NEGATIVE result, kept so nobody
# spends the six minutes finding it again
python -m sim.train --config config/m4d.yaml --run-name m4d --updates 200 \
    --policy-mode individual --init-from checkpoints/m3-masked/latest.pt
python -m sim.train --config config/m4d_unshaped.yaml --run-name m4d-unshaped \
    --updates 200 --policy-mode individual --init-from checkpoints/m3-masked/latest.pt

# make the last unit worth something (the mechanic), then also perceptible (the
# channel) — both NEGATIVE, and together they retire the last-unit hypothesis
python -m sim.train --config config/m4e_premium.yaml --run-name m4e-premium \
    --updates 200 --policy-mode individual --init-from checkpoints/m4c-anneal/latest.pt
python -m sim.train --config config/m4e.yaml --run-name m4e --updates 200 \
    --policy-mode individual --init-from checkpoints/m4c-anneal/latest.pt

# the one that worked: put the material nodes where the agents already live,
# then give every cluster a site (m4g is the best M4 policy)
python -m sim.train --config config/m4f.yaml --run-name m4f --updates 200 \
    --policy-mode individual --init-from checkpoints/m4e-premium/latest.pt
python -m sim.train --config config/m4g.yaml --run-name m4g --updates 200 \
    --policy-mode individual --init-from checkpoints/m4f/latest.pt
python -m sim.train --config config/m4h.yaml --run-name m4h --updates 200 \
    --policy-mode individual --init-from checkpoints/m4g/latest.pt

# Milestone 5: gifts unpaid (the result) and gifts paid (the ablation), both
# continued from the annealed M4 policy
python -m sim.train --config config/m5.yaml --run-name m5 --updates 200 \
    --policy-mode individual --init-from checkpoints/m4c-anneal/latest.pt
python -m sim.train --config config/m5_shaped.yaml --run-name m5-shaped \
    --updates 200 --policy-mode individual --init-from checkpoints/m4c-anneal/latest.pt
python -m sim.exchange --checkpoint checkpoints/m5/latest.pt

# the fair retest: exchange on M4's WORKING economy (m4h), gifts still unpaid
python -m sim.train --config config/m5b.yaml --run-name m5b --updates 200 \
    --policy-mode individual --init-from checkpoints/m4h/latest.pt
python -m sim.exchange --checkpoint checkpoints/m5b/latest.pt

# diagnostics for the navigation finding (open problem 1)
python -m sim.train --config config/nav_probe.yaml --run-name nav-probe --updates 200
python -m sim.train --config config/nav_compete.yaml --run-name nav-compete \
    --updates 200 --policy-mode individual --init-from checkpoints/m2/latest.pt

# faster travel: the ratio hypothesis, a NEGATIVE result. The two `--updates 0`
# runs are the controls -- they retrain nothing and exist only so the checkpoint
# carries the new world, which is how every zero-shot control here is done.
python -m sim.train --config config/nav_move.yaml --run-name nav-move \
    --updates 200 --policy-mode individual --init-from checkpoints/m2/latest.pt
python -m sim.train --config config/nav_move.yaml --run-name nav-move-zero \
    --updates 0 --policy-mode individual --init-from checkpoints/m3-masked/latest.pt
python -m sim.train --config config/nav_move.yaml --run-name nav-move-m2zero \
    --updates 0 --policy-mode individual --init-from checkpoints/m2/latest.pt
python -m sim.navigation --checkpoint checkpoints/nav-move/latest.pt --baselines
python -m sim.opportunity --checkpoint checkpoints/nav-move/latest.pt
python -m sim.evaluate --checkpoint checkpoints/nav-move/latest.pt --baselines --episodes 40

# the uptake postmortem: per-opportunity uptake, then the counterfactual that says
# whether the declined remainder is worth anything (it is not, in either world)
python -m sim.opportunity --checkpoint checkpoints/m3-masked/latest.pt --force steal
python -m sim.opportunity --checkpoint checkpoints/m4h/latest.pt --force build

# the night thread: exposure, its floor, and the dusk-approach table. A NEGATIVE --
# the clock and the shelter are both already observed and the policy ignores them.
python -m sim.navigation --checkpoint checkpoints/m4h/latest.pt --nights

# what the gap actually is: navigation, priced by steering only the move directions
python -m sim.steering --checkpoint checkpoints/m4h/latest.pt

# re-forking the chain from a navigator instead of m1. A NEGATIVE: the far-field
# navigation nav-probe brings decays as the M3 world trains foraging back in, and
# the M4 leg is a dead heat with its matched control.
python -m sim.train --config config/m3_masked.yaml --run-name m3-nav --updates 200 \
    --policy-mode individual --init-from checkpoints/nav-probe/latest.pt
python -m sim.train --config config/m3_masked.yaml --run-name m3-nav2 --updates 200 \
    --policy-mode individual --init-from checkpoints/m3-nav/latest.pt
python -m sim.train --config config/m4h.yaml --run-name m4h-nav --updates 200 \
    --policy-mode individual --init-from checkpoints/m3-nav2/latest.pt
python -m sim.train --config config/m4h.yaml --run-name m4h-direct --updates 200 \
    --policy-mode individual --init-from checkpoints/m3-masked/latest.pt   # the control

# making camping insufficient: same 48 berries over six clusters instead of three.
# A NEGATIVE, and the sharpest one -- the return on navigation rises 58% and the
# policy still will not travel. `--updates 0` is the zero-shot control again.
python -m sim.train --config config/nav_spread.yaml --run-name nav-spread-zero \
    --updates 0 --policy-mode individual --init-from checkpoints/m3-masked/latest.pt
python -m sim.train --config config/nav_spread.yaml --run-name nav-spread --updates 200 \
    --policy-mode individual --init-from checkpoints/m2/latest.pt
python -m sim.train --config config/nav_spread.yaml --run-name nav-spread2 --updates 200 \
    --policy-mode individual --init-from checkpoints/nav-spread/latest.pt
python -m sim.navigation --checkpoint checkpoints/nav-spread2/latest.pt --baselines --value
python -m sim.steering --checkpoint checkpoints/nav-spread2/latest.pt

# the advantage check: the toward-vs-away learning signal PPO actually sees,
# per distance band, against the noise its normalisation divides by
python -m sim.advantage --checkpoint checkpoints/nav-spread2/latest.pt
python -m sim.advantage --checkpoint checkpoints/m3-masked/latest.pt \
    --report viewer/reports/advantage_m3.json

# the k-tick commitment lever, a NEGATIVE with the mechanism verified. The world
# itself enforces decision_interval, so the zero-shot control needs nothing special.
python -m sim.train --config config/nav_commit.yaml --run-name nav-commit-zero \
    --updates 0 --policy-mode individual --init-from checkpoints/m3-masked/latest.pt
python -m sim.train --config config/nav_commit.yaml --run-name nav-commit --updates 200 \
    --policy-mode individual --init-from checkpoints/m2/latest.pt
python -m sim.navigation --checkpoint checkpoints/nav-commit/latest.pt --baselines --value
python -m sim.advantage --checkpoint checkpoints/nav-commit/latest.pt
python -m sim.steering --checkpoint checkpoints/nav-commit/latest.pt

# the equilibrium-stability test: fork the strong navigator into the world where
# camping cannot pay, and watch whether the far-field bands survive training
python -m sim.train --config config/nav_spread.yaml --run-name spread-nav --updates 200 \
    --policy-mode individual --init-from checkpoints/nav-probe/latest.pt
python -m sim.train --config config/nav_spread.yaml --run-name spread-nav2 --updates 200 \
    --policy-mode individual --init-from checkpoints/spread-nav/latest.pt
python -m sim.navigation --checkpoint checkpoints/spread-nav2/latest.pt --baselines

# the capacity probe: same fork at 512x512. REFUTES interference -- the wide net
# navigates better in the probe world (99.9% at 20+) and loses it faster in the
# spread world (+0.7 over floor at 200 updates), while gaining +67.7 +- 8.3 of
# survival from local competence alone.
python -m sim.train --config config/nav_probe.yaml --run-name nav-probe-512 \
    --updates 200 --set "policy.hidden_sizes=[512,512]"
python -m sim.train --config config/nav_spread.yaml --run-name spread-nav512 --updates 200 \
    --policy-mode individual --init-from checkpoints/nav-probe-512/latest.pt \
    --set "policy.hidden_sizes=[512,512]"
python -m sim.train --config config/nav_spread.yaml --run-name spread-nav512-2 --updates 200 \
    --policy-mode individual --init-from checkpoints/spread-nav512/latest.pt \
    --set "policy.hidden_sizes=[512,512]"
python -m sim.navigation --checkpoint checkpoints/spread-nav512-2/latest.pt --baselines

# the subset experiment: is the navigation prize individual or collective?
# Steers only the first n agents and pairs each agent against itself unsteered.
python -m sim.steering --checkpoint checkpoints/nav-spread2/latest.pt --subset 1,2,3,6
python -m sim.steering --checkpoint checkpoints/m4h/latest.pt --subset 1,2,3,6

# INTERLEAVING: half the envs run the probe geography in the scarce world's
# observation/action clothing. The scarce bands stay flat while the SAME weights
# navigate at 98.3% in the probe half -- the competence is present and withheld.
python -m sim.train --config config/nav_spread_mix.yaml --run-name spread-mix --updates 200 \
    --policy-mode individual --init-from checkpoints/m2/latest.pt
python -m sim.navigation --checkpoint checkpoints/spread-mix/latest.pt --baselines
# the control that says the probe leg still teaches navigation on its own (99.3%)
python -m sim.train --config config/nav_probe_mix.yaml --run-name probe-mix-solo --updates 200 \
    --policy-mode individual --init-from checkpoints/m2/latest.pt

# Island 2.0 stage 5, the third lever: persist-until-goal options. --persist has
# to be passed to BOTH the training and every runner in the evaluation, or the
# same weights are playing a different game (sim.society warns if they disagree).
python -m sim.arbiter --config config/island2/society4.yaml --run-name arb5-persist \
    --learn-agents 20 --gamma 0.997 --updates 150 --persist
python -m sim.society --config config/island2/society4.yaml --episodes 10 --persist \
    --arbiter mixed --checkpoint checkpoints/arb5-persist/latest.pt --vs utility
python -m sim.society --config config/island2/society4.yaml --episodes 10 --persist \
    --arbiter mixedrandom --vs utility          # the floor, in the same world
```

## Compute budget — runs are longer than they need to be

Measured on the shipped configs: the point at which a run's trailing 40-update
mean is within 3% of where it finishes.

| run | updates used | actually settled by |
|---|---|---|
| m1 | 300 | 175 |
| m3-masked | 300 | 89 |
| m4c | 400 | 79 |
| m5 | 200 | ~120 |

A 200-update run is **about six minutes** on this laptop (4.9M agent-steps at
~14k steps/s), which is the number to have in mind when deciding whether to run a
control. You can always afford the control.

**200 updates is enough for anything in this project**, and 250 is generous. The
300/400 figures are historical, not tuned. Cutting to 200 halves the wall-clock
and the heat for no loss of signal — the curves are flat long before the end.

Two related notes for anyone running this on a laptop:

* **`ppo.threads` defaults to 4, and that is not a compromise.** Measured, 4
  threads is as fast as 8 (40.0s vs 40.8s over 20 updates): these nets are small
  enough that the numpy env step dominates, so extra cores produce heat and
  nothing else. Results are bit-identical at any thread count and a test pins it.
* **Run experiments sequentially, not in parallel.** Two concurrent runs do not
  finish sooner in total, they just concentrate the same work into a hotter
  window — and on a fanless machine, thermal throttling can make the pair slower
  than running them back to back. Several results in these notes were produced
  by parallel pairs; that was for my convenience, not because it was faster.

**And do not repeat the long-run experiment.** `scarce-long` deliberately spent
1000 updates (3.3× budget) to test whether M3 was compute-starved. It was flat
from update 150. That question is answered; more compute is never the fix here.

## Layout notes

Built at the repo root rather than in a nested `island/` directory as the brief's
tree suggested — the repo *is* the project. Two files exist that the brief's tree
did not list:

- `sim/config.py` — YAML into frozen dataclasses. Needed by world, ppo, train and
  evaluate, so it did not belong inside any of them.
- `sim/make_fake_replay.py` — generates a replay without training, so the viewer
  could be built and verified before a policy existed (brief §5.2).
- `sim/divergence.py` + `viewer/divergence.html` — the M2 behavioural-divergence
  view.
- `sim/navigation.py` — the distance-bucketed toward-target measurement, plus a
  "could the policy see the target at all" block (target present in the
  observation, offset clipping, and the toward-share split by visibility), plus
  `--nights`: why a night tick is spent outside, with a night-behaviour floor that
  keeps the policy by day and randomises only the night. That floor exists because
  the ordinary baselines never finish a shelter and so produce no night rows.
- `sim/navigation.py --value` — V by distance to food, with hunger *and* the tick
  window held. Without the tick window the measurement inverts, because far-from-food
  ticks bunch at the start of an episode and V is then reading remaining horizon.
- `sim/steering.py` — what NAVIGATION is worth: every decision the policy makes is
  kept and only the direction of its moves is replaced, so the tick budget is
  identical and the difference is walking. The movement counterpart of
  `sim.opportunity --force`, and the source of the +89/+113 figures. `--subset
  N1,N2,...` steers only the first n agents and pairs each agent against itself
  unsteered — the individual-vs-collective split, and the source of the +173
  lone-navigator figure.
- `sim/advantage.py` — the toward-vs-away learning signal, measured as PPO sees
  it: one-step TD residual and GAE advantage per distance band, toward minus away,
  against the std that advantage normalisation divides by. Decision-aligned: under
  `decision_interval` it accumulates rewards per decision exactly as the trainer
  does. This is the tool that turned "the signal is ~0.06" into "the signal is
  positive near food, ~zero at 6–20 and negative beyond 20".
- `sim/opportunity.py` — opportunity versus uptake per action, read off the action
  mask. It exists because "the policy rarely does X" has meant three unrelated
  things here: X almost never legal (`m4f`'s `chop`, a world problem), X legal and
  declined (a policy problem), and X legal for many consecutive ticks after one
  take already used the chance (a *metric* problem — the one that actually
  applied). It prints uptake per legal tick **and per distinct opportunity**, and
  `--force ACTION` runs the counterfactual: the action taken whenever legal and
  suppressed entirely, paired per island, which is what says whether a low uptake
  is headroom at all.

**`world.decision_interval` commits every action for k ticks, enforced by
`World.step` itself.** On non-decision ticks (tick % k ≠ 0) the world repeats the
last decision whatever the caller passes, so every driver — trainer, evaluation,
replay, scripted baselines, a test stepping the world by hand — produces the same
dynamics and none can drift. `VecWorld.step` advances up to k ticks per call,
sums the rewards into the one transition PPO stores, and stops early at an episode
boundary so nothing leaks across it; γ then discounts per decision, as frame-skip
is normally trained. Reset zeroes the clock, so an episode can never start
mid-commitment. `sim.navigation` and `sim.advantage` hold their own action arrays
across sticky ticks so the action they *score* is the one that executed —
`sim.opportunity`'s per-tick uptake rates have NOT been made commitment-aware, so
do not read them in a k > 1 world without thinking. Default 1 reproduces every
earlier world bit-identically (tested, `tests/test_commitment.py`).

**`sim.evaluate` prints paired per-island differences** whenever a learned policy
runs alongside baselines. Same seed block for every policy, so island noise
cancels; it costs no extra rollouts. Read those lines rather than the ± column —
see rule 7.

**`--init-from` remaps observation columns by name, and the trigger is the layout,
not the width.** `train.py` builds `column_map(observation_layout(source),
observation_layout(target))` from the two configs and passes it to `grow_policy`, so
optional channels inserted mid-vector are handled and new columns are zeroed. Keying
the check off the width instead would let two same-width, different-feature layouts
through untouched — 3 bushes with a `blocked` channel is 26 dims and so is 4 bushes
without it — and every trained weight would read off the wrong feature silently. That
case now raises from `column_map`, which is correct: a swapped-out feature has no home
in the target. Pinned by a test.

**Checkpoints are scoped by run** (`checkpoints/<run_name>/latest.pt`). They were
flat until M2, at which point a run that forks from `checkpoints/latest.pt`
promptly overwrote the file it had just forked from. Do not flatten this again.

## Decisions, and why

**Auto-eat rather than an explicit eat action.** The brief left this open. Agents
eat automatically when `hunger < eat_threshold` and they are carrying food. Two
reasons: it keeps the action space at exactly the 10 actions the brief pins down,
and — the real reason — the eat reward is *scaled by hunger deficit*. With an
explicit action, an agent would be paid more for eating later, i.e. rewarded for
starving itself closer to death before acting. Auto-eat removes the pathology
instead of tuning around it.

**`hunger` counts down.** Per the brief it starts at 100, drains per tick, and
`<= 0` is death. So it is a satiety meter with a misleading name. The name is
kept for consistency with the brief; every docstring flags it. Do not "fix" the
direction without changing the observation normalisation and the viewer's colour
thresholds together.

**Observation distances use `observation.distance_scale` (20), not the island
diameter.** The brief said "distance-normalised". Dividing by the diameter (80)
squashes a bush 5 units away to 0.06 and destroys exactly the near-field gradient
the policy needs. Offsets are divided by 20 and clipped to ±1: full resolution
where it matters, saturation far away. Tunable in config.

**Padding is zeros, and magnitudes are [0, 1] not [-1, 1].** Berry counts and
neighbour hunger are normalised to [0, 1] so a zero-padded slot reads as "an
entity with nothing in it at zero offset", which no real entity is. Had those
channels been centred at zero, padding would have been indistinguishable from a
half-full bush sitting on top of you.

**Walking into the sea is impossible, not fatal.** Movement is projected back
onto the disc; the agent slides along the shoreline and loses the tick. The brief
wanted agents to "learn not to walk into the sea"; drowning would teach that
harder but adds a death mode the brief never specified. If foraging near the edge
looks bad, this is the knob.

**Bush layout is resampled every episode** (`bushes.resample_each_episode`).
Observations are egocentric with no absolute coordinates, so a fixed map cannot
be memorised anyway, but resampling stops the policy overfitting to one
arrangement of clusters. Set false if you want a stable map for eyeballing.

**Death is termination, the tick limit is truncation.** An agent that starves has
genuinely zero future value. An episode that hits `max_ticks` does not — those
agents were alive and would have carried on. `ppo.py` bootstraps `V(final_obs)`
into the reward at a truncation boundary. Collapsing the two teaches the policy
that the world ends at tick 600 and quietly poisons every value estimate near the
horizon.

**Dead agents keep their slot.** They are stepped every tick with a zero
observation and a forced `idle`, and dropped from GAE and every loss term via an
`active` mask. Masking the *loss* rather than the buffer keeps every tensor
rectangular. This is the part of `ppo.py` most likely to break under a change;
`tests/test_ppo.py` pins it from three directions (activity monotonicity, exact
row count reaching the update, forced-idle actions).

## Gotchas

**Berries gathered is a weak metric once a policy is competent.** Auto-eat makes
consumption homeostatic: a competent forager eats exactly as often as hunger
arithmetic demands and gathers exactly enough to top its inventory back up. The
scripted forager returns *identical* totals across wildly different noise levels
for this reason. Judge policies on lifespan and deaths; berries only separates
the incompetent from the competent, not the good from the great.

**Value loss rising is not a bug.** As the policy improves, returns grow, so the
critic's targets grow, so absolute value loss climbs. It only falls once the
return scale settles. Use `explained_variance` to judge the critic — it is
scale-free. (On the smoke-test bandit EV pins to 0 no matter what, because that
task's return is i.i.d. and genuinely unpredictable from the observation.)

**The viewer cannot list a directory.** It is a static page: no directory
listing over http, and `fetch` on `file://` is blocked outright. `sim/replay.py`
maintains `viewer/replays/index.json` and the viewer reads that. Anything that
writes a replay outside `ReplayRecorder.save` must call `update_manifest` or the
file will not appear in the dropdown. Drag-and-drop always works, including from
`file://`.

**The viewer needs network on first load** — Three.js comes from a CDN via an
import map, as the brief specified. There is no bundler and no vendored copy.

**Replay schema is versioned and the viewer fails loudly.** `SCHEMA_VERSION` in
`sim/replay.py` (v1 base, v2 construction, v3 exchange, **v4 island2 stage-4
households/stockpiles/raids**), `SUPPORTED_SCHEMA` in `viewer/main.js`. The viewer also
cross-checks the file's own `tick_fields.agent` against the column order it
expects, so reordering columns cannot silently shift what gets rendered. Bump the
version on *any* change to the tick encoding, and update both sides.

**Determinism is CPU-only and covers the whole stack.** Same seed + same config
gives byte-identical replays and identical training curves. `PPOTrainer` owns its
own `np.random.Generator` for minibatch shuffling — do not reach for global numpy
state anywhere in the training path, it silently breaks the guarantee.
`VecWorld` seeds per-env streams via `SeedSequence`, not `seed + i`.

## Reference points

Measured on the shipped `config/default.yaml`, 20 episodes, seeds 10000+.
Training run `m1`: 300 updates, 7.37M agent-steps, 4.2 min on CPU.

| policy | mean lifespan | of 600 ticks | deaths/ep | berries |
|---|---|---|---|---|
| random actions | 302.3 ± 61.6 | 50% | 5.10 | 11.5 |
| **learned (300 updates)** | **595.6 ± 19.0** | **99%** | **0.10** | **64.0** |
| scripted greedy forager | 600.0 ± 0.0 | 100% | 0.00 | 66.0 |

**1.97× the random baseline**, effectively at the scripted ceiling. Reproduce
with `python -m sim.train --run-name m1` (seed 0) and
`python -m sim.evaluate --checkpoint checkpoints/latest.pt --baselines`.

The scripted forager (`policy.greedy_forager_actions`) reads *only* the 26-dim
observation, not world state. That is on purpose: it proves the observation is
sufficient for the task, so any failure to learn is the algorithm's fault rather
than the sensor's. Treat it as a soft ceiling for memoryless reactive foraging —
it is not optimal, it just never wastes a tick.

Learning curve shape, for recognising a healthy run: lifespan sits at the random
floor for ~60 updates while explained variance climbs to ~0.8 (the critic learns
first), then lifespan rises steeply between updates 80 and 170, and pins at 600
from ~240 onward. Entropy only falls from 2.30 to ~2.04 — the policy stays
noticeably stochastic even when solving the task, because with bushes everywhere
many actions are near-equivalent and nothing punishes the indifference.

### Observed behaviour (not rewarded, worth knowing)

- **The learned policy spams `gather`** — 34% of its actions, against 1.9% for
  the scripted forager, which gathers only when it needs to. Gathering pays +1.0
  whenever an inventory slot is free, so camping a bush and grabbing
  opportunistically is straightforwardly worth more than walking away. It is
  rational given the reward, not a bug, but it means the learned policy looks
  *busier* than an optimal one.
- **Agents cluster on bush hotspots** and travel much less than the scripted
  forager (mean distance to nearest bush 2.25 vs 7.05 for random; 53% of ticks
  spent inside gather range against 11% by chance). No reward encourages
  proximity to other agents — this falls out of everyone independently wanting
  the same clusters. Do not read it as social behaviour yet; M3 is where
  competition gets a real test.

## Milestone 2 — individual brains

`PolicyGroup` (in `policy.py`) holds one `ActorCritic` per agent and dispatches on
an `agent_ids` tensor that `ppo.py` threads through collect and update. A shared
`ActorCritic` accepts the same argument and ignores it, which is the whole reason
there is still only one training loop rather than two that drift apart.

**Gradient clipping is per-brain, not global.** `PolicyGroup.clip_grad_norm`
clips each policy separately. Clipping the union would mean one agent's bad
update scales down every other agent's gradient that step — quietly coupling six
policies whose entire purpose is to be independent. If you add another optimiser
concern, ask the same question of it.

**Forking, not fresh initialisation.** M2 starts from the trained M1 shared
checkpoint copied six ways, so every agent begins competent and diverges from
there. Six randomly-initialised brains would also "diverge", but you would mostly
be measuring initialisation noise.

One Adam over all six brains is exactly equivalent to six separate Adams — its
state is per-parameter and nothing couples the policies — so the optimiser stays
simple. Throughput drops from ~30k to ~26k agent-steps/s (six small matmuls where
there was one large one), which was judged an acceptable price for a readable loop.

### Results

300 updates forked from `checkpoints/m1/latest.pt`. Both rows below are 20
episodes on the *same fixed island*, seed 10000:

| | M1 shared (control) | M2 individual |
|---|---|---|
| mean lifespan | 594.5 | 592.9 |
| **action JS divergence** | **0.0012 bits** | **0.0233 bits (19×)** |
| territory JS divergence | 0.6722 | 0.9080 |
| gather-share spread | 2.2 pts | 7.4 pts |
| mean-radius spread | 5.5 | 21.1 |
| time-in-gather-range spread | 10.5 pts | 22.8 pts |

Individual brains genuinely specialised. Agent 5 became a bush-camper (37.8%
gather, 77.3% of ticks in range, closest to bushes); agent 2 a rover (68.2%
travel, lowest gather share, best hit rate); agent 4 stayed near the island centre
(mean radius 9.0) while agent 3 ranged to 30.1.

**Read the action matrix, not the territory matrix.** Territory divergence is
already 0.67 bits for six agents *sharing one brain* — they spawn apart and each
walks to whichever cluster is nearest, so different ground is the default, not a
finding. Action divergence is near-zero under sharing and is the signal that
survives the control.

**Lifespan is saturated, so it cannot show M2 working.** Both milestones sit at
the 600-tick ceiling; M2 is not "better", it is *differentiated*. Foraging
proximity did improve (time in gather range went from 41–51% to 55–77%), but the
headline survival number has no room left to move. Any future milestone that
wants a survival signal needs a harder world first.

## Gotchas (Milestone 2)

**Territory heatmaps need a fixed map.** With `bushes.resample_each_episode` on
(the training default), every episode scatters clusters somewhere new, so
averaging positions in absolute coordinates smears every agent toward the same
centred blob. `sim/divergence.py` pins the layout by default; `--no-fixed-map`
exists but makes the territory section meaningless, and the report carries a
`fixed_map` flag that the viewer turns into a warning banner. The
map-independent statistics (action mix, hit rate, bush distance) are fine either way.

**Old checkpoints have no `mode` key.** `policy_from_config_dict` defaults them to
shared, which is what pre-M2 checkpoints are. Do not make the key required.

## Milestone 3 — competition

### Configs are layered now

`extends:` in a YAML config inherits from a parent and overrides only the keys it
restates. The chain is `m3.yaml -> scarce.yaml -> default.yaml`. This exists so
the M1/M2 world stays reproducible instead of being edited out from under the
results already documented above. Don't collapse it back into one file.

### The M3 world sits at exactly 100% of subsistence

Measured, not estimated (`config/m3.yaml`, 600 ticks):

| | berries |
|---|---|
| demand: 8 meals x 6 agents | 48 |
| supply: 12 initial + 6 bushes x 6 regrowths | 48 |

There is **no slack at all**. Perfect play feeds everyone exactly, and any
inefficiency starves somebody. The scripted forager harvests 37.4 of the 48 the
island produces (78%) and still loses 2.2 agents an episode; nothing in this
world keeps six agents alive.

This was not deliberate — when sizing `scarce.yaml` I estimated 6 meals per agent
from `eat_restore / drain_per_tick` and the true figure is 8, because an agent
eats at the *threshold* (60) rather than at empty, so each meal only buys
`(60 + 35 - 60) / 0.5 = 70` ticks rather than a full tank. Recompute demand with
`sim` rather than by hand before changing the bush economy again.

Two consequences worth carrying forward:

* **Survival here is dominated by distribution, not production.** That is exactly
  why the scripted thief beats the scripted forager despite harvesting slightly
  *less*: theft moves berries to whoever is about to eat one.
* **It is a knife-edge testbed.** Judge a policy on berries harvested as a share
  of the 48 the island produces, not on lifespan alone — lifespan compresses
  every policy into a narrow band because the food simply is not there.

### The world had to get harder first

M1 and M2 were oversupplied by roughly 7×: ~360 berries against the ~48 six
agents actually eat. Everything survived, every policy pinned to the 600-tick
ceiling, and survival stopped being able to register any effect. `scarce.yaml`
brings supply down to meet demand (6 bushes, capacity 2, regrow 100 → 48 berries)
and the scripted forager falls from 600 to ~498 with 2.4 deaths per episode. Six
bushes for six agents is deliberate: one each, if they can hold it.

### Mechanics, and one that did not work

* **`contest_bushes`** — one taker per bush per tick. **Nearly inert on its own**,
  and this is the interesting part: agents are crowded onto the same bush for
  ~2500 of 4800 ticks, yet only 3 gather attempts per 8 episodes were ever
  blocked. In a scarce world bushes are *empty* most of the time, so two agents
  almost never manage a *successful* gather on the same tick even while both
  parked on it. What agents actually compete over is who is standing there when a
  berry regrows.
* **`exclusive_bushes`** — only the agent closest to a bush may take from it. This
  is the mechanic with teeth: blocked attempts went from 3 to 1456. Standing on a
  bush now denies it. Dead agents cannot block (tested — a corpse holding a bush
  forever would be a nasty silent bug).
* **`enable_steal`** — action 10, take one berry from a neighbour within
  `steal_radius` who has some. Appended, never inserted, so every earlier action
  keeps its index and an M1/M2 checkpoint means the same thing here.
* **`observe_neighbour_food`** — widens the observation 26 → 29. Required: a policy
  that cannot tell a loaded neighbour from an empty one could only learn "rob at
  random", which resembles the behaviour without being it.

**Theft pays no reward.** Gathering pays +1.0, stealing pays 0. The brief allows
no reward terms beyond survival, so robbery has to earn its keep through the food
it yields and the eating that food enables — deliberately the harder option. If
it emerges anyway, it emerged from survival pressure rather than from us paying
for it. There is a test pinning this; if it ever starts paying out, that test
should fail loudly.

The `scripted thief` baseline (`policy.greedy_thief_actions`) is the reference:
opportunistic theft only, never chasing a victim, so it is a floor on what theft
is worth rather than a ceiling.

### Growing a policy across a milestone boundary

M3's observation and action space are both wider than M2's, so an M2 checkpoint no
longer fits. `grow_policy` copies every trained weight and **zero-initialises the
new ones**: a zeroed input column contributes nothing and a zeroed action row
gives `steal` a logit of 0 beside trained logits, so the grown policy starts out
behaving as it did and then learns to use what it has been given. The new action
is reachable rather than masked, which is what lets PPO find out whether it is
worth taking. Shrinking is refused outright.

### Results — and two things that did not work

All 20 episodes, `config/m3.yaml`, seeds 10000+:

| policy | mean lifespan | vs random | deaths/ep | steals/ep |
|---|---|---|---|---|
| random actions | 223.0 | 1.00× | 5.85 | 0 |
| **learned, forked from M2** | **452.1** | **2.03×** | 3.40 | 28.9 |
| learned, from scratch | 223.3 | 1.00× | 5.95 | 0.1 |
| scripted forager | 495.5 | 2.22× | 2.20 | 0 |
| scripted thief | 553.3 | 2.48× | 1.45 | 1099.3 |

**1. Theft did not emerge.** The learned policy steals 29 times an episode where the
scripted thief manages ~1100, and it finishes *below both* scripted references.
This is not "theft is useless" — the scripted thief proves theft is worth ~58 ticks
of extra lifespan and a death per episode. It is a credit-assignment failure: with
`reward.steal = 0.0` the chain is steal → carry → auto-eat some ticks later → don't
starve, and at γ=0.99 that signal is too weak and too delayed to compete with the
+1.0 a gather pays immediately. This is exactly the failure mode the brief predicts
for M4's construction rewards, arriving early.

### The shaping ablation, and why it is a warning for M4

`config/m3_shaped.yaml` pays a successful steal the same +1.0 a gather earns.
Everything else is identical. 20 episodes:

| | M3 (theft unpaid) | M3 shaped (theft paid) |
|---|---|---|
| steals / episode | 28.9 | **140.0** |
| berries gathered / episode | 26.4 | 23.4 |
| policy entropy (last 40 updates) | 2.167 | 1.742 |
| **mean lifespan** | **452.1** | **428.6** |
| deaths / episode | 3.40 | 3.70 |

**The shaping worked and the outcome got slightly worse.** Theft emerged — 4.8×
more of it, and the entropy drop shows the policy genuinely committing rather than
sampling it by accident, which confirms the credit-assignment diagnosis. But
survival did *not* improve: lifespan fell 452 → 429 and deaths rose.

The mechanism is worth understanding before M4 leans on shaping. Stealing moves
food between agents; it never creates any. Paying for it buys ticks spent
redistributing the same berries instead of harvesting new ones — gathering fell
26.4 → 23.4 — so the population ends up with less food overall and dies sooner.
The policy learned to steal *because stealing pays*, not because stealing helps.

Note also that the scripted thief steals ~8× more than the shaped policy and
*does* survive better (553): opportunistic theft targeted at loaded neighbours is
useful, indiscriminate theft-for-reward is not. Volume was never the point.

**The lesson for M4:** a shaped reward reliably produces the behaviour it pays
for. That is not evidence the behaviour helps. When M4 adds intermediate rewards
for gathering materials and partial construction, the shaped run has to be
compared against the unshaped one *on the terminal metric* — and the brief's
instruction to test whether shaping can be annealed away is the right instinct.

**2. The scarce world cannot be learned from scratch.** A from-scratch run lands on
*exactly* the random baseline (223.3 vs 223.0, 1.00×) after the full 7.4M steps,
with entropy still at ~2.28 of a possible ln(11)=2.40 and negative mean reward.
Agents starve before they can discover foraging, the −10 death term dominates
everything, and the policy never escapes. **The milestone chain is load-bearing,
not a narrative convenience** — M3 only works because M1 and M2 transferred
competence into it. If you make the world harder again, expect to need a
curriculum, not a bigger budget.

### Competition produced convergence, not partitioning

The prediction going in was that territoriality would show up as territory
divergence *rising*. It fell — hard: 0.9080 in M2 to **0.3628** in M3. With 6
bushes instead of 20, agents pile onto the same few spots rather than spreading
out. Territory divergence measures how *differently* agents are distributed, and
scarcity makes them all want the same ground.

What did appear is inequality. Lifespan spread went from 0 ticks in M2 (everyone
hit the ceiling) to **144 ticks**:

| | agents 3, 5 | agents 2, 4 |
|---|---|---|
| lifespan | 388.6, 392.1 | 276.0, 248.0 |
| distance to nearest bush | 1.50, 1.55 | 4.97, 3.21 |

A dominant pair holds bushes and survives; the excluded ones are pushed to the
margins and starve. **That is the territorial result — it is just expressed as who
eats rather than as who stands where.** Read the lifespan spread, not the
territory matrix, as M3's headline behavioural number.

Gather hit rate also collapsed from ~5.5% to 0.8–3.4%, which is `exclusive_bushes`
working: most gather attempts now lose to a closer agent.

### Why the learned policy loses to the scripted forager

M3's headline sits below both scripted references. That is worth understanding
before building on it, so here is the investigation, including the parts that
found nothing — they are the expensive ones to repeat.

**What the policy actually does wrong.** Run the learned policy, and at each tick
ask what the scripted forager would have done from the same observation:

| forager wanted | policy did | share of living ticks |
|---|---|---|
| move (81% of ticks) | move | 47.6% |
| | **gather** | **17.7%** |
| | **steal** | **12.8%** |
| | idle | 2.5% |

Exact agreement is 9.3%. **The policy does not travel.** Where the forager would
be walking to a berry-bearing bush, the policy stands still and mashes `gather`
or `steal`, which pay nothing from where it is standing. That is 30% of every
tick it lives.

**This is inherited, not a tuning failure.** M1's abundant world taught exactly
this: "camp a bush and spam gather" is *optimal* when there are 20 bushes and one
is always underfoot, and it is written up as an observed M1 behaviour above (34%
gather actions). M2 kept it and M3 forked from M2. In a six-bush world where only
the closest agent may harvest, that prior is actively wrong — and a policy that
never travels cannot find the four bushes nobody is standing on. It explains the
harvest gap (26 berries of 48 against the forager's 37) and the *fall* in
territory divergence at the same time.

**Seven interventions, all within noise (437–452 lifespan):**

| intervention | result |
|---|---|
| baseline (as first run) | 452 |
| fixing the feature-misalignment bug | 447 |
| `observe_bush_contested` | 450 |
| `ent_coef` 0.01 → 0.002 | 446 |
| `gamma` 0.99 → 0.995 | 445 |
| annealing `ent_coef` to 0.0005 | 437 |
| annealed entropy + `gamma` 0.997 | 442 |
| one shared brain instead of six | 449 |

Two hypotheses were killed outright rather than merely failing to help:

* **Entropy is not the constraint.** Annealing the bonus to 0.0005 left policy
  entropy at 2.03 of a possible 2.40. The policy is near-uniform *by choice* — it
  has no confident preference to express — so "the bonus forbids commitment" is
  simply wrong.
* **Distance clipping is not the constraint.** `observation.distance_scale` (20)
  was tuned for the abundant world, and a scarce island has bushes much further
  away, so the ±1 clip looked like a suspect. Measured: 0.3% of nearest-bush
  offsets saturate on one axis and 0.0% on both. Direction is intact.

**The gap survives removing competition entirely.** In `scarce.yaml` with no
blocking and no stealing — the pure foraging task — the policy still plateaus at
~27 berries against the forager's 38.7. So this is not about M3's mechanics at
all; it is about foraging in a world where food is far apart, and the failure was
simply invisible in M1 because food was never far apart. A 1000-update run
(3.3× budget) was flat from update 150, so more compute is not the answer either.

### What the seven interventions were all missing: navigation

Written up long after the fact, because it took a measurement nobody had made.
For every move a policy makes, ask whether it ends up closer to a *berry-bearing*
bush — **bucketed by how far away the agent currently is**, because an agent
hovering on its target necessarily scores ~50% and that is indistinguishable
from a random walk:

Each policy against **a random baseline measured in its own world** — the floor
is not 50% everywhere, because movement is projected back onto the disc and a
walker near the shoreline drifts inward. `--baselines` measures it. 10 episodes:

| toward-food share | 3–6 | 6–10 | 10–20 | 20+ |
|---|---|---|---|---|
| **`nav-probe`** learned | 63.9% | 73.6% | **83.0%** | **85.1%** |
| ...its random floor | 46.5% | 51.2% | 50.2% | 50.3% |
| ...**over floor** | **+17** | **+22** | **+33** | **+35** |
| **m1** learned | 62.5% | 66.5% | **78.3%** | 93.8% *(n=16)* |
| ...its random floor | 50.2% | 49.8% | 51.7% | 54.1% |
| ...**over floor** | **+12** | **+17** | **+27** | (+40) |
| **m3-masked** learned | 50.3% | 56.2% | 55.0% | **51.1%** |
| ...its random floor | 51.0% | 47.2% | 49.4% | 50.7% |
| ...**over floor** | **−1** | **+9** | **+6** | **+0** |
| m4h learned | 55.1% | 55.2% | 53.4% | 50.6% |
| scripted forager | — | — | **100%** | **100%** |

Read the "over floor" rows. m1 and the probe clear their floors by 12–35 points
and *widen* the margin with distance, which is what navigation looks like.
m3-masked clears its floor by 9 points at mid-range and by **nothing at all**
beyond 20 units — and in the scarce world the nearest berry is beyond 20 units on
9,865 of its 23,700 move-ticks, i.e. exactly where it has no signal.

Three things fall out, and the third is the one that matters.

* **Navigation is learnable in this setup.** `config/nav_probe.yaml` removes
  scarcity, competition, construction and exchange, leaving one tight cluster and
  a long walk to it. PPO gets to 86% toward-food from beyond 20 units. So the
  observation carries enough direction, the action encoding is consistent with
  it, and PPO can fit it. (The scripted forager already proved the first two by
  scoring 100% off the observation alone; the probe proves the third.)
* **M1 had navigation.** 60.9% at 3–6 units and 71.1% at 6–10, against random's
  ~49%. Not the probe's 86%, but unambiguous.
* **M3 lost it, and never got it back.** m3-masked is 56.5% / 54.3% / 50.5% at
  the ranges where M1 managed 61–71% — a random walk with a rounding error on
  top — *despite forking from M2, which forked from M1*. The milestone chain
  transferred the opportunism and dropped the navigation, and m4h still has not
  recovered it.

**This is what the M3 gap always was.** The seven interventions in the table above
tried entropy, discounting, perception of contest, brain sharing and budget.
None of them was about going anywhere, which is why they all landed inside noise.

**Why it decays: not competition — the target does not survive the walk.**
The first hypothesis was `exclusive_bushes` (only the closest agent may harvest,
so travelling means arriving second). `config/nav_compete.yaml` is m3_masked with
that single rule off, trained from the same M2 checkpoint. **It made no
difference**, which is a clean kill:

| toward-food | 3–6 | 6–10 | 10–20 | 20+ |
|---|---|---|---|---|
| m2 (what both start from) | **67.1%** | 54.9% | **72.8%** | — |
| m3-masked (exclusivity on) | 50.8% | 55.8% | 53.6% | 51.3% |
| nav-compete (**exclusivity off**) | 53.0% | 54.4% | 57.0% | 50.3% |

The pre-registered alternative is the answer, and it is quantitative. Measure how
long a berry survives on a bush against how long it takes to walk to one:

| | M1 world | **M3 scarce world** |
|---|---|---|
| bushes holding a berry | 19.8 of 20 | **2.6 of 6** |
| distance to the nearest one | 2.5 | **17.8** |
| → travel time at `move_step` 0.8 | 3.1 ticks | **22.3 ticks** |
| a berry survives, on average | 69.0 ticks | **25.7 ticks** |
| **lifetime ÷ travel time** | **22×** | **1.15×** |

In M1 a berry outlives the walk to it twenty-two times over, so setting off is
free and always pays. In the scarce world the margin is 15% — the berry is gone
about as often as not by the time you arrive, and that is *before* accounting for
variance. **Navigation stops being learnable when a target's expected lifetime is
comparable to the time it takes to reach it**, and unlearning it is correct
behaviour, not a failure. Exclusivity was irrelevant because with or without it
the berry is already eaten when you get there.

That ratio is the lever, and it can be moved without touching the food economy at
all: `world.move_step` changes travel time and nothing else. Doubling it to 1.6
takes the M3 ratio from 1.15× to ~2.3×. **It was tried as `nav-move`, the ratio
moved as predicted, and navigation did not — see the next section.**

**The metric warning is part of the finding** — see rule 6. Briefly: the first
version of this measured the toward-food share aggregated over all distances,
scored m1 at 51.3%, and concluded the project had never learned to navigate.
That was sample-weighting (17,347 near-field ticks against 199 far ones), not a
result. Bucket by distance, always, and print the counts.

### `nav-move` — the ratio moved, navigation did not, and the premise was wrong

`config/nav_move.yaml` is `m3_masked` with `world.move_step` 0.8 → 1.6 and nothing
else, trained 200 updates from the M2 checkpoint. 1.6 rather than more because
`gather_radius` is 2.0 and a step above it lets an agent straddle a bush without
ever landing in range — that would have added a movement-precision failure on top
of the thing being tested.

**The lever worked mechanically.** Measured in both worlds with one estimator
(10 episodes, learned policy driving; berries still on a bush at episode end are
censored out, equally in both):

| | m3-masked world (0.8) | **nav-move world (1.6)** |
|---|---|---|
| bushes holding a berry | 2.58 of 6 | 2.42 of 6 |
| distance to the nearest one | 18.0 | 18.4 |
| travel time | 22.5 ticks | **11.5 ticks** |
| a berry survives | 27.5 ticks | **34.0 ticks** |
| **lifetime ÷ travel time** | **1.22×** | **2.96×** |

Note the pre-registered confound resolving in the *favourable* direction: berry
lifetime is endogenous and could have fallen along with travel time, leaving the
ratio flat. It rose, so the ratio landed at 2.96× against the config's
pre-registered ~2.3×.

**Navigation did not move.** Over the random floor measured in the same world:

| toward-food, over floor | 3–6 | 6–10 | 10–20 | 20+ |
|---|---|---|---|---|
| m3-masked (0.8) | −1 | +9 | +6 | +0 |
| **nav-move (1.6)** | **+6** | **+11** | **+3** | **+1** |
| m1, for scale | +12 | +17 | +27 | (+40) |

**So the lifetime/travel ratio is not what stops navigation being learned.** The
hypothesis was quantitative, the lever moved it 2.4× (1.22× → 2.96×), and the
measurement it predicted is flat. That retires the leading explanation for the
collapse, as `nav-compete` retired competition — and it does so with the
confound measured rather than assumed.

**And training added nothing on top of the world change.** Same world, 40 paired
islands, `--baselines`:

| in the nav-move world | lifespan | vs the forager, paired |
|---|---|---|
| M2 weights, zero-shot (what training starts from) | 469.0 | −41.0 ± 9.4 (10/40) |
| **m3-masked weights, zero-shot, no training at all** | **494.6** | **−15.4 ± 10.0 (15/40)** |
| **nav-move, 200 updates from M2** | **491.0** | **−19.0 ± 10.1 (16/40)** |
| scripted forager | 510.0 | — |
| scripted thief | 541.9 | −50.8 ± 12.2 (13/40) |
| random actions | 229.5 | — |

The unchanged M3 policy scores 494.6 in the faster world with zero training;
200 updates from M2 lands at 491.0. Rule 4 again, and the same shape as `m4f` —
the world did all the work — except here the metric it was aimed at never moved.

**The premise of the open problem is what actually broke.** The handoff said the
chain's lost navigation "is what the gap to the scripted references actually is".
It is not: in the slow world the gap to the forager is −32.7 ± 10.8 paired, in the
fast world it is −15.4 ± 10.0, and **navigation is flat across that halving.**
Faster travel helps a policy that wanders more than it helps one that already
walks straight to food, so half the gap closed without a point of navigation being
recovered. The toward-food *share* therefore does not track the reference gap across
worlds.

**Do not read that as "navigation is not the gap", which is what this section said
first.** `sim.steering` later priced the competence directly — replace only the
direction of m4h's moves and it gains **+89.1 ± 11.4**, or **+113.2 ± 10.7** with a
night home-run, beating the scripted builder. What `nav-move` retires is the *ratio*
explanation and the *metric*, not the constraint.

**Perception is acquitted below 20 units, and there is a correction.**
`sim.navigation` now also reports whether the bush it scores against was
*perceivable*: the observation carries the `k_bushes` **nearest** bushes whether or
not they hold berries, so in a scarce world the nearest berry-bearing bush can
rank outside that set and be absent entirely. nav-move, learned policy:

| | 3–6 | 6–10 | 10–20 | 20+ |
|---|---|---|---|---|
| target present in the observation | 99.8% | 93.6% | 96.8% | **68.8%** |
| offset saturates the ±1 clip, one axis | 0% | 0% | 0% | **90.4%** |
| ...both axes (direction lost to a diagonal) | 0% | 0% | 0% | **15.9%** |
| toward-food **when the target is visible** | 54.3% | 58.1% | 52.3% | 50.8% |

Below 20 units the target is visible almost always, the clip cannot bite at all
(|dx| ≥ 20 requires d ≥ 20, by arithmetic), and the policy is *still* flat — so
neither perception channel explains the bands where the collapse lives. Beyond 20
units there is a real deficit: a third of the time the target is not in the
observation, and 90% of the time its direction is degraded by the clip. `m4h` is
worse on both counts (45.7% visible at 20+, with eight bushes competing for the
same four slots), which is worth knowing before anyone reads its 20+ band as a
policy failure.

**The correction.** The seven-intervention table above retires distance clipping
on "0.3% of nearest-bush offsets saturate on one axis and 0.0% on both". That
measurement was of the **nearest bush**, not the nearest **berry-bearing** bush —
in the M1 world those are the same thing (19.8 of 20 loaded) and in the scarce
world they are not (2.6 of 6). For the target that matters, at 20+ units, it is
90.4% and 15.9%. The verdict for the 3–20 bands stands, because clipping provably
cannot occur there; what is retracted is the claim that clipping was measured and
found harmless *everywhere*.

**What the numbers now point at instead: declined theft.** `sim.opportunity`
reports, per action, the share of living ticks on which the mask says it is legal
and the share of those on which the policy takes it (10 episodes):

| | legal on | taken on | |
|---|---|---|---|
| `gather`, nav-move | 1.4% of ticks | **99.2%** | nothing left to convert |
| `gather`, m3-masked | 1.3% | 98.6% | |
| `steal`, nav-move | 10.7% | **40.1%** | 1,843 legal steals declined per 10 eps |
| `steal`, m3-masked | 11.6% | 46.4% | |

**Gather uptake is saturated.** The policy takes essentially every gather the
world offers it, so its 34 berries are opportunity-bound, not choice-bound, and
the only way up through foraging is more legal ticks. Theft is the opposite: 60%
of legal steals are declined, and the scripted thief — which takes them — is
**+50.8 ± 12.2** ahead on the same islands, three times the forager's edge. The
largest measured headroom in the M3 world is theft uptake, not navigation.

### `nav-refork` — starting from a navigator changes nothing, and the world unlearns it in flight

The standing proposal was to re-fork the chain from `nav-probe` (which reaches 86%
toward-food) instead of `m1`, on the theory that the milestones lose a competence
they were never given strongly enough. **First, the blocker was imaginary:**
`--init-from` has always remapped observation columns by *name* —
`train.py` builds `column_map(observation_layout(source), observation_layout(target))`
and hands it to `grow_policy` — so nav_probe's 26 dims grow into m3's 29 (9 columns
moved) or m4h's 55 (24 moved) with new weights zeroed. The "k_bushes trap" this file
warned about did not exist. (The one real hole is closed: the remap now triggers on a
layout change rather than a width change, so a same-width feature swap raises instead
of silently misaligning.)

**Second, the experiment.** `nav-probe` → M3 (individual, masked), then M4:

| in the M3 world | updates | lifespan | berries | 3–6 | 6–10 | 10–20 | **20+** |
|---|---|---|---|---|---|---|---|
| `m3-nav` (from nav-probe) | 200 | 393.8 ± 84.5 | 18.8 | −1.0 | +5.9 | +7.6 | **+7.1** |
| `m3-nav2` (200 more) | 400 | **470.0 ± 57.9** | 29.1 | +0.5 | +8.3 | +6.4 | **+1.4** |
| `m3-masked` (from m2) | 300 | 471.6 | 29.1 | −1 | +9 | +6 | **+0** |

Bands are points over the random floor measured in that same world. Read the two
`nav` rows as a *trajectory*, because that is the finding: at 200 updates the
re-forked policy still clears the floor by 7 points beyond 20 units and is a poor
forager (18.8 berries); 200 updates later it forages exactly as well as m3-masked
(29.1 berries, 470.0 vs 471.6 lifespan) and its far-field navigation has decayed to
+1.4. **The scarce world does not fail to transfer navigation — it trades it away,
and the trade is visible in flight.**

**And the M4 leg is a dead heat.** Both lineages grown into the m4h world by the
same single 200-update jump, so the curriculum length matches:

| | lifespan (20 eps) | shelters | nights in | berries | steering headroom |
|---|---|---|---|---|---|
| `m4h-nav` (from m3-nav2) | 482.5 | 3.0 | 80% | 35.5 | +92.0 ± 11.8 |
| `m4h-direct` (from m3-masked) | 474.0 | 2.5 | 79% | 32.4 | +108.9 ± 12.0 |
| `m4h` (the long curriculum) | 490.4 | 3.0 | 83% | — | +89.1 ± 11.4 |

Paired on 40 islands, `m4h-nav` − `m4h-direct` = **+5.9 ± 7.5 ticks, better on 20 of
40** — nothing. Their navigation bands are equally flat (+4.4 / +7.2 / +3.1 / +2.6
against +0.5 / +6.0 / +3.3 / +4.6), and the steering headroom — the direct measure of
how much navigation is *missing* — is undiminished in both. Starting stronger bought
no navigation, no survival, and no reduction in the gap.

**What this rules out and what it sharpens.** It kills "the chain never had enough
navigation to carry": the chain had it and the world removed it. So the question is
no longer transfer, it is the *equilibrium* — and the arithmetic of camping is the
obvious suspect, because in `scarce.yaml` camping is sufficient:

| | berries |
|---|---|
| one cluster's regrowth income (2 bushes ÷ 100 ticks) | 1 per **50** ticks |
| one agent's demand (`eat_restore` 35 ÷ `drain` 0.5) | 1 per **70** ticks |

**A lone camper at a cluster earns 1.4× what it needs, so travelling is not just
risky — it is unnecessary.** (Six agents on three clusters is 2 per cluster, i.e. 1
per 100 ticks each against a need of 1 per 70, which is precisely the documented
inequality: a dominant pair eats and the excluded starve.) The single-variable test
is in the plan: spread the same six bushes over six clusters, so per-cluster income
halves to 1 per 100 ticks — below one agent's need — and camping cannot feed even one
agent. Supply stays exactly 48 berries.

### `nav-spread` — camping made insufficient: the incentive rose 58% and the behaviour did not move

The last explanation standing after `nav-refork` was the *equilibrium*: in
`scarce.yaml` a lone camper's cluster earns 1 berry per 50 ticks against a need of 1
per 70, so travelling is not merely risky, it is unnecessary. `config/nav_spread.yaml`
spreads the same six bushes over six clusters (`num_clusters` 3 → 6,
`bushes_per_cluster` 2 → 1), halving per-cluster income to 1 berry per 100 ticks —
below one agent's need — while **supply stays exactly 48 berries**.

**The world change did what it was supposed to.** Zero-shot, the unchanged m3-masked
policy in the spread world, 20 paired islands:

| | its own world | **the spread world** |
|---|---|---|
| m3-masked weights, zero-shot | 464.7 | 436.3 |
| scripted forager | 497.4 | **509.5** |
| **paired gap to the forager** | **−32.7 ± 10.8** | **−73.2 ± 13.7** |

The forager gets slightly *better* and the camper 28 ticks worse, so the world now
discriminates exactly as intended: it punishes standing still without being harder
for something that walks.

**Navigation did not appear.** Points over the random floor measured in the same
world, and the run was extended to 400 updates because `nav-refork` had shown 200
from a fresh start can be under-trained:

| | updates | lifespan | berries | 3–6 | 6–10 | **10–20** | **20+** |
|---|---|---|---|---|---|---|---|
| `nav-spread` | 200 | 424.8 | 22.6 | +2.5 | +4.4 | **+0.8** | **+1.1** |
| `nav-spread2` | 400 | 433.4 | 23.5 | +7.3 | +3.2 | **+1.7** | **+1.7** |
| m3-masked, in its own world | 300 | 471.6 | 29.1 | −1 | +9 | **+6** | **+0** |

And training bought nothing at all: `nav-spread2` against the *zero-shot* m3-masked
weights in the same world is **+7.4 ± 8.5 ticks, better on 22 of 40 islands**. Four
hundred updates in a world that starves campers produced a policy that still does not
travel — it simply eats less (23.5 berries against the forager's 38.8).

**The sharpest way to see it: the incentive is now bigger and still unclaimed.**
`sim.steering`, which replaces only the direction of the policy's moves:

| steering headroom | m4h world | **spread world** |
|---|---|---|
| steered at food − learned | +89.1 ± 11.4 (38/40) | **+140.9 ± 8.2 (40/40)** |
| steered at food, absolute | 556.3 | **582.9** |
| the scripted forager, same world | 465.8 | 511.5 |

**Navigation is worth 58% more here than in the m4h world, on every single island,
and PPO left all of it on the table.** That is as clean a statement as this project
has: raising the return on a competence does not make PPO acquire it.

**And the critic is not the problem either.** The remaining hypothesis was that V
cannot separate "far from food" from "on food", leaving no gradient to climb. It can.
V by distance to the nearest berry-bearing bush, with hunger held in [55, 85] *and*
the tick window held to 150–450 (both conditions matter — see below):

| `nav-spread2` | 0–3 | 3–6 | 6–10 | 10–20 | 20+ |
|---|---|---|---|---|---|
| V | **2.27** | 1.43 | 1.16 | 1.04 | 1.21 |
| vs the nearest band | — | −0.84 | −1.11 | **−1.23** | −1.05 |

A clean monotone fall of ~1.2 in value from standing on food to being 10–20 units
away, so the gradient exists. `m3-masked` has the same shape (2.15 → 1.23). Two
caveats kept: the 20+ band ticks back up, and the `nav-probe` control is
**inconclusive** — that policy is at its cluster on almost every mid-episode tick, so
its far bands have too few samples to compare. Without the tick window the whole
measurement inverts, because far-from-food ticks bunch at the start of an episode and
V is reading remaining horizon; `sim.navigation --value` holds both.

**What is left, and it is now arithmetic rather than a hypothesis.** The value
difference across the whole 10–20 band is ~1.2, and one move covers 0.8 units, so a
single step toward food is worth about **1.2 × 0.8 / 15 ≈ 0.06** of advantage — against
per-tick reward noise of ±1.0 from a gather landing or not. The gradient is real and
roughly twenty times smaller than the noise it has to be found in. That points at
temporal resolution rather than at the world: a policy that commits to a direction for
k ticks, or a longer-horizon advantage, would see the same slope at twenty times the
step size. **Verified the next session, and half wrong.** The noise figure was right
(std of the GAE advantage is 0.851); the slope is not 0.06 everywhere — it is band-local,
~5× shallower than the average-slope arithmetic beyond 10 units and **negative** beyond
20 — and the k-tick lever was run (`nav-commit`) and is a clean negative with the
mechanism confirmed. See the next section.

### The advantage check and `nav-commit` — the signal measured, multiplied 4×, and still pointing home

**The "0.06 against ±1.0" arithmetic was measured instead of trusted, and half of
it was wrong.** `sim.advantage` computes, for every move an alive agent makes, the
two quantities PPO actually trains on — the one-step TD residual
δ = r + γV(s′) − V(s) and the GAE advantage with the checkpoint's own γ and λ —
and takes E[· | toward food] − E[· | away] by current distance (20 episodes, tick
window 150–450 held as in `--value`):

| nav-spread2, toward − away | 3–6 | 6–10 | 10–20 | 20+ |
|---|---|---|---|---|
| δ gap | −0.003 ± 0.016 | +0.006 ± 0.007 | **+0.013 ± 0.006** | **−0.028 ± 0.007** |
| GAE-advantage gap | **+0.083 ± 0.045** | −0.024 ± 0.026 | +0.010 ± 0.021 | −0.010 ± 0.022 |

The noise was right: the std of the GAE advantage over all transitions — what
per-minibatch normalisation divides by — is **0.851**. The signal was not "0.06
everywhere": it is **band-local**. Positive near food (the GAE gap at 3–6 held
+0.083…+0.096 across three runs of the tool, and that is the one band where
nav-spread2 clears its floor, +7.3). Near zero at 6–20: the 1.2-over-15-units
arithmetic used the *average* slope of a convex V curve, and the local slope
beyond 10 units is ~5× shallower. And **negative beyond 20**. m3-masked has the
same shape, more sharply (δ gaps +0.033 / +0.028 / −0.008 / −0.026 going outward
from 3–6, with the GAE gap +0.179 ± 0.039 at 3–6 and zero-or-negative past 6).
*Read the structure, not single cells*: the policy samples, and before the tool
seeded torch, three runs moved individual band gaps by up to ±0.05 — what is
stable across every run is near-field positive, mid-band ~zero, and the 20+ δ gap
at −0.024…−0.028.

**The anti-signal at 20+ is the camping equilibrium pricing itself, not a critic
bug.** Split the 20+ movers by whether they stand within 3 units of *any* bush,
loaded or not: **82% are parked at an empty bush**, and for them a step toward
distant food scores δ **−0.045 ± 0.008** — walking off means abandoning a regrowth
queue, and under the current continuation policy, which wanders, that is a genuine
loss, priced correctly. In the open the gap is −0.019 ± 0.012, i.e. flat. So
beyond ~6 units the on-policy gradient points *into* the camp from everywhere:
A^π is computed under a π that cannot navigate, leaving really is bad *for that
π*, and the local optimum defends itself.

**The k-tick lever multiplied the signal exactly as promised and bought no
learning.** `world.decision_interval: 4` makes every action persist four ticks —
enforced by `World.step` itself, so the trainer, every eval loop, replays and the
scripted baselines all live under the same commitment — and PPO stores one
transition per decision with the four ticks' rewards summed (`config/nav_commit.yaml`
= nav_spread + that one key; 200 updates from m2, the nav-spread lineage).
Mechanically it delivered everything the arithmetic asked for:

| per decision, at 10–20 | k=1 (nav-spread2) | **k=4 (nav-commit)** |
|---|---|---|
| δ gap | +0.013 ± 0.006 | **+0.061 ± 0.033 (×4.7 ≈ k)** |
| δ noise (std, all transitions) | 0.310 | 0.841 (×2.7 ≈ √k, a little over — rewards within a commitment correlate) |
| → signal-to-noise | 0.042 | **0.073 (×1.7)** |
| V fall, on-food → 10–20 | −1.23 | **−2.24** |
| GAE-advantage gap | +0.010 ± 0.021 | +0.240 ± 0.090 (unstable between runs: an unseeded run read +0.014 ± 0.105) |
| **δ gap at 20+** | **−0.028 ± 0.007** | **−0.161 ± 0.025 (×5.7 — the repulsion multiplied faster)** |

And the outcome, measured against this world's own floor and references:

| in the commit world | lifespan | bands over floor 3–6 / 6–10 / 10–20 / 20+ |
|---|---|---|
| m3-masked weights, zero-shot, no training | **485.9** | +12.3 / +7.6 / +3.5 / +6.4 |
| **nav-commit, 200 updates from m2** | **492.9** | +12.2 / +6.4 / +4.8 / +6.3 |
| scripted forager | 504.5 | 92.9% absolute at 20+ |

Paired, trained − zero-shot control: **−1.5 ± 7.5 (better on 21/40)**, with berries
identical (28.8 both) — training reallocated ~80 steals an episode (17 → 96) and
bought nothing, consistent with the steal margin being worth ~0. The toward-food
bands of the trained policy are the untrained control's bands. Steering headroom is
intact at **+93.6 ± 7.5 (39/40)**. Four hundred more updates were not run: the curve
was flat from update ~40 and rule 4 stands.

**What commitment did buy arrived zero-shot, and it is nav-move's shape again.**
The unchanged m3-masked weights under k=4 score 485.9 against 436.3 at k=1 —
**~+50 ticks with no training**, because commitment turns a near-uniform move
distribution into 4-tick ballistic segments that cover ground roughly twice as
fast. That alone collapsed the paired gap to the forager from −73.2 to
**−11.6 ± 8.6 (19/40)**. Faster effective travel helps a wanderer; direction stays
unlearned. (The forager pays a small commitment tax itself: 509.5 → 499.7 on the
same 20 islands.)

**And the +50 does not generalise to m4h.** The same interface dropped onto the
flagship world with zero training (`m4h-commit-zero`, via
`--set world.decision_interval=4`) is worth **+6.2 — 496.6 ± 67.6 against the
canonical 490.4 on the same 20 islands, inside noise** — with nights indoors
falling 83% → 69% and steals 115 → 19 while berries rise. Ballistic diffusion
pays where the map forces long crossings (six spread single-bush clusters) and
pays nothing where everything the policy needs already sits at its cluster
(`materials_at_clusters`). Do not adopt k=4 as a general upgrade.

**What this retires, and the diagnosis that replaces "signal too small".** At k=4
the mid-band per-decision signal is real and measurable (δ +0.061 ± 0.033, ~7% of
the noise) — and nothing moved, so temporal resolution was not the binding
constraint. What survives every run is the **sign structure**: negative beyond 20
units, because the critic accurately prices leaving a camp as a loss under a
policy that cannot walk straight — and commitment multiplied that wrong-way
signal *faster* than the right one (δ at 20+: −0.028 → −0.161, ×5.7). Bigger k is
not the follow-up — the config header pre-registered that, and k=8 would commit
6.4 units past a 2.0 gather radius. PPO's one-step improvement cannot cross a
desert its own valuation says not to enter: π cannot navigate, so A^π says stay,
so π never learns to navigate.

### `spread-nav` — the equilibrium explanation dies too: a navigator decays even where camping cannot pay

The self-defending-optimum diagnosis above has an obvious escape hatch: hand the
world a policy that already navigates, so A^π prices travel *positively* from the
start. `nav-refork` tried that and the competence decayed — but its destination
was the M3 world, where camping is sufficient (income 1.4× need) and unlearning
navigation is arguably correct play. The spread world is the missing cell: camping
verifiably cannot feed even one agent (0.7× need) and steering prices navigation
at +140.9 there. Pre-registered before the run: if the far-field bands survive 400
updates here, the camping equilibrium is what erases navigation; if they decay
like nav-refork's, the equilibrium explanation is dead and what is left is
**retention itself**.

`spread-nav` forks `nav-probe` (86% toward-food at 20+) into `nav_spread.yaml`,
200 then 400 updates. Bands are points over the random floor measured in the same
world:

| | updates | lifespan | berries | 3–6 | 6–10 | 10–20 | **20+** |
|---|---|---|---|---|---|---|---|
| `spread-nav` | 200 | 334.5 ± 65.9 | 12.6 | +1.1 | +4.5 | +3.2 | **+6.3** |
| `spread-nav2` | 400 | 376.2 ± 64.6 | 17.4 | +1.0 | +3.9 | +1.8 | **+2.1** |
| `m3-nav` → `m3-nav2` (the refork, for comparison) | 200→400 | — | — | | | | **+7.1 → +1.4** |
| `nav-spread2` (from m2, same world) | 400 | 433.4 | 23.5 | +7.3 | +3.2 | +1.7 | +1.7 |

**Nearly the same decay trajectory as the refork, in the world built to prevent
it.** Survival climbs (334 → 376, steals 70 → 85) while the far-field competence
the policy arrived with drains away — the world trades navigation for local
scarcity skills even where standing still is verifiably fatal. Note also the
lineage cost: the probe lineage lands *below* the m2 lineage (376 vs 433), because
it never had the M1/M2 scarcity competence — rule 3 cuts both ways.

**What is actually left, with every mechanism-level explanation now dead.** Not
competition, not the lifetime/travel ratio, not perception below 20 units, not
entropy/γ/budget/sharing, not the starting policy, not the camping *equilibrium*,
not the critic's gradient existing, not the per-decision signal size. The one
suspect still standing is the **on-policy sample distribution itself**: ~90% of
transitions are near-field ticks (51k-transition batches with a few hundred
far-field rows), whose gradients simultaneously (a) price leaving the camp as a
loss under a wandering continuation policy and (b) overwrite whatever far-field
structure the shared 128×128 trunk carried, by sheer sample weight. It is rule 6's
87:1 weighting problem operating on the *gradient* instead of on a metric — and it
explains the one world where navigation survives: in `nav_probe` the walk IS the
dominant state, so the distribution protects the competence instead of eroding it.
The state distribution is the curriculum. Untested corollaries: more capacity
should slow the decay (train a wider nav-probe, fork it, watch the 20+ band), and
no reward-side lever should be able to fix it. **The capacity corollary was tested
the same day and refuted — see the next section.**

### The capacity probe — a wider net unlearns navigation just as completely, and banks +68 of survival instead

The interference hypothesis made one cheap prediction: a wider trunk has room for
both competences, so forking a wider navigator into the spread world should slow
the far-field decay. Run exactly as `spread-nav` was, with one change
(`--set policy.hidden_sizes=[512,512]`, ~11× the parameters):

**The wide probe navigates better than the narrow one ever did.** In the probe
world, 200 updates from scratch: **99.9% toward-food at 20+** (n=820, floor
50.3%), where the 128-wide probe reached 86.1%. Capacity helps *acquisition*
where the state distribution supports navigation — worth knowing on its own.

**And the spread world strips it anyway, faster if anything.** Bands over the
floor measured in the same world:

| | updates | lifespan | berries | 3–6 | 6–10 | 10–20 | **20+** |
|---|---|---|---|---|---|---|---|
| `spread-nav512` | 200 | 398.6 ± 63.1 | 19.1 | +2.3 | +4.1 | +3.3 | **+0.7** |
| `spread-nav512-2` | 400 | 442.1 ± 50.9 | 24.1 | +2.6 | +6.0 | +1.1 | **+0.9** |
| `spread-nav` / `-2` (128-wide) | 200 → 400 | 334.5 → 376.2 | 12.6 → 17.4 | | | | **+6.3 → +2.1** |

At 200 updates the wide net's far-field navigation is already gone (+0.7), where
the narrow one still held +6.3 — despite starting from 99.9% instead of 86%. **So
the decay is not passive interference: a net with room to spare followed the same
gradient to the same place, faster.** The unlearning is gradient-following. That
is consistent with what `sim.advantage` measured directly — the on-policy δ gap
at 20+ is *negative* — and it retires "capacity" as a lever for retention.

**What capacity did buy is survival, and a lot of it.** Paired on 40 islands,
`spread-nav512-2` − `spread-nav2`: **+67.7 ± 8.3, better on 36 of 40** (444.6 vs
376.9, berries 24.3 vs 16.9, deaths 3.80 vs 4.90) — at or above the m2 lineage's
433.4 in this world (unpaired, 20 eps). A bigger trunk converts straight into
better *local* competence while far-field navigation sits at the floor. The two
are simply not in tension for PPO here: it spends all capacity on the behaviour
its own advantage endorses.

**Where this leaves the diagnosis.** Part (b) of the sample-distribution story —
overwrite-by-sample-weight — is dead: capacity would have helped. Part (a) stands
alone and sharper: **the on-policy advantage actively trains far-field navigation
away**, at any capacity, because A^π under the training distribution prices
toward-moves negative beyond 20 units. The open question that now matters is
whether that pricing is *correct*: `sim.steering` steers all six agents at once
(+140.9 here), which is a group counterfactual, while PPO's gradient sees the
individual one. **Measured next — see the subset experiment below.**

### One steered agent — the prize is individual, first-mover-largest, and invisible to one-step improvement

`sim.steering --subset` steers only the first n agents' moves at food and pairs
every agent against *itself* unsteered on the same islands (`run_per_agent`).
The question it decides: is the +89…+141 headroom an individual prize PPO's
gradient fails to climb, or a collective one no individual gradient can see?

| steer the first n of 6 | steered agents, paired | unsteered agents | population |
|---|---|---|---|
| **spread world, n=1** | **+173.0 ± 19.0 (30/40)** | **+19.5 ± 6.4** | +45.1 ± 6.1 |
| spread, n=2 | +172.8 ± 15.5 (34/40) | +41.8 ± 8.6 | +85.5 ± 6.2 |
| spread, n=3 | +156.1 ± 11.6 (38/40) | +64.9 ± 10.3 | +110.5 ± 7.1 |
| spread, n=6 | +140.9 ± 8.2 (40/40) | — | +140.9 ± 8.2 |
| **m4h, n=1** | **+138.3 ± 23.0 (22/40)** | **+12.5 ± 6.6** | +33.5 ± 7.3 |
| m4h, n=3 | +107.7 ± 17.4 (35/40) | +61.4 ± 8.5 | +84.5 ± 9.1 |
| m4h, n=6 | +89.1 ± 11.4 (37/40) | — | +89.1 ± 11.4 |

**The coordination reading is dead, emphatically.** A lone navigator among five
campers gains **more** than each agent does when all six navigate — +173 against
+141 in the spread world — and the five campers *also gain* (+19.5 ± 6.4; the
navigator quits the local regrowth queue and leaves it to them). Navigation is
the best individual strategy in every world measured, largest for the first
mover, with positive spillover — so there is no tragedy-of-the-commons blocking
it, and training the whole population toward it is not self-defeating. The usual
caveat rides along: steering reads world state, so +173 is the perfect-navigation
upper bound for an individual, not what the observation supports.

**And it does not contradict the advantage check — together they name the
barrier exactly.** The one-step δ gap at 20+ is genuinely negative under π: one
toward-step followed by the wandering policy really does not pay, and the critic
prices that correctly. A *sustained* deviation pays +173. Both measurements are
right; they differ only in the length of the deviation. So the failure is
neither the world, nor the signal, nor the critic, nor capacity: **one-step
policy improvement cannot see a ~20-step prize.** The desert is ~19–25 ticks
wide, every individual step into it is priced ≤ 0 by an accurate critic of the
current policy, and PPO climbs one step at a time. `nav-commit`'s k=4 does not
escape this — it shortens the desert to 5–6 decisions, each still priced by the
same one-step rule — which is why uniform frame-skip failed while a full-episode
override pays.

### `spread-mix` — interleaving: the SAME weights navigate at 98% and camp at chance, in one policy

The lever the subset result pointed at: stop *sequencing* the worlds and put both
in every batch, so a positive navigation gradient is present while the scarce
world's negative one is applied. `cfg.mix` runs a share of the training envs on a
second world config — one policy, both distributions, same update.

**The blocker was real and is not the one this file warned about.** `nav_probe`
extends `default.yaml` at 26 dims / 10 actions; the spread world is 29 / 11. One
policy cannot train on both, so `config/nav_probe_mix.yaml` transplants the
probe's *geography* onto m3_masked's interface — identical observation layout,
action set, masking and agent count, with `exclusive_bushes`/`contest_bushes`
turned off (neither adds a channel) so arriving always pays on the probe leg.
`config/nav_spread_mix.yaml` is `nav_spread` plus `mix.fraction: 0.5`.

**In the scarce world it changed nothing.** 200 updates from m2, bands over the
floor measured in the spread world:

| | 3–6 | 6–10 | 10–20 | 20+ | lifespan |
|---|---|---|---|---|---|
| `spread-mix` (50% interleaved) | +3.0 | +5.6 | +1.5 | **+0.3** | 431.7 |
| `nav-spread` (same budget, no mix) | +2.5 | +4.4 | +0.8 | +1.1 | 424.8 |

Note the confound resolving *favourably* and being irrelevant anyway: the mixed
run sees **half** the scarce-world transitions per update and still matches on
lifespan, so dilution is not hiding an effect. The bands are flat.

**And then the same checkpoint measured in the other half, which is the result.**
The layouts are identical by construction, so the same weights run in either
world with no remapping:

| `spread-mix`, one policy | 3–6 | 6–10 | 10–20 | **20+** |
|---|---|---|---|---|
| in the **spread** world, over floor | +3.6 | +3.6 | +2.9 | **+1.2** (51.5%) |
| in the **probe** world, over floor | +22.9 | +36.0 | +45.1 | **+47.6 (98.3%)** |

**The competence is not missing, not decayed, and not out of reach. It is
present, at 98.3%, in the very weights that walk at chance in the scarce world.**
Every framing this file has carried is now too weak: the chain does not "lose"
navigation and the scarce world does not "erase" it — under interleaving the
policy *keeps* navigation and *declines to apply it* where it would pay +173.
The behaviour is conditional on what the observation says about the food, and
both branches are locally rational: walk when arrival pays, camp when the
one-step gradient says camp. That is exactly the policy `sim.advantage`
described, now demonstrated to coexist with full navigation competence in one
network.

**What this rules out, and what is left.** Not representation (the same trunk
holds both behaviours). Not capacity (already refuted, and now doubly). Not
retention, and note the stronger form: the probe leg did not merely *preserve*
navigation, it **taught** it — m2, which both lineages start from, sits at 72.8%
in the 10–20 band and this policy reaches 96.7% there — and 200 updates of
scarce-world gradient applied to the same weights did not take it back. Not
perception, not the world's incentives, not coordination. What remains is the thing three measurements now agree on: **PPO
will not take a ~20-step excursion whose every prefix its own accurate critic
prices at ≤ 0, even holding the finished competence to do it.** The gap between
+1.2 and +47.6 in one network is the sharpest form of that statement in the
file, and it is an argument about the improvement operator, not about the agent.

**The control says the probe leg is doing what it should.** `probe-mix-solo`
(the same world trained alone from m2, 200 updates) reaches 99.3% at 20+, so
interleaving cost the probe leg essentially nothing — 98.3% against 99.3%. The
policy learned **both branches near-optimally at once**, which is the part that
makes the flat scarce-world bands a choice rather than a failure to fit.

**And the 2×2 says what the switch is conditioned on: expected payoff of the
trip, not a perceptual quirk.** The probe and spread worlds differ in two things
at once — cluster geometry and bush depth — so they were crossed, zero-shot, on
the `spread-mix` weights with no training:

| | thin bushes (cap 2) | deep bushes (cap 12) |
|---|---|---|
| **1 tight cluster** | +47.4 (n=697) | +47.2 (n=681) |
| **6 spread clusters** | **−0.6 (n=9434)** | +46.3 (n=99) |

*(points over the floor at 20+, measured per cell; the deep cells have small n
because a policy that walks to food and stays generates few far-field ticks —
read them against their floors, which is what the numbers already are.)*

**Three of the four cells navigate at +46 to +50. The one that does not is
exactly the real scarce world.** Neither factor explains it alone: a tight
cluster of *thin* bushes still gets +47.4, and *spread* deep bushes still get
+46.3. Only the conjunction — food that is both far away and individually
meagre — turns navigation off. That is not a broken trigger; it is close to the
correct *local* value judgement, and it is the same quantity `sim.advantage`
measured: a trip is worth taking when arrival pays enough, and in the spread
world one thin bush at the end of a 20-tick walk does not, one step at a time.

So the policy has a working navigation controller and deploys it exactly where
one-step reasoning endorses it. The single world where it withholds it is the
world where `sim.steering --subset` says a *sustained* crossing pays
**+173.0 ± 19.0**. Locally right, globally wrong, by the exact margin this
investigation has been chasing.

### Declined theft was a counting artefact — and the steals it declines are worth nothing

Plan item 1 said theft uptake was "the largest measured headroom in the M3 world":
`steal` legal on 11.6% of living ticks, taken on 46.4%, against a scripted thief
+50.8 ± 12.2 ahead. **Both halves of that framing are now dead**, and the second
kill is the useful one.

**1. The rate was confounded.** `steal` uptake is 48.1% per legal *tick* and
**85.0% per distinct opportunity** — a "span", i.e. a maximal run of consecutive
ticks on which the mask kept saying yes to one agent. Steal spans average 3.4
ticks, so the per-tick number divides one chance by however long it persisted.
`gather` spans last 1.2 ticks, which is why its 97.8% per tick and 99.3% per span
agree — and why the error was invisible until an action with long spans turned up.
`sim.opportunity` now prints both columns and says which to read. Rule 6 again, in
a new place.

**2. The remaining steals are worth nothing.** `sim.opportunity --force steal`
runs the same weights with `steal` taken whenever legal, and again suppressed
entirely, with common random numbers and paired islands:

| m3-masked, 40 paired islands | lifespan | steals/ep | vs the unmodified policy |
|---|---|---|---|
| learned (control) | 463.9 | 125.7 | — |
| **ALWAYS steal** | 455.6 | **422.9** | **−8.3 ± 4.1 (better on 10/40)** |
| **NEVER steal** | 448.3 | 0 | **−15.6 ± 5.9 (better on 13/40)** |
| scripted thief | 551.7 | 1111.0 | +87.8 ± 10.3 |

**3.4× the theft buys nothing** (if anything it costs 8 ticks), while removing
theft costs 15.6. So the ~45% rate is near its own optimum: the steals the policy
takes are worth ~16 ticks of life and the ones it declines are worth nothing. There
was no headroom to convert.

**And the refusal was never a decision.** Mean P(steal) on legal ticks is 45.1%
against a 44.4% sampled uptake — the policy holds a preference and the dice do the
rest, exactly as `m4h`'s `build` did. Note the direction, which is opposite to
`build`: `steal` is the *argmax* action on only 27% of its legal ticks, so argmax
play would steal **less**, not more. Hunger does not select either — 95.6% of
declined steals and 95.7% of taken ones happen above the eat threshold, with mean
hunger 74.2 against 73.4. The policy is not passing on steals it needs.

**Where the thief's +87.8 actually comes from: opportunity, not uptake.** Forcing
every legal steal gets 423 an episode against the thief's 1111. The thief has ~2.6×
more *chances* because it positions itself next to loaded neighbours — it
manufactures the opportunity rather than converting it. That is a behaviour, not a
rate, and it is the same shape as everything else in this file: the gap is in going
somewhere.

### The fix that worked: action masking

`competition.mask_invalid_actions` hides `gather` and `steal` when they cannot
possibly succeed (no berry in range / no loaded neighbour in reach; moving and
idling are always available, dead agents keep `idle` so no row is ever fully
masked). It is not a reward term and not a hint about what is best — the
observation says what is *there*, the mask says what is *reachable*. The masked
logits use −1e8 rather than −inf so a fully-masked row cannot mint NaN gradients,
and PPO stores the rollout masks because the ratio must be computed against the
behaviour policy, which was masked.

| | unmasked (m3-fork) | masked (m3-masked) |
|---|---|---|
| mean lifespan | 452.1 | **471.6 (2.11×)** |
| deaths / ep | 3.40 | 3.10 |
| berries / ep | 26.4 | 29.1 |
| steals / ep — **still unpaid** | 28.9 | **115.5** |
| doomed gathers (% of ticks) | 17.7% | **0.0%** |
| gather hit rate | 0.8–3.4% | **50–98%** |

Two things worth reading twice. **Theft finally emerged without being paid for**
— masking made `steal` only ever appear when a loaded victim is in reach, so its
empirical return became visible to PPO, and usage rose 4× with `reward.steal`
still 0.0. And the behavioural profile flipped from stand-and-mash to travel
(60% → 87% of ticks moving).

Stacking the contested-bush channel and entropy annealing *on top of* masking
(600 updates, `m3-final`) gave 456.9 — nothing again. **`checkpoints/m3-masked`
is the canonical M3 checkpoint.**

**The honest residual: 471.6 still trails the scripted forager (495.5) and thief
(553.3).** On 40 paired islands that is −32.7 ± 10.8 against the forager and
−87.0 ± 11.3 against the thief. The gap is not doomed actions (there are none
left), not entropy, not budget, not perception of any mechanic we could name, and
— since `nav-move` — not navigation either. Lifespan spread on the fixed map is
307–550: one agent roves at a 98% hit rate while others get excluded and starve,
so part of the shortfall lives in the crowding/exclusion dynamics. The one piece
of it that is now *measured* rather than guessed at is theft uptake: `steal` is
legal on 11.6% of ticks and taken on 46.4%, and the thief that takes them is 87
ticks ahead. Masking is where principled single-change fixes stopped paying; see
plan item 1 for where to look next.

## Milestone 4 — multi-resource + construction

Wood and stone as depletable nodes, communal shelter sites, and a day/night
cycle where the unsheltered drain hunger at 3×. `chop`/`mine`/`build` appended as
actions 11–13; observation 29 → 55. Replay schema v2. All off by default, so
M1–M3 worlds stay bit-identical.

### The economy, sized in code

Per the M3 postmortem rule, demand was computed with `sim` rather than by hand:

| | berries |
|---|---|
| sheltered demand (8 meals × 6 agents) | 48 |
| **supply** (8 bushes × 2 cap + 8 × 6 regrowths) | **64** |
| exposed demand (sleeping rough every night) | 72 |

Supply sits *between* the two, so a population that shelters can afford the ticks
construction costs and one that does not starves. Shelter is load-bearing by
arithmetic, not decoration.

### Results — construction partially emerged

20 episodes each. The scripted builder is the bar; it forages first, runs home at
dusk, and feeds the most-finished site (targeting the *nearest* site instead
spreads material across three and completes almost nothing — worth knowing).

| policy | lifespan | deaths | shelters/ep | nights indoors |
|---|---|---|---|---|
| scripted builder | **529.8** | 2.05 | **1.9** | **92%** |
| scripted forager | 457.6 | 3.12 | 0 | 0% |
| **learned, shaped (m4c)** | **393.5** | 4.70 | 0.1 | **22%** |
| learned, unshaped control | 362.5 | 5.05 | 0.0 | 2% |

**The shaped run beat its control on the terminal metric** — +31 ticks of life,
22% of night ticks under shelter against 2% — which is the *opposite* of the M3
shaping ablation, where paying for theft produced more theft and less survival.
Here the shaping bought a real outcome, not just the behaviour it paid for.

**But full shelters essentially never complete** (0.1 per episode against the
builder's 1.9), and the learned policy remains far below the scripted reference.
Agents deliver materials and shelter under half-built walls; they do not finish
the job. That is the honest headline.

### Three attempts, and what each one taught

| attempt | deliveries/ep | shelters/ep | nights in |
|---|---|---|---|
| `m4` — sites scattered, cheap shaping | 0.56 | 0.012 | 0.3% |
| `m4b` — sites on the berry clusters | 2.00 | 0.056 | 2.5% |
| `m4c` — + partial shelter protection | 1.94 | 0.064 | **16%** |

Neither fix touched the shaping coefficients, deliberately — the M3 ablation is
the standing warning against that lever. Both were the same *kind* of move that
solved M3: **change the shape of the problem, do not pay more at the summit.**

* **`m4b` deleted the uncreditable walk.** With sites scattered independently, a
  loaded agent had to cross open ground to a place it otherwise never went, and
  that leg earned nothing — structurally identical to M3's doomed gathers.
  Putting sites on the clusters agents already live at raised deliveries 3.6×.
* **`m4c` removed the cliff.** A site costs four units and only the fourth bought
  anything, so three quarters of the work was invisible to the value function.
  Scaling protection with build progress made the landscape continuous, and
  night protection went 2.5% → 16%. **It also removed the summit, which nobody
  noticed for two milestones** — a linear protection curve makes the last unit
  worth exactly what the first one is, so "nobody completes a shelter" stopped
  being a defect and became correct play. See "What is left" below.

An earlier sizing (6-unit sites, shaping 0.3/0.5/2.0) produced *zero* completions
in 200 updates; CSVs in `runs/_m4_probe1`. At 0.3 a material action loses to a
+1.0 gather everywhere the two compete.

### The annealing test — the shaping was a real bootstrap

The brief asks whether M4's shaping can be annealed away once it has done its
job. `config/m4c_anneal.yaml` continues the shaped policy for 200 more updates
with all four shaping terms at 0.0, changing nothing else.

| | deliveries/ep | shelters/ep | nights indoors | lifespan |
|---|---|---|---|---|
| m4c shaped | 1.94 | 0.064 | 16.1% | 377.6 |
| **m4c annealed (nothing paid)** | **2.21** | **0.091** | **18.0%** | **382.3** |
| m4c unshaped control | 0.73 | 0.011 | 4.7% | 368.8 |

**The behaviour survived, and slightly improved.** Two hundred updates with
nothing paying for wood, stone, delivery or completion, and construction stayed
at the shaped level rather than decaying toward the control. Final evaluation:
403.1 lifespan, 17% of nights sheltered — the best M4 number of any run.

So the shaping here was genuine scaffolding that could be removed, not a
subsidy the behaviour depended on. Once the policy has *found* shelter, the
survival benefit alone sustains it — which is the answer the brief was asking
for, and the reason the shaping is defensible.

**Read this against the M3 shaping ablation, which went the other way.** Paying
for theft produced 4.8× the theft and *worse* survival, because stealing moves
food without creating any. Paying for construction produced building that pays
for itself and persists unpaid, because a shelter genuinely reduces the drain.
The difference is not the shaping technique — it is whether the shaped behaviour
was actually worth doing. That is the test, and only the terminal metric
answers it.

### What is left, and a correction to what this section used to say

**This section was wrong for two milestones, so read the correction before the
open problem.** It used to say: "the remaining gap is the last unit — finishing a
site is worth much more than the marginal unit suggests (a complete shelter
protects fully and permanently), and the policy stops at *good enough* partial
cover." That was true of `m4`/`m4b`, where `partial_shelter` was off and
protection was binary. **It stopped being true the moment `m4c` turned
`partial_shelter` on, and nobody re-derived it.** Measured, `m4c`, 4-unit site,
night drain 1.5 exposed / 0.5 sheltered:

| delivered | protection | night drain | marginal gain |
|---|---|---|---|
| 1/4 | 0.25 | 1.25 | +0.25 |
| 2/4 | 0.50 | 1.00 | +0.25 |
| 3/4 | 0.75 | 0.75 | +0.25 |
| **4/4** | **1.00** | **0.50** | **+0.25** ← the last unit |

Exactly linear, and a partial shelter persists exactly as a finished one does, so
"fully and permanently" separates nothing. In the canonical annealed world
(`reward.complete` 0.0) **laying the final unit buys precisely what the first one
bought and not one tick more.**

So "nobody finishes a shelter" was never a perception failure or a credit-
assignment failure. **It was correct play**, and m4c's own fix caused it:
`partial_shelter` removed the cliff, and the reason to reach the top went with
it. The only thing that ever paid for completion was `reward.complete: 3.0`,
which is shaping, and which is zero in the canonical checkpoint.

Two things follow, and both are load-bearing for whoever picks this up:

* **0.1 shelters an episode is not a defect to fix.** Read it as the policy
  correctly declining to pay for something worth nothing. The comparison to the
  scripted builder's 1.9 is not like-for-like: the builder finishes because it
  was *written* to finish, not because finishing pays.
* **A fix that removes a cliff can remove the summit with it.** That is the
  general lesson, and it is why this went unnoticed — `m4c` was a success on
  every metric anyone looked at (nights indoors 2.5% → 16%), and the thing it
  quietly deleted was only visible by re-deriving the arithmetic.

The standing advice that survives the correction:

* **Do not raise the shaping.** `m4` already showed 3.5× the material activity of
  its control with no completions; volume was never the constraint.
* Longer training is *not* indicated — the M3 investigation burned 3.3× budget
  for nothing, and these curves are flat by update ~250.
* **Cheaper sites were the untried structural lever. They were tried, and they
  did not pay** — see below. Note that the correction above explains *why* they
  could not have: m4d halved the distance to a summit that was worth nothing on
  arrival.
* If you want completion, the mechanic has to pay for it. That is what
  `construction.completion_premium` is for — it withholds a slice of the
  protection until a site is finished, so the continuous landscape survives and
  the last unit is worth more than the others. `config/m4e_premium.yaml` (the
  mechanic alone) and `config/m4e.yaml` (mechanic + the `site{j}.finishes`
  observation channel) are the pair; both headers carry their predictions.

### The cheap-sites test (`m4d`) — the lever did not pay

`config/m4d.yaml` cuts a site from 4 units to 2 (1 wood + 1 stone), which at
`material_capacity` 2 is exactly **one round trip**: an agent already carrying a
full load can finish a shelter without ever forming a multi-trip intention. Every
other thing is m4c. 1+1 rather than 2+0 so `mine` does not become a permanently
doomed action and the rock observation channels do not become noise.

The trap this run was designed around, written into the config header before it
ran: under `partial_shelter`, protection is the *fraction* of units delivered and
"indoors" is `protection >= 0.5`. That is 2 deliveries at 4-unit sites and **1**
at 2-unit sites, so `shelters` and `night_sheltered_frac` both get mechanically
cheaper along with the world. Neither can be compared across sizings.
`tests/test_construction.py::test_indoors_statistic_scales_with_site_cost` pins
this so the result cannot be misread later.

Matched budget, trailing 40 updates of a 200-update run — the tightest estimate
available, since it averages far more episodes than a 20-episode evaluation:

| | m4c (4-unit) | **m4d (2-unit)** |
|---|---|---|
| **mean lifespan** | **374.6** | **373.6** |
| shelters / ep *(not comparable)* | 0.064 | 0.083 |
| nights indoors *(not comparable)* | 17.9% | 24.7% |
| deliveries / ep *(not comparable — 6 units exist, not 12)* | 1.91 | 0.96 |

**Dead flat on the only metric that transfers.** And the 20-episode evaluation
says who did benefit:

| | m4c world | m4d world (cheap sites) |
|---|---|---|
| learned, shaped | 393.5 | 379.6 ± 61.9 |
| learned, unshaped control | 362.5 | 363.6 ± 58.9 |
| **scripted builder** | **529.8** | **545.0 ± 45.7** (2.5 shelters, 95% nights in) |
| **gap, learned → builder** | **136** | **165** |

The scripted builder converted cheaper sites into more shelters and ~15 more
ticks of life. The learned policy converted them into nothing, so the gap did not
narrow — if anything it widened. Neither individual delta clears ~1.5 SE on 20
episodes, so the honest claim is the conservative one: **cheaper sites bought the
learned policy no survival, while being clearly usable by something that knows
how to use them.**

The sharpest way to see it: the "indoors" bar was **halved** — one delivery
instead of two — and the shaped policy still cleared it on 22% of nights, exactly
what it managed at m4c. The control drifted 2% → 10% on the same halving, which
is the metric inflating with no behaviour behind it. That is the whole result in
two numbers.

Shaping still beat its own control (379.6 vs 363.6, +16), so nothing about m4c is
retracted. What is retracted is the hypothesis that completion was out of reach
because it was *too far*. It is not the distance to the summit. **No annealing
run was done, deliberately** — there was no gain to remove the scaffolding from,
and m4c's own header calls that kind of run a ritual.

Where this leaves the milestone: the three structural levers that worked (site
placement, partial protection, action masking) all removed something *uncreditable*
from the chain. Cheap sites removed *length*, not uncreditability, and length was
never what was broken.

**And the deeper reason they could not have worked** — found while writing up the
next experiment, not while running this one — is the correction above: under
`partial_shelter` the last unit is worth exactly what every other unit is worth.
m4d halved the distance to a summit that pays nothing on arrival. Any lever that
only shortens the chain is arguing with the wrong premise.

### The `m4e` pair — and the constraint that was actually binding

Two runs, both continued 200 updates from `checkpoints/m4c-anneal`, both with
every construction shaping term at zero, so the only reason to build is the night
drain. `m4e-premium` adds `completion_premium: 0.5` (the mechanic: a site
protects 0.125 / 0.25 / 0.375 on the way up and 1.0 when finished, so the last
unit is worth **5×** a normal one). `m4e` adds one further thing, the
`site{j}.finishes` observation channel.

| | lifespan | shelters/ep | nights indoors |
|---|---|---|---|
| m4c-anneal (no premium) | 403.1 | 0.09 | 18%\* |
| **m4e-premium** (mechanic only) | **401.7 ± 68.3** | **0.10** | 2% |
| **m4e** (mechanic + perception) | **377.9 ± 64.5** | **0.15** | 2% |
| scripted builder, same world | **526.0** | **1.85** | **88%** |

\* not comparable — under the premium, "indoors" requires a *finished* shelter,
where m4c-anneal's 18% counted half-built walls. That 18% was never 18% of nights
in a shelter; it was 18% of nights under something ≤ 37.5% built. Rule 5 again.

**Neither lever moved anything.** Making the last unit worth 5× produced 0.10
shelters against 0.09. Adding the channel produced 0.15 — which is *three
completed shelters across twenty episodes against two*, not a result. The
perception change cost 24 ticks of lifespan (~1.6 SE), so the conservative read is
that it did nothing.

The `m4e` header predicted the mechanic would move completions and the channel
would be flat, and pre-registered what a null on the mechanic would mean: "if this
does NOT move, the problem was never the incentive." It did not move. **So the
incentive was not the constraint either** — which, with m4d, retires the third of
three candidate explanations.

**What is actually binding, measured.** Counting where every delivered unit
lands, 20 episodes, same seeds:

| | units delivered / ep | on the busiest site | completions/ep | **if perfectly concentrated** |
|---|---|---|---|---|
| m4c-anneal | 2.00 | 75% | 0.05 | **0.25** |
| m4e-premium | 2.60 | 77% | 0.10 | **0.30** |
| m4e | 3.05 | 71% | 0.15 | **0.45** |
| **scripted builder** | **9.05** | 43% | **1.85** | **2.10** |

Read the last column. **A shelter costs 4 units and the policy moves about 3 an
episode, so even with perfect concentration it would finish well under half a
shelter per episode.** Completion is not out of reach because the summit is far
(m4d), or unrewarding (m4e-premium), or invisible (m4e). It is out of reach
because *not enough material ever gets carried up the hill*. The scripted builder
delivers **3.5× the material**, and that single ratio is the whole gap.

Note the policy is not even spreading material thin — 71–77% of everything it
delivers lands on one site, better concentrated than the builder's 43% (the
builder finishes sites and moves on). Concentration was the plausible culprit and
it is not the problem.

**And throughput is the number nothing has moved.** It sits at 2–3 units an
episode across the shaped run (1.94), the annealed run (2.00), the premium (2.60)
and the perception run (3.05). Shaping material actions at 0.5/0.5/1.0/3.0 did not
raise it; removing the shaping did not lower it. Meanwhile the policy takes ~30
berries and 90–140 steals an episode, because a gather pays +1.0 immediately and
material work pays nothing until nightfall.

**If you pick this up, move throughput or move nothing.** That is what `m4f` did.

### `m4f` — construction emerged, and the policy already knew how

Profiling the material chain before choosing a lever (rule 2's refinement, which
the three preceding nulls earned) pointed at one leg. On the m4c-anneal policy,
mean distance to the nearest bush is 2.4, to the nearest **tree 10.2** and the
nearest **rock 15.5**; agents stand in harvest range on 2.7% of ticks; `chop` is
*reachable* on 0.2% of ticks and is taken on 82–93% of those. **The policy was
never declining to harvest. It was almost never standing anywhere it could.**

The cause is that `m4b` did half a job. It moved the *shelter sites* onto the
berry clusters and left trees and rocks scattered, which **relocated** the
uncreditable walk to the harvest leg rather than deleting it — the same failure
m4b was written to fix, hiding one step upstream for two milestones.
`construction.materials_at_clusters` deals trees and rocks onto the clusters too
(`config/m4f.yaml`). Nothing else changes; the shaping is still zero throughout.

**The result, and the control that makes it a result.** Because `m4f` trained 200
updates beyond `m4e-premium`, the gain could have been the extra compute. It was
not: run the **unchanged** `m4e-premium` policy in the `m4f` world with *zero*
additional training and almost the entire effect is already there.

| m4e-premium policy, no retraining | in its own world | **in the m4f world** |
|---|---|---|
| mean lifespan | 388.3 | **445.4** |
| units delivered / ep | 1.80 | **6.95** |
| **shelters / ep** | **0.05** | **1.00** |
| nights indoors (finished shelters) | 1.4% | **31.7%** |

**The policy already knew how to build. It had nowhere to do it.** Twenty times
the completions out of weights that were not touched. Training on the new
geography then added nothing measurable — `m4f` trained finishes at 434.8, if
anything below the 445.4 zero-shot, well inside noise. Rule 4 again.

Where that leaves the milestone, 20 episodes, same seeds:

| | m4c-anneal | m4e-premium | **m4f** | scripted builder |
|---|---|---|---|---|
| mean lifespan | 403.1 | 401.7 | **434.8 ± 60.9** | **518.7** |
| **shelters / ep** | 0.09 | 0.10 | **1.10** | 2.80 |
| nights indoors | 18%\* | 2% | **42%** | 96% |
| units harvested / ep | 5.6 | 5.9 | **18.95** | 14.00 |
| units delivered / ep | 1.65 | 2.15 | **7.90** | 11.35 |
| **gap to the builder** | — | **124** | **84** | — |

\* the incomparable pre-premium figure; see rule 5.

**This is the first time construction genuinely emerged**, at the same order of
magnitude as the scripted reference rather than two orders below it — and with
every construction shaping term at zero, so nothing paid for it but the night
drain. Note the builder got slightly *worse* in this world (526.0 → 518.7), so
the narrowing gap is not the world getting easier for everybody; contrast `m4d`,
where the builder gained and the policy did not.

### What is left in M4

The harvest leg is fixed and the **delivery leg is now the constraint**:

| | m4f | scripted builder |
|---|---|---|
| delivery rate (delivered / harvested) | **41.7%** | **81.1%** |
| ticks carrying material | **83.8%** | 43.7% |
| units still held when the episode ends | **11.05** | 2.65 |
| distance to nearest site | 7.20 | 4.99 |

Agents now harvest *more* than the builder (18.95 vs 14.00) and deliver *less*
(7.90 vs 11.35). They hoard: full for 84% of ticks, and 11 units an episode die
in inventories.

**There is an obvious and cheap cause.** `num_sites` is 3 and `bushes.num_clusters`
is 4. Materials are dealt round-robin onto all four clusters; sites onto only
three. **An agent living on cluster 3 can harvest and has nowhere within reach to
deliver.** That was tested as `m4g`.

### `m4g` — the fourth site buys capacity, not competence

`config/m4g.yaml` is `m4f` with `num_sites: 4`, one per cluster. The zero-shot
control was run first, since `m4f` showed geography doing all the work: drop the
unchanged `m4f` policy into the four-site world and measure before spending any
compute.

The hypothesis was right about the mechanism. Distance to the nearest site falls
**7.08 → 4.14**, closer than the scripted builder keeps it (4.99), and delivery
rises:

| | m4f (3 sites) | **m4g (4 sites)** | builder |
|---|---|---|---|
| units delivered / ep | 7.90 | **9.60** | 11.35 |
| delivery rate | 41.7% | **46.8%** | 81.1% |
| mean lifespan | 434.8 | **460.2 ± 60.9** | 542.6 |
| shelters / ep | 1.1 | **1.4** | 3.4 |
| nights indoors | 42% | **49%** | 99% |

**460.2 is the best learned lifespan anywhere in this project.** But read the two
lines that say what it is not:

| | m4f | m4g |
|---|---|---|
| **completion rate per site** | 1.1 / 3 = **36.7%** | 1.4 / 4 = **35.0%** |
| **gap to the scripted builder** | **83.9** | **82.4** |
| ticks carrying material | 83.8% | 83.4% |
| units dying in inventories / ep | 11.05 | 10.90 |

The policy finishes **the same fraction of the sites it is given**, and the gap to
the builder is unchanged, because the builder gained from the fourth site too
(518.7 → 542.6). A fourth site produced a fourth site's worth of building and no
improvement in the *ability* to build. Contrast `m4f`, where the builder got
slightly worse and the policy got much better — that was competence; this is
capacity.

**And the hoarding did not move at all**: full 83% of ticks, ~11 units an episode
still dying in inventories, in both. That is what pins the delivery rate at half
the builder's, and it survives having a site at every cluster.

### `m4h` — a COMPOSITION deadlock, and the best result in the project

**I first wrote this up as a targeting problem and that was wrong.** The reasoning
was that agents move 88% of ticks and `build` is legal on only 0.7% of them, so
they must not be navigating to sites. One measurement killed it:

| share of living ticks, m4g | |
|---|---|
| carrying ≥ 1 unit | 83.3% |
| within `build_radius` of any site | 30.9% |
| **in range AND loaded** | **29.7%** |
| **in range AND the site wants what I carry** | **0.7%** |

Agents are at a site, holding material, on nearly a third of all ticks. **On 97.5%
of those the site does not want what they are carrying.** Navigation was never
the problem.

Why: a site needs 3 wood + 1 stone, every agent feeds its *nearest* site whatever
it happens to hold, and nothing routes the missing material to the site that
wants it. So sites deadlock on composition:

| site state at end of episode | m4g | scripted builder |
|---|---|---|
| done | 40.0% | 85.0% |
| **all wood in, ONE STONE missing** | **22.5%** | 5.0% |
| stone in, no wood | 10.0% | 1.2% |
| stone left in inventories | **3.70** | 0.65 |

A fifth of all sites end one stone short *while the agents are carrying 3.7
stone*. The scripted builder dodges this by targeting one focal site globally —
coordination six independent brains do not have and cannot easily learn, since no
agent can see another's inventory relative to a site.

`construction.fungible_materials` lets any outstanding unit take any carried
material. Both economies stay live — wood and stone still both have to be found,
harvested and carried — but a site stops caring which arrived first. Same move as
`partial_shelter`: delete a discontinuity PPO cannot route around.

**Results, 20 episodes.** Zero-shot first, as usual: the unchanged `m4g` policy in
the fungible world goes from 43.8% to **71.2%** of sites completed with no
training at all. Trained:

| | m4g | **m4h** | scripted builder |
|---|---|---|---|
| **mean lifespan** | 460.2 | **490.4 ± 67.9** | **540.9** |
| deaths / ep | 3.70 | **2.90** | 1.75 |
| shelters / ep | 1.4 | **3.0** | 3.45 |
| nights indoors | 49% | **83%** | 99% |
| units delivered / ep | 9.60 | **12.25** | 11.35 |
| delivery rate | 46.8% | **52.1%** | 81.1% |
| sites ending "one stone short" | 22.5% | **3.8%** | 5.0% |
| **gap to the builder** | 82.4 | **50.5** | — |

**This is competence, not capacity, and it passes its own pre-registered test:**
the gap to the builder fell from 82 to 50 while the builder barely moved
(542.6 → 540.9). Completions went from 41% of the builder's rate to **87%**, and
the policy now delivers *more* material per episode than the builder does.

**And it clears two of the three scripted references.** 490.4 against the scripted
forager's 463.6 and the thief's 452.9 — the first time since M1 that the learned
policy has beaten a scripted baseline. Read it with the caveat it deserves: the
forager and thief ignore construction entirely, so in a world where shelter is
load-bearing they are handicapped by design. The builder, which does build, is
still ahead by 50.

### What is left in M4 — not `build` uptake. It is going home at dusk.

**The `build`-uptake open problem is closed, and it was a counting error.** This
section used to say the headroom was "`build` available on 1.5% of ticks and taken
on only 28.6%". Measured per *distinct opportunity* rather than per legal tick —
one loaded agent standing at a site that wants its material is one chance, however
many ticks it lingers — uptake is **72.6%**, on spans averaging 4.2 ticks. Same
correction as `steal` in the M3 section, same rule 6.

And the residue is worth nothing. `sim.opportunity --force build`, common random
numbers, 40 paired islands:

| m4h | lifespan | builds/ep | shelters/ep | vs the unmodified policy |
|---|---|---|---|---|
| learned (control) | 467.1 | 11.5 | 2.77 | — |
| **ALWAYS build** | 470.7 | **11.9** | 2.90 | **+3.6 ± 7.7 (21/40)** |
| **NEVER build** | 362.7 | 0 | 0 | **−104.5 ± 8.7 (2/40)** |
| scripted builder | 539.8 | 14.8 | 3.52 | +72.6 ± 10.5 |

Forcing every legal build raises builds by 0.4 an episode, because the extra legal
ticks were the *same* opportunities: a build spends the material, so one take ends
the span. Construction itself is worth **104 ticks** — by far the largest measured
effect in the project — and it is essentially saturated.

**So where is the builder's +72.6?** Not in build volume (14.8 against a forced
11.9) and not in shelters (3.52 against 2.90). It is at night. Measured with
`sim.navigation --nights`:

| m4h, night agent-ticks | learned | random AT NIGHT (floor) | scripted builder |
|---|---|---|---|
| exposed (not inside a finished shelter) | **18–19%** | **30%** | **0%** |
| of those, no shelter finished anywhere | **0.0%** | — | — |
| ...one existed, mean distance | **13.6 units (median 8.0)** | — | — |

**Every single exposed night tick has a finished shelter available**, a mean 13.6
units away — about 17 ticks of walking — and the policy stays out. Its night
behaviour is better than chance (19% against the 30% floor), so this is a partial
competence, not an absence; the builder closes the whole thing by running home at
dusk. Note the floor had to be built specially: random and foraging baselines never
finish a shelter, so they generate no night rows at all, and the floor is therefore
the learned policy by day with uniform legal actions at night.

Crude size check, from one point rather than a curve: the night-random floor costs
23.6 ticks of life for 11.5 extra points of exposure, so ~19 points is order-40
ticks — roughly half the builder's edge. Treat that as an order of magnitude, not a
measurement.

Beyond that, 11.25 units an episode still die in inventories and the delivery rate
is 52% against the builder's 81%.

### The night channel already existed — and the night is not the gap. Navigation is.

The plan's first lever was "a night channel in the observation (does the policy even
know it is dusk?)". **It has known all along.** The M4 observation carries
`night.phase` and `night.is_night`, and every site carries an explicit
`site{j}.complete` flag. Nor is the shelter hidden the way the nearest berry-bearing
bush is: `k_sites` is 2 of 4, but the nearest *finished* shelter is in the
observation on **99.3%** of night ticks (96.3% of the exposed ones, mean rank 0.11 of
4), because sites get finished where agents already live. No run was needed to
retire this; reading `sim/agents.py` and one measurement did it.

**What the policy does with the clock: nothing.** Distance to the nearest finished
shelter by cycle phase, night from 0.75, at a FIXED number of finished shelters —
the conditioning matters, because shelters accumulate over an episode and pooling
phases across a run turns construction progress into a fake inward "drift":

| 3 shelters up | 0.0 | 0.2 | 0.4 | 0.6 | **0.7** | **0.8** | **0.9** |
|---|---|---|---|---|---|---|---|
| learned (m4h) | 3.6 | 3.3 | 3.2 | 3.1 | 3.3 | **3.8** | **4.0** |
| scripted builder | 4.1 | 5.7 | 5.2 | 2.9 | **2.8** | **2.8** | **2.8** |

The builder starts closing at phase 0.5 and holds at 2.8, well inside
`shelter_radius` 6.0. The learned policy holds ~3.2 all day and **drifts outward
once night falls**. So it sees dusk and ignores it.

**And making it go home buys nothing.** Same weights, one override: from phase `h`
onward, any agent outside `shelter_radius` walks at the nearest finished shelter.
40 paired islands:

| m4h | lifespan | nights indoors | berries | shelters | vs the unmodified policy |
|---|---|---|---|---|---|
| learned (control) | 467.1 | 71.4% | 34.7 | 2.77 | — |
| go home from 0.75 (dusk) | 472.7 | **89.4%** | 30.4 | 2.27 | **+5.6 ± 6.8** |
| go home from 0.6 | 467.5 | **92.0%** | 29.2 | 2.30 | **−0.3 ± 7.6** |
| go home from 0.5 | 465.7 | **93.9%** | 28.2 | 2.15 | **+1.4 ± 6.6** |
| scripted builder | 539.8 | 99.7% | **41.8** | **3.52** | +72.6 ± 10.5 |

Nights indoors goes from 71% to 94% — nearly the builder's number — and survival
does not move, because the ticks come straight out of foraging and building
(berries 34.7 → 28.2, shelters 2.77 → 2.15). **The night hours are productive
hours.** Note the builder's row: it is indoors 99.7% *and* gathers more *and*
builds more, which is not a trade-off it has to make.

**Which is the whole story: this policy is tick-starved.** Every intervention that
spends ticks differently has now come back at zero — forcing builds (+3.6 ± 7.7),
forcing steals (−8.3 ± 4.1), going home (−0.3 ± 7.6). So measure the thing that
*adds* ticks. `sim.steering` keeps every decision the policy makes and replaces only
the direction of its moves, steering them at the nearest berry-bearing bush (and at
night, optionally, the nearest finished shelter). The tick budget is identical:

| m4h, 40 paired islands | lifespan | berries | shelters | nights in | deaths | vs the policy |
|---|---|---|---|---|---|---|
| learned (control) | 467.1 | 34.7 | 2.77 | 71.4% | 3.35 | — |
| **steered: food** | **556.3** | **49.9** | 3.25 | 53.6% | 1.57 | **+89.1 ± 11.4 (38/40)** |
| **steered: food + home** | **580.3** | 45.0 | 3.17 | 86.9% | **0.62** | **+113.2 ± 10.7 (39/40)** |
| scripted builder | 539.8 | 41.8 | 3.52 | 99.7% | 1.52 | +72.6 ± 10.5 |
| scripted forager | 465.8 | 50.1 | 0 | 0% | 3.12 | +1.4 ± 11.1 |

**Navigation alone is worth +89 ticks, and with the night rule +113 — which beats
the scripted builder by about 40.** Nothing else measured in this project comes
close: the largest previous effect was construction existing at all (−104.5 when
suppressed), and the largest *lever* was `m4h`'s fungible deliveries at +30.
Deaths fall from 3.35 to 0.62.

Read the last two rows together, because they say why the night thread looked
promising and was not: food-only steering *lowers* nights indoors to 53.6% and still
gains 89 ticks, while adding the home rule on top is worth a further ~24. **Going
home pays only once navigation has freed the ticks to pay with.**

**The caveat that rides with the number.** The steering rule reads world state — the
nearest berry-bearing bush among all of them, where the observation carries the
`k_bushes` nearest bushes loaded or not, and on m4h the target is missing from the
observation on 54% of ticks beyond 20 units. So +89/+113 is an upper bound on what
perfect navigation is worth, not a demonstration that it is learnable from the
current 61-dim observation. What keeps it from being a fantasy: `nav_probe` shows
PPO reaching 86% toward-food from long range, and the scripted forager scores 100%
off the observation alone.

## Milestone 5 — exchange

Two appended actions, `give_food` (14) and `give_material` (15), each moving one
unit to the nearest neighbour in reach with room for it. Observation 55 → 61
(neighbours' carried wood and stone). Replay schema v3 carries a per-tick list of
transfers. All off by default, so M1–M4 worlds stay bit-identical.

The world is `m5.yaml` = the annealed M4 world plus exchange, so an M5 world pays
for exactly three things, all from Milestone 1: staying alive, gathering, eating.

### Design decisions, and why

**Food and materials are separate actions; wood and stone are not.** They are two
different economies — one keeps you alive, the other builds shelter — and an
agent carrying both would otherwise be unable to choose which it is taking part
in. Wood versus stone is a much narrower distinction (the receiver's `build`
already resolves which the site needs), so collapsing those two saved an action
slot that PPO would have had to discover the value of separately.

**The giver does not choose the recipient.** A gift goes to the *nearest*
neighbour with room, exactly as `steal` takes from the nearest loaded victim.
What the giver actually controls is where it stands, so **positioning is the
targeting mechanism**. This bit the scripted trader first: a rule that asked "is
anyone near me hungry?" gave away 80 berries an episode of which 3 were eaten,
because the berry kept going to somebody else. The rule has to be evaluated on
the neighbour the *world* would pick.

**Nothing pays for a gift** (`reward.give = 0.0`), the same call as
`reward.steal`. `m5_shaped.yaml` raises it as a labelled ablation, and its header
contains the prediction it was run to test, written before the run.

**The transfer ledger is its own artefact.** A per-episode average of "gifts"
cannot answer who gave to whom or whether it came back, so `sim/exchange.py`
writes a JSONL of `(episode, tick, giver, receiver, item)` plus an aggregate
report that `viewer/exchange.html` renders. `exchange.log_transfers` is off by
default: training runs 32 worlds at once and does not need the ledger.

### Results — exchange did not emerge

20 episodes, seeds 10000+, all on the same islands:

| policy | lifespan | deaths | berries | gifts/ep | nights indoors |
|---|---|---|---|---|---|
| random actions | 176.6 | 6.00 | 1.6 | — | — |
| scripted forager | 463.6 | 3.05 | 50.4 | 0 | 0% |
| scripted builder | 529.8 | 2.05 | 38.5 | 0 | 92% |
| **scripted trader** | **543.8** | **1.85** | 38.0 | 8.3 | 93% |
| M4 policy grown into M5, untrained (control) | 406.4 | 4.60 | 34.9 | 11.3 | 15% |
| **learned, gifts unpaid (`m5`)** | **413.3** | 4.60 | 34.6 | **7.9** | 20% |
| learned, gifts paid (`m5-shaped`) | 359.1 | 5.30 | 26.9 | **330.9** | 15% |

**Read the control row.** A grown-but-untrained policy already gives 11.3 times
an episode, because a zero-initialised action row makes `give` just another thing
to sample. After 200 updates with nothing paying for it, that fell to 7.9. Giving
was not merely un-learned — it was mildly selected *against*, which is correct:
a gift costs a tick and hands the payoff to somebody else.

**The mechanic is not worthless, which is what makes this a finding.** The
scripted trader beats the scripted builder on the same islands by **+14.0 ± 5.2
ticks (paired, better on 12 of 20 islands)** and 0.2 fewer deaths, on about
**eight** well-aimed gifts an episode. So a handful of gifts genuinely buys
survival, and PPO still cannot find them. This is the credit-assignment wall the
M4 handoff predicted, arriving exactly where it said it would.

### The shaping ablation — a gift farm, as predicted

`m5_shaped.yaml` pays a successful transfer +1.0, what a gather pays. The
prediction written into its header beforehand was: many gifts, fewer berries,
lifespan at or below the unpaid run. All three:

| | unpaid (`m5`) | paid (`m5-shaped`) |
|---|---|---|
| gifts / episode | 7.9 | **330.9 (42×)** |
| berries gathered / ep | 34.6 | 26.9 |
| **mean lifespan** | **413.3** | **359.1** |
| deaths / episode | 4.60 | 5.30 |

The ledger says exactly what went wrong, which a lifespan number alone could not:

| ledger (20 episodes) | unpaid | paid |
|---|---|---|
| transfers / episode | 11.1 | 355.2 |
| share of living ticks spent giving | 0.4–1.5% | 16–20% |
| net flow per agent (gave − received) | −13 … +14 | −23 … +31 of ~1200 |
| reciprocity | 0.851 | **0.942** |
| gifted food eaten within 50 ticks | 56% | **5%** |
| gifted material delivered within 50 ticks | 3% | **0%** |

Every agent gives and receives about 1200 times and ends up net flat, and almost
nothing gifted is ever used. It is a circulation farm: two agents standing next
to each other pass a berry back and forth and collect a gather's worth of reward
each time, without a berry being created or eaten. **A shaped reward reliably
produces the behaviour it pays for, and that is still not evidence the behaviour
helps** — the M3 theft ablation, restated with a bigger multiplier.

Note the asymmetry with M4, which is the useful comparison: paying for
construction produced building that survived annealing, because a shelter really
does reduce the drain. Paying for gifts produced motion, because a transfer
creates nothing. The technique did not change; the mechanic did.

### `m5b` — the retest on a working economy, and what it changed

**M5's verdict was reached in a world with almost no material economy, and that
needed saying.** The original run sat on `m4c_anneal`, where the learned policy
completed **0.09 shelters an episode** and delivered 2.0 units. Exchange was asked
to emerge in an economy that barely existed. `m4h` fixed that — 23.5 units
harvested, 12.25 delivered, 3.0 of 4 shelters finished — and left 11.25 units an
episode dying in inventories, exactly the surplus a transfer could move.
`config/m5b.yaml` is `m4h` plus exchange, gifts still paying nothing.

| | `m5` (old world) | **`m5b`** (working economy) |
|---|---|---|
| grown-but-untrained control, gifts/ep | 11.3 | 9.5 |
| **after 200 updates, gifts/ep** | **7.9 — down 30%** | **11.3 — held** |
| mean lifespan | 413.3 | **497.5 ± 73.5** |
| the same world *without* exchange | 406.4 | 490.4 |
| scripted trader | 543.8 | **569.0** |
| trader's edge over the builder | +14.0 | **+28.1** |
| gap, learned → trader | 130.5 | **71.5** |
| reciprocity | 0.851 | **0.653** |
| gifted food used | 56% | 48% |
| gifted material used | 3% | **2%** |

**The one real correction: giving is no longer selected against.** m5's headline
was that 200 updates trained giving *down* from 11.3 to 7.9. On a working economy
it holds instead (9.5 → 11.3). And the flow changed character — reciprocity fell
0.851 → 0.653, with consistent net donors and net recipients (agent 5 gave 61 and
received 28; agent 4 gave 17 and received 48) rather than the balanced
pass-it-back pattern. That is directional flow, which is what the beginnings of
exchange would look like.

**But the verdict itself survives.** 11.9 transfers an episode is tiny, lifespan
is +7 over the same world without exchange (inside a ±73 spread), and the gap to
the scripted trader is still 71.5 ticks.

**And the retest makes the credit-assignment finding stronger, not weaker.** The
trader's edge over the builder *doubled* in this world, +14.0 → **+28.1** — so
trading is now worth twice as much as it was when M5 concluded PPO could not find
it, and PPO still cannot find it.

**My hypothesis for this run was wrong in an informative way.** I expected the
*material* relay to be what paid, since fungible deliveries make any gifted unit
usable. The opposite happened: material giving was selected *against* (4 → 2 an
episode) and gifted material is used **2%** of the time, while food giving rose
(6 → 9) and food is used 48%. Handing someone a berry they eat is a credit chain
two steps long; handing someone wood they must then carry to a site and spend is
the same long chain that failed before, and making the *delivery* fungible did
nothing to shorten it.

### What would actually be worth trying

Not a bigger coefficient, and not more updates (the curves are flat by ~120).
The unpaid chain is too weak, so change the *mechanic*:

* **Make gifts non-fungible.** If wood could only be harvested by an agent
  standing far from the sites, and building only worked near them, a relay would
  be the *cheapest* way to build rather than a nicety. Specialisation would then
  be forced by geography rather than hoped for.
* **Let the giver choose the recipient** (nearest is currently forced). Adding a
  target choice widens the action space, but it is the difference between
  "positioning as targeting" and actual directed trade.
* **Pay only gifts the receiver uses** — reward the giver when the receiver eats
  or delivers within N ticks. That is a much narrower shaping than a flat payment
  and would answer whether the farm is the only thing a payment can buy. It needs
  a deferred-reward mechanism the trainer does not currently have.

### Gotchas (Milestone 5)

**A thief can take the berry you were about to hand over.** Theft resolves in
phase 1c and gifts in 1e, so a robbery lands first and the gift silently does not
happen — the giver's `gave` stays 0 even though the mask had said yes. This is
**75% of every failed give the learned policy makes**, and it is not a mask bug:
you cannot hand over what was just taken from you. Pinned by a test. The other
failure mode is the same-tick race two givers can have for one free slot, which
`gather` has had since M3 (the mask is a start-of-tick promise).

**Give hit rate is therefore not a mask-quality metric.** In M3 a low gather hit
rate meant doomed actions; here a give hit rate of 63–100% is mostly other agents
acting first. Look at the ledger's utilisation figures instead.

**`reciprocity` near 1.0 is not good news by itself.** Perfectly balanced pairs
are what both mutual aid and a reward farm look like. Only utilisation separates
them, which is why the report prints both and the viewer shows them together.

## Gotchas (Milestone 3)

**Don't edit `metrics.py` while a run is in flight.** A run holds its CSV header
from the moment it opens the file, so fields added mid-run are missing from that
run's CSV and read as blanks forever. Cost me a run.

**`contests_lost ≈ 0` is behavioural, not a broken code path.** It is tested
directly. See above for why the same-tick rule cannot fire in a scarce world.

## The seven rules that survived five milestones

Written down because each one was learned the expensive way, and because every
result in this file that ignored one of them turned out to be wrong.

1. **A shaped reward reliably produces the behaviour it pays for. That is never
   evidence the behaviour helps.** Theft, paid: 4.8× the theft, worse survival.
   Gifts, paid: 42× the gifts, 54 fewer ticks of life. Construction, paid: more
   building *and* better survival, and it survived annealing. Same technique,
   three different verdicts — decided entirely by whether the shaped behaviour
   was worth doing. Only the terminal metric can tell you which case you are in,
   so every shaped run ships with an unshaped control and is read on lifespan.
2. **Fix the shape of the problem, not the size of the number.** Every real
   improvement here came from changing what the agent could perceive or reach —
   action masking (M3), siting shelters where agents already live and making
   partial walls give partial protection (M4). Every attempt to buy the outcome
   with a bigger coefficient produced activity without result.
   **The refinement, learned from `m4d`/`m4e`:** not every structural change
   qualifies. The three that worked all deleted a step that *could not be
   credited* — a doomed action, an unpaid approach walk, three-quarters of a build
   invisible to the value function. Three later attempts on the same problem all
   came back flat: cheaper sites (shorter chain), a completion premium (bigger
   payoff at the end), a "one unit finishes this" channel (better perception).
   Each changed something real and none was the binding constraint, which turned
   out to be plain throughput — 3 units delivered an episode against a 4-unit
   shelter. **Measure the constraint before choosing the lever.** Three runs and
   two configs went to changing things that were not what was stopping it.
3. **The milestone chain is load-bearing.** The scarce world is unlearnable from
   scratch — a from-scratch run lands on *exactly* the random baseline after the
   full budget. Each milestone works because the previous one transferred
   competence into it via `--init-from` and `grow_policy`.
4. **More compute is never the fix.** `scarce-long` spent 3.3× budget and was
   flat from update 150. Runs settle by ~120–175 updates. If a run is not
   working, the world or the observation is wrong, not the budget.
5. **When you change a mechanic, re-derive the arithmetic that justified the
   open problems around it.** `m4c` made shelter protection linear in build
   progress. That fixed the cliff it was aimed at, and in the same stroke made
   the last unit of a site worth exactly what the first one was — deleting the
   reason to complete a shelter. The note calling incomplete shelters M4's
   failure was written before that change and was carried forward, unexamined,
   through the whole of M5 and into `m4d`, an experiment that could not have
   worked because it was arguing with a premise that had already expired. Cost:
   one run and two milestones of a wrong headline. The M3 economy postmortem
   already says "recompute demand with `sim` rather than by hand" — this is the
   same rule, applied to *incentives* rather than supply. A fix that removes a
   cliff can remove the summit with it.
6. **A rate is dominated by whichever states the policy spends its time in,
   and for a competent policy those are the states where the behaviour barely
   applies.** The toward-food share looked like a clean measure of navigation
   and gave m1 51.3%, barely above a random walk — the project had apparently
   never learned to navigate. It had: bucketed by distance, m1 is 88.4% at 10–20
   units. The aggregate was 17,347 near-field samples against 199 far ones, so
   it was essentially the near-field number wearing a general label.
   Note the explanation that is *wrong*, because it was the first one reached
   for: this is not "you can only move away from a target you are standing on".
   The scripted forager scores **100%** in the same near band, because it moves
   only when moving is right and otherwise gathers. The near-field 48% is a real
   deficiency; the error was averaging it with real far-field competence at 87:1
   weighting and reporting the result as one thing.
   Caught only because a probe result was internally contradictory — the policy
   plainly reached the food (mean distance 3.8 against random's 24.0) while
   "scoring" 54.8% toward it. **When two of your numbers cannot both be true,
   stop and fix the metric.**
   **It happened again, in a different disguise, and cost two open problems.**
   "`build` legal on 1.5% of ticks and taken on 28.6%" was the headline M4
   inefficiency for a milestone, and "`steal` taken on 46.4%" was promoted to the
   largest headroom in M3. Both divide by *ticks*, and a build spends the material
   that made it legal — so one take ends a run of legal ticks averaging 4.2 of them.
   Per distinct opportunity the figures are **72.6%** and **85.0%**, and forcing the
   remainder buys nothing. `gather` was the reason it hid: its spans last 1.2 ticks,
   so per-tick and per-opportunity agree there (98% and 99%) and the metric looked
   sound wherever anyone had checked it. **Ask what one opportunity is before
   dividing by anything.**
7. **Compare policies island by island, not mean against mean.** Island-to-island
   variation here is ±60 ticks, so a 20-episode mean carries ~13 ticks of unpaired
   standard error — wider than most effects in this file. On 20 episodes `nav-move`
   scored 507.0 against the scripted forager's 506.6 and read as a dead heat, and
   the first draft of that write-up said so. On 40 *paired* islands it is
   **−19.0 ± 10.1, better on 16 of 40** — behind, not level. Nothing about the
   policy changed between those two sentences; the seed block did.
   `sim.evaluate` now prints the paired differences by default, at no extra
   compute, and the win count next to the mean is there so one lucky island cannot
   read as a general result.
