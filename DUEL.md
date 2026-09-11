# Two-network trial

This is an isolated experiment using existing frozen goal networks, with no
training. All previous configurations and checkpoint weights are preserved.

Run from the project root:

```sh
.venv/bin/python -m sim.duel --ticks 1200 --swap
```

The village network and household network control opposing tribes in the same
world. The second match swaps their sides on the same seed. The world starts
with 40 people and allows 120 living slots, with reuse after deaths. CPU neural
inference uses two threads. Replays and a report get unique filenames; the
existing viewer can open each replay through its dropdown or replay query URL.

The models choose goals; existing scripted movement and interaction controllers
execute them. New goals absent from each checkpoint are masked, including the
untrained conquest goal. The world's conquest rule remains enabled, so captures
can happen through proximity, but are not evidence of learned military tactics.
Technology unlocks remain authored world mechanics. No weight updates occur.

All people on a tribe share its network. Births and assimilation use current
tribe membership; assimilated people reconsider their goal on the next tick.
Heredity is disabled for this initial compatibility experiment. Both models get
the same seeded individual trait inputs. One model was trained in an older
world and both were trained with scripted neighbors, so all-learned competition
is an unfamiliar setting and collapse is a valid possible outcome.

Compare survival and goal use across both side assignments. A short pair is
exploratory and cannot establish a generally superior network. The report flags
whether the population reached the slot cap, and measures simulation time and
peak process memory (excluding the browser). Startup may be slow if macOS has
offloaded the local Python libraries or checkpoint files.

## First run (2026-09-12, one seed, exploratory)

Seed 10000, 1200 ticks, 40 founders split 22 (tribe 0) / 18 (tribe 1), 120
slots. Both matches took 4.3 s of simulation each. Population at the end:

| tribe 0 network | tribe 1 network | tribe 0 | tribe 1 | total | births | captures |
|---|---|---|---|---|---|---|
| arb7-village | arb5-hh | 22 -> 15 | 18 -> 12 | 27 | 0 | 0 |
| arb5-hh | arb7-village | 22 -> 18 | 18 -> 10 | 28 | 4 | 1 |
| scripted utility arbiter, both tribes (control) | | 22 -> 29 | 18 -> 15 | 44 | 31 | n/a |

Three readings, all on one seed, so none of them is settled:

* Both learned networks lose to the scripted arbiter in this world. The
  scripted village grows 40 -> 44 with 31 births; the learned pairs fall to
  27 or 28 with 0 and 4 births. Neither network was trained here (arb5-hh
  never saw tribes, houses or children; arb7-village saw 600-tick episodes),
  so this is the expected outcome and not a verdict on either.
* Side matters more than network. Whoever holds tribe 0 (22 founders) ends
  ahead, and the scripted control shows the same 29 / 15 split with ONE brain
  on both sides. The ground, not the chooser, decides most of the gap, which
  is the Island 3.0 tribe result again.
* Per network across sides, arb5-hh keeps more of its people (-18% and -33%)
  than arb7-village (-32% and -44%). arb5-hh steals heavily (2,126 and 7,184
  goal-ticks) and delivers material; arb7-village never harvests or delivers
  (the construction wall, as in its training write-up) and explores a lot.

Bug fixed before the first run: `World` has no `num_agents` attribute; the
slot count is `World.slots`. Caught by `tests/test_duel.py`.

## The fair duel (2026-09-12): 5 seeds, both sides, each network against itself, scripted control

`sim/duel_seeds.py` runs, per seed 10000-10004 at 1200 ticks: A vs B with
sides swapped, A vs A, B vs B, and `sim.society` with the scripted utility
arbiter on the same island (same seed, so the same map and founders). 25
matches, 2 min 14 s in total. Founders are NOT split evenly: tribe 0 starts
with 22-26 of the 40, tribe 1 with 14-18, which is why the self-duels matter.

Final population, tribe 0 / tribe 1 = total, per seed:

| arm | 10000 | 10001 | 10002 | 10003 | 10004 | mean total |
|---|---|---|---|---|---|---|
| arb7-village (t0) vs arb5-hh (t1) | 15/12=27 | 9/24=33 | 13/14=27 | 11/11=22 | 15/10=25 | 26.8 |
| arb5-hh (t0) vs arb7-village (t1) | 18/10=28 | 29/9=38 | 17/11=28 | 18/7=25 | 13/13=26 | 29.0 |
| arb7-village vs itself | 16/12=28 | 17/9=26 | 16/10=26 | 16/8=24 | 15/9=24 | 25.6 |
| arb5-hh vs itself | 19/12=31 | 31/7=38 | 22/6=28 | 19/7=26 | 16/10=26 | 29.8 |
| scripted arbiter, both sides | 29/15=44 | 41/12=53 | 34/22=56 | 30/21=51 | 31/22=53 | 51.4 |

Paired per seed (mean +- SE, seeds where the sign holds):

1. **Both learned networks lose to the scripted arbiter, on every seed.**
   Learned pair minus scripted: **-24.6 +- 2.5 (0/5)** with arb7-village on
   tribe 0, **-22.4 +- 2.8 (0/5)** with arb5-hh there. The self-duels say the
   same (-25.8 +- 2.5 and -21.6 +- 3.2, both 0/5). The scripted village has
   29-37 births a run; the learned ones 0-7. Neither network was trained in a
   world with tribes at this length, so this is the expected result, but it
   is now measured rather than assumed.
2. **arb5-hh beats arb7-village by a little, and it is not settled.** Holding
   the side fixed and summing both sides: **+10.6 +- 6.6 (4/5)**, i.e. +6.4 +-
   3.7 on tribe 0 and +4.2 +- 3.0 on tribe 1. About 1.6 SE on five seeds.
   arb5-hh's self-duel also holds more people (29.8 against 25.6).
3. **The ground is worth more than the network.** With ONE brain on both
   sides, tribe 0 ends ahead by **+6.4 +- 0.7** (arb7 vs arb7), **+13.0 +-
   3.3** (arb5 vs arb5) and **+14.6 +- 3.7** (scripted), 5/5 each. That is the
   founder split (22-26 against 14-18) carrying through, the Island 3.0
   result that a tribe's size is mostly its starting ground. Any single
   unswapped match would credit the tribe-0 network with that.
4. **Captures happen only when the brains differ.** Mixed matches: 0, 1, 9,
   1, 3, 2, 2, 1, 0, 1. Self-duels: 2 in one of ten, 0 in the rest. Neither
   network has the `conquer` goal (masked), so these are raids that ended in
   a capture by presence; two different behaviours brought raiders to enemy
   stockpiles more often than two copies of one behaviour did.

Honest limits: five seeds, one length (1200 ticks, so no agriculture or
granary), two networks trained in different worlds and neither in this one.
Read 3 as the finding, 1 as the control doing its job, and 2 as a hint.
