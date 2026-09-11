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
