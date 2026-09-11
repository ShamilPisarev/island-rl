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
