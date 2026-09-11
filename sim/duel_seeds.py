"""Fair duel over seeds: A vs B on both sides, A vs A, B vs B, and the scripted arbiter on the same island. Run from the repo root: .venv/bin/python -m sim.duel_seeds"""
import json, re, subprocess, sys
import numpy as np
PY = ".venv/bin/python"
A, B = "checkpoints/arb7-village/latest.pt", "checkpoints/arb5-hh/latest.pt"
SEEDS = list(range(10000, 10005))
rows = []
def duel(a, b, seed, swap):
    cmd = [PY, "-m", "sim.duel", "--a", a, "--b", b, "--seed", str(seed), "--ticks", "1200"] + (["--swap"] if swap else [])
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [json.loads(l) for l in out.splitlines() if l.startswith("{")]
def control(seed):
    out = subprocess.run([PY, "-m", "sim.society", "--config", "config/duel.yaml", "--ticks", "1200",
                          "--episodes", "1", "--seed", str(seed)], capture_output=True, text=True, check=True).stdout
    pop = re.search(r"tribes\s+population\s+(\d+)\s+(\d+)", out)
    births = re.search(r"births\s+(\d+)\s+(\d+)", out)
    return [int(pop.group(1)), int(pop.group(2))], [int(births.group(1)), int(births.group(2))]
for s in SEEDS:
    for r in duel(A, B, s, swap=True): rows.append(("mixed", s, r))
    rows.append(("A_vs_A", s, duel(A, A, s, swap=False)[0]))
    rows.append(("B_vs_B", s, duel(B, B, s, swap=False)[0]))
    pop, births = control(s)
    rows.append(("scripted", s, {"population": pop, "initial_population": rows[-1][2]["initial_population"],
                                 "births": births, "captures": None}))
    print(f"seed {s} done", file=sys.stderr, flush=True)
json.dump([(k, s, r) for k, s, r in rows], open("viewer/reports/duel_seeds_rows.json", "w"))

def pm(x): x = np.asarray(x, float); return f"{x.mean():+.1f} +- {x.std(ddof=1)/np.sqrt(len(x)):.1f} ({(x>0).sum()}/{len(x)})"
print("\n== founders per seed (tribe0, tribe1) ==")
for s in SEEDS:
    print(s, [r for k, ss, r in rows if k == "mixed" and ss == s][0]["initial_population"])
print("\n== final population, tribe0 / tribe1 / total, by arm and seed ==")
arms = {}
for k, s, r in rows:
    name = k if k != "mixed" else f"t0={r['networks_by_tribe'][0]}"
    arms.setdefault(name, {})[s] = r
for name, d in arms.items():
    line = "  ".join(f"{d[s]['population'][0]:2d}/{d[s]['population'][1]:2d}={sum(d[s]['population']):2d}" for s in SEEDS)
    tot = [sum(d[s]["population"]) for s in SEEDS]
    print(f"{name:22s} {line}   mean total {np.mean(tot):.1f}")
sc = arms["scripted"]; a0 = arms["t0=arb7-village"]; b0 = arms["t0=arb5-hh"]
print("\n== paired reads (per seed) ==")
print("1. learned pair total - scripted total (arb7 on t0):", pm([sum(a0[s]["population"]) - sum(sc[s]["population"]) for s in SEEDS]))
print("   learned pair total - scripted total (arb5 on t0):", pm([sum(b0[s]["population"]) - sum(sc[s]["population"]) for s in SEEDS]))
# network effect: arb5-hh minus arb7-village holding the SIDE fixed, both sides summed
net = [(b0[s]["population"][0] - a0[s]["population"][0]) + (a0[s]["population"][1] - b0[s]["population"][1]) for s in SEEDS]
print("2. arb5-hh - arb7-village, same side, both sides summed:", pm(net))
print("   ... arb5 on t0 vs arb7 on t0:", pm([b0[s]["population"][0] - a0[s]["population"][0] for s in SEEDS]))
print("   ... arb5 on t1 vs arb7 on t1:", pm([a0[s]["population"][1] - b0[s]["population"][1] for s in SEEDS]))
print("3. ground: tribe0 - tribe1 with ONE brain both sides")
for name in ("A_vs_A", "B_vs_B", "scripted"):
    d = arms[name]; print(f"   {name:10s}", pm([d[s]["population"][0] - d[s]["population"][1] for s in SEEDS]))
print("4. self-duel totals vs scripted: arb7 pair", pm([sum(arms["A_vs_A"][s]["population"]) - sum(sc[s]["population"]) for s in SEEDS]),
      "| arb5 pair", pm([sum(arms["B_vs_B"][s]["population"]) - sum(sc[s]["population"]) for s in SEEDS]))
print("5. captures per match: mixed", [r["captures"] for k, s, r in rows if k == "mixed"], "self", [r["captures"] for k, s, r in rows if k in ("A_vs_A", "B_vs_B")])
print("   births: scripted", [sum(sc[s]["births"]) for s in SEEDS], "| mixed peak-40", [r["peak_population"] - 40 for k, s, r in rows if k == "mixed"])
