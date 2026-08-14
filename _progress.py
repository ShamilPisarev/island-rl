import csv, sys
path = sys.argv[1] if len(sys.argv) > 1 else 'runs/m1/metrics.csv'
rows = list(csv.DictReader(open(path)))
def g(r, k, nd=2):
    v = r[k]
    return f"{float(v):.{nd}f}" if v else "-"
print(f"{len(rows)} updates logged")
print(f"{'upd':>4} {'lifespan':>9} {'deaths':>7} {'berries':>8} {'entropy':>8} {'rew/step':>9} {'expvar':>7} {'step/s':>8}")
step = max(len(rows)//14, 1)
for r in rows[::step] + rows[-1:]:
    print(f"{r['update']:>4} {g(r,'mean_lifespan',1):>9} {g(r,'deaths_per_episode'):>7} "
          f"{g(r,'berries_gathered',1):>8} {g(r,'entropy',3):>8} {g(r,'mean_reward',4):>9} "
          f"{g(r,'explained_variance',3):>7} {g(r,'steps_per_s',0):>8}")
