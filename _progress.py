import csv, sys
path = sys.argv[1] if len(sys.argv) > 1 else 'runs/m3/metrics.csv'
rows = list(csv.DictReader(open(path)))
def g(r, k, nd=2):
    v = r.get(k, ''); return f"{float(v):.{nd}f}" if v else "-"
print(f"{len(rows)} updates logged")
cols = [('update','upd',0),('mean_lifespan','lifespan',1),('deaths_per_episode','deaths',2),
        ('berries_gathered','berries',1),('steals','steals',1),('contests_lost','lost',1),
        ('entropy','entropy',3),('mean_reward','rew/step',4),('explained_variance','expvar',3)]
print(" ".join(f"{lbl:>9}" for _,lbl,_ in cols))
step = max(len(rows)//14, 1)
for r in rows[::step] + rows[-1:]:
    print(" ".join(f"{g(r,k,nd):>9}" for k,_,nd in cols))
