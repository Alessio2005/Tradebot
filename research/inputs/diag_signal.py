import numpy as np, pandas as pd
from scipy import stats
ev = pd.read_parquet('/home/claude/research/events.parquet')
f = pd.read_parquet('/home/claude/research/features.parquet')
import sys; sys.path.insert(0,'/home/claude/alessio2005/tradebot/src')
grid = pd.date_range('2020-03-26', '2026-06-23', freq='D', tz='UTC')
ev['date'] = grid[ev['event_bar'].to_numpy()]
ev['week'] = ev['date'].dt.to_period('W').astype(str)
cost = 0.0013
ev['R'] = (ev['fill_return'] - cost) / (np.sqrt(5)*ev['sigma'])
oos = ev['date'] >= '2022-01-01'
print("n events", len(ev), "n weeks", ev['week'].nunique(), "events/week", len(ev)/ev['week'].nunique())
print("events per day with >1 coin same side:", (ev.groupby(['event_bar','side']).size()>1).mean())
# primary signal performance
for name, m in [('all', slice(None)), ('OOS>=2022', oos)]:
    e = ev[m]
    for s in [1.0, -1.0, None]:
        x = e if s is None else e[e.side == s]
        print(f"{name:10s} side={s}: n={len(x)} hit={x.target.mean():.3f} meanR={x.R.mean():+.3f} "
              f"t_naive={x.R.mean()/x.R.std()*np.sqrt(len(x)):+.2f} weeks={x.week.nunique()}")
def cluster_boot_ic(a, b, clusters, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'a': a, 'b': b, 'c': clusters}).dropna()
    ic = stats.spearmanr(df.a, df.b).statistic
    groups = [g.index.to_numpy() for _, g in df.groupby('c')]
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        d = df.loc[idx]; vals.append(stats.spearmanr(d.a, d.b).statistic)
    return ic, np.std(vals)
rows = []
for col in f.columns:
    if col == 'side': continue
    raw = f[col].to_numpy(); signed = raw * ev['side'].to_numpy()
    ic_r, se_r = cluster_boot_ic(raw, ev.target.to_numpy().astype(float), ev.week.to_numpy(), n=300)
    ic_s, se_s = cluster_boot_ic(signed, ev.target.to_numpy().astype(float), ev.week.to_numpy(), n=300)
    rows.append((col, ic_r, ic_r/se_r, ic_s, ic_s/se_s))
tab = pd.DataFrame(rows, columns=['feature','IC_raw','t_raw','IC_signed','t_signed']).set_index('feature')
print(tab.round(3).sort_values('t_signed'))
print("side itself: IC", stats.spearmanr(ev.side, ev.target).statistic)
