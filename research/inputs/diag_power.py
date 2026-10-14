import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats
ev = pd.read_parquet('/home/claude/research/events.parquet')
p = pd.read_parquet('/home/claude/research/oos_ensemble.parquet')
grid = pd.date_range('2020-03-26', '2026-06-23', freq='D', tz='UTC')
d = p.merge(ev[['event_bar','symbol','side']], left_on='row', right_index=True)
d['week'] = grid[d.event_bar.to_numpy()].tz_localize(None).to_period('W').astype(str)
rng = np.random.default_rng(1)
weeks = d.week.unique(); g = {w: d.index[d.week == w].to_numpy() for w in weeks}
auc = roc_auc_score(d.y, d.p)
boots = []
for _ in range(2000):
    idx = np.concatenate([g[w] for w in rng.choice(weeks, len(weeks))])
    boots.append(roc_auc_score(d.y.loc[idx], d.p.loc[idx]))
se_c = np.std(boots)
n1, n0 = d.y.sum(), (1-d.y).sum()
# Hanley-McNeil SE at AUC=0.5 (iid)
se_iid = np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
print(f"OOS n={len(d)} weeks={len(weeks)} AUC(unweighted)={auc:.4f}")
print(f"SE iid={se_iid:.4f}  SE week-cluster bootstrap={se_c:.4f}  design effect={(se_c/se_iid)**2:.2f}  n_eff~{len(d)/(se_c/se_iid)**2:.0f}")
print(f"95% CI cluster: [{np.quantile(boots,0.025):.3f}, {np.quantile(boots,0.975):.3f}]")
# Minimum detectable AUC (80% power, 5% two-sided) at this SE
print("MDE AUC at this design:", 0.5 + (1.96+0.84)*se_c)
# Label-mixing: 2 coins same day same side -> correlated outcomes
d2 = ev.copy(); d2['wk']=grid[d2.event_bar.to_numpy()].tz_localize(None).to_period('W').astype(str)
same = d2.groupby(['event_bar','side']).target.agg(['size','mean'])
print("share of events sharing day+side with another coin:", (same['size'][same['size']>1].sum())/len(d2))
# within-day same-side outcome concordance
pairs = []
for (bar, side), grp in d2.groupby(['event_bar','side']):
    t = grp.target.to_numpy()
    if len(t) > 1:
        for i in range(len(t)):
            for j in range(i+1, len(t)): pairs.append(t[i] == t[j])
print("outcome concordance of same-day same-side events:", np.mean(pairs), "n pairs", len(pairs))
