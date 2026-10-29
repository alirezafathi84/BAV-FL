import json, numpy as np, pandas as pd
from .analyze import load, compare
df = load(); S = {}
def ms(x): return f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}"
S['util'] = {m: {str(e): {k: ms(g[k]) for k in ['site_auc', 'auc', 'bacc']} for e, g in df[df.method == m].groupby('eps')}
             for m in df.method.unique()}
S['stats'] = compare(df, others=['dp_default', 'dp_fedprox', 'dp_decay', 'dp_tuned', 'central_dp']).round(4).to_dict('records')
S['stats_auc'] = compare(df, metric='auc', others=['dp_default', 'dp_fedprox', 'dp_decay', 'dp_tuned', 'central_dp']).round(4).to_dict('records')
A = pd.DataFrame([json.loads(l) for l in open('results/attacks.jsonl')])
A['tpr'] = A.det.apply(lambda d: d['tp'] / max(d['tp'] + d['fn'], 1)); A['fpr'] = A.det.apply(lambda d: d['fp'] / max(d['fp'] + d['tn'], 1))
S['attacks'] = {f"{a}|{d}": dict(site_auc=ms(g.site_auc), tpr=round(g.tpr.mean(), 3), fpr=round(g.fpr.mean(), 4))
                for (a, d), g in A.groupby(['attack', 'defense'])}
S['privacy'] = json.load(open('results/privacy_attacks.json'))
S['plans'] = json.load(open('results/plans.json'))
S['tuned'] = json.load(open('results/tuned_baseline.json'))
S['secagg_ms'] = float(df.secagg_ms.dropna().mean())
S['honest_fp'] = int(sum(d['fp'] for d in df[df.method.str.startswith('bavfl')].det)); S['honest_n'] = int(sum(d['fp'] + d['tn'] for d in df[df.method.str.startswith('bavfl')].det))
json.dump(S, open('results/summary.json', 'w'), indent=1)
for m in ['fedavg', 'central', 'local']: print(m, S['util'][m])
for m in ['bavfl', 'dp_default', 'dp_tuned', 'bavfl_Dhalf', 'bavfl_D2']: print(m, {e: v['site_auc'] for e, v in S['util'][m].items()})
print(S['honest_fp'], S['honest_n'])

# ---- second dataset
T_ = load('results/tcga/utility.jsonl')
S2 = {'util': {m: {str(e): {k: ms(g[k]) for k in ['site_auc', 'auc']} for e, g in T_[T_.method == m].groupby('eps')} for m in T_.method.unique()},
      'stats': compare(T_, others=['dp_default', 'dp_fedprox', 'dp_decay', 'dp_tuned', 'central_dp']).round(4).to_dict('records'),
      'plans': json.load(open('results/tcga/plans.json')), 'tuned': json.load(open('results/tcga/tuned.json')),
      'honest_fp': int(sum(d['fp'] for d in T_[T_.method.str.startswith('bavfl')].det)),
      'honest_n': int(sum(d['fp'] + d['tn'] for d in T_[T_.method.str.startswith('bavfl')].det))}
json.dump(S2, open('results/tcga/summary.json', 'w'), indent=1)
