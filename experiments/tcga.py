"""Second real multi-site benchmark: Fed-TCGA-BRCA (6 regions), survival (C-index).
Same protocol as the heart study: oracle tuned on validation of seeds 0-7; evaluation on fresh seeds 100-119."""
import itertools, json, math, os, sys, time, numpy as np
from bavfl.data_tcga import load_sites, split_sites
from bavfl.models import make_model
from bavfl.fl import run_fl, run_centralized, run_local
from bavfl.planner import plan
from .common import SEEDS, DELTA
LR, DIN = 0.2, 39; OUT = 'results/tcga/utility.jsonl'; EPS = [0.5, 1.0, 2.0, 4.0, 8.0]
S = load_sites(); M = lambda: make_model('expph', DIN)
def log(r):
    with open(OUT, 'a') as f: f.write(json.dumps(r) + '\n')
t0 = time.time()
tr0, _ = split_sites(S, SEEDS[0]); n = {s: len(v[1]) for s, v in tr0.items()}
plans = {}
for e in EPS:
    for Dm in (1.0, 0.5, 2.0):
        plans[(e, Dm)] = plan(n, eps=e, delta=DELTA, d=DIN + 1, lr=LR, D=Dm * math.sqrt(DIN + 1))
json.dump({f'{e}|{Dm}': p for (e, Dm), p in plans.items()}, open('results/tcga/plans.json', 'w'), indent=1)
print('plans done', time.time() - t0, flush=True)
tuned = {}
for e in EPS:
    rows = []
    for B, E, T in itertools.product([8, 16, 32, 64, 128], [1, 2], [10, 20, 40]):
        v = []
        for sd in range(8):
            tr, _ = split_sites(S, sd); tr2, va = split_sites(tr, sd + 1000, test_frac=0.25)
            v.append(run_fl(M(), tr2, va, {"dp": "fixed"}, sd, eps=e, B=B, E=E, T=T, lr=LR)['site_auc'])
        rows.append((float(np.nanmean(v)), B, E, T))
    b = max(rows); tuned[e] = dict(B=b[1], E=b[2], T=b[3], val=b[0]); print('tuned', e, tuned[e], time.time() - t0, flush=True)
json.dump({str(k): v for k, v in tuned.items()}, open('results/tcga/tuned.json', 'w'), indent=1)
for sd in SEEDS:
    tr, te = split_sites(S, sd)
    for name, fn in [('fedavg', lambda: run_fl(M(), tr, te, {}, sd, lr=LR)), ('central', lambda: run_centralized(M(), tr, te, sd, lr=LR)),
                     ('local', lambda: run_local(M(), tr, te, sd, lr=LR))]:
        r = fn(); log(dict(method=name, eps=1.0, seed=sd, auc=r['auc'], site_auc=r['site_auc']))
for e in EPS:
    for sd in SEEDS:
        tr, te = split_sites(S, sd); kw = dict(eps=e, delta=DELTA, lr=LR)
        runs = {'dp_default': ({"dp": "fixed"}, {}), 'dp_fedprox': ({"dp": "fixed", "prox_mu": 0.1}, {}),
                'dp_decay': ({"dp": "decay"}, {}),
                'dp_tuned': ({"dp": "fixed"}, dict(B=tuned[e]['B'], E=tuned[e]['E'], T=tuned[e]['T']))}
        for Dm, nm in [(1.0, 'bavfl'), (0.5, 'bavfl_Dhalf'), (2.0, 'bavfl_D2')]:
            P = plans[(e, Dm)]; runs[nm] = ({"dp": "fixed", "secagg": True, "verify": "coupled"}, dict(T=P['T'], site_cfg=P['sites']))
        for nm, (cfg, extra) in runs.items():
            r = run_fl(M(), tr, te, cfg, sd, **kw, **extra)
            log(dict(method=nm, eps=e, seed=sd, auc=r['auc'], site_auc=r['site_auc'], eps_spent=r['eps_spent'], det=r['det']))
        r = run_centralized(M(), tr, te, sd, eps=e, delta=DELTA, lr=LR)
        log(dict(method='central_dp', eps=e, seed=sd, auc=r['auc'], site_auc=r['site_auc']))
    print('eps', e, 'done', time.time() - t0, flush=True)
print('ALLDONE', flush=True)
