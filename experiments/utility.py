"""Main utility benchmark on fresh seeds. Usage: python -m experiments.utility <methods,...> <eps,...>"""
import json, os, sys, time, numpy as np
from bavfl.data import load_sites, split_sites
from bavfl.models import make_model
from bavfl.fl import run_fl, run_centralized, run_local
from .common import SEEDS, DELTA, get_plan
methods = sys.argv[1].split(','); eps_list = [float(e) for e in sys.argv[2].split(',')]
OUT = 'results/utility.jsonl'
sites = load_sites()
def log(row):
    with open(OUT, 'a') as f: f.write(json.dumps(row) + '\n')
t0 = time.time()
for eps in eps_list:
    for meth in methods:
        kw = dict(eps=eps, delta=DELTA); cfg = None
        if meth in ('dp_default',): cfg = {"dp": "fixed"}
        elif meth == 'dp_fedprox': cfg = {"dp": "fixed", "prox_mu": 0.1}
        elif meth == 'dp_decay': cfg = {"dp": "decay"}
        elif meth == 'dp_tuned':
            tb = json.load(open('results/tuned_baseline.json'))[str(eps)]
            cfg = {"dp": "fixed"}; kw.update(B=tb['B'], E=tb['E'], T=tb['T'])
        elif meth.startswith('bavfl'):
            Dm = {'bavfl': 1.0, 'bavfl_Dhalf': 0.5, 'bavfl_D2': 2.0}[meth]
            P = get_plan(eps, Dm)
            cfg = {"dp": "fixed", "secagg": True, "verify": "coupled"}
            kw.update(T=P['T'], site_cfg=P['sites'])
        elif meth == 'bavfl_noSA':
            P = get_plan(eps); cfg = {"dp": "fixed"}; kw.update(T=P['T'], site_cfg=P['sites'])
        for sd in SEEDS:
            tr, te = split_sites(sites, sd); m = make_model('logreg', 17)
            if meth == 'fedavg': r = run_fl(m, tr, te, {}, sd)
            elif meth == 'central': r = run_centralized(m, tr, te, sd)
            elif meth == 'central_dp': r = run_centralized(m, tr, te, sd, eps=eps, delta=DELTA)
            elif meth == 'local': r = run_local(m, tr, te, sd)
            else: r = run_fl(m, tr, te, cfg, sd, **kw)
            log(dict(method=meth, eps=eps, seed=sd, auc=r['auc'], bacc=r['bacc'], site_auc=r['site_auc'],
                     eps_spent=r.get('eps_spent'), secagg_ms=r.get('secagg_ms'), det=r.get('det')))
        print(f'{meth} eps={eps} done t={time.time()-t0:.0f}s', flush=True)
