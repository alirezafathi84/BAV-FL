"""Poisoning experiments: one malicious hospital (Cleveland, largest weight) at eps=2."""
import json, sys, time
from bavfl.data import load_sites, split_sites
from bavfl.models import make_model
from bavfl.fl import run_fl
from .common import SEEDS, DELTA, get_plan
EPS = 2.0; OUT = 'results/attacks.jsonl'
DEF = {"no_defense_SA": {"dp": "fixed", "secagg": True},
       "static_bound_SA": {"dp": "fixed", "secagg": True, "verify": "static"},
       "bavfl_coupled_SA": {"dp": "fixed", "secagg": True, "verify": "coupled"},
       "static_vn_SA": {"dp": "fixed", "secagg": True, "verify": "static_vn"},
       "coord_median": {"dp": "fixed", "robust": "median"},
       "trimmed_mean": {"dp": "fixed", "robust": "trimmed"}}
ATT = [None, "signflip_scaled", "gaussian", "labelflip", "bounded_signflip"]
sites = load_sites(); P = get_plan(EPS); t0 = time.time()
for att in ATT:
    for dname, cfg in DEF.items():
        for sd in SEEDS:
            tr, te = split_sites(sites, sd)
            r = run_fl(make_model('logreg', 17), tr, te, cfg, sd, eps=EPS, delta=DELTA, T=P['T'],
                       site_cfg=P['sites'], malicious=() if att is None else ("cleveland",), attack=att)
            with open(OUT, 'a') as f:
                f.write(json.dumps(dict(attack=str(att), defense=dname, seed=sd, site_auc=r['site_auc'],
                                        auc=r['auc'], bacc=r['bacc'], det=r['det'])) + '\n')
        print(att, dname, f'{time.time()-t0:.0f}s', flush=True)
