"""Grid-tune DP-FedAvg (B, E, T) on a VALIDATION split carved from training data (seeds 0-7).
NOTE: the privacy cost of these tuning runs is NOT accounted -- this baseline is optimistic."""
import itertools, json, sys, numpy as np
from bavfl.data import load_sites, split_sites
from bavfl.models import make_model
from bavfl.fl import run_fl
eps_list=[float(e) for e in sys.argv[1].split(',')]
sites=load_sites(); out={}
for eps in eps_list:
    rows=[]
    for B,E,T in itertools.product([8,16,32,64,128],[1,2],[10,20,40]):
        v=[]
        for sd in range(8):
            tr,_=split_sites(sites,sd)
            tr2,va=split_sites(tr,sd+1000,test_frac=0.25)
            v.append(run_fl(make_model('logreg',17),tr2,va,{"dp":"fixed"},sd,eps=eps,B=B,E=E,T=T)['site_auc'])
        rows.append((float(np.mean(v)),B,E,T))
    best=max(rows); out[str(eps)]=dict(val_site_auc=best[0],B=best[1],E=best[2],T=best[3])
    print(eps,out[str(eps)],flush=True)
json.dump(out,open('results/tuned_baseline.json','w'),indent=1)
