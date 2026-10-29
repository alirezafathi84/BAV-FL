import sys, numpy as np
from bavfl.data import load_sites, split_sites
from bavfl.models import make_model
from bavfl.fl import run_fl
from experiments.common import get_plan
P=get_plan(2.0); sites=load_sites(); seeds=range(5)   # development seeds
for v,k in [("static_vn",None),("rms_vn",1.5),("rms_vn",2.0),("rms_vn",3.0),("rms",2.0)]:
    out=[]
    for att in [None,"bounded_signflip","signflip_scaled"]:
        a=[];fp=0;n=0;tp=0;m=0
        for sd in seeds:
            tr,te=split_sites(sites,sd)
            r=run_fl(make_model('logreg',17),tr,te,{"dp":"fixed","secagg":True,"verify":v,"kappa":k},sd,eps=2.0,T=P['T'],site_cfg=P['sites'],malicious=() if att is None else ("cleveland",),attack=att)
            a.append(r['site_auc']); fp+=r['det']['fp']; n+=r['det']['fp']+r['det']['tn']; tp+=r['det']['tp']; m+=r['det']['tp']+r['det']['fn']
        out.append(f"{att}: auc={np.mean(a):.3f} fpr={fp/n:.3f} tpr={tp/max(m,1):.2f}")
    print(v,k,' | '.join(out),flush=True)
