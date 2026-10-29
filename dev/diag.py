import numpy as np, math
from bavfl.data import load_sites, split_sites
from bavfl.models import make_model
from bavfl import fl
sites=load_sites(); m=make_model('logreg',17)
# per-sample grad norms at a reasonable model
tr,te=split_sites(sites,0)
w=fl.run_fl(m,tr,te,{},0)['w']
X=np.vstack([v[0] for v in tr.values()]); y=np.concatenate([v[1] for v in tr.values()])
g=np.linalg.norm(m.per_sample_grad(w,X,y),axis=1); print('grad norm quantiles at FedAvg optimum', np.round(np.quantile(g,[.5,.75,.9,.99]),3))
g0=np.linalg.norm(m.per_sample_grad(np.zeros(18),X,y),axis=1); print('grad norm quantiles at init', np.round(np.quantile(g0,[.5,.75,.9,.99]),3))
# clip-only vs noise
orig=fl.ALLOCATORS
class Tiny:
    def __init__(s,*a,**k): s.h=[]
    def next_sigma(s,t,h): return 1e-6
    def spent(s): return float('inf')
fl.ALLOCATORS={**orig,'tiny':Tiny}
for C in [0.1,0.3,1.0,3.0]:
  for name,cfg,eps in [('clip-only',{'dp':'tiny'},1),('dp eps1',{'dp':'fixed'},1),('dp eps3',{'dp':'fixed'},3)]:
    r=[fl.run_fl(m,*split_sites(sites,sd),cfg,sd,eps=eps,C=C)['site_auc'] for sd in range(8)]
    print(f'C={C} {name:9s} siteAUC={np.mean(r):.4f}')
for lr in [0.1,0.25,1.0]:
    r=[fl.run_fl(m,*split_sites(sites,sd),{'dp':'fixed'},sd,eps=1,lr=lr)['site_auc'] for sd in range(8)]
    r2=[fl.run_fl(m,*split_sites(sites,sd),{},sd,lr=lr)['site_auc'] for sd in range(8)]
    print(f'lr={lr} dp1 {np.mean(r):.4f} fedavg {np.mean(r2):.4f}')
