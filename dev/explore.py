import sys, json, time, numpy as np
from bavfl.data import load_sites, split_sites
from bavfl.models import make_model
from bavfl.fl import run_fl
model_name=sys.argv[1]; eps_list=[float(e) for e in sys.argv[2].split(',')]; seeds=range(int(sys.argv[3]))
extra=json.loads(sys.argv[4]) if len(sys.argv)>4 else {}
sites=load_sites()
M=json.loads(sys.argv[5]) if len(sys.argv)>5 else {"fedavg":{}, "fixed":{"dp":"fixed"}, "decay":{"dp":"decay"}, "bona":{"dp":"bona"}}
res={}
t0=time.time()
for eps in eps_list:
  for name,cfg in M.items():
    if name=="fedavg" and eps!=eps_list[0]: continue
    a=[];sa=[]
    for sd in seeds:
        tr,te=split_sites(sites,sd); m=make_model(model_name,tr["cleveland"][0].shape[1])
        r=run_fl(m,tr,te,cfg,sd,eps=eps,**extra); a.append(r['auc']); sa.append(r['site_auc'])
    res[(eps,name)]=(a,sa)
    print(f"eps={eps} {name:7s} auc={np.mean(a):.4f}±{np.std(a):.4f} siteAUC={np.mean(sa):.4f}±{np.std(sa):.4f}  t={time.time()-t0:.0f}s",flush=True)
names=[k for k in M if k!="fedavg"]
for eps in eps_list:
    base=np.array(res[(eps,names[0])][1])
    for k in names[1:]:
        a=np.array(res[(eps,k)][1]); print(f"eps={eps} {k} - {names[0]}: {np.mean(a-base):+.4f} wins {np.mean(a>base):.2f}")
