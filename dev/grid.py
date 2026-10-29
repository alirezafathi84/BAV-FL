import sys, json, itertools, numpy as np
from bavfl.data import load_sites, split_sites
from bavfl.models import make_model
from bavfl.fl import run_fl
eps=float(sys.argv[1]); seeds=range(int(sys.argv[2])); model=sys.argv[3] if len(sys.argv)>3 else 'logreg'
sites=load_sites(); out=[]
for B,E,T in itertools.product([8,16,32,64],[1,2],[10,20,40]):
    r=[run_fl(make_model(model,17),*split_sites(sites,sd),{"dp":"fixed"},sd,eps=eps,B=B,E=E,T=T) for sd in seeds]
    out.append(dict(B=B,E=E,T=T,site_auc=float(np.mean([x['site_auc'] for x in r])),auc=float(np.mean([x['auc'] for x in r]))))
    print(out[-1],flush=True)
json.dump(out,open(f'results/dev/grid_{model}_{eps}.json','w'))
