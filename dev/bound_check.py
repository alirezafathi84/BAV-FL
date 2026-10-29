import json, math, numpy as np
from scipy.stats import spearmanr
from bavfl.data import load_sites, split_sites
from bavfl.privacy import FixedAllocator
sites=load_sites(); tr,_=split_sites(sites,0)
n={s:len(v[1]) for s,v in tr.items()}; N=sum(n.values()); p={s:n[s]/N for s in n}
d=18; lr=0.5; C=1
for eps in [1.0,3.0]:
    g=json.load(open(f'results/dev/grid_logreg_{eps}.json'))
    for c in g:
        c['sig']={}
        for s in n:
            B,E,T=c['B'],c['E'],c['T']; q=min(B/n[s],1); S=E*math.ceil(n[s]/B)
            c['sig'][s]=(FixedAllocator(q,S,T,eps,1e-4).sigma, S*T, min(B,n[s]))
    y=[c['site_auc'] for c in g]
    for D in [18**0.5]:
        phi=[sum(p[s]*(D**2/(2*lr*St)+lr/2*(C**2+d*(sg*C/Be)**2)) for s,(sg,St,Be) in c['sig'].items()) for c in g]
        best=g[int(np.argmin(phi))]
        print(f"eps={eps} D={D}: spearman(-bound,siteAUC)={spearmanr(-np.array(phi),y)[0]:+.3f}  planner pick B{best['B']} E{best['E']} T{best['T']} -> {best['site_auc']:.4f} (grid best {max(y):.4f}, median {np.median(y):.4f})")
